"""Shared-box GPU etiquette.

This machine is shared. Three mechanisms, each from an incident:

* **Hard-fail on a foreign process.** An earlier launcher only WARNED, and a warning scrolls
  past: four GPUs got double-booked. Refusing is the only thing that works.
* **A lock with a PID liveness check and a TTL.** A plain lock file outlives the process that
  made it; after a reboot it blocks every future run until someone deletes it by hand.
* **Never claim every card.** The default leaves the last one, because the other user on this
  box runs AMBER and MD at 100% on a shared card slows both jobs and can OOM theirs.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ['GpuError', 'gpu_processes', 'foreign_processes', 'assert_gpus_free', 'GpuLock']

LOCK_DIR = Path(os.environ.get('CHLIDE_LOCK_DIR', '/tmp/chlide-gpu-locks'))
LOCK_TTL = 24 * 3600


class GpuError(RuntimeError):
    """A GPU is not available, and we are not taking it anyway."""


@dataclass(frozen=True)
class GpuProc:
    index: str
    pid: int
    user: str
    memory: str
    name: str


def _owner(pid: int) -> str:
    try:
        r = subprocess.run(['ps', '-o', 'user=', '-p', str(pid)],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or '?'
    except (OSError, subprocess.SubprocessError):
        return '?'


def gpu_processes() -> list[GpuProc]:
    """Every compute process on every card. Empty if nvidia-smi is unavailable."""
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return []
        uuid_to_index = {}
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
        for line in q.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2:
                uuid_to_index[parts[1]] = parts[0]
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            pid = int(parts[1])
            out.append(GpuProc(uuid_to_index.get(parts[0], '?'), pid, _owner(pid),
                               parts[2], parts[3]))
        return out
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def foreign_processes(devices: list[str] | None = None) -> list[GpuProc]:
    """Compute processes on `devices` that are not ours."""
    me = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
    want = None
    if devices:
        want = {d.split(':')[-1] for d in devices}
    return [p for p in gpu_processes()
            if p.user != me and (want is None or p.index in want)]


def assert_gpus_free(devices: list[str], *, allow_foreign: bool = False) -> None:
    """Refuse to start if someone else is on a card we want.

    Naming the user and the process matters: the point is that you go and ask them, not that
    you wait and retry into their job.
    """
    foreign = foreign_processes(devices)
    if foreign and not allow_foreign:
        lines = [f'  GPU {p.index}: {p.user} pid {p.pid} {p.name} ({p.memory})' for p in foreign]
        raise GpuError(
            'refusing to start -- another user has compute processes on the requested GPUs:\n'
            + '\n'.join(lines)
            + '\nTalk to them, choose other devices, or pass --allow-foreign-gpu if you have '
              'already agreed to share.')


class GpuLock:
    """O_EXCL lock with a PID liveness check and a TTL.

    A stale lock from a killed process or a reboot is reclaimed, because the alternative is a
    file nobody understands blocking every run until it is deleted by hand.
    """

    def __init__(self, devices: list[str], lock_dir: Path = LOCK_DIR, ttl: int = LOCK_TTL):
        self.devices = list(devices)
        self.lock_dir = Path(lock_dir)
        self.ttl = ttl
        self.held: list[Path] = []

    def _stale(self, path: Path) -> bool:
        try:
            body = path.read_text().strip().split()
            pid = int(body[0])
            age = time.time() - path.stat().st_mtime
        except (OSError, ValueError, IndexError):
            return True
        if age > self.ttl:
            return True
        try:
            os.kill(pid, 0)          # liveness only; signal 0 does not touch the process
        except ProcessLookupError:
            return True
        except PermissionError:
            return False             # exists, owned by someone else
        return False

    def acquire(self) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        for dev in self.devices:
            path = self.lock_dir / f'{dev.replace(":", "_")}.lock'
            if path.exists() and self._stale(path):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self.release()
                raise GpuError(f'{dev} is locked by a live process: {path.read_text().strip()}')
            with os.fdopen(fd, 'w') as fh:
                fh.write(f'{os.getpid()} {os.environ.get("USER", "?")} '
                         f'{time.strftime("%Y-%m-%dT%H:%M:%S")}\n')
            self.held.append(path)

    def release(self) -> None:
        for path in self.held:
            try:
                path.unlink()
            except OSError:
                pass
        self.held = []

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False
