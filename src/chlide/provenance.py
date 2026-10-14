"""What produced this run.

A finished campaign has to stay reproducible after the code moves on. Recording the config is
not enough: the model weights are referenced BY PATH today, so a swapped .pt is currently
undetectable, and a dirty working tree means the commit SHA describes code that was not run.

So the manifest records the SHA **and** the dirty flag, the sha256 of every weights file it can
find, the resolved config, and the environment. Hashing a 2 GB checkpoint takes a few seconds
and happens once per run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

__all__ = ['write_manifest', 'file_sha256', 'git_state']

REPO = Path(__file__).resolve().parents[2]


def file_sha256(path: Path, *, max_bytes: int | None = None) -> str | None:
    """Hash a file in chunks. None if unreadable -- never raises into a run."""
    try:
        h = hashlib.sha256()
        read = 0
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
                read += len(chunk)
                if max_bytes and read >= max_bytes:
                    break
        return h.hexdigest()
    except OSError:
        return None


def git_state(repo: Path = REPO) -> dict:
    """Commit, branch and whether the tree was dirty.

    The dirty flag is the point. A manifest naming a commit that does not contain the code that
    ran is worse than no manifest, because it looks authoritative.
    """
    def g(*args) -> str | None:
        try:
            r = subprocess.run(['git', '-C', str(repo), *args],
                               capture_output=True, text=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = g('status', '--porcelain')
    return {
        'commit': g('rev-parse', 'HEAD'),
        'branch': g('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(status) if status is not None else None,
        'dirty_files': [ln[3:] for ln in (status or '').splitlines()[:20]] or None,
        'remote': g('config', '--get', 'remote.origin.url'),
    }


def _gpu_info() -> list[dict]:
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                out.append({'index': parts[0], 'name': parts[1],
                            'memory_total': parts[2], 'driver': parts[3]})
        return out
    except (OSError, subprocess.SubprocessError):
        return []


def _weights(params: dict) -> dict:
    """Hash every weights file the params point at, plus the AF3 parameter directory.

    Referenced by path today, so without this a swapped checkpoint is invisible.
    """
    out: dict = {}
    ckpt = params.get('model_checkpoint')
    if ckpt:
        p = Path(str(ckpt))
        out['laser_checkpoint'] = {'path': str(p), 'exists': p.is_file(),
                                   'sha256': file_sha256(p) if p.is_file() else None,
                                   'bytes': p.stat().st_size if p.is_file() else None}
    cache = os.environ.get('BOLTZ_CACHE')
    if cache:
        for ck in sorted(Path(cache).glob('*.ckpt'))[:4]:
            out[f'boltz:{ck.name}'] = {'path': str(ck), 'sha256': file_sha256(ck),
                                       'bytes': ck.stat().st_size}
    af3 = Path(os.environ.get('AF3_MODEL_DIR', '/home/data/programs/AF3_PARAMETERS'))
    if af3.is_dir():
        files = sorted(af3.glob('*.bin')) + sorted(af3.glob('*.npz'))
        out['af3_parameters'] = {'path': str(af3), 'n_files': len(files),
                                 'files': [f.name for f in files[:5]]}
    return out


def write_manifest(run: Path, cfg, params: dict, *, campaign_file: Path | None = None,
                   extra: dict | None = None) -> Path:
    """Write runs/<run>/run_manifest.json and the resolved config beside it."""
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)

    manifest = {
        'schema': 1,
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'campaign': cfg.name,
        'campaign_file': str(campaign_file) if campaign_file else None,
        'run_dir': str(run),
        'git': git_state(),
        'engine': {'fold': cfg.engine.fold,
                   'cross_check': cfg.engine.resolved_cross_check,
                   'constrain_his_mg': cfg.engine.constrain_his_mg,
                   'devices': list(cfg.engine.devices)},
        'weights': _weights(params),
        'host': {'hostname': platform.node(), 'python': platform.python_version(),
                 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                 'user': os.environ.get('USER')},
        'gpus': _gpu_info(),
        'env_overrides': {k: v for k, v in sorted(os.environ.items())
                          if k in ('BOLTZ_CACHE', 'BOLTZ_PYTHON', 'ROSETTA_PYTHON', 'NISE_PATH',
                                   'AF3_MODEL_DIR', 'AF3_DB_DIR', 'AF3_RUNNER', 'AF3_PYTHON',
                                   'CHLIDES_MMCIF', 'CHLIDE_PYTHON')},
        'params': {k: _jsonable(v) for k, v in sorted(params.items())},
        'config': cfg.model_dump(mode='json'),
    }
    if extra:
        manifest.update(extra)

    out = run / 'run_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + '\n')
    return out


def _jsonable(v):
    if isinstance(v, (set, frozenset)):
        return sorted(str(x) for x in v)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
