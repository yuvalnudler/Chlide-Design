"""Execute a stage list.

What this is defending against, in order of how much each has cost:

* **Lying about the outcome.** A pipeline that reports success because its last stage ran is
  worse than one that crashes. The figures stage runs regardless, so it once sent a Telegram
  saying DONE after the cascade had been SIGKILLed at 2770/11201. Every stage's outcome is
  recorded, and the summary is computed from all of them.
* **Running a stage whose inputs are absent.** `validate_order` catches a bad ORDER at import
  time; this catches a run where an earlier stage was skipped or failed, so the artifact is
  genuinely not there. That is BLOCKED, not FAILED, and the two mean different things.
* **Silently continuing past a spine failure.** An optional scoring stage may fail and narrow
  the ranking. A stage that PRODUCES the ranking may not.
* **Taking the whole machine.** This is a shared box. The worker count is sized from OTHER
  users' load, never from total cores, and GPU stages refuse unless explicitly allowed.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from chlide.stages.base import Outcome, Stage, StageResult, run_dir_has, summarise

__all__ = ['run_stages', 'StageRunner']

TOOLS = Path(__file__).resolve().parents[2] / 'tools'


class StageRunner:
    def __init__(self, run: Path, *, python: str | None = None, allow_gpu: bool = False,
                 reserve_cpus: int = 16, dry_run: bool = False,
                 on_event: Callable[[dict], None] | None = None):
        self.run = Path(run).resolve()
        self.python = python or os.environ.get(
            'CHLIDE_PYTHON', '/home/yigall/mambaforge/envs/lasermpnn/bin/python')
        self.allow_gpu = allow_gpu
        self.reserve_cpus = reserve_cpus
        self.dry_run = dry_run
        self.on_event = on_event or (lambda e: None)

    # ---------------------------------------------------------------- helpers

    def workers(self) -> int:
        """Worker count sized from OTHER users' load.

        Counting our own load collapsed the pool to 4 workers and a 14 h cascade: the cascade
        itself was the load being measured. Never fewer than 1, never more than what is left
        after the reserve.
        """
        total = os.cpu_count() or 8
        me = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
        other = 0.0
        try:
            out = subprocess.run(['ps', '-eo', 'user,pcpu', '--no-headers'],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] != me:
                    other += float(parts[1])
        except (OSError, ValueError, subprocess.SubprocessError):
            other = 0.0
        free = total - int(other / 100) - self.reserve_cpus
        return max(1, free)

    def _inputs_present(self, stage: Stage) -> list[str]:
        return [p for p in stage.reads if not run_dir_has(self.run, p)]

    # ---------------------------------------------------------------- one stage

    def run_one(self, stage: Stage, enabled: bool = True) -> StageResult:
        if not enabled:
            # A disabled stage contributes NOTHING -- it does not write an empty file that a
            # downstream stage would then rank on.
            return StageResult(stage.name, Outcome.SKIPPED, detail='disabled')

        if stage.needs_gpu and not self.allow_gpu:
            return StageResult(stage.name, Outcome.SKIPPED,
                               detail='GPU stage; pass --allow-gpu on a shared box')

        missing = self._inputs_present(stage)
        if missing:
            return StageResult(stage.name, Outcome.BLOCKED,
                               detail=f'inputs absent: {", ".join(missing[:2])}')

        tool = TOOLS / f'{stage.tool}.py'
        if not tool.is_file():
            return StageResult(stage.name, Outcome.FAILED, detail=f'no tools/{stage.tool}.py')

        cmd = [self.python, str(tool), '--run-dir', str(self.run), *stage.args]
        if self.dry_run:
            return StageResult(stage.name, Outcome.SKIPPED, detail='dry run: ' + ' '.join(cmd))

        self.on_event({'event': 'stage_start', 'stage': stage.name, 'tool': stage.tool,
                       'cmd': cmd, 'gpu': stage.needs_gpu})
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.run))
        except OSError as e:
            return StageResult(stage.name, Outcome.FAILED, time.time() - t0,
                               detail=f'{type(e).__name__}: {e}')
        dt = time.time() - t0

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()
            detail = f'exit {proc.returncode}' + (f': {tail[-1][:120]}' if tail else '')
            outcome = Outcome.FAILED
        else:
            # Exit 0 is not proof. A stage that was supposed to write something and did not
            # has failed, whatever it returned.
            unwritten = [p for p in stage.writes if not run_dir_has(self.run, p)]
            if unwritten:
                outcome = Outcome.FAILED
                detail = f'exit 0 but did not write {unwritten[0]}'
            else:
                outcome = Outcome.OK
                detail = ''

        produced = [p for p in stage.writes if run_dir_has(self.run, p)]
        res = StageResult(stage.name, outcome, dt, detail, produced)
        self.on_event({'event': 'stage_end', 'stage': stage.name, 'outcome': outcome.value,
                       'seconds': round(dt, 1), 'detail': detail})
        return res

    # ---------------------------------------------------------------- the list

    def run_all(self, stages: Sequence[Stage],
                enabled: dict[str, bool] | None = None) -> list[StageResult]:
        enabled = enabled or {}
        results: list[StageResult] = []
        for s in stages:
            res = self.run_one(s, enabled.get(s.name, True))
            results.append(res)
            print(f'  {res}', flush=True)
            if res.outcome in (Outcome.FAILED, Outcome.BLOCKED) and not s.optional:
                # A spine failure stops the run. Continuing would produce a ranking built on
                # an artifact that is missing or stale, which is the failure mode that makes a
                # broken pipeline look like a working one.
                results.append(StageResult(
                    '(halted)', Outcome.BLOCKED,
                    detail=f'{s.name} is not optional and did not succeed'))
                break
        return results


def run_stages(run: Path, stages: Sequence[Stage], **kw) -> tuple[list[StageResult], str]:
    r = StageRunner(run, **kw)
    results = r.run_all(stages)
    return results, summarise(results)
