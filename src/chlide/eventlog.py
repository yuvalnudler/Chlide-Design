"""Append-only JSONL event stream, one file per run.

`cascade_partial.jsonl` already proved the pattern survives `kill -9`: append, flush, fsync-free
but line-atomic, so a killed process leaves a truncated LAST line and every earlier line intact.

It never raises. A logger that can take down the run it is logging is worse than no logger --
the same rule the Telegram client follows.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

__all__ = ['EventLog']


class EventLog:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def __call__(self, event: dict) -> None:
        self.emit(**event)

    def emit(self, **fields) -> None:
        rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'run_id': self.run_id, **fields}
        try:
            with self.path.open('a') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
        except (OSError, TypeError, ValueError):
            pass    # never raise into the run

    def read(self) -> list[dict]:
        """Every complete line. A truncated final line from a kill -9 is dropped."""
        out = []
        try:
            for line in self.path.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return out
