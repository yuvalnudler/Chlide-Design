"""Stages, and the ordering rule that makes them safe.

A stage is a named step that reads some artifacts and writes others. Declaring both is the
point: the terminal pipeline once ran `af3_monomer` and `burial_rescore` BEFORE
`final_shortlist`, which produces the `top*.pdb` and `shortlist.csv` they read. On a first
run they failed with "no top*.pdb" and a FileNotFoundError; the campaign before it only
worked because an earlier run had left those files on disk. A stale artifact made a broken
order look correct.

So `reads`/`writes` are not documentation. `validate_order()` checks every stage's inputs are
produced by an earlier stage (or already exist in the run), and it is run as a TEST, so a
mis-ordered pipeline fails in CI rather than at hour three of a terminal run.

Two further rules, both learned the hard way:

* **A disabled stage contributes nothing.** Not a no-op that still writes an empty file --
  nothing. A downstream stage must see "absent", not "present but empty", or it will rank on
  a column of zeros.
* **Report the truth.** A pipeline that reports success because its last stage ran is lying.
  A cascade SIGKILLed at 2770/11201 still printed "DONE ... shortlist.csv" and sent a Telegram
  from the figures stage, which runs regardless, so a run with no rows downstream looked
  finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ['Stage', 'StageResult', 'Outcome', 'validate_order', 'OrderError']


class OrderError(ValueError):
    """A stage reads an artifact nothing earlier produces."""


class Outcome(str, Enum):
    OK = 'ok'
    SKIPPED = 'skipped'          # disabled, or its gate said no: contributes NOTHING
    FAILED = 'failed'            # ran and did not succeed
    BLOCKED = 'blocked'          # could not run: inputs absent


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline.

    `reads` / `writes` are run-relative glob patterns. They are what `validate_order` checks,
    so they must name the artifacts the stage ACTUALLY touches, not a tidy summary.
    """

    name: str
    tool: str                                   # tools/<tool>.py
    args: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    needs_gpu: bool = False
    # A stage whose failure must not stop the pipeline. Scoring passes are optional: a missing
    # column should narrow the ranking, not abort a run that has already cost GPU-days.
    optional: bool = False
    # Resolved per campaign, e.g. the fold check is the engine that did NOT lead.
    engine_dependent: bool = False


@dataclass
class StageResult:
    stage: str
    outcome: Outcome
    seconds: float = 0.0
    detail: str = ''
    produced: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    def __str__(self) -> str:
        return f'{self.stage}: {self.outcome.value}' + (f' ({self.detail})' if self.detail else '')


def validate_order(stages: list[Stage], preexisting: set[str] | None = None) -> None:
    """Every stage's inputs must be produced by an earlier stage, or already be on disk.

    `preexisting` names artifacts the LOOP wrote before the terminal pipeline starts -- the
    sampling dataframes and backbone queues. Anything else must have a producer.
    """
    available = set(preexisting or ())
    for i, s in enumerate(stages):
        for want in s.reads:
            if want in available:
                continue
            producer = next((t.name for t in stages[i + 1:] if want in t.writes), None)
            if producer:
                raise OrderError(
                    f'stage {s.name!r} reads {want!r}, which is written by {producer!r} -- '
                    f'a LATER stage. This is the bug that made af3_monomer and burial_rescore '
                    f'fail on a first run and silently use stale files on a second.')
            raise OrderError(
                f'stage {s.name!r} reads {want!r}, which nothing produces and which is not '
                f'listed as pre-existing.')
        available |= set(s.writes)


def summarise(results: list[StageResult]) -> str:
    """A truthful one-liner. Names what failed; does not claim success for a partial run."""
    ok = [r for r in results if r.outcome is Outcome.OK]
    bad = [r for r in results if r.outcome in (Outcome.FAILED, Outcome.BLOCKED)]
    skipped = [r for r in results if r.outcome is Outcome.SKIPPED]
    head = 'DONE' if not bad else 'INCOMPLETE'
    parts = [f'{head}: {len(ok)}/{len(results)} stages ok']
    if skipped:
        parts.append(f'{len(skipped)} skipped ({", ".join(r.stage for r in skipped)})')
    if bad:
        parts.append(f'FAILED: {", ".join(f"{r.stage} [{r.detail}]" for r in bad)}')
    return ' | '.join(parts)


def run_dir_has(run: Path, pattern: str) -> bool:
    """Does this artifact exist? A pattern with no match is absent, not empty."""
    return any(run.glob(pattern))
