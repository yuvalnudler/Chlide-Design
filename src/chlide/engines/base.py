"""One interface over Boltz-2x and AlphaFold3.

Both engines already emit the SAME on-disk layout -- `af3_adapter.to_boltz_layout` makes AF3
write into `boltz_results_boltz_inputs/predictions/<stem>/`, which is why swapping engines
changed nothing downstream. This package puts that behind a declared protocol instead of
behind two forked copies of a 1200-line driver.

What this is NOT: an abstraction that hides the differences. The engines disagree in ways that
change results, and a layer that smoothed them over would be lying. Those differences live on
the engine objects as DECLARED capabilities (`supports_chirality_restraints`,
`supports_metal_distance_restraint`) and are documented in docs/ENGINES.md, so a caller can
branch on them deliberately rather than discovering them in a bad structure three days later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

__all__ = ['FoldResult', 'LigandSpec', 'Folder', 'EngineError']


class EngineError(RuntimeError):
    """An engine was asked for something it cannot do."""


@dataclass(frozen=True)
class LigandSpec:
    """What an engine needs to place the ligand. Mirrors LigandConfig, minus run paths."""

    code: str                       # CCD code, e.g. CLAOX
    user_ccd: Path                  # the CIF/pickle carrying our own component
    axial_his_resnum: int | None = None   # 1-based RESIDUE NUMBER, engine-facing


@dataclass
class FoldResult:
    """What came back. `root` is in Boltz layout whichever engine produced it."""

    engine: str
    root: Path
    n_ok: int = 0
    n_failed: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_ok + self.n_failed

    def __str__(self) -> str:
        rate = f'{self.seconds / self.n_ok:.1f} s/design' if self.n_ok else 'n/a'
        return (f'{self.engine}: {self.n_ok} ok, {self.n_failed} failed, '
                f'{self.seconds:.0f} s ({rate})')


@runtime_checkable
class Folder(Protocol):
    """A structure predictor the NISE loop can drive."""

    name: str
    input_suffix: str                       # '.yaml' for Boltz, '.json' for AF3

    # Declared, not inferred. See docs/ENGINES.md for the measurements behind each.
    supports_chirality_restraints: bool     # Boltz: 12 ChiralAtomPotential constraints
    supports_metal_distance_restraint: bool  # Boltz only, and only PATCHED (two-sided window)

    def write_input(self, dest_dir: Path, stem: str, sequence: str,
                    ligand: LigandSpec, *, constrain_metal: bool = False) -> Path:
        """Write ONE job file into dest_dir. Returns the path written."""
        ...

    def fold(self, inputs_dir: Path, out_dir: Path, devices: Sequence[str],
             ligand: LigandSpec, *, debug: bool = False) -> FoldResult:
        """Fold every job file under inputs_dir, emitting Boltz-layout output."""
        ...


def check_metal_restraint(folder: Folder, constrain_metal: bool) -> None:
    """Refuse a metal restraint on an engine that cannot express one safely.

    AF3's `bondedAtomPairs` has no distance window: it crushes His NE2 -- Mg to 1.50 A and
    flips the 13-2 stereocentre to R. Measured, then shipped on by default anyway in an early
    v8 build -- a self-test caught it. This makes the refusal structural.
    """
    if constrain_metal and not folder.supports_metal_distance_restraint:
        raise EngineError(
            f'{folder.name} cannot restrain the metal distance. AF3 bondedAtomPairs has no '
            'distance window: it crushes His-Mg to 1.50 A and flips 13-2 to R. Let Rosetta '
            'relax form the coordination (4.22 -> 2.50 A) and gate on it instead.')
