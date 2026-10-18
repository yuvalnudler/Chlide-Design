"""Campaign configuration: one validated YAML per campaign.

A campaign used to be a *directory* -- a full copy of its predecessor, driver and
all. v7 and v8 differ in engine, and the only way to express that was to fork the
1200-line driver: a module-level import, a `CONSTRAIN_HIS_MG_BOND` flag and a
replaced `predict_complex_structures`. Their `params` dicts are byte-identical.

Here that difference is two lines of YAML.

Validation exists to reject, at parse time, the specific bugs that have cost this
project real campaign time. Every validator below cites the incident it prevents;
if you remove one, you are re-enabling that bug. See docs/LESSONS.md.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ['CampaignConfig', 'load_campaign', 'resolve_input', 'ConfigError']


class ConfigError(ValueError):
    """Raised when a campaign file is internally inconsistent."""


class _Base(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


# --------------------------------------------------------------------------- ligand


class LigandConfig(_Base):
    """The ligand, its protonation state, and the SMILES that builds Rosetta params."""

    code: str = Field(min_length=1, max_length=5, description='CCD code, e.g. CLAOX')
    component: str = Field(description='data_ block to read from the source mmCIF')
    source: Path
    smiles: str
    deprotonated: bool = Field(
        description='C17 propionate as [O-]. At pH 7 a propionic acid (pKa ~4.8) is '
                    'deprotonated, so this should be True for anything physiological.')
    required_config: Literal['R', 'S'] | None = Field(
        default=None, description='CIP code demanded at the 13-2 stereocentre, or None to not gate')
    stereocentre_atom: str = Field(default='CBD', description='atom whose CIP code is read')

    # RMSD / burial atom sets. POST-protonation names (MG -> MG1, O1A -> OA1, ...).
    rmsd_mask_atoms: frozenset[str] = frozenset()
    atoms_enforce_buried: frozenset[str] = frozenset()
    atoms_enforce_exposed: frozenset[str] = frozenset()

    @model_validator(mode='after')
    def _check(self) -> LigandConfig:
        # INCIDENT (v7): the component and the input pose were both correctly
        # deprotonated, but ligand_smiles still carried a neutral C(=O)O. The SMILES
        # builds the ROSETTA PARAMS via AddHs, so an acid proton (H13 on OA2) went
        # into every relaxed structure anyway. Caught only when the user spotted it
        # in a rendered image. Three places must agree; this checks the one that hid.
        if self.deprotonated and '[O-]' not in self.smiles:
            raise ConfigError(
                f'ligand {self.code}: deprotonated=True but the SMILES has no [O-]. '
                'The SMILES builds the Rosetta params, so a neutral C(=O)O reinstates '
                'the acid proton even when the CCD and the input pose are both correct.')
        if not self.deprotonated and '[O-]' in self.smiles:
            raise ConfigError(
                f'ligand {self.code}: deprotonated=False but the SMILES carries [O-].')

        # INCIDENT (v4/v5): a SMILES with the four Mg-N bonds present cannot be
        # embedded by RDKit ETKDGv3, and Boltz hard-fails with "Failed to compute 3D
        # conformer". The CCD component MUST keep those bonds (as DATIVE); the SMILES
        # must NOT. A dative-bond SMILES derived from the CCD also fails
        # AssignBondOrdersFromTemplate with 'No matching found'.
        if re.search(r'\[Mg[^\]]*\]\s*[(\-=:]', self.smiles) or '->' in self.smiles:
            raise ConfigError(
                f'ligand {self.code}: the magnesium must be DISCONNECTED in the SMILES '
                '(a trailing ".[Mg+2]"). Bonded Mg cannot be embedded by ETKDGv3 and '
                'Boltz fails with "Failed to compute 3D conformer". The CCD component '
                'keeps its four Mg-N dative bonds -- the SMILES does not.')
        if '[Mg' not in self.smiles:
            raise ConfigError(f'ligand {self.code}: no magnesium in the SMILES.')

        overlap = self.atoms_enforce_buried & self.atoms_enforce_exposed
        if overlap:
            raise ConfigError(f'atoms both buried and exposed: {sorted(overlap)}')
        return self


# --------------------------------------------------------------------------- scaffold


class ScaffoldConfig(_Base):
    """Input pose and the axial histidine, addressed positionally."""

    pdb: Path
    axial_his_resindex: int = Field(
        ge=0,
        description='0-BASED POSITIONAL index, not a residue number. Boltz renumbers '
                    'its output 1-N, so a resnum-based selection silently moves.')
    burial_alpha: float = Field(
        default=50.0, gt=0,
        description='alpha-hull alpha. BDFP44 measured: at alpha 9 only 10/45 ligand '
                    'atoms read buried and the Mg reads EXPOSED, so the burial gate '
                    'would reject everything at iteration 0. 50 is the smallest value '
                    'where the metal core is buried and the propionate stays exposed.')

    @property
    def fixed_identity_selection(self) -> str:
        return f'resindex {self.axial_his_resindex}'


# --------------------------------------------------------------------------- engine


class EngineConfig(_Base):
    fold: Literal['boltz', 'af3']
    cross_check: Literal['auto', 'boltz', 'af3', 'none'] = 'auto'
    constrain_his_mg: bool = Field(
        description='Boltz only: a patched two-sided 2.0-2.6 A ConnectionsPotential. '
                    'Upstream the flat bottom is one-sided (lower_bounds=None, '
                    'buffer=2.0) so a declared bond collapses to ~1.52 A. AF3 has no '
                    'equivalent -- bondedAtomPairs has no distance window and crushes '
                    'His-Mg to 1.50 A, flipping 13-2 to R. Must be False for AF3.')
    devices: list[str] = Field(default_factory=lambda: ['cuda:0'])

    @model_validator(mode='after')
    def _check(self) -> EngineConfig:
        # INCIDENT (v8 bring-up): af3_json defaulted bondedAtomPairs ON despite the
        # harm having already been measured. A self-test caught it before the run.
        if self.fold == 'af3' and self.constrain_his_mg:
            raise ConfigError(
                'engine.constrain_his_mg=True with fold=af3. AF3 bondedAtomPairs has no '
                'distance window: it crushes His-Mg to 1.50 A and flips the 13-2 '
                'stereocentre to R. Coordination is the Rosetta relax gate\'s job.')
        if not self.devices:
            raise ConfigError('engine.devices is empty.')
        if len(set(self.devices)) != len(self.devices):
            raise ConfigError(f'engine.devices has duplicates: {self.devices}')
        for d in self.devices:
            if not re.fullmatch(r'cuda:\d+', d):
                raise ConfigError(f'engine.devices entry {d!r} is not of the form cuda:N')
        return self

    @property
    def resolved_cross_check(self) -> str | None:
        """'auto' means whichever engine did NOT lead."""
        if self.cross_check == 'none':
            return None
        if self.cross_check == 'auto':
            return 'boltz' if self.fold == 'af3' else 'af3'
        return self.cross_check


# --------------------------------------------------------------------------- loop


class ConvergenceConfig(_Base):
    # The auto-stop is a CONVENIENCE, not a correctness feature. The published protocol says
    # to keep going past the plateau -- "both EPIC and APEX came from later iterations"
    # (Nature 656, 237, 2026) -- and v8 agreed: it halted itself at iteration 16 of 25 while
    # half its top-10 by BUNs came from iteration 11 or later. With this False the metric is
    # still computed, logged and plotted; only the halt is suppressed.
    enabled: bool = True
    window: int = Field(default=3, ge=1)
    eps: float = Field(default=0.002, gt=0)
    patience: int = Field(default=7, ge=1)
    min_iterations: int = Field(default=10, ge=0)
    metric: str = Field(default='p90_ligand_plddt')


class LoopConfig(_Base):
    iterations: int = Field(ge=1)
    per_backbone: int = Field(ge=1, description='sequences sampled per promoted backbone')
    # Seeding is a different problem from steady state and deserves its own width. Iteration 0
    # starts from ONE backbone and its output seeds everything downstream, so starving it
    # starves the whole run: v10 gave it 64 designs against v9's 500 and never recovered, 0.05
    # ligand pLDDT behind for all 25 iterations. Sample wide until the queue holds
    # top_backbones gate-passing candidates, then drop to per_backbone.
    # None = no seed phase, use per_backbone throughout (every campaign before v11).
    seed_per_backbone: int | None = Field(default=None, ge=1)
    top_backbones: int = Field(default=3, ge=1)
    sequences_sampled_at_once: int = Field(default=30, ge=1)
    objective: str = 'ligand_plddt'
    convergence: ConvergenceConfig = ConvergenceConfig()

    @property
    def per_iteration(self) -> int:
        return self.per_backbone * self.top_backbones

    @model_validator(mode='after')
    def _check_seed(self) -> LoopConfig:
        if self.seed_per_backbone is not None and self.seed_per_backbone < self.per_backbone:
            raise ConfigError(
                f'seed_per_backbone ({self.seed_per_backbone}) is below per_backbone '
                f'({self.per_backbone}): the seed phase is meant to be WIDER than steady '
                f'state, not narrower.')
        return self

    @model_validator(mode='after')
    def _check(self) -> LoopConfig:
        if self.convergence.min_iterations > self.iterations:
            raise ConfigError(
                f'convergence.min_iterations ({self.convergence.min_iterations}) exceeds '
                f'loop.iterations ({self.iterations}) -- the run can never converge.')
        return self


# --------------------------------------------------------------------------- gates


class GateConfig(_Base):
    """Thresholds. Every value is evidenced in docs/CALIBRATION.md."""

    ligand_plddt: float = Field(default=0.80, ge=0, le=1)
    protein_plddt: float = Field(default=0.80, ge=0, le=1)
    self_consistency_ligand_rmsd: float = Field(default=2.5, gt=0)
    self_consistency_protein_rmsd: float = Field(default=1.5, gt=0)
    rmsd_use_chirality: bool = Field(
        default=False,
        description='STAYS FALSE, and this was measured. useChirality=True requires '
                    'every PERCEIVED centre to agree; RDKit perceives MG1 (4-coordinate '
                    'Mg, not a real centre) and CA2 alongside the one centre we care '
                    'about, so BOTH the inverted and the correct population return zero '
                    'matches, every ligand_rmsd becomes inf, the backbone queue empties '
                    'at iteration 0 and the campaign freezes. Gate 13-2 directly instead.')
    packstat: float = Field(default=0.55, ge=0, le=1)
    apo_clashes_max: int = Field(default=0, ge=0)
    allow_buried_charged: bool = False
    coordination_window: tuple[float, float] = (2.0, 2.6)
    # The RESTRAINT TARGET, which is not the window's midpoint. 2.15 A is the physical
    # Mg-N(His) distance; Rosetta pulls to it with a FlatHarmonic and the window above is what
    # the result is then ACCEPTED against. Deriving one from the other (midpoint = 2.30) silently
    # moves the restraint 0.15 A off the physical bond.
    metal_r0: float = Field(default=2.15, gt=0)

    # Two-copy aggregation proxy. The paper's 14 is AF2-multimer-derived and was paired
    # with visual inspection we do not perform. On AF3-folded v8 the distribution is
    # bimodal and AF3's own ipTM puts the valley at pAE ~9; 14 sits on the rising flank
    # and is conservative. Disagreement is one-directional, so 14 costs recall but never
    # admits a dimer. Keep 14; treat (grey_band, 14] as inspect-not-fail.
    dimer_interface_pae: float = Field(default=14.0, gt=0)
    dimer_pae_grey_band: float = Field(default=9.0, gt=0)
    dimer_contact_cutoff: float = Field(default=5.0, gt=0, description='CB-CB, Angstrom')

    # COORDINATION GEOMETRY, enforced in the loop as well as the cascade. The His-Mg DISTANCE
    # window above is saturated by the shortlist stage and discriminates nothing; these three
    # describe whether the bond is actually axial and whether the imidazole is turned the right
    # way. Defaults are the 90th percentile of 57 His-coordinated Mg-chlorophylls measured in
    # the reference crystals (1M3X, 7COU) -- see docs/CALIBRATION.md.
    #
    # v9 optimised ligand pLDDT + P(bind) and walked its histidine from 15 to 40 degrees over
    # eight iterations, ending with 0.6% of its pool inside these bounds. Nothing was watching,
    # because a distance gate cannot see a rotation.
    axial_angle_max: float = Field(default=11.5, gt=0, le=90)
    lonepair_angle_max: float = Field(default=17.0, gt=0, le=90)
    mg_imidazole_oop_max: float = Field(default=12.7, gt=0, le=90)

    @model_validator(mode='after')
    def _check(self) -> GateConfig:
        lo, hi = self.coordination_window
        if lo >= hi:
            raise ConfigError(f'coordination_window {self.coordination_window} is not ordered')
        if not lo <= self.metal_r0 <= hi:
            raise ConfigError(
                f'metal_r0 {self.metal_r0} lies outside coordination_window '
                f'{self.coordination_window}: the restraint would pull every design to a '
                f'distance the gate then rejects.')
        # A geometry bound tighter than the experimental MEDIAN rejects most real sites.
        for name, value, median in (('axial_angle_max', self.axial_angle_max, 6.7),
                                    ('lonepair_angle_max', self.lonepair_angle_max, 10.9),
                                    ('mg_imidazole_oop_max', self.mg_imidazole_oop_max, 6.0)):
            if value < median:
                raise ConfigError(
                    f'{name} {value} is below the experimental median {median} deg: the gate '
                    f'would reject more than half of real His-coordinated Mg-chlorophylls.')
        if self.dimer_pae_grey_band > self.dimer_interface_pae:
            raise ConfigError(
                f'dimer_pae_grey_band ({self.dimer_pae_grey_band}) exceeds '
                f'dimer_interface_pae ({self.dimer_interface_pae}).')
        return self


# --------------------------------------------------------------------------- stages


class StageConfig(_Base):
    """A disabled stage must contribute NOTHING -- not a no-op that still writes."""

    enabled: bool = True


class StagesConfig(_Base):
    generate: StageConfig = StageConfig(enabled=False)   # backbone generation, off by default
    design: StageConfig = StageConfig()
    fold: StageConfig = StageConfig()
    relax: StageConfig = StageConfig()
    cascade: StageConfig = StageConfig()
    crossfold: StageConfig = StageConfig()
    dimer: StageConfig = StageConfig()
    report: StageConfig = StageConfig()


# --------------------------------------------------------------------------- compute


class ComputeConfig(_Base):
    reserve_cpus: int = Field(
        default=16, ge=0,
        description="CPUs left for OTHER users, sized from THEIR load only. Counting "
                    "our own collapsed the pool to 4 workers and a 14 h cascade.")
    relax_workers: int = Field(default=12, ge=1)
    relax_timeout: int = Field(default=1800, gt=0)
    relax_pool_watcher: bool = Field(
        default=True,
        description="Relax the design pool DURING the run instead of all at once at the end. "
                    "The in-loop relax only touches top_backbones*4 designs, enough to gate "
                    "promotion; everything else still has to be minimised before the cascade "
                    "can score it. v8 ran this and its harvest caught up in six minutes; v9 "
                    "did not and harvested 12,972 unrelaxed designs. Default on, because the "
                    "cost is the same either way and only the timing differs.")
    relax_pool_interval: int = Field(
        default=600, ge=30,
        description='seconds between pool passes; relax_pool is incremental, so a missed '
                    'poll costs nothing')
    relax_pool_workers: int = Field(
        default=12, ge=1,
        description="Kept well under the box. This must never starve the driver's in-loop "
                    "relax, which times out at relax_timeout and then silently ranks on "
                    "UNRELAXED structures -- changing the trajectory for a background job.")


class NotifyConfig(_Base):
    """Per-iteration Telegram reporting.

    Started by `chlide run`, for the same reason the relax watcher is: it used to be launched
    by hand beside each campaign, and v10 ran with no reporter at all until someone asked
    whether Telegram was pointed at it. Nothing fails when it is missing -- you simply stop
    hearing from a run that is otherwise fine, which is the worst way for it to break.

    Credentials are NOT config. They live in ~/.config/pipeline_telegram.env (mode 0600) and
    are read by chlide.notify; nothing here, in the repo, or on a command line ever holds them.
    """

    telegram: bool = True
    interval: int = Field(default=300, ge=30, description='seconds between polls')
    harvest: bool = Field(
        default=True,
        description='refresh results/global_top after each iteration, so a run interrupted '
                    'at any point already has a harvest on disk')


# --------------------------------------------------------------------------- top


class CampaignConfig(_Base):
    name: str = Field(min_length=1)
    description: str = ''
    ligand: LigandConfig
    scaffold: ScaffoldConfig
    engine: EngineConfig
    loop: LoopConfig
    gates: GateConfig = GateConfig()
    stages: StagesConfig = StagesConfig()
    compute: ComputeConfig = ComputeConfig()
    notify: NotifyConfig = NotifyConfig()

    @model_validator(mode='after')
    def _check(self) -> CampaignConfig:
        # INCIDENT (five times): a copied campaign kept its predecessor's absolute
        # paths. One driver's input_dir still pointed at v5's input tree and would have
        # written new iterations into a closed archive. Paths here must be RELATIVE and
        # must not climb out of the run directory.
        for label, p in (('ligand.source', self.ligand.source),
                         ('scaffold.pdb', self.scaffold.pdb)):
            if p.is_absolute():
                raise ConfigError(
                    f'{label}={p} is absolute. Campaign paths are relative to the run '
                    'directory, which is a CLI argument -- an absolute path is how a '
                    'copied campaign ends up writing into its predecessor.')
            if '..' in p.parts:
                raise ConfigError(f'{label}={p} escapes the run directory.')

        # P(bind) is Boltz-2's affinity head. AF3 has no equivalent, so an AF3-led campaign
        # asking for it would read a key that is never written -- a silent no-op that looks
        # like a configured objective. Reject it at parse time.
        if 'pbind' in self.loop.objective and self.engine.fold != 'boltz':
            raise ConfigError(
                f'loop.objective={self.loop.objective!r} needs P(bind), which is a Boltz-2 '
                f'head (affinity_probability_binary). engine.fold is {self.engine.fold!r} and '
                'has no equivalent, so the term would contribute nothing. Use fold: boltz.')

        if self.stages.crossfold.enabled and self.engine.resolved_cross_check is None:
            raise ConfigError(
                'stages.crossfold is enabled but engine.cross_check is "none".')
        if self.stages.crossfold.enabled and self.engine.resolved_cross_check == self.engine.fold:
            raise ConfigError(
                f'engine.cross_check={self.engine.cross_check} resolves to '
                f'{self.engine.fold}, the SAME engine that folds. A cross-check against '
                'yourself measures nothing.')
        return self

    def summary(self) -> str:
        x = self.engine.resolved_cross_check or 'none'
        charge = ' [COO-]' if self.ligand.deprotonated else ' [COOH]'
        chiral = (f', 13-2 must be {self.ligand.required_config}'
                  if self.ligand.required_config else '')
        return (f'{self.name}: {self.engine.fold} fold / {x} cross-check, '
                f'{self.loop.iterations} iterations x {self.loop.per_iteration} designs, '
                f'ligand {self.ligand.code}{charge}{chiral}')


def resolve_input(run: Path, rel: Path, env_var: str | None = None,
                  fallback: str | None = None) -> Path | None:
    """Find a declared input: run directory first, then a named shared location.

    Campaign files hold RELATIVE paths only -- an absolute one is how five campaigns inherited
    their predecessor's tree. But some inputs genuinely are shared across campaigns (the
    chlorophyllide mmCIF, the Boltz checkpoints), so a relative name is resolved against the run
    directory first and a shared location second. Returns None if neither has it, so the caller
    can report what is missing rather than failing on a path that was never going to exist.
    """
    local = run / rel
    if local.is_file():
        return local
    shared = os.environ.get(env_var) if env_var else None
    for cand in (shared, fallback):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def load_campaign(path: str | Path, overrides: dict | None = None) -> CampaignConfig:
    """Load a campaign YAML, deep-merging `_base.yaml` from the same directory if present."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    base_file = path.parent / '_base.yaml'
    if base_file.is_file() and base_file.resolve() != path.resolve():
        raw = _deep_merge(yaml.safe_load(base_file.read_text()) or {}, raw)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return CampaignConfig.model_validate(raw)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
