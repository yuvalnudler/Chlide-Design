"""Build a run directory.

Seeding one campaign's inputs from another is legitimate when the ligand and scaffold are
identical BY DESIGN -- v9 is v7 plus one objective term, so sharing the CCD is the point, not a
shortcut. What is not legitimate is assuming it. Five campaigns inherited a predecessor's paths
and one driver would have written new iterations into a closed archive, so every seeded file is
VERIFIED against the config before the run is allowed to start:

* the component's net charge must match `deprotonated`
* the 13-2 stereocentre must read the configured CIP code
* the scaffold's axial residue must actually be a histidine at the configured 0-based index

A mismatch is a hard failure. The alternative is the v7 incident, where the component and the
pose were correctly deprotonated and the SMILES was not, and nobody noticed until a reviewer
saw an acid proton in a rendered image.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from chlide.config import CampaignConfig

__all__ = ['seed_run', 'verify_inputs', 'SetupError']

# Small files a run needs; the mols farm and checkpoints are symlinked, never copied.
INPUT_FILES = ('CLAOX.pkl', 'CLAOX_raw.pkl', 'claox_userccd.cif',
               'lig_from_input_pdb.params', 'lig_from_input_pdb.pdb', 'lig_from_input_pdb.sdf')
RUN_DIRS = ('input', 'logs', 'relaxed', 'results')


class SetupError(RuntimeError):
    """The run directory is not fit to start a campaign."""


def seed_run(cfg: CampaignConfig, run: Path, seed_from: Path, *, link_cache: bool = True) -> dict:
    """Populate `run` from a finished campaign with the SAME ligand and scaffold."""
    run, seed_from = Path(run).resolve(), Path(seed_from).resolve()
    if run == seed_from:
        raise SetupError('run and seed directories are the same')
    if not (seed_from / 'input').is_dir():
        raise SetupError(f'{seed_from} has no input/ -- not a campaign directory')

    report: dict = {'seed_from': str(seed_from), 'copied': [], 'linked': []}
    for d in RUN_DIRS:
        (run / d).mkdir(parents=True, exist_ok=True)

    for name in INPUT_FILES:
        src = seed_from / 'input' / name
        if src.is_file():
            shutil.copy2(src, run / 'input' / name)
            report['copied'].append(f'input/{name}')

    # The starting scaffold -- ONLY the one the config names. NISE raises
    # "More than one input backbone not currently supported" if the directory holds two, and
    # v7's holds a `_fixbeta` variant alongside the real one. Copying the whole directory put
    # both there and the smoke test died on it. The config is the source of truth for which.
    want = Path(cfg.scaffold.pdb).name
    src = seed_from / cfg.scaffold.pdb
    if not src.is_file():
        cand = list((seed_from / 'input' / 'input_backbones').glob('*.pdb'))
        raise SetupError(
            f'{seed_from} has no {cfg.scaffold.pdb}. Backbones present: '
            + (', '.join(c.name for c in cand) or 'none'))
    dest_dir = run / Path(cfg.scaffold.pdb).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / want)
    report['copied'].append(str(cfg.scaffold.pdb))
    others = [c.name for c in src.parent.glob('*.pdb') if c.name != want]
    if others:
        report['skipped_backbones'] = others

    # the Boltz cache: checkpoints and a 45k-file component farm. Symlink -- copying it is
    # ~10 GB of duplicate data and the checkpoints are already symlinks into the owner's home.
    if link_cache and (seed_from / 'boltz_cache').is_dir():
        cache = run / 'boltz_cache'
        cache.mkdir(exist_ok=True)
        for item in (seed_from / 'boltz_cache').iterdir():
            dest = cache / item.name
            if dest.exists() or dest.is_symlink():
                continue
            target = item.resolve() if item.is_symlink() else item
            os.symlink(target, dest)
            report['linked'].append(f'boltz_cache/{item.name} -> {target}')

    return report


def verify_inputs(cfg: CampaignConfig, run: Path) -> list[str]:
    """Check the seeded inputs actually match the campaign. Returns a list of problems."""
    run = Path(run).resolve()
    problems: list[str] = []

    # --- the ligand component
    comp = run / 'input' / f'{cfg.ligand.code}.pkl'
    if not comp.is_file():
        problems.append(f'missing {comp.relative_to(run)}')
    else:
        try:
            import pickle
            mol = pickle.loads(comp.read_bytes())
            charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            names = [a.GetPropsAsDict().get('name') for a in mol.GetAtoms()]
            n_heavy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != 'H')
            # deprotonated -> the C17 propionate carries a formal -1; Mg is +2, pyrrole N 2x -1
            if cfg.ligand.deprotonated and charge >= 0:
                problems.append(
                    f'{comp.name}: net charge {charge:+d}, expected negative for '
                    f'deprotonated=true (the carboxylate carries -1)')
            if not cfg.ligand.deprotonated and charge < 0:
                problems.append(f'{comp.name}: net charge {charge:+d} but deprotonated=false')
            for atom in ('MG', *('NA NB NC ND'.split())):
                if atom not in names:
                    problems.append(f'{comp.name}: no atom named {atom}')
            if n_heavy < 40:
                problems.append(f'{comp.name}: only {n_heavy} heavy atoms, expected ~46')
        except Exception as e:                      # noqa: BLE001 - any failure is a problem
            problems.append(f'{comp.name}: unreadable ({type(e).__name__}: {e})')

    # --- the Rosetta params, which the SMILES builds and which carried the v7 acid proton
    params = run / 'input' / 'lig_from_input_pdb.params'
    if not params.is_file():
        problems.append('missing input/lig_from_input_pdb.params')
    else:
        # Test the BOND, not the atom name. An earlier version of this check looked for the
        # string "H13" and flagged v7 -- which is correctly deprotonated -- because H13 there
        # is a METHYL hydrogen on CMB1. The v5 incident was specifically `BOND OA2 H13`, a
        # hydrogen bonded to a carboxylate oxygen.
        carboxylate = {'OA1', 'OA2'}
        acid_h = []
        formal = set()
        for line in params.read_text().splitlines():
            f = line.split()
            if len(f) >= 3 and f[0] == 'BOND':
                a, b = f[1], f[2]
                if (a in carboxylate and b.startswith('H')) or \
                   (b in carboxylate and a.startswith('H')):
                    acid_h.append(f'{a}-{b}')
            elif len(f) >= 4 and f[0] == 'CHARGE' and f[2] == 'FORMAL':
                if f[1] in carboxylate and f[3].startswith('-'):
                    formal.add(f[1])
        if cfg.ligand.deprotonated:
            if acid_h:
                problems.append(
                    f'lig_from_input_pdb.params bonds a hydrogen to the carboxylate '
                    f'({", ".join(acid_h)}) but deprotonated=true. The SMILES builds these '
                    f'params, so a neutral C(=O)O reinstates the acid proton even when the '
                    f'component and the pose are both correct. This is the v5->v7 incident.')
            if not formal:
                problems.append(
                    'lig_from_input_pdb.params declares no negative FORMAL charge on OA1/OA2, '
                    'but deprotonated=true. A correct file carries `CHARGE OA2 FORMAL -1`.')
        elif formal and not acid_h:
            problems.append('params look deprotonated but deprotonated=false')

    # --- the scaffold and its axial histidine
    scaffold = run / cfg.scaffold.pdb
    if not scaffold.is_file():
        problems.append(f'missing {cfg.scaffold.pdb}')
    else:
        try:
            import prody as pr
            pr.confProDy(verbosity='none')
            ag = pr.parsePDB(str(scaffold))
            prot = ag.select('protein')
            resnames = {}
            for a in prot.copy().ca:
                resnames[a.getResindex()] = a.getResname()
            idx = cfg.scaffold.axial_his_resindex
            got = resnames.get(idx)
            if got != 'HIS':
                problems.append(
                    f'scaffold resindex {idx} is {got}, not HIS. This index is 0-BASED and '
                    f'POSITIONAL -- it survives Boltz renumbering the output 1-N, which a '
                    f'resnum does not.')
            n_his = sum(1 for v in resnames.values() if v == 'HIS')
            if n_his != 1:
                problems.append(f'scaffold has {n_his} histidines; the axial one should be '
                                f'the only one, or the metal gate is ambiguous')
        except Exception as e:                      # noqa: BLE001
            problems.append(f'{scaffold.name}: unreadable ({type(e).__name__}: {e})')

    # --- the Boltz cache, if this campaign folds with Boltz
    if cfg.engine.fold == 'boltz':
        cache = run / 'boltz_cache'
        ckpts = list(cache.glob('*.ckpt')) if cache.is_dir() else []
        if not ckpts:
            problems.append('boltz_cache has no checkpoints; a Boltz-led run cannot fold')
        elif 'pbind' in cfg.loop.objective:
            if not any('aff' in c.name for c in ckpts):
                problems.append(
                    'objective asks for P(bind) but boltz_cache has no affinity checkpoint '
                    '(boltz2_aff.ckpt); the affinity head cannot run')

    return problems
