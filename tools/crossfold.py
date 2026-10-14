#!/usr/bin/env python
"""Re-fold a campaign's finalists with the OTHER engine, as an independent second opinion.

A campaign optimises its own folding model's confidence, so that model cannot also be the check.
In Boltz-led campaigns we used AF3 for this (on v5 it rejected 7 of 29 finalists that Boltz rated
0.94+). In an AF3-led campaign, AF3 is no longer independent -- the orthogonal check is Boltz.
The engine is detected from the driver, not assumed, because the Boltz-shaped directory layout and
the `ligand_plddt` column name survive into AF3 runs and silently mislabel which model produced a
number.

Two modes, both cheap:

  monomer  protein only, no ligand. Asks the narrow question "does this SEQUENCE encode the fold",
           which is what af3_monomer.py asked in the Boltz campaigns.
  complex  protein + ligand. Asks the broader question "does an independent model also put the
           ligand there, in the right hand, coordinated" -- fold check and pose check in one.

Usage:
    python tools/crossfold.py --top 25 --mode complex --devices 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import prody as pr

pr.confProDy(verbosity='none')

BOLTZ_PY = os.environ.get('BOLTZ_PYTHON', '/home/yigall/mambaforge/envs/boltz/bin/python')


def resolve_boltz_cache(run: Path, explicit: str | None = None) -> Path:
    """Find a Boltz cache with the model checkpoints in it.

    An AF3-led campaign has no Boltz setup of its own: v8's boltz_cache holds exactly one
    component pickle and ZERO checkpoints, so cross-folding it with Boltz needs a cache from
    elsewhere. That used to be `BOLTZ_DONOR = Path('.../nise_bdfp44_v7')` -- a hardcoded
    pointer from inside one campaign at another, which is the class of reference that let a
    tool rebuild a finished campaign's ligand and a driver write into a closed archive.

    The cache is a SHARED ASSET (the checkpoints are symlinks into ~/.boltz anyway), not a
    property of v7. Resolution order, first hit wins:

        1. --boltz-cache / $BOLTZ_CACHE      explicit, always wins
        2. <run>/boltz_cache                 if it actually carries checkpoints
        3. ~/.boltz                          the upstream default location

    Raises with the places it looked rather than silently folding against an empty cache,
    which fails deep inside Boltz with an unhelpful message.
    """
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    elif os.environ.get('BOLTZ_CACHE'):
        cands.append(Path(os.environ['BOLTZ_CACHE']).expanduser())
    cands += [run / 'boltz_cache', Path.home() / '.boltz']
    # NB on this box the checkpoints live in the boltz owner's home (~yigall/.boltz) and
    # v7's boltz_cache is a tree of symlinks into it. Path.home() is therefore NOT a
    # reliable fallback for every user; the explicit argument is the supported route.
    for c in cands:
        if c.is_dir() and any(c.glob('*.ckpt')):
            return c
    raise SystemExit(
        'no Boltz cache with checkpoints found. Looked in:\n  '
        + '\n  '.join(str(c) for c in cands)
        + '\nPass --boltz-cache <dir> or set $BOLTZ_CACHE. An AF3-led campaign has no '
          'Boltz cache of its own.')
MASK = {'CAA', 'CBA', 'CGA', 'O1A', 'O2A', 'CAA1', 'CBA1', 'CGA1', 'OA1', 'OA2'}


def engine_of(run: Path) -> str:
    for drv in run.glob('run_nise_*.py'):
        if 'from af3_adapter import' in drv.read_text():
            return 'AF3'
    return 'Boltz'


def yaml_for(seq: str, ccd: str, his: int, with_ligand: bool) -> str:
    y = (f'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n'
         f'      msa: empty\n')
    if with_ligand:
        y += f'  - ligand:\n      id: B\n      ccd: {ccd}\n'
        y += (f'constraints:\n  - bond:\n      atom1: [A, {his}, NE2]\n'
              f'      atom2: [B, 1, MG]\n')
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(os.environ.get('CHLIDE_RUN_DIR') or Path(__file__).resolve().parent.parent))
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--mode', choices=('monomer', 'complex'), default='complex')
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--ccd', default='CLAOX')
    ap.add_argument('--his', type=int, default=85)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--boltz-cache', default=None,
                    help='Boltz cache with the model checkpoints. An AF3-led campaign '
                         'has none of its own -- for v8 this is nise_bdfp44_v7/boltz_cache. '
                         'Also read from $BOLTZ_CACHE.')
    a = ap.parse_args()
    run = a.run_dir
    eng = engine_of(run)
    other = 'Boltz' if eng == 'AF3' else 'AF3'
    print(f'campaign engine: {eng}  ->  cross-checking with {other}, mode={a.mode}')
    if other != 'Boltz':
        print('AF3 cross-check lives in tools/af3_monomer.py; this tool provides the Boltz side.')
        return 1

    sl = pd.read_csv(run / 'results/final/shortlist.csv', comment='#')
    sl = sl.nsmallest(a.top, 'buns_score_paper').reset_index(drop=True)
    root = run / f'crossfold_{a.mode}'
    in_dir, out_dir = root / 'inputs', root / 'out'

    if not a.score_only:
        if root.exists():
            shutil.rmtree(root)
        devs = a.devices.split(',')
        for k in range(len(devs)):
            (in_dir / f'gpu{devs[k]}').mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(sl.itertuples()):
            d = in_dir / f'gpu{devs[i % len(devs)]}'
            (d / f'x{i}.yaml').write_text(
                yaml_for(str(r.sequence), a.ccd, a.his, a.mode == 'complex'))
        cache = resolve_boltz_cache(run, getattr(a, 'boltz_cache', None))
        print(f'boltz cache: {cache}')
        env = dict(os.environ, BOLTZ_CACHE=str(cache))
        # the patch is CODE and ships with this repo -- never borrowed from a campaign
        patch = Path(__file__).resolve().parent / 'boltz_coord_patch.py'
        procs = []
        t0 = time.time()
        for dev in devs:
            d = in_dir / f'gpu{dev}'
            if not any(d.glob('*.yaml')):
                continue
            cmd = (f'CUDA_VISIBLE_DEVICES={dev} NCCL_P2P_DISABLE=1 {BOLTZ_PY} {patch} predict {d} '
                   f'--devices 1 --out_dir {out_dir}/gpu{dev} --output_format pdb --override '
                   f'--use_potentials')
            log = open(root / f'gpu{dev}.log', 'w')
            procs.append(subprocess.Popen(cmd, shell=True, env=env, stdout=log,
                                          stderr=subprocess.STDOUT))
        print(f'folding {len(sl)} designs with Boltz on {len(procs)} GPU(s) ...')
        for p in procs:
            p.wait()
        print(f'  done in {time.time() - t0:.0f}s')

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ligand_chirality import ligand_config
    comp = run / 'input' / 'CLAOX.pkl'

    rows = []
    for i, r in enumerate(sl.itertuples()):
        hits = list(out_dir.glob(f'gpu*/boltz_results_gpu*/predictions/x{i}/x{i}_model_0.pdb'))
        rec = {'name': r.name, f'{eng}_ligand_plddt': getattr(r, 'ligand_plddt', np.nan),
               'buns_score_paper': r.buns_score_paper}
        if hits:
            ag = pr.parsePDB(str(hits[0]))
            prot = ag.select('chid A') or ag.select('protein')
            rec['boltz_protein_plddt'] = float(prot.copy().ca.getBetas().mean() / 100)
            lig = ag.select('(not protein) and not element H')
            if lig is not None and len(lig):
                keep = np.array([n not in MASK for n in lig.getNames()])
                rec['boltz_ligand_plddt'] = float(lig.getBetas()[keep].mean() / 100)
                rec['boltz_config'] = ligand_config(ag, component=comp)
                d = dict(zip(lig.getNames(), lig.getCoords()))
                his = prot.select(f'resnum {a.his} and name NE2')
                if his is not None and 'MG' in d:
                    rec['boltz_his_mg'] = float(np.linalg.norm(his.getCoords()[0] - d['MG']))
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = run / 'results' / 'final' / f'crossfold_{a.mode}.csv'
    df.to_csv(out, index=False)
    ok = df.dropna(subset=[c for c in ('boltz_protein_plddt',) if c in df])
    print(f'\n{len(ok)}/{len(df)} cross-folded')
    if len(ok):
        print('  Boltz protein pLDDT : median %.3f  min %.3f' % (ok.boltz_protein_plddt.median(),
                                                                 ok.boltz_protein_plddt.min()))
        if 'boltz_ligand_plddt' in ok:
            print('  Boltz ligand pLDDT  : median %.3f  min %.3f' % (ok.boltz_ligand_plddt.median(),
                                                                     ok.boltz_ligand_plddt.min()))
            c = ok[f'{eng}_ligand_plddt'].corr(ok.boltz_ligand_plddt)
            print(f'  {eng} vs Boltz ligand pLDDT: r = {c:+.2f}   '
                  f'(independent if near zero)')
        if 'boltz_config' in ok:
            n = (ok.boltz_config == 'S').sum()
            print('  Boltz agrees 13-2 = S : %d/%d' % (n, len(ok)))
        if 'boltz_his_mg' in ok:
            print('  Boltz His-Mg          : median %.2f A' % ok.boltz_his_mg.median())
    print(f'-> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
