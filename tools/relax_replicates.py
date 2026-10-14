#!/usr/bin/env python
"""Re-relax the leading designs several times to see whether their ranking survives the noise.

FastRelax is stochastic. The shortlist ranks by combined BUNs, and the top five sit at 9, 10, 11,
11, 11 -- one or two buried atoms apart. If a single relax of one design can move its BUNs by two,
that ordering is noise and the "best" design is an artefact of one random seed.

This runs N independent relaxations per design and reports the spread. Read it as: if the
per-design standard deviation is comparable to the gaps between designs, treat the top group as a
set rather than a ranking.

CPU only. Runs in the Rosetta env:
    python tools/relax_replicates.py --run-dir <run> --top 10 --replicates 5 --workers 24
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np

HIS_INDEX = 85
R0, TOL = 2.15, 0.25

_INIT = False


def _init(params):
    global _INIT
    if _INIT:
        return
    import pyrosetta
    pyrosetta.init(f'-mute all -extra_res_fa {params} -ex1 -ex2', silent=True)
    _INIT = True


def one(args) -> dict:
    pdb, params, seed = args
    rec = {'input': pdb, 'seed': seed, 'ok': False}
    try:
        _init(params)
        import pyrosetta
        from pyrosetta import pose_from_file
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType
        from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
        from pyrosetta.rosetta.core.scoring.func import FlatHarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.core.scoring.packstat import compute_packing_score
        from pyrosetta.rosetta.protocols.simple_filters import BuriedUnsatHbondFilter
        from pyrosetta.rosetta.core.select.residue_selector import (
            ResidueIndexSelector, ResidueSpanSelector)

        # a different RNG seed per replicate is the whole point
        pyrosetta.rosetta.numeric.random.rg().set_seed('mt19937', seed)

        pose = pose_from_file(pdb)
        li = pose.total_residue(); n_prot = li - 1
        lig = pose.residue(li)
        mg = next(i for i in range(1, lig.natoms() + 1) if lig.atom_name(i).strip() == 'MG1')
        ne2 = pose.residue(HIS_INDEX).atom_index('NE2')
        sf = ScoreFunctionFactory.create_score_function('ref2015')
        sf.set_weight(ScoreType.atom_pair_constraint, 1.0)
        pose.add_constraint(AtomPairConstraint(AtomID(ne2, HIS_INDEX), AtomID(mg, li),
                                               FlatHarmonicFunc(R0, 0.2, TOL)))
        fr = FastRelax(sf, 1)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)

        rec['his_mg'] = float((pose.residue(HIS_INDEX).xyz('NE2') - pose.residue(li).xyz(mg)).norm())
        rec['score'] = float(sf(pose))
        rec['packstat'] = float(compute_packing_score(pose, 1))
        bl = BuriedUnsatHbondFilter(); bl.set_residue_selector(ResidueIndexSelector(str(li)))
        bp = BuriedUnsatHbondFilter(); bp.set_residue_selector(ResidueSpanSelector(1, n_prot))
        rec['buns_ligand'] = float(bl.compute(pose))
        rec['buns_protein'] = float(bp.compute(pose))
        rec['buns_combined'] = rec['buns_ligand'] + rec['buns_protein']
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--replicates', type=int, default=5)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()

    sl = a.run_dir / 'results' / 'final' / 'shortlist.csv'
    with sl.open() as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith('#'))][:a.top]
    params = a.run_dir / 'input' / 'lig_from_input_pdb.params'

    jobs = []
    for r in rows:
        src = Path(r['output'])
        if src.is_file():
            for s in range(1, a.replicates + 1):
                jobs.append((str(src), str(params), 1000 + s))
    print(f'{len(rows)} designs x {a.replicates} replicates = {len(jobs)} relaxations, '
          f'{a.workers} workers')

    import multiprocessing as mp
    recs = []
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, jobs), 1):
            recs.append(rec)
            if i % 10 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)}', flush=True)

    by = {}
    for rec in recs:
        if rec.get('ok'):
            by.setdefault(rec['input'], []).append(rec)

    name_of = {r['output']: r['name'] for r in rows if r.get('output')}
    out = a.run_dir / 'results' / 'final' / 'replicates.csv'
    print(f'\n{"design":<34}{"BUNs mean":>10}{"sd":>7}{"range":>10}{"packstat":>10}{"His-Mg":>9}')
    summary = []
    for src, rs in by.items():
        b = np.array([r['buns_combined'] for r in rs])
        p = np.array([r['packstat'] for r in rs])
        h = np.array([r['his_mg'] for r in rs])
        nm = name_of.get(src, Path(src).stem)
        summary.append((nm, b.mean(), b.std(), b.min(), b.max(), p.mean(), h.mean(), len(rs)))
    summary.sort(key=lambda x: x[1])
    for nm, m, sd, lo, hi, pm, hm, n in summary:
        print(f'{nm[:34]:<34}{m:>10.1f}{sd:>7.1f}{f"{lo:.0f}-{hi:.0f}":>10}{pm:>10.3f}{hm:>9.2f}')
    with out.open('w', newline='') as f:
        f.write('# Independent FastRelax replicates per design. If the per-design sd is close to\n')
        f.write('# the gaps between designs, the shortlist ordering is not resolvable.\n')
        w = csv.writer(f)
        w.writerow(['name', 'buns_mean', 'buns_sd', 'buns_min', 'buns_max',
                    'packstat_mean', 'his_mg_mean', 'n_replicates'])
        for row in summary:
            w.writerow([row[0]] + [f'{v:.4f}' if isinstance(v, float) else v for v in row[1:]])

    sds = [s[2] for s in summary]
    gaps = [summary[i + 1][1] - summary[i][1] for i in range(len(summary) - 1)]
    if sds and gaps:
        print(f'\nmedian within-design sd : {np.median(sds):.2f} BUNs')
        print(f'median gap between designs: {np.median(gaps):.2f} BUNs')
        if np.median(sds) >= np.median(gaps):
            print('-> the spread within a design is as large as the gap between designs.')
            print('   Treat the top group as a set; the ordering is not resolvable.')
        else:
            print('-> gaps exceed the within-design spread; the ordering carries signal.')
    print(f'\n{out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
