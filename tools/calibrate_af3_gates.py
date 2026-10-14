#!/usr/bin/env python
"""Derive v8's pLDDT gates from AF3's own scale instead of inheriting Boltz's.

Every numeric gate in this project was tuned on Boltz-2x confidence: the 0.8 ligand/protein pLDDT
bars in tools/harvest_global_top.py and tools/relax_pool.py. AF3's pLDDT is NOT numerically
interchangeable -- measured on the native BDFP44 sequence, AF3 gives 0.64 where Boltz gives 0.83.

That difference is not cosmetic. The campaign LOOP does not threshold on pLDDT (its objective is to
maximise it), so a mis-set gate will not stop the run -- it will quietly empty the harvest. With
0.8 on AF3 confidence, relax_pool minimises nothing and harvest_global_top pools nothing, and the
campaign finishes after days of GPU with an empty shortlist.

This folds a sample of designs the Boltz campaign already scored, with AF3, and reports the paired
distributions so the gates can be set to admit a comparable FRACTION rather than a comparable
NUMBER. Reference: on v5/v7, roughly 60-70% of designs clear the 0.8 ligand pLDDT bar.

Usage:
    python tools/calibrate_af3_gates.py --n 30 --devices 0,1,2,3
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import prody as pr

pr.confProDy(verbosity='none')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from af3_adapter import af3_json, fold_shards, to_boltz_layout  # noqa: E402

MASK = {'CAA', 'CBA', 'CGA', 'O1A', 'O2A', 'CAA1', 'CBA1', 'CGA1', 'OA1', 'OA2'}


def plddt(pdb: Path) -> tuple[float, float] | None:
    ag = pr.parsePDB(str(pdb))
    lig = ag.select('(not protein) and not element H')
    prot = ag.select('chid A')
    if lig is None or prot is None:
        return None
    keep = np.array([n not in MASK for n in lig.getNames()])
    return float(lig.getBetas()[keep].mean() / 100), float(prot.copy().ca.getBetas().mean() / 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(os.environ.get('CHLIDE_RUN_DIR') or Path(__file__).resolve().parent.parent))
    ap.add_argument('--source', type=Path,
                    default=Path(os.environ.get(
                        'BOLTZ_REFERENCE_DF',
                        '/home/data/yuvaln/nise_bdfp44_v7/input/sampling_dataframes')),
                    help='Boltz-led campaign to calibrate AF3 gates against. The two '
                         'pLDDT scales are NOT interchangeable (Boltz median 0.865 vs '
                         'AF3 0.753, r = +0.13 to +0.18).')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--ccd', default='CLAOX')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--score-only', action='store_true')
    a = ap.parse_args()

    # pool the Boltz campaign's own designs, across iterations, with their Boltz pLDDT
    frames = []
    for pkl in sorted(a.source.glob('iter_*_data.pkl')):
        df = pd.read_pickle(pkl)
        df['iter'] = int(pkl.stem.split('_')[1])
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    allf['ligand_plddts'] = pd.to_numeric(allf['ligand_plddts'], errors='coerce')
    allf['protein_plddts'] = pd.to_numeric(allf['protein_plddts'], errors='coerce')
    allf = allf[allf['sequences'].notna() & allf['ligand_plddts'].notna()]
    print(f'Boltz reference pool: {len(allf)} designs over {allf["iter"].nunique()} iterations')
    print('  Boltz ligand pLDDT : median %.3f  p10 %.3f  p90 %.3f  | %.0f%% clear 0.80'
          % (allf.ligand_plddts.median(), allf.ligand_plddts.quantile(.1),
             allf.ligand_plddts.quantile(.9), 100 * (allf.ligand_plddts > 0.8).mean()))
    print('  Boltz protein pLDDT: median %.3f  | %.0f%% clear 0.80'
          % (allf.protein_plddts.median(), 100 * (allf.protein_plddts > 0.8).mean()))

    sub = allf.sample(n=min(a.n, len(allf)), random_state=a.seed)
    root = a.run_dir / 'af3_calib'
    in_dir, out_dir = root / 'inputs', root / 'out'

    if not a.score_only:
        if root.exists():
            shutil.rmtree(root)
        in_dir.mkdir(parents=True)
        for i, r in enumerate(sub.itertuples()):
            stem = f'calib_{i}'
            (in_dir / f'{stem}.json').write_text(json.dumps(
                af3_json(stem, str(r.sequences), a.ccd,
                         a.run_dir / 'input' / 'claox_userccd.cif'), indent=2))
        print(f'\nfolding {len(sub)} designs with AF3 on GPUs {a.devices} ...')
        fold_shards(in_dir, out_dir, a.devices.split(','), a.ccd)
        to_boltz_layout(out_dir, root, [f'calib_{i}' for i in range(len(sub))])

    rows = []
    for i, r in enumerate(sub.itertuples()):
        p = root / 'boltz_results_boltz_inputs' / 'predictions' / f'calib_{i}' / f'calib_{i}_model_0.pdb'
        if not p.is_file():
            continue
        v = plddt(p)
        if v:
            rows.append(dict(boltz_lig=r.ligand_plddts, boltz_prot=r.protein_plddts,
                             af3_lig=v[0], af3_prot=v[1]))
    d = pd.DataFrame(rows)
    if d.empty:
        print('no AF3 outputs scored'); return 1

    print(f'\n=== paired on {len(d)} designs (same sequences, both engines) ===')
    print('                  Boltz            AF3            shift')
    for lab, b, f in (('ligand pLDDT', 'boltz_lig', 'af3_lig'),
                      ('protein pLDDT', 'boltz_prot', 'af3_prot')):
        print('  %-14s  median %.3f    median %.3f    %+.3f'
              % (lab, d[b].median(), d[f].median(), d[f].median() - d[b].median()))
    print('  correlation ligand pLDDT Boltz vs AF3: r = %+.2f' % d.boltz_lig.corr(d.af3_lig))

    print('\n=== gate that admits the SAME FRACTION as Boltz 0.80 ===')
    for lab, b, f in (('ligand', 'boltz_lig', 'af3_lig'), ('protein', 'boltz_prot', 'af3_prot')):
        frac = (allf[b.replace('boltz_', '') + 's'] > 0.8).mean() if False else \
               (allf.ligand_plddts > 0.8).mean() if lab == 'ligand' else (allf.protein_plddts > 0.8).mean()
        gate = float(np.quantile(d[f], 1 - frac))
        print('  %-8s Boltz 0.80 admits %.0f%%  ->  AF3 gate %.2f  (admits %.0f%% of this sample)'
              % (lab, 100 * frac, gate, 100 * (d[f] > gate).mean()))
    print('\n  Set these in tools/harvest_global_top.py and tools/relax_pool.py (PAPER dict).')
    print('  A sample of %d is indicative, not definitive -- re-run tools/recalibrate_gates.py' % len(d))
    print('  after iterations 0-2 of the real campaign and move them if the pass rate is off.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
