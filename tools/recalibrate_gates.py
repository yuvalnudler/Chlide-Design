#!/usr/bin/env python
"""Report what the Boltz-tuned gates actually do to AF3 confidence, after a few iterations.

v6a swaps the folding engine. Every numeric gate in this project was tuned against Boltz-2x:
the harvest's ligand/protein pLDDT bar of 0.8 (tools/harvest_global_top.py, tools/relax_pool.py)
and the driver's self-consistency RMSD thresholds. AF3's pLDDT is not numerically interchangeable
with Boltz's, so those numbers carry over as an assumption, not a measurement.

Two failure modes, and they look nothing alike in the log:

  gates too LOOSE   almost everything passes, the harvest pool is enormous, and there is no
                    selection pressure -- the campaign wanders instead of optimising.
  gates too TIGHT   almost nothing passes, the backbone queue runs on the fallback path every
                    round, and the funnel collapses the way v4's predecessor did.

Run this after iterations 0-2 and compare the pass rates with v5's. It changes nothing on its own
-- it prints what the thresholds are doing so the choice to move them is a measured one.

Usage:
    python tools/recalibrate_gates.py                       # this campaign
    python tools/recalibrate_gates.py --compare /home/data/yuvaln/nise_bdfp44_v5
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GATES = {'protein_rmsd': 1.5, 'ligand_rmsd': 1.5, 'ligand_plddt': 0.8, 'protein_plddt': 0.8}


def frames(run: Path, upto: int | None) -> pd.DataFrame:
    sdf, sbb = run / 'input' / 'sampling_dataframes', run / 'input' / 'sampled_backbones'
    out = []
    for pkl in sorted(sdf.glob('iter_*_data.pkl')):
        try:
            i = int(pkl.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        if upto is not None and i > upto:
            continue
        if not (sbb / f'iter_{i}' / 'backbone_queue.txt').exists():
            continue
        df = pd.read_pickle(pkl)
        df['iteration'] = i
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f'{name}: no complete iterations yet')
        return
    for c in ('protein_rmsds', 'ligand_rmsds', 'ligand_plddts', 'protein_plddts'):
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    n = len(df)
    print(f'\n{name}: {n} designs over iterations {sorted(df.iteration.unique())}')
    print('  %-16s %8s %8s %8s %8s   %s' % ('metric', 'median', 'p10', 'p90', 'nan%', 'passes gate'))
    for col, key, hi in (('ligand_plddts', 'ligand_plddt', True),
                         ('protein_plddts', 'protein_plddt', True),
                         ('ligand_rmsds', 'ligand_rmsd', False),
                         ('protein_rmsds', 'protein_rmsd', False)):
        v = df[col].dropna()
        if v.empty:
            continue
        passes = (v > GATES[key]).mean() if hi else (v < GATES[key]).mean()
        print('  %-16s %8.3f %8.3f %8.3f %7.0f%%   %5.1f%% %s %.2f'
              % (col, v.median(), v.quantile(.1), v.quantile(.9),
                 100 * df[col].isna().mean(), 100 * passes, '>' if hi else '<', GATES[key]))
    keep = ((df.protein_rmsds < GATES['protein_rmsd']) & (df.ligand_rmsds < GATES['ligand_rmsd'])
            & (df.ligand_plddts > GATES['ligand_plddt'])
            & (df.protein_plddts > GATES['protein_plddt'])).fillna(False)
    print('  ALL FOUR GATES : %d/%d = %.1f%%' % (keep.sum(), n, 100 * keep.mean()))
    if 'ligand_config' in df.columns:
        ok = (df['ligand_config'] == 'S').mean()
        print('  13-2 centre S  : %.1f%%' % (100 * ok))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--upto', type=int, default=None, help='only iterations <= this')
    ap.add_argument('--compare', type=Path,
                    default=Path(os.environ.get('COMPARE_CAMPAIGN',
                                                '/home/data/yuvaln/nise_bdfp44_v5')),
                    help='campaign to compare pass rates against')
    a = ap.parse_args()
    report(a.run_dir.name, frames(a.run_dir, a.upto))
    if a.compare and a.compare.exists() and a.compare.resolve() != a.run_dir.resolve():
        report(f'{a.compare.name} (Boltz reference, same iterations)',
               frames(a.compare, a.upto))
    print('\nA combined pass rate far from the reference means the thresholds are measuring a '
          '\ndifferent thing on AF3 confidence than they did on Boltz. Move them deliberately, '
          '\nin tools/harvest_global_top.py and tools/relax_pool.py, and write down why.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
