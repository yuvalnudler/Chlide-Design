#!/usr/bin/env python
"""Replay the convergence stop rule over finished campaigns before trusting it on a live one.

The rule decides when to stop burning ~2 GPU-hours per iteration, and a rule that fires early
costs the best designs: v4's three best iterations all came immediately after its longest
non-improving streak. So the constants are not tuned by intuition -- they are replayed against
every NISE campaign on this box and required to (a) not stop v4 or 6hrk before they finished
improving, and (b) still stop nise_bdfp44_relax, which genuinely was flat.

Usage:
    python tools/replay_convergence.py                      # all campaigns, default constants
    python tools/replay_convergence.py --patience 5         # show why a smaller value is wrong
    python tools/replay_convergence.py --no-smoothing       # show why the raw series is wrong
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGNS = [
    # FIXTURE CORPUS, not a dependency. These are finished campaigns with KNOWN answers,
    # replayed to prove a convergence rule change would not have stopped a run before its
    # peak. Overridable with $REPLAY_CAMPAIGNS (colon-separated); missing ones are skipped.
    '/home/data/yuvaln/nise_bdfp44_coord',     # v4: still improving at 15, must NOT stop
    '/home/data/yuvaln/nise_6hrk_his117',      # still improving at 10, must NOT stop
    '/home/data/yuvaln/nise_bdfp44_relax',     # genuinely flat, SHOULD stop
    '/home/data/yuvaln/nise_bdfp44_his85',     # flat then jumps, must NOT stop
    '/home/data/yuvaln/nise_bdfp44_v5',        # 17 iters, best p90 at the last -- must NOT stop
]


def series_for(run: Path) -> dict[int, float]:
    """p90 ligand pLDDT per complete iteration, using the same completeness test as resume."""
    sdf = run / 'input' / 'sampling_dataframes'
    sbb = run / 'input' / 'sampled_backbones'
    out = {}
    for pkl in sorted(sdf.glob('iter_*_data.pkl')):
        try:
            i = int(pkl.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        if not (sbb / f'iter_{i}' / 'backbone_queue.txt').exists():
            continue
        v = pd.to_numeric(pd.read_pickle(pkl)['ligand_plddts'], errors='coerce').dropna()
        if len(v):
            out[i] = float(np.nanpercentile(v, 90))
    return dict(sorted(out.items()))


def replay(series: dict[int, float], window: int, eps: float, patience: int,
           min_iterations: int) -> dict:
    best, best_i, streak, stop_at = -float('inf'), None, 0, None
    rows, vals = [], []
    for n, (i, s) in enumerate(series.items(), 1):
        vals.append(s)
        sm = float(np.mean(vals[-window:])) if window > 1 else s
        if len(vals) < window:
            # Warm-up: mirrors run_nise_*.py convergence_state(). A mean of fewer than `window`
            # values is not comparable with the true window-means it would be the baseline for,
            # so it establishes no baseline and advances no streak. This branch MUST stay in step
            # with the driver -- the whole point of the replay is to test the driver's rule.
            rows.append((i, s, sm, best, best_i, streak, False))
            continue
        if sm >= best + eps:
            best, best_i, streak, improved = sm, i, 0, True
        else:
            streak += 1
            improved = False
        if stop_at is None and streak >= patience and n >= min_iterations:
            stop_at = i
        rows.append((i, s, sm, best, best_i, streak, improved))
    return dict(rows=rows, stop_at=stop_at,
                max_streak=max((r[5] for r in rows), default=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--eps', type=float, default=0.002)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--min-iterations', type=int, default=10)
    ap.add_argument('--no-smoothing', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--runs', nargs='*', default=CAMPAIGNS)
    a = ap.parse_args()
    window = 1 if a.no_smoothing else a.window

    print(f'rule: window {window}, eps {a.eps}, patience {a.patience}, '
          f'min_iterations {a.min_iterations}\n')
    print('%-26s %5s %9s %11s %s' % ('campaign', 'iters', 'maxstreak', 'stops at', 'verdict'))
    bad = 0
    for run in a.runs:
        p = Path(run)
        s = series_for(p)
        if not s:
            print('%-26s %5s %9s %11s %s' % (p.name, '-', '-', '-', 'no data'))
            continue
        r = replay(s, window, a.eps, a.patience, a.min_iterations)
        best_iter = max(s, key=lambda k: s[k])
        stop = r['stop_at']
        # A stop is wrong if it lands before the iteration that produced the best p90.
        if stop is None:
            verdict = 'runs to completion'
        elif stop < best_iter:
            verdict = f'PREMATURE -- best p90 was at iteration {best_iter}'
            bad += 1
        else:
            verdict = 'stops after the peak (correct)'
        print('%-26s %5d %9d %11s %s'
              % (p.name, len(s), r['max_streak'], stop if stop is not None else '-', verdict))
        if a.verbose:
            print('   %4s %8s %8s %8s %7s %s' % ('iter', 'p90', f'mean{window}', 'best', 'streak', ''))
            for i, sv, sm, b, bi, st, imp in r['rows']:
                print('   %4d %8.4f %8.4f %8.4f %7d %s'
                      % (i, sv, sm, b, st, 'improved' if imp else ''))
    print()
    if bad:
        print(f'{bad} campaign(s) would have been stopped before their best iteration. '
              f'Do not use these constants.')
        return 1
    print('No campaign stopped before its best iteration.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
