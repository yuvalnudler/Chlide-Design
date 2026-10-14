#!/usr/bin/env python
"""One figure for ONE iteration, for sending while a campaign is still running.

paper_figures.py plots a finished trajectory; at iteration 0 there is no trajectory yet, only a
distribution. That distribution is still the thing worth seeing early: whether ligand pLDDT has
a usable right tail, whether the chirality gate is passing designs, and -- for a P(bind)
campaign -- whether the affinity head is producing a SPREAD rather than saturating at 0 or 1.
A saturated P(bind) carries no gradient and the objective is doing nothing, which is worth
knowing at iteration 0 rather than at iteration 25.

Usage: python tools/iteration_figure.py --run-dir <run> --iter N [--out fig.png]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK2, MUTED, GRID = '#52514e', '#8a8984', '#e3e2de'
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
                     'axes.edgecolor': MUTED, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def load(run: Path, it: int) -> pd.DataFrame:
    f = run / 'input' / 'sampling_dataframes' / f'iter_{it}_data.pkl'
    if not f.is_file():
        raise SystemExit(f'no dataframe for iteration {it} at {f}')
    return pd.read_pickle(f)


def figure(d: pd.DataFrame, run: Path, it: int, out: Path) -> Path:
    has_pbind = 'affinity_probability_binary' in d.columns and d['affinity_probability_binary'].notna().any()
    panels = 3 if has_pbind else 2
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 3.6))
    axes = np.atleast_1d(axes)

    lp = pd.to_numeric(d.get('ligand_plddts'), errors='coerce').dropna()
    axes[0].hist(lp, bins=28, color=C1, alpha=.85, edgecolor='white', linewidth=.5)
    axes[0].axvline(0.80, color=C2, lw=1.4)
    axes[0].annotate('gate 0.80', (0.80, axes[0].get_ylim()[1] * .95), xytext=(3, 0),
                     textcoords='offset points', color=C2, fontsize=7.5, va='top')
    axes[0].set_xlabel('ligand pLDDT')
    axes[0].set_ylabel('designs')
    axes[0].set_title(f'a  ligand pLDDT  (p90 {lp.quantile(.9):.3f})', fontsize=9.5, loc='left')

    rm = pd.to_numeric(d.get('ligand_rmsds'), errors='coerce')
    fin = np.isfinite(rm)
    axes[1].hist(rm[fin], bins=28, color=C3, alpha=.85, edgecolor='white', linewidth=.5)
    axes[1].axvline(2.5, color=C2, lw=1.4)
    cfg = d['ligand_config'].value_counts().to_dict() if 'ligand_config' in d else {}
    n_s = cfg.get('S', 0)
    axes[1].set_xlabel('ligand RMSD to design pose (A)')
    axes[1].set_title(f'b  self-consistency  ({fin.sum()}/{len(rm)} finite, '
                      f'{n_s}/{len(d)} correct epimer)', fontsize=9.5, loc='left')

    if has_pbind:
        pb = pd.to_numeric(d['affinity_probability_binary'], errors='coerce').dropna()
        axes[2].hist(pb, bins=28, color='#8a5fd3', alpha=.85, edgecolor='white', linewidth=.5)
        axes[2].set_xlim(0, 1)
        axes[2].set_xlabel('P(bind)  (Boltz-2 affinity head)')
        # A saturated P(bind) means the objective term carries no gradient.
        spread = pb.max() - pb.min()
        verdict = 'informative' if spread > 0.15 else 'SATURATED - carries no gradient'
        axes[2].set_title(f'c  P(bind)  median {pb.median():.3f}, spread {spread:.3f} '
                          f'({verdict})', fontsize=9.5, loc='left')

    for a in axes:
        a.grid(color=GRID, lw=.6)
        a.set_axisbelow(True)
    fig.suptitle(f'{run.name} - iteration {it}  ({len(d)} designs)', fontsize=11, x=.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def summary(d: pd.DataFrame, run: Path, it: int) -> str:
    lp = pd.to_numeric(d.get('ligand_plddts'), errors='coerce').dropna()
    pp = pd.to_numeric(d.get('protein_plddts'), errors='coerce').dropna()
    rm = pd.to_numeric(d.get('ligand_rmsds'), errors='coerce')
    fin = int(np.isfinite(rm).sum())
    cfg = d['ligand_config'].value_counts().to_dict() if 'ligand_config' in d else {}
    lines = [f'<b>{run.name} - iteration {it} complete</b>',
             f'{len(d)} designs',
             f'  ligand pLDDT  p90 <b>{lp.quantile(.9):.4f}</b>  median {lp.median():.4f}',
             f'  protein pLDDT median {pp.median():.4f}' if len(pp) else '',
             f'  self-consistent (finite RMSD): {fin}/{len(rm)}',
             f'  13-2 stereocentre: ' + ', '.join(f'{k}={v}' for k, v in cfg.items())]
    if 'affinity_probability_binary' in d.columns and d['affinity_probability_binary'].notna().any():
        pb = pd.to_numeric(d['affinity_probability_binary'], errors='coerce').dropna()
        lines.append(f'  P(bind) median <b>{pb.median():.3f}</b>  '
                     f'range {pb.min():.3f}-{pb.max():.3f}')
    return '\n'.join(x for x in lines if x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path,
                    default=Path(os.environ.get('CHLIDE_RUN_DIR')
                                 or Path(__file__).resolve().parent.parent))
    ap.add_argument('--iter', type=int, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--telegram', action='store_true')
    a = ap.parse_args()
    run = a.run_dir.resolve()
    d = load(run, a.iter)
    out = a.out or (run / 'results' / 'figures' / f'iter_{a.iter}.png')
    figure(d, run, a.iter, out)
    text = summary(d, run, a.iter)
    print(text.replace('<b>', '').replace('</b>', ''))
    print(f'  -> {out}')
    if a.telegram:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        from chlide import notify
        notify.send_document(out, caption=text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
