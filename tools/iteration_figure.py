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
    has_pbind = ('affinity_probability_binary' in d.columns
                 and pd.to_numeric(d['affinity_probability_binary'],
                                   errors='coerce').notna().any())
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
    pbcol = pd.to_numeric(d.get('affinity_probability_binary'), errors='coerce') \
        if 'affinity_probability_binary' in d.columns else None
    if pbcol is not None and pbcol.notna().any():
        pb = pbcol.dropna()
        av = pd.to_numeric(d.get('affinity_pred_value'), errors='coerce').dropna()
        spread = pb.max() - pb.min()
        # The paper filtered designs at P(bind) > 0.5, tightening to 0.8 only when more than
        # 3,000 passed. Reporting both tells you which cutoff this campaign can afford.
        over5 = 100 * (pb > 0.5).mean()
        over8 = 100 * (pb > 0.8).mean()
        lines += ['',
                  f'<b>P(bind)</b>  (Boltz-2 affinity head -- the v9 objective term)',
                  f'  median <b>{pb.median():.3f}</b>   range {pb.min():.3f}-{pb.max():.3f}'
                  f'   spread {spread:.3f}',
                  f'  above the paper\'s 0.5 filter: <b>{over5:.0f}%</b>'
                  f'   above 0.8: {over8:.0f}%']
        if len(av):
            # affinity_pred_value is log10(IC50) on a uM scale; more negative is tighter.
            lines.append(f'  predicted affinity: median {av.median():+.2f} '
                         f'log10(IC50/uM)  ~= {10 ** av.median() * 1000:.0f} nM')
        if spread <= 0.15:
            lines.append('  <b>WARNING: saturated</b> -- a P(bind) with no spread carries no '
                         'gradient, so the objective term is contributing nothing.')
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


# --------------------------------------------------------------------------- trajectory

def load_all(run: Path) -> dict[int, pd.DataFrame]:
    out = {}
    d = run / 'input' / 'sampling_dataframes'
    if d.is_dir():
        for f in sorted(d.glob('iter_*_data.pkl')):
            try:
                out[int(f.stem.split('_')[1])] = pd.read_pickle(f)
            except Exception:                               # noqa: BLE001
                continue
    return dict(sorted(out.items()))


def trajectory(hist: dict[int, pd.DataFrame], run: Path, out: Path) -> Path | None:
    """How the campaign is moving, not just where it is.

    The point of the NISE loop is that each iteration should be better than the last. A single
    iteration's histogram cannot show that; this can, and it is what tells you early whether a
    trajectory is climbing, flat, or being driven by the P(bind) term alone.
    """
    if len(hist) < 2:
        return None
    its = sorted(hist)
    p90 = [pd.to_numeric(hist[i]['ligand_plddts'], errors='coerce').quantile(.9) for i in its]
    med = [pd.to_numeric(hist[i]['ligand_plddts'], errors='coerce').median() for i in its]
    has_pb = all('affinity_probability_binary' in hist[i]
                 and pd.to_numeric(hist[i]['affinity_probability_binary'],
                                   errors='coerce').notna().any() for i in its)
    pb = ([pd.to_numeric(hist[i]['affinity_probability_binary'], errors='coerce').median()
           for i in its] if has_pb else None)
    frac_s = [100 * (hist[i]['ligand_config'] == 'S').mean() if 'ligand_config' in hist[i]
              else np.nan for i in its]
    finite = [100 * np.isfinite(pd.to_numeric(hist[i]['ligand_rmsds'],
                                              errors='coerce')).mean() for i in its]

    n = 3 if has_pb else 2
    fig, ax = plt.subplots(1, n, figsize=(4.4 * n, 3.6))
    ax = np.atleast_1d(ax)

    ax[0].plot(its, p90, 'o-', color=C1, lw=1.8, ms=4, label='p90')
    ax[0].plot(its, med, 'o--', color=C1, lw=1.2, ms=3, alpha=.55, label='median')
    ax[0].axhline(0.80, color=C2, lw=1.2, ls=':')
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('ligand pLDDT')
    ax[0].set_title(f'a  the objective  (p90 {p90[0]:.3f} -> {p90[-1]:.3f})',
                    fontsize=9.5, loc='left')
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(its, frac_s, 'o-', color=C3, lw=1.8, ms=4, label='correct epimer (S)')
    ax[1].plot(its, finite, 'o--', color=MUTED, lw=1.2, ms=3, label='self-consistent')
    ax[1].set_ylim(0, 103)
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('% of designs')
    ax[1].set_title('b  gates', fontsize=9.5, loc='left')
    ax[1].legend(frameon=False, fontsize=8, loc='lower right')

    if has_pb:
        ax[2].plot(its, pb, 'o-', color='#8a5fd3', lw=1.8, ms=4)
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel('median P(bind)')
        ax[2].set_ylim(0, 1)
        # NISE maximises ligand pLDDT + P(bind); if this line is flat the second term is
        # contributing nothing and the campaign is effectively a plain ligand-pLDDT run.
        delta = pb[-1] - pb[0]
        ax[2].set_title(f'c  P(bind)  ({pb[0]:.3f} -> {pb[-1]:.3f}, {delta:+.3f})',
                        fontsize=9.5, loc='left')

    for a in ax:
        a.grid(color=GRID, lw=.6)
        a.set_axisbelow(True)
    fig.suptitle(f'{run.name} - trajectory through iteration {its[-1]}', fontsize=11,
                 x=.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def trajectory_summary(hist: dict[int, pd.DataFrame], run: Path) -> str:
    its = sorted(hist)
    p90 = [pd.to_numeric(hist[i]['ligand_plddts'], errors='coerce').quantile(.9) for i in its]
    best = max(p90)
    lines = [f'<b>{run.name} - trajectory through iteration {its[-1]}</b>',
             f'{sum(len(hist[i]) for i in its):,} designs over {len(its)} iterations',
             f'  ligand pLDDT p90: {p90[0]:.4f} -> <b>{p90[-1]:.4f}</b>  '
             f'(best {best:.4f} at iter {its[p90.index(best)]})']
    if all('affinity_probability_binary' in hist[i]
           and pd.to_numeric(hist[i]['affinity_probability_binary'],
                             errors='coerce').notna().any() for i in its):
        pb = [pd.to_numeric(hist[i]['affinity_probability_binary'],
                            errors='coerce').median() for i in its]
        o5 = [100 * (pd.to_numeric(hist[i]['affinity_probability_binary'],
                                   errors='coerce') > 0.5).mean() for i in its]
        lines.append(f'  P(bind) median: {pb[0]:.3f} -> <b>{pb[-1]:.3f}</b> '
                     f'({pb[-1] - pb[0]:+.3f})')
        lines.append(f'  above 0.5 filter: {o5[0]:.0f}% -> <b>{o5[-1]:.0f}%</b>')
        if abs(pb[-1] - pb[0]) < 0.02:
            lines.append('  <b>P(bind) is flat</b> -- NISE maximises ligand pLDDT + P(bind); '
                         'if this term does not move, v9 is a plain ligand-pLDDT run.')
    if 'ligand_config' in hist[its[-1]]:
        s = 100 * (hist[its[-1]]['ligand_config'] == 'S').mean()
        lines.append(f'  correct epimer at iter {its[-1]}: {s:.0f}%')
    return '\n'.join(lines)
