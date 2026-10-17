#!/usr/bin/env python
"""Paper-style figures for a NISE chlorophyllide campaign, and the Telegram report.

Figures follow Polizzi et al. 2025:

  1. self-consistency shift   protein vs ligand RMSD, first vs last iteration, with marginal
                              distributions (their Fig. 2e)
  2. trajectory               ligand pLDDT and binding-site NLL per iteration, as stacked panels
                              sharing an x axis -- NOT a twin y axis, which makes two different
                              scales look comparable when they are not
  3. filter cascade           survivors at each stage of the Fig. E7 funnel
  4. metal coordination       His NE2-Mg before and after relax against the target band (our
                              addition; the paper's ligand has no metal)
  5. shortlist                combined BUNs vs packstat, shaded by AF3 interface pAE

Palette is the validated categorical default (blue/orange/aqua/yellow): all six checks pass,
worst adjacent CVD dE 9.1 (protan). The contrast warning on the lighter hues is discharged by
direct labels and legends, which every figure here carries.

Usage:
    python tools/paper_figures.py --run-dir <run> [--out results/figures] [--telegram]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# validated categorical slots (light mode, surface #fcfcfb)
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GRID = '#e3e2de'
SEQ = plt.cm.viridis   # sequential, single perceptual ramp, light->dark

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def engine(run: Path) -> str:
    """Which model folded this campaign's designs.

    Every structure these figures measure -- the sampling pickles, the global_top predictions the
    Rosetta cascade minimises -- is written by the driver's folding engine into a Boltz-shaped
    directory tree, whichever engine that is. The layout therefore proves nothing; the driver's
    import does. v8 is AF3-led and its fig. 4 still said "Boltz prediction".
    """
    for drv in run.glob('run_nise_*.py'):
        if 'from af3_adapter import' in drv.read_text():
            return 'AF3'
    return 'Boltz'


def load_iterations(run: Path) -> dict:
    out = {}
    for f in glob.glob(str(run / 'input' / 'sampling_dataframes' / 'iter_*_data.pkl')):
        m = re.search(r'iter_(\d+)_data\.pkl$', f)
        if m:
            try:
                out[int(m.group(1))] = pd.read_pickle(f)
            except Exception:
                pass
    return dict(sorted(out.items()))


def fig_self_consistency(hist: dict, out: Path) -> Path | None:
    """Paper Fig. 2e: the distribution should move toward the origin as the loop runs."""
    if len(hist) < 2:
        return None
    first, last = min(hist), max(hist)
    fig = plt.figure(figsize=(5.6, 5.6))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=.05, wspace=.05)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    for it, col, lab in ((first, C1, f'iteration {first}'), (last, C2, f'iteration {last}')):
        d = hist[it].dropna(subset=['protein_rmsds', 'ligand_rmsds'])
        ax.scatter(d['protein_rmsds'], d['ligand_rmsds'], s=5, alpha=.28, c=col,
                   linewidths=0, label=f'{lab}  (n={len(d)})')
        bins_x = np.linspace(0, 8, 45)
        bins_y = np.linspace(0, 14, 45)
        axx.hist(d['protein_rmsds'], bins=bins_x, color=col, alpha=.55, density=True, lw=0)
        axy.hist(d['ligand_rmsds'], bins=bins_y, color=col, alpha=.55, density=True,
                 orientation='horizontal', lw=0)

    ax.axvline(1.5, color=MUTED, ls=':', lw=1)
    ax.axhline(1.5, color=MUTED, ls=':', lw=1)
    ax.text(1.62, 2.1, 'self-consistency\nthresholds', color=MUTED, fontsize=7, va='bottom')
    ax.set_xlim(0, 8); ax.set_ylim(0, 14)
    ax.set_xlabel('protein C$\\alpha$ RMSD (Å)'); ax.set_ylabel('ligand heavy-atom RMSD (Å)')
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    for a in (axx, axy):
        a.axis('off')
    axx.set_title('Self-consistency shifts toward the origin as the loop runs',
                  fontsize=10, loc='left', pad=8)
    p = out / '1_self_consistency.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def fig_trajectory(hist: dict, out: Path) -> Path | None:
    """Two panels, one x axis. Deliberately not a twin y axis: pLDDT and NLL do not share a scale."""
    if not hist:
        return None
    its = sorted(hist)
    med = [hist[i]['ligand_plddts'].median() for i in its]
    q3 = [hist[i]['ligand_plddts'].quantile(.75) for i in its]
    best = [hist[i]['ligand_plddts'].max() for i in its]
    nll = [hist[i]['laser_bs_nll'].median() for i in its]
    promoted = [int((hist[i]['ligand_is_buried'].fillna(False)
                     & (hist[i]['ligand_rmsds'] < 2.5)
                     & (hist[i]['protein_rmsds'] < 1.5)).sum()) for i in its]

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.4), sharex=True,
                             gridspec_kw={'hspace': .18})
    a = axes[0]
    a.plot(its, best, '-o', color=C2, lw=2, ms=4, label='best')
    a.plot(its, q3, '-o', color=C1, lw=2, ms=4, label='third quartile')
    a.plot(its, med, '-o', color=C3, lw=2, ms=4, label='median')
    a.axhline(0.8, color=MUTED, ls=':', lw=1)
    a.annotate('paper threshold 0.8', (its[-1], 0.8), textcoords='offset points',
               xytext=(-4, -11), ha='right', color=MUTED, fontsize=7)
    a.set_ylabel('ligand pLDDT')
    # axis starts at 0.70, not 0: every value lies in 0.78-0.98 and the full 0-1 range
    # compresses the variation this panel exists to show. The 0.8 threshold stays visible.
    a.set_ylim(0.70, 1.0)
    a.legend(frameon=False, fontsize=8, ncol=3, loc='lower left')
    a.grid(color=GRID, lw=.6); a.set_axisbelow(True)
    a.set_title('Design trajectory', fontsize=10, loc='left')
    a.annotate(f'{best[-1]:.3f}', (its[-1], best[-1]), textcoords='offset points',
               xytext=(6, -2), color=C2, fontsize=8, fontweight='bold')

    b = axes[1]
    b.plot(its, nll, '-o', color=C1, lw=2, ms=4)
    b.set_ylabel('binding-site NLL')
    b.grid(color=GRID, lw=.6); b.set_axisbelow(True)
    b.annotate(f'{nll[-1]:.3f}', (its[-1], nll[-1]), textcoords='offset points',
               xytext=(6, -2), color=C1, fontsize=8)

    c = axes[2]
    c.bar(its, promoted, color=C3, width=.62)
    c.set_ylabel('designs promoted'); c.set_xlabel('NISE iteration')
    c.grid(color=GRID, lw=.6, axis='y'); c.set_axisbelow(True)
    c.annotate(f'{promoted[-1]}', (its[-1], promoted[-1]), textcoords='offset points',
               xytext=(0, 3), ha='center', color=INK2, fontsize=8)
    p = out / '2_trajectory.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def fig_cascade(casc: pd.DataFrame, n_pool: int, out: Path) -> Path | None:
    """Fig. E7 funnel: how many designs survive each filter, applied cumulatively."""
    if casc is None or not len(casc):
        return None
    stages, cum = [], casc.copy()
    stages.append(('cascade-passing designs', n_pool))
    stages.append(('minimised successfully', len(cum)))
    for label, mask in [
        ('His–Mg 2.0–2.6 Å', (cum.his_mg >= 2.0) & (cum.his_mg <= 2.6)),
        ('protein RMSD→min < 1.5 Å', cum.ca_rmsd_to_min < 1.5),
        ('ligand RMSD→min < 1.5 Å', cum.lig_rmsd_to_min < 1.5),
        ('packstat > 0.55', cum.packstat > 0.55),
        ('apo-repack clashes = 0', cum.apo_clashes == 0),
        ('no buried charged residues', cum.buried_charged == 0),
    ]:
        cum = cum[mask.reindex(cum.index, fill_value=False)]
        stages.append((label, len(cum)))

    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=C1, height=.62)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    for yy, v in zip(y, vals):
        ax.annotate(f'{v:,}', (v, yy), textcoords='offset points', xytext=(5, 0),
                    va='center', fontsize=8, color=INK)
    ax.set_xlabel('designs surviving (cumulative)')
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(color=GRID, lw=.6, axis='x'); ax.set_axisbelow(True)
    ax.set_title('Filter cascade (Polizzi et al. 2025, Fig. E7)', fontsize=10, loc='left')
    p = out / '3_cascade.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def fig_metal(casc: pd.DataFrame, out: Path, eng: str = 'Boltz') -> Path | None:
    """`his_mg_before` is measured on the folding engine's own output, before Rosetta touches it.

    Chain of custody: relax_pool.py records his_mg_before off index.csv's `source_path`, which is
    input/sampled_backbones/iter_*/boltz_results_boltz_inputs/predictions/*_model_0.pdb -- written
    by the LEADING engine (af3_adapter.to_boltz_layout in v8), not by Boltz. Hence `eng`.
    """
    if casc is None or not len(casc):
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(1.4, 3.2, 70)
    ax.axvspan(2.0, 2.6, color=C3, alpha=.13, lw=0)
    # y in AXES fraction, not data: this runs before the histograms, when get_ylim() is still
    # the default (0, 1), so a data-space y=1 dropped the label onto the x-axis tick labels.
    ax.annotate('target window\n2.0–2.6 Å', (2.3, 0.97), color=INK2, fontsize=7,
                xycoords=('data', 'axes fraction'), ha='center', va='top')
    ax.hist(casc.his_mg_before.dropna(), bins=bins, color=C1, alpha=.65, lw=0,
            label=f'{eng} prediction')
    ax.hist(casc.his_mg.dropna(), bins=bins, color=C2, alpha=.65, lw=0,
            label='after Rosetta relax')
    ax.axvline(2.15, color=MUTED, ls=':', lw=1)
    ax.annotate('ideal 2.15 Å', (2.15, 0), textcoords='offset points', xytext=(3, 6),
                color=MUTED, fontsize=7, rotation=90)
    ax.set_xlabel('His NE2 – Mg distance (Å)'); ax.set_ylabel('designs')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.set_title('Axial metal coordination', fontsize=10, loc='left')
    p = out / '4_metal.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def fig_shortlist(casc: pd.DataFrame, pae: dict, out: Path) -> Path | None:
    if casc is None or not len(casc):
        return None
    d = casc.dropna(subset=['buns_combined', 'packstat'])
    if not len(d):
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    surv = d[(d.apo_clashes == 0) & (d.buried_charged == 0) & (d.packstat > 0.55)]
    ax.scatter(d.buns_combined, d.packstat, s=9, c=MUTED, alpha=.22, linewidths=0,
               label=f'all minimised  (n={len(d)})')
    if len(surv):
        if pae:
            col = [pae.get(n, np.nan) for n in surv.name]
            finite = np.isfinite(np.array(col, dtype=float))
            sc = ax.scatter(surv.buns_combined[finite], surv.packstat[finite],
                            s=34, c=np.array(col, dtype=float)[finite], cmap=SEQ,
                            edgecolors='white', linewidths=.6,
                            label=f'cascade survivors  (n={len(surv)})')
            cb = fig.colorbar(sc, ax=ax, pad=.02)
            cb.set_label('AF3 two-copy interface pAE (higher = less aggregation-prone)', fontsize=7)
            cb.outline.set_edgecolor(MUTED)
        else:
            ax.scatter(surv.buns_combined, surv.packstat, s=34, c=C2,
                       edgecolors='white', linewidths=.6,
                       label=f'cascade survivors  (n={len(surv)})')
    ax.axhline(0.55, color=MUTED, ls=':', lw=1)
    ax.annotate('packstat 0.55', (ax.get_xlim()[1], 0.552), ha='right', va='bottom',
                color=MUTED, fontsize=7)
    ax.set_xlabel('buried unsatisfied polar atoms (protein + ligand)')
    ax.set_ylabel('packstat')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.set_title('Final ranking space (lower BUNs and higher packstat are better)',
                 fontsize=10, loc='left')
    p = out / '5_shortlist.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def read_dimer(run: Path) -> pd.DataFrame | None:
    """Full two-copy table: interface pAE AND the ipTM AF3 reports natively."""
    f = run / 'results' / 'af3_dimer' / 'interface_pae.csv'
    if not f.is_file():
        return None
    with f.open() as fh:
        d = pd.read_csv(fh, comment='#')
    for c in ('interface_pae', 'iptm', 'contact_pairs'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d if 'iptm' in d and d.iptm.notna().any() else None


def fig_dimer(dim: pd.DataFrame, out: Path) -> Path | None:
    """Where the aggregation gate should sit for THIS engine.

    The paper's pAE < 14 comes from AF2 multimer and was paired with visual
    inspection. AF3 reports ipTM for the same two-copy prediction, and the two
    agree at rho ~ -0.97 -- so ipTM is not a second opinion, it is the same
    measurement on a scale where the modes separate cleanly. Its empty bin is
    the engine's own answer to "where does a confident dimer begin".
    """
    if dim is None:
        return None
    d = dim.dropna(subset=['interface_pae', 'iptm'])
    fin = d[np.isfinite(d.interface_pae)]
    if len(fin) < 20:
        return None
    rho = fin.interface_pae.corr(fin.iptm, method='spearman')
    o = fin.sort_values('iptm')
    cross = float(np.interp(0.50, o.iptm.values, o.interface_pae.values))

    # Is there actually a valley? v8 (AF3-led) is bimodal; v5/v7 (Boltz-led) are flat.
    # Asserting "bimodal" in the title regardless is the same class of hardcoded-label
    # bug as fig. 4's "Boltz prediction", so measure it instead of claiming it.
    h, edges = np.histogram(fin.interface_pae, bins=np.arange(0, 30, 2))
    mid = (edges[:-1] + edges[1:]) / 2
    lo_pk = h[mid < 8].max() if (mid < 8).any() else 0
    hi_pk = h[mid > 16].max() if (mid > 16).any() else 0
    trough = h[(mid >= 8) & (mid <= 16)].min() if ((mid >= 8) & (mid <= 16)).any() else 0
    bimodal = lo_pk and hi_pk and trough < 0.6 * min(lo_pk, hi_pk)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    a1.hist(fin.interface_pae, bins=np.arange(0, 30, 1.5), color=C1, alpha=.8,
            edgecolor='white', linewidth=.5)
    a1.axvline(14, color=C2, lw=1.6)
    a1.annotate("paper's 14\n(AF2 multimer)", (14, 0.97), xytext=(3, 0),
                xycoords=('data', 'axes fraction'),
                textcoords='offset points', color=C2, fontsize=7.5, va='top')
    a1.axvline(cross, color=C3, lw=1.6, ls='--')
    a1.annotate(('ipTM valley\n= pAE %.0f' % cross) if bimodal else
                ('ipTM 0.50\n= pAE %.0f' % cross),
                (cross, 0.60), xytext=(-5, 0), xycoords=('data', 'axes fraction'),
                textcoords='offset points', color=C3, fontsize=7.5, va='top', ha='right')
    a1.set_xlabel('two-copy interface pAE  (mean over CB pairs < 5.0 A)')
    a1.set_ylabel('designs')
    a1.set_title('a  bimodal: a gate can sit in the valley' if bimodal else
                 'a  unimodal: no valley, so no threshold is self-evident here',
                 fontsize=9.5, loc='left')
    a1.grid(color=GRID, lw=.6); a1.set_axisbelow(True)

    sc = a2.scatter(fin.interface_pae, fin.iptm, s=24, c=fin.contact_pairs, cmap=SEQ,
                    edgecolors='white', linewidths=.4)
    cb = fig.colorbar(sc, ax=a2, pad=.02); cb.set_label('cross-chain contacts', fontsize=7)
    cb.outline.set_edgecolor(MUTED)
    a2.axvline(14, color=C2, lw=1.4)
    a2.axhline(0.50, color=C3, lw=1.4, ls='--')
    n_dis = int(((fin.interface_pae <= 14) & (fin.iptm < 0.50)).sum())
    a2.annotate(f'{n_dis} designs pAE rejects\nbut ipTM calls monomeric',
                (0.98, 0.97), xycoords='axes fraction', fontsize=7.5, color=INK2,
                va='top', ha='right')
    a2.set_xlabel('two-copy interface pAE'); a2.set_ylabel('AF3 ipTM (two copies)')
    a2.set_title(f'b  the two are one measurement  (Spearman {rho:.2f})',
                 fontsize=9.5, loc='left')
    a2.grid(color=GRID, lw=.6); a2.set_axisbelow(True)

    p = out / '6_dimer_calibration.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    return p


def read_cascade(run: Path):
    f = run / 'results' / 'cascade' / 'cascade_metrics.csv'
    if not f.is_file():
        return None
    with f.open() as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    for c in ('his_mg', 'his_mg_before', 'score', 'ca_rmsd_to_min', 'lig_rmsd_to_min',
              'packstat', 'apo_clashes', 'buried_charged', 'buns_ligand', 'buns_protein',
              'buns_combined'):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def read_pae(run: Path) -> dict:
    f = run / 'results' / 'af3_dimer' / 'interface_pae.csv'
    if not f.is_file():
        return {}
    with f.open() as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    out = {}
    for r in rows:
        try:
            out[r['name']] = float(r['interface_pae'])
        except Exception:
            pass
    return out


def _make_report():
    """make_report.py holds order_set() and order_reasons(); import rather than reimplement so
    the Telegram table and the HTML report can never disagree about why a design was picked."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_mr', str(Path(__file__).resolve().parent / 'make_report.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_order_table(run: Path, out: Path) -> Path | None:
    """The designs picked for synthesis, every measure, and why each one is there.

    This exists because the order set was only ever visible by opening the HTML report. The
    reasons are generated from each design's standing in the shortlist by make_report's
    order_reasons(), not written by hand, so they cannot drift from the numbers beside them.
    Returns None when the campaign has no ORDER directory yet.
    """
    try:
        mr = _make_report()
        df = mr.load(run)
        picks, src = mr.order_set(run)
    except Exception as e:
        print(f'  order table skipped: {type(e).__name__}: {e}')
        return None
    if not picks:
        return None
    reasons = mr.order_reasons(df, picks)
    idx = df.set_index('name')

    cols = [('buns_score_paper', 'BUNs', '{:.0f}', 'lo'),
            ('buried_frac', 'Lig bur', '{:.0%}', 'hi'),
            ('buried_frac_core', 'Core bur', '{:.0%}', 'hi'),
            ('buried_frac_apolar', 'Apolar', '{:.3f}', 'hi'),
            ('packstat', 'Packstat', '{:.3f}', 'hi'),
            ('sasa_mg_A2', 'SASA Mg', '{:.1f}', 'lo'),
            ('ligand_plddt', 'lig pLDDT', '{:.3f}', 'hi'),
            ('xfold_ligand_plddt', 'x-fold lig', '{:.3f}', 'hi'),
            ('interface_pae', 'Dimer pAE', '{:.1f}', 'hi'),
            # His-Mg is a WINDOW (2.0-2.6), not a direction: nothing in it is "best".
            ('his_mg', 'His-Mg', '{:.2f}', None),
            ('axial_angle', 'Axial', '{:.1f}', 'lo'),
            ('his_lonepair_angle', 'LonePair', '{:.1f}', 'lo'),
            ('mg_imidazole_oop', 'OffImid', '{:.1f}', 'lo')]
    cols = [c for c in cols if c[0] in df.columns]
    present = [k for k, _l, _f, _d in cols]

    header = ['#', 'Design'] + [lbl for _k, lbl, _f, _d in cols]
    body, why, kept = [], [], []
    for i, nm in enumerate(picks, 1):
        if nm not in idx.index:
            continue
        r = idx.loc[nm]
        short = nm.replace('_model_0', '')
        row = [str(i), short]
        for k, _lbl, fmt, _d in cols:
            v = r.get(k)
            row.append(fmt.format(v) if pd.notna(v) else '-')
        body.append(row)
        kept.append(nm)
        why.append(f'{i}. {short} — {reasons.get(nm, "")}')
    if not body:
        return None

    # Two stacked axes: a table that keeps its own proportions, and a text block that does not
    # have to be positioned against it by hand.
    nrow, nwhy = len(body), len(why)
    t_h, w_h = 0.52 * (nrow + 1), 0.34 * (nwhy + 3)
    fig, (ax_t, ax_w) = plt.subplots(
        2, 1, figsize=(max(15, 1.15 * len(header) + 4), t_h + w_h),
        gridspec_kw={'height_ratios': [t_h, w_h], 'hspace': 0.08})
    for ax in (ax_t, ax_w):
        ax.axis('off')

    # bbox makes the table fill its axes: without it matplotlib leaves a wide margin and the
    # table floats away from the text block below.
    tbl = ax_t.table(cellText=body, colLabels=header, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(header))))   # names were being clipped
    for j, (k, _lbl, _f, direction) in enumerate(cols, start=2):
        if direction is None:
            continue
        vals = [float(idx.loc[n][k]) for n in kept if pd.notna(idx.loc[n].get(k))]
        if not vals:
            continue
        target = min(vals) if direction == 'lo' else max(vals)
        for i2, n in enumerate(kept, start=1):
            v = idx.loc[n].get(k)
            if pd.notna(v) and float(v) == target:
                tbl[(i2, j)].set_facecolor('#d8f0d8')
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor('#e8e8e8')
        tbl[(0, j)].set_text_props(weight='bold')
    for i2 in range(1, nrow + 1):
        tbl[(i2, 1)].set_text_props(ha='left')

    ax_w.text(0, 1.0, 'Why each design is in the set '
                      '(top-decile standing in the shortlist, generated from the numbers):',
              transform=ax_w.transAxes, fontsize=10.5, weight='bold', va='top')
    step = 1.0 / (nwhy + 3)
    for n_i, line in enumerate(why):
        ax_w.text(0, 1.0 - step * (n_i + 1.25), line, transform=ax_w.transAxes,
                  fontsize=9.5, va='top')
    note = 'Green = best in the set for that column; His-Mg has none because 2.0-2.6 A is a window, not a direction.'
    geom = [l for k, l, _f, _d in cols if k in ('axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop')]
    if geom:
        note += (f'  {" / ".join(geom)} are coordination geometry in degrees, 0 = ideal; they are '
                 'REPORTED, not part of the ranking (docs/CALIBRATION.md).')
    ax_w.text(0, 1.0 - step * (nwhy + 2.0), note, transform=ax_w.transAxes,
              fontsize=8.5, va='top', style='italic', color='#444444', wrap=True)

    ax_t.set_title(f'{run.name} — suggested order set ({len(body)} designs, from {src})',
                   fontsize=13, weight='bold', loc='left')
    p_out = out / '7_order_set.png'
    fig.savefig(p_out, dpi=135, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--telegram', action='store_true')
    a = ap.parse_args()
    out = a.out or (a.run_dir / 'results' / 'figures')
    out.mkdir(parents=True, exist_ok=True)

    eng = engine(a.run_dir)
    hist = load_iterations(a.run_dir)
    casc = read_cascade(a.run_dir)
    pae = read_pae(a.run_dir)
    n_pool = 0
    mf = a.run_dir / 'results' / 'global_top' / 'metrics.csv'
    if mf.is_file():
        n_pool = sum(1 for l in mf.open() if l and not l.startswith('#')) - 1

    figs = [f for f in (
        fig_self_consistency(hist, out),
        fig_trajectory(hist, out),
        fig_cascade(casc, n_pool, out),
        fig_metal(casc, out, eng),
        fig_shortlist(casc, pae, out),
        fig_dimer(read_dimer(a.run_dir), out),
        fig_order_table(a.run_dir, out),
    ) if f]
    print(f'{len(figs)} figures in {out}')
    for f in figs:
        print('  ', f)

    if a.telegram:
        try:                      # vendored in this repo; no path dependency
            from chlide import notify as telegram
        except ImportError:       # running as a loose script from a campaign dir
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
            from chlide import notify as telegram
        its = sorted(hist)
        best = max(hist[i]['ligand_plddts'].max() for i in its) if its else float('nan')
        lines = [f'<b>{a.run_dir.name}: campaign complete</b>',
                 f'{len(its)} iterations · best ligand pLDDT <b>{best:.3f}</b> · '
                 f'{n_pool:,} designs passed the in-loop cascade']
        if casc is not None:
            surv = casc[(casc.apo_clashes == 0) & (casc.buried_charged == 0)
                        & (casc.packstat > 0.55) & (casc.ca_rmsd_to_min < 1.5)
                        & (casc.lig_rmsd_to_min < 1.5)]
            lines += ['', '<b>Terminal Rosetta cascade</b>',
                      f'  minimised: {len(casc):,}',
                      f'  His–Mg in 2.0–2.6 Å: {int(((casc.his_mg>=2.0)&(casc.his_mg<=2.6)).sum()):,}',
                      f'  packstat &gt; 0.55: {int((casc.packstat>0.55).sum()):,}',
                      f'  apo-repack clashes = 0: {int((casc.apo_clashes==0).sum()):,}',
                      f'  no buried charges: {int((casc.buried_charged==0).sum()):,}',
                      f'  <b>survivors: {len(surv):,}</b>']
            if len(surv):
                b = surv.nsmallest(3, 'buns_combined')
                lines += ['', '<b>Top by combined BUNs</b>']
                for _, r in b.iterrows():
                    lines.append(f'  {r["name"][:34]} — BUNs {r.buns_combined:.0f} '
                                 f'(lig {r.buns_ligand:.0f}/prot {r.buns_protein:.0f}), '
                                 f'packstat {r.packstat:.3f}, His–Mg {r.his_mg:.2f} Å')
        if pae:
            keep = sum(1 for v in pae.values() if v > 14)
            lines += ['', f'<b>AF3 two-copy interface pAE</b>: {keep}/{len(pae)} pass &gt; 14 '
                          f'(high pAE = unlikely to aggregate)']
        else:
            lines += ['', '<i>AF3 interface pAE not yet run</i>']
        # The order set also goes in as TEXT: the table figure is the readable form, but an
        # image cannot be copied out of Telegram and these names get pasted into other tools.
        try:
            mr = _make_report()
            picks, src = mr.order_set(a.run_dir)
            if picks:
                sdf = mr.load(a.run_dir)
                why = mr.order_reasons(sdf, picks)
                sidx = sdf.set_index('name')
                lines += ['', f'<b>Suggested order set</b> ({len(picks)} designs, {src})']
                for i, nm in enumerate(picks, 1):
                    if nm not in sidx.index:
                        continue
                    r = sidx.loc[nm]
                    bits = []
                    for k, lbl, fmt in (('buns_score_paper', 'BUNs', '{:.0f}'),
                                        ('packstat', 'packstat', '{:.3f}'),
                                        ('interface_pae', 'pAE', '{:.1f}')):
                        if k in sidx.columns and pd.notna(r.get(k)):
                            bits.append(f'{lbl} {fmt.format(r[k])}')
                    lines.append(f'  {i}. <code>{nm.replace("_model_0", "")}</code> — '
                                 f'{", ".join(bits)}')
                    lines.append(f'      <i>{why.get(nm, "")}</i>')
        except Exception as e:
            print(f'  order set text skipped: {type(e).__name__}: {e}')
        ok = telegram.send_message('\n'.join(lines))
        print('telegram message:', ok)
        if ok and figs:
            print('telegram album:', telegram.send_photo_album(figs, caption=f'{a.run_dir.name} — paper-style figures'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
