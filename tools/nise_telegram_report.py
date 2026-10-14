#!/usr/bin/env python
"""Telegram reporter for a NISE design campaign.

The main_pipeline reporter (scripts/telegram_report_bot.py) cannot watch this run: it reads
the Snakemake layout (report/iteration_*.done, iteration_*/ranked_candidates.csv) and calls
snakemake_alive(). NISE writes something else entirely -- one pickled dataframe per iteration
plus Boltz prediction directories. This reads the NISE layout directly and reuses only the
stdlib sender from scripts/telegram.py, so credentials and rate-limit handling stay in one place.

Per iteration it reports:
  * NISE's own in-loop admission gates (what actually drives the search)
  * the Polizzi et al. 2025 filter cascade (Fig. E7), the numbers the paper ranked designs by
  * Boltz-2 confidence (confidence_score, iptm, ligand_iptm, ptm, complex_plddt)
  * Mg coordination geometry -- Mg to the chlorin N4 centroid, and axial His NE2 to Mg
and sends six plots: objective trajectory, RMSD trajectory, the paper's protein-vs-ligand
RMSD scatter, Mg geometry, the filter cascade, and Boltz confidence distributions.

Usage:
    python nise_telegram_report.py --run-dir <run-dir> [--once]
                                   [--interval 300] [--from-iteration 0]
"""
from __future__ import annotations

import argparse
import json
import html
import re
import subprocess
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))   # vendored notify
sys.path.insert(0, os.environ.get('MAIN_PIPELINE_SCRIPTS',
                                  '/home/data/yuvaln/main_pipeline/scripts'))
import telegram  # noqa: E402  stdlib-only sender, never raises

import prody as pr  # noqa: E402
pr.confProDy(verbosity='none')

# ---------------------------------------------------------------------------
# Thresholds
#
# NISE in-loop gates: the values in the runner's params dict. These decide which poses become
# next round's backbones, i.e. they steer the search.
MIN_ITERATIONS = 10   # must match DesignCampaign.CONVERGENCE["min_iterations"]
GATE_LIG_RMSD = 2.5
GATE_PROT_RMSD = 1.5

# Polizzi et al. 2025, "Zero-shot design of drug-binding proteins via neural selection-expansion",
# Fig. E7 filter cascade applied to designs pooled across all NISE iterations. The main text
# quotes ligand RMSD < 1.0 A while Fig. E7 shows < 1.5 A; both are reported.
PAPER = {
    'protein_rmsd': 1.5,      # protein Calpha RMSD
    'ligand_rmsd': 1.5,       # ligand heavy-atom RMSD (Fig. E7)
    'ligand_rmsd_strict': 1.0,  # main-text value
    'ligand_plddt': 0.8,
    'protein_plddt': 0.8,
}
# Downstream paper stages (Rosetta packstat > 0.55, apo-repack clash count, AF2 interface pAE)
# need Rosetta/AF2 and are not computed here; noted in the report so the cascade is not
# mistaken for the full protocol.

# Axial His, 0-based positional index; survives Boltz renumbering to 1..135.
HIS_RESINDEX = 84
MG_NAMES = ('MG1', 'MG')
N4_SETS = (('NA1', 'NB1', 'NC1', 'ND1'), ('NA', 'NB', 'NC', 'ND'))
# Target Mg-N(His) coordination window, and how far Mg may sit from the chlorin N4 centroid.
COORD_LO, COORD_HI = 2.0, 2.6
IN_RING = 0.5


def iter_dirs(run_dir: Path):
    d = run_dir / 'input' / 'sampling_dataframes'
    out = {}
    for f in d.glob('iter_*_data.pkl'):
        m = re.search(r'iter_(\d+)_data\.pkl$', f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def structure_metrics(run_dir: Path, iteration: int) -> pd.DataFrame:
    """Boltz confidence + Mg geometry, read from the prediction directories.

    Collected independently of the iteration dataframe (rather than joined row-wise) because
    the dataframe carries no stable key back to the prediction directory names.
    """
    base = run_dir / 'input' / 'sampled_backbones' / f'iter_{iteration}'
    pdbs = sorted(base.glob('boltz_results_boltz_inputs/predictions/*/*_model_0.pdb'))
    if not pdbs:  # shards not yet consolidated
        pdbs = sorted(base.glob('gpu*/boltz_results_*/predictions/*/*_model_0.pdb'))
    rows = []
    for p in pdbs:
        rec = {'name': p.parent.name}
        # P(bind) lives in its own file beside the confidence one, written by the affinity
        # pass. Boltz names it affinity_<stem without _model_0>.json.
        aj = p.parent / ('affinity_' + p.stem[:-len('_model_0')] + '.json')
        if aj.exists():
            try:
                rec.update({k: v for k, v in json.load(aj.open()).items()
                            if isinstance(v, (int, float))})
            except Exception:
                pass
        cj = p.parent / f'confidence_{p.stem}.json'
        if cj.exists():
            try:
                rec.update({k: v for k, v in json.load(cj.open()).items()
                            if isinstance(v, (int, float))})
            except Exception:
                pass
        try:
            st = pr.parsePDB(str(p))
            lig = st.select('chid B')
            mg = next((lig.select(f'name {n}') for n in MG_NAMES
                       if lig is not None and lig.select(f'name {n}') is not None), None)
            n4 = next((lig.select('name ' + ' '.join(s)) for s in N4_SETS
                       if lig is not None and lig.select('name ' + ' '.join(s)) is not None
                       and len(lig.select('name ' + ' '.join(s))) == 4), None)
            if mg is not None and n4 is not None:
                c = mg.getCoords()[0]
                rec['mg_ring_offset'] = float(np.linalg.norm(c - n4.getCoords().mean(0)))
                ne2 = st.select(f'resindex {HIS_RESINDEX} and name NE2')
                if ne2 is not None:
                    rec['his_mg'] = float(np.linalg.norm(ne2.getCoords()[0] - c))
        except Exception:
            pass
        rows.append(rec)
    return pd.DataFrame(rows)


def cascade(df: pd.DataFrame) -> list:
    """Paper Fig. E7 filter cascade, each stage applied on top of the previous."""
    n = len(df)
    if not n:
        return []
    m = pd.Series(True, index=df.index)
    out = [('designs', n)]
    for label, col, op, thr in [
        ('protein RMSD < %.1f A' % PAPER['protein_rmsd'], 'protein_rmsds', 'lt', PAPER['protein_rmsd']),
        ('ligand RMSD < %.1f A' % PAPER['ligand_rmsd'], 'ligand_rmsds', 'lt', PAPER['ligand_rmsd']),
        ('ligand pLDDT > 0.8', 'ligand_plddts', 'gt', PAPER['ligand_plddt']),
        ('protein pLDDT > 0.8', 'protein_plddts', 'gt', PAPER['protein_plddt']),
    ]:
        if col not in df:
            continue
        v = df[col]
        m = m & ((v < thr) if op == 'lt' else (v > thr)) & v.notna()
        out.append((label, int(m.sum())))
    return out


def fmt(v, spec='%.2f'):
    return 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec % v


def pace_and_eta(run_dir: Path, iteration: int, total: int) -> list:
    """Per-iteration wall time, ETA to the iteration cap, and the earliest auto-stop.

    Timed from the mtimes of input/sampling_dataframes/iter_N_data.pkl, which the driver writes
    last in each iteration. Iteration 0 is excluded from the average: it samples one backbone
    (500 sequences) against three (1500) for every round after it, so including it flatters the
    pace by roughly a third.

    The stop line is the EARLIEST possible stop, not a prediction -- the convergence rule needs a
    7-iteration non-improving streak AND at least min_iterations complete, so the floor is what is
    knowable in advance. The run may go all the way to the cap.
    """
    import os, re, glob
    from datetime import datetime, timedelta
    try:
        pk = sorted(glob.glob(str(run_dir / 'input' / 'sampling_dataframes' / 'iter_*_data.pkl')),
                    key=lambda q: int(re.search(r'_(\d+)_', q).group(1)))
        ts = [(int(re.search(r'_(\d+)_', q).group(1)), os.path.getmtime(q)) for q in pk]
        if len(ts) < 2:
            return []
        durs = [(ts[k][0], (ts[k][1] - ts[k - 1][1]) / 60) for k in range(1, len(ts))]
        full = [d for i, d in durs if i > 0]
        if not full:
            return []
        avg = sum(full) / len(full)
        done = ts[-1][0]
        rem_cap = max(0, (total - 1 - done)) * avg / 60
        rem_min = max(0, (MIN_ITERATIONS - 1 - done)) * avg / 60
        out = [f"pace: {avg:.0f} min/iter (n={len(full)})   "
               f"ETA iter {total - 1}: {rem_cap:.1f} h, "
               f"{(datetime.now() + timedelta(hours=rem_cap)).strftime('%a %H:%M')}"]
        if rem_min > 0:
            out.append(f"  earliest auto-stop (iter {MIN_ITERATIONS - 1}): {rem_min:.1f} h, "
                       f"{(datetime.now() + timedelta(hours=rem_min)).strftime('%a %H:%M')}")
        return out
    except Exception:
        return []


def convergence_line(run_dir: Path) -> list:
    """Latest convergence state, read from the artifact the driver writes each iteration."""
    import json
    try:
        cs = json.loads((run_dir / 'logs' / 'convergence.json').read_text())
    except Exception:
        return []
    ser = cs.get('series') or {}
    if not ser:
        return []
    last = max(ser, key=lambda k: int(k))
    best = cs.get('best')
    shown = f"best {float(best):.4f}@{cs.get('best_iteration')}" if best is not None else "warm-up"
    return [f"convergence: p90 {float(ser[last]):.4f}   {shown}"
            f"   streak {cs.get('streak')}/{cs.get('patience')}"]


def summarise(run_dir: Path, iteration: int, df: pd.DataFrame, sm: pd.DataFrame,
              total: int) -> str:
    n = len(df)
    gates = (df.get('ligand_is_buried', pd.Series(dtype=bool)).fillna(False)
             & (df.get('ligand_rmsds', pd.Series(dtype=float)) < GATE_LIG_RMSD)
             & (df.get('protein_rmsds', pd.Series(dtype=float)) < GATE_PROT_RMSD))
    npass = int(gates.sum())
    buried = int(df.get('ligand_is_buried', pd.Series(dtype=bool)).fillna(False).sum())

    L = [f"<b>NISE iter {iteration}/{total - 1}</b>  <code>{run_dir.name}</code>",
         f"designs: {n}   ligand buried: {buried}/{n}"]
    L += convergence_line(run_dir)
    L += pace_and_eta(run_dir, iteration, total)
    L += [
         "",
         f"<b>NISE gates</b> (lig&lt;{GATE_LIG_RMSD} &amp; prot&lt;{GATE_PROT_RMSD} &amp; buried)"
         f": <b>{npass}/{n} promoted</b>"]

    def stat(col, spec='%.2f'):
        if col not in df or df[col].dropna().empty:
            return 'n/a'
        s = df[col].dropna()
        return f"min {spec % s.min()} / med {spec % s.median()} / max {spec % s.max()}"

    L += [f"  ligand RMSD  {stat('ligand_rmsds')}",
          f"  protein RMSD {stat('protein_rmsds')}",
          f"  ligand pLDDT {stat('ligand_plddts', '%.3f')}",
          f"  protein pLDDT {stat('protein_plddts', '%.3f')}",
          f"  LASErMPNN NLL {stat('laser_nll', '%.3f')}  (binding site {stat('laser_bs_nll', '%.3f')})"]

    casc = cascade(df)
    if casc:
        L += ["", "<b>Polizzi 2025 filter cascade</b> (Fig. E7)"]
        L += [f"  {html.escape(lab)}: {cnt}" for lab, cnt in casc]
        if 'ligand_rmsds' in df:
            strict = int(((df['ligand_rmsds'] < PAPER['ligand_rmsd_strict'])
                          & (df['protein_rmsds'] < PAPER['protein_rmsd'])).sum())
            L.append(f"  (main-text ligand RMSD &lt; 1.0 A with protein &lt; 1.5: {strict})")
        L.append("  <i>Rosetta packstat / apo-clash / AF2 pAE stages not computed</i>")

    if not sm.empty:
        L += ["", "<b>Boltz-2 confidence</b>"]
        for c, lab in [('confidence_score', 'confidence'), ('iptm', 'iptm'),
                       ('ligand_iptm', 'ligand_iptm'), ('ptm', 'ptm'),
                       ('complex_plddt', 'complex_pLDDT')]:
            if c in sm and sm[c].notna().any():
                s = sm[c].dropna()
                L.append(f"  {lab}: med {s.median():.3f}  max {s.max():.3f}")
        if 'mg_ring_offset' in sm and sm['mg_ring_offset'].notna().any():
            r = sm['mg_ring_offset'].dropna()
            h = sm['his_mg'].dropna() if 'his_mg' in sm else pd.Series(dtype=float)
            inring = int((r < IN_RING).sum())
            coord = int(((h >= COORD_LO) & (h <= COORD_HI)).sum()) if not h.empty else 0
            L += ["", "<b>Mg coordination</b>",
                  f"  Mg in chlorin (&lt;{IN_RING} A): {inring}/{len(r)}   med {r.median():.2f} A",
                  f"  His NE2-Mg {COORD_LO}-{COORD_HI} A: {coord}/{len(h) if not h.empty else 0}"
                  + (f"   min {h.min():.2f} A  med {h.median():.2f} A" if not h.empty else "")]
    return "\n".join(L)


def make_plots(run_dir: Path, hist: dict, sm_hist: dict, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    its = sorted(hist)
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})

    def series(col, fn):
        return [fn(hist[i][col].dropna()) if col in hist[i] and hist[i][col].notna().any()
                else np.nan for i in its]

    # 1. objective trajectory -- this is what NISE maximises
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(its, series('ligand_plddts', np.median), 'o-', label='ligand pLDDT (median)')
    ax.plot(its, series('ligand_plddts', np.max), 'o--', label='ligand pLDDT (best)', alpha=.7)
    ax.plot(its, series('protein_plddts', np.median), 's-', label='protein pLDDT (median)', alpha=.7)
    ax.axhline(PAPER['ligand_plddt'], color='crimson', ls=':', label='paper 0.8')
    ax.set_xlabel('NISE iteration'); ax.set_ylabel('pLDDT'); ax.set_ylim(0, 1)
    ax.set_title('Objective trajectory'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    p = out_dir / '1_objective.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 2. self-consistency RMSD trajectory
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(its, series('ligand_rmsds', np.median), 'o-', label='ligand RMSD (median)')
    ax.plot(its, series('ligand_rmsds', np.min), 'o--', label='ligand RMSD (best)', alpha=.7)
    ax.plot(its, series('protein_rmsds', np.median), 's-', label='protein RMSD (median)')
    ax.plot(its, series('protein_rmsds', np.min), 's--', label='protein RMSD (best)', alpha=.7)
    ax.axhline(GATE_LIG_RMSD, color='navy', ls=':', label=f'NISE ligand gate {GATE_LIG_RMSD}')
    ax.axhline(GATE_PROT_RMSD, color='crimson', ls=':', label=f'NISE protein gate {GATE_PROT_RMSD}')
    ax.set_xlabel('NISE iteration'); ax.set_ylabel('RMSD (A)'); ax.set_yscale('log')
    ax.set_title('Self-consistency'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    p = out_dir / '2_rmsd.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. paper Fig. 2e style: protein vs ligand RMSD, first vs latest iteration
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, col, lab in [(its[0], 'tab:blue', f'iter {its[0]}'),
                        (its[-1], 'tab:red', f'iter {its[-1]}')]:
        d = hist[i]
        if {'protein_rmsds', 'ligand_rmsds'} <= set(d.columns):
            ax.scatter(d['protein_rmsds'], d['ligand_rmsds'], s=6, alpha=.35, c=col, label=lab)
    ax.axvline(PAPER['protein_rmsd'], color='k', ls=':', lw=.8)
    ax.axhline(PAPER['ligand_rmsd'], color='k', ls=':', lw=.8)
    ax.set_xlabel('protein Ca RMSD (A)'); ax.set_ylabel('ligand heavy-atom RMSD (A)')
    ax.set_xlim(0, 10); ax.set_ylim(0, 15)
    ax.set_title('Self-consistency, first vs latest'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    p = out_dir / '3_scatter.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 4. Mg coordination -- the reason this campaign exists
    sits = sorted(sm_hist)
    if sits:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
        for ax_, col, lab, lo, hi in [
                (axes[0], 'mg_ring_offset', 'Mg to chlorin N4 centroid (A)', None, IN_RING),
                (axes[1], 'his_mg', 'His NE2 to Mg (A)', COORD_LO, COORD_HI)]:
            data = [sm_hist[i][col].dropna().values if col in sm_hist[i] else np.array([])
                    for i in sits]
            data = [d if len(d) else np.array([np.nan]) for d in data]
            ax_.boxplot(data, positions=sits, widths=.6, showfliers=False)
            if lo is not None:
                ax_.axhspan(lo, hi, color='tab:green', alpha=.15, label='target')
            else:
                ax_.axhline(hi, color='tab:green', ls=':', label='in ring')
            ax_.set_xlabel('iteration'); ax_.set_ylabel(lab)
            ax_.legend(fontsize=7); ax_.grid(alpha=.3)
        fig.suptitle('Mg coordination geometry', fontsize=10)
        p = out_dir / '4_metal.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 5. filter cascade on the latest iteration
    casc = cascade(hist[its[-1]])
    if casc:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labs = [c[0] for c in casc]; vals = [c[1] for c in casc]
        ax.barh(range(len(vals))[::-1], vals, color='tab:blue')
        ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labs, fontsize=7)
        for y, v in zip(range(len(vals))[::-1], vals):
            ax.text(v, y, f' {v}', va='center', fontsize=7)
        ax.set_xlabel('designs surviving')
        ax.set_title(f'Polizzi 2025 cascade, iteration {its[-1]}')
        p = out_dir / '5_cascade.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 6. Boltz-2 confidence distributions
    if sits:
        last = sm_hist[sits[-1]]
        cols = [c for c in ('confidence_score', 'iptm', 'ligand_iptm', 'ptm', 'complex_plddt')
                if c in last and last[c].notna().any()]
        if cols:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.boxplot([last[c].dropna().values for c in cols], labels=cols, showfliers=False)
            ax.set_ylim(0, 1); ax.set_ylabel('score'); ax.grid(alpha=.3)
            ax.set_title(f'Boltz-2 confidence, iteration {sits[-1]}')
            plt.setp(ax.get_xticklabels(), rotation=20, fontsize=7)
            p = out_dir / '6_confidence.png'; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 7. P(bind). NISE maximises ligand pLDDT + P(bind), so the question is not what the value
    # is but whether it MOVES, and whether it has spread -- a saturated P(bind) carries no
    # gradient and the objective term contributes nothing.
    pb_its = [i for i in sits if 'affinity_probability_binary' in sm_hist[i]
              and sm_hist[i]['affinity_probability_binary'].notna().any()]
    if pb_its:
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
        med = [sm_hist[i]['affinity_probability_binary'].median() for i in pb_its]
        o5 = [100 * (sm_hist[i]['affinity_probability_binary'] > 0.5).mean() for i in pb_its]
        ax[0].plot(pb_its, med, 'o-', color='#8a5fd3', lw=1.8, ms=4)
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('median P(bind)')
        ax[0].axhline(0.5, color='#eb6834', lw=1.2, ls=':')
        ax[0].set_title('P(bind) trajectory  (%.3f -> %.3f)' % (med[0], med[-1]), fontsize=9.5)
        ax2 = ax[0].twinx()
        ax2.plot(pb_its, o5, 's--', color='#8a8984', lw=1, ms=3)
        ax2.set_ylabel('% above 0.5 filter', color='#8a8984')
        ax2.set_ylim(0, 103)
        last = sm_hist[pb_its[-1]]['affinity_probability_binary'].dropna()
        ax[1].hist(last, bins=30, color='#8a5fd3', alpha=.85, edgecolor='white', linewidth=.5)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel('P(bind)')
        spread = float(last.max() - last.min())
        ax[1].set_title('iteration %d: spread %.3f%s' % (
            pb_its[-1], spread, '' if spread > 0.15 else '   SATURATED'), fontsize=9.5)
        for a in ax:
            a.grid(alpha=.3)
        p = out_dir / '7_pbind.png'
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path,
                    default=Path(os.environ.get('CHLIDE_RUN_DIR')
                                 or Path(__file__).resolve().parent.parent))
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--total', type=int, default=35)
    ap.add_argument('--from-iteration', type=int, default=0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--no-send', action='store_true', help='render locally, send nothing')
    ap.add_argument('--no-harvest', action='store_true',
                    help='do not refresh results/global_top after each iteration')
    ap.add_argument('--harvest-top', type=int, default=50)
    a = ap.parse_args()

    state_file = a.run_dir / 'logs' / '.telegram_sent.json'
    sent = set()
    if state_file.exists():
        try:
            sent = set(json.load(state_file.open()).get('iterations', []))
        except Exception:
            pass

    hist, sm_hist = {}, {}
    harvested = -1
    while True:
        found = iter_dirs(a.run_dir)
        for it, f in found.items():
            if it < a.from_iteration:
                continue
            try:
                hist[it] = pd.read_pickle(f)
            except Exception:
                continue
            if it not in sm_hist:
                sm_hist[it] = structure_metrics(a.run_dir, it)

        # Refresh results/global_top as soon as a new iteration lands, so the ranked designs track
        # the campaign instead of only existing once it ends -- a run can be interrupted at any
        # point (this one lost 9 iterations to a host reboot) and the harvest should already be
        # there. Run as a subprocess so a harvest failure cannot take down reporting, and keyed on
        # the newest iteration so it runs once per iteration rather than once per poll.
        newest = max(hist) if hist else -1
        if not a.no_harvest and newest > harvested:
            # Tools live in THIS repo, not in the run directory. They used to sit inside
            # the campaign, which is why this looked there.
            tool = Path(__file__).resolve().parent / 'harvest_global_top.py'
            if tool.is_file():
                try:
                    r = subprocess.run(
                        [sys.executable, str(tool), '--run-dir', str(a.run_dir),
                         '--top', str(a.harvest_top)],
                        capture_output=True, text=True, timeout=1800)
                    if r.returncode == 0:
                        tail = [l for l in r.stdout.splitlines() if 'cascade' in l]
                        print(f'[harvest] refreshed through iteration {newest}: '
                              f'{tail[0].strip() if tail else "ok"}', flush=True)
                        harvested = newest
                    else:
                        print(f'[harvest] failed (rc={r.returncode}): {r.stderr.strip()[:200]}',
                              flush=True)
                except Exception as e:
                    print(f'[harvest] error: {type(e).__name__}: {e}', flush=True)
            else:
                print(f'[harvest] {tool} not found; skipping', flush=True)

        todo = [i for i in sorted(hist) if i not in sent]
        for it in todo:
            text = summarise(a.run_dir, it, hist[it], sm_hist.get(it, pd.DataFrame()), a.total)
            plots = make_plots(a.run_dir, hist, sm_hist, a.run_dir / 'logs' / 'plots')
            if a.no_send:
                print(text); print('plots:', *[str(p) for p in plots], sep='\n  ')
                ok = True
            else:
                # Only record an iteration as reported if the text actually landed. Marking it
                # sent regardless would silently drop a report on any transient Telegram error,
                # and this runs unattended for days.
                ok = telegram.send_message(text)
                if ok and plots:
                    telegram.send_photo_album(plots, caption=f'NISE iteration {it}')
            if not ok:
                print(f'[report] iteration {it} send failed; will retry next poll', flush=True)
                continue
            sent.add(it)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            json.dump({'iterations': sorted(sent)}, state_file.open('w'))

        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == '__main__':
    sys.exit(main())
