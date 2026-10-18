#!/usr/bin/env python
"""Side-by-side comparison of finished campaigns, in the same form as the per-campaign report.

Every per-campaign report answers "how did THIS run go". None of them answer "is this run
better than the last one", which is the question that decides what to build next and which
proteins to order. This renders that comparison from the campaigns' own shortlists and
cascade metrics -- nothing is typed in, so it cannot drift from the data the way a written
summary does.

Usage:
    python tools/compare_campaigns.py --run /path/to/nise_bdfp44_v7 \
                                      --run /path/to/nise_bdfp44_v8 \
                                      --run /path/to/nise_bdfp44_v9 \
                                      --out comparison.html
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, str(HERE / f'{name}.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Higher is better unless listed here. Drives which cell gets highlighted per row.
LOWER_IS_BETTER = {'buns_score_paper_best', 'buns_score_paper_med', 'sasa_mg_A2',
                   'axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop'}
# No direction at all: a window, or a count whose size is not itself a virtue.
NO_DIRECTION = {'his_mg', 'n_shortlist', 'n_pool', 'iterations', 'mg_out_of_plane',
                'n4_planarity'}

ROWS = [
    ('n_pool',              'Designs scored by the cascade',   '{:,.0f}'),
    ('n_shortlist',         'Shortlist survivors',             '{:,.0f}'),
    ('survival',            'Survival rate',                   '{:.1%}'),
    ('buns_score_paper_best', 'BUNs Score, best',              '{:.0f}'),
    ('buns_score_paper_med',  'BUNs Score, median',            '{:.0f}'),
    ('lig_buns_zero',       'Ligand BUNs = 0',                 '{:.0%}'),
    ('buried_frac',         'Ligand burial, median',           '{:.3f}'),
    ('buried_frac_core',    'Core burial, median',             '{:.3f}'),
    ('buried_frac_apolar',  'Apolar burial, median',           '{:.3f}'),
    ('packstat',            'Packstat, median',                '{:.3f}'),
    ('sasa_mg_A2',          'SASA Mg, median (A^2)',           '{:.2f}'),
    ('his_mg',              'His-Mg, median (A)',              '{:.2f}'),
    ('ligand_plddt',        'Ligand pLDDT, median',            '{:.3f}'),
    ('interface_pae',       'Dimer interface pAE, median',     '{:.1f}'),
    ('axial_angle',         'Axial angle, median (deg)',       '{:.1f}'),
    ('his_lonepair_angle',  'Lone-pair angle, median (deg)',   '{:.1f}'),
    ('mg_imidazole_oop',    'Mg off-imidazole, median (deg)',  '{:.1f}'),
]

CASCADE_STAGES = [
    ('minimised',                  lambda d: d.his_mg.notna()),
    ('His-Mg 2.0-2.6 A',           lambda d: d.his_mg.between(2.0, 2.6)),
    ('protein RMSD->min < 1.5 A',  lambda d: d.ca_rmsd_to_min < 1.5),
    ('ligand RMSD->min < 1.5 A',   lambda d: d.lig_rmsd_to_min < 1.5),
    ('packstat > 0.55',            lambda d: d.packstat > 0.55),
    ('apo-repack clashes = 0',     lambda d: d.apo_clashes == 0),
    ('no buried charged residues', lambda d: d.buried_charged == 0),
]


def numeric(d: pd.DataFrame) -> pd.DataFrame:
    for c in d.columns:
        if c not in ('name', 'sequence', 'output', 'xfold_config', 'stem', 'input'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def collect(run: Path, mr) -> dict:
    """Everything the report needs from one campaign. Missing pieces stay None rather than
    raising, so a campaign that never ran a stage still appears in the table."""
    out = {'path': run, 'name': run.name}
    try:
        out['engine'] = mr._engine(run)
        out['cross'] = mr._cross_engine(run)
    except Exception:
        out['engine'] = out['cross'] = '?'

    mf = run / 'run_manifest.json'
    out['objective'] = '?'
    if mf.is_file():
        try:
            m = json.loads(mf.read_text())
            cfg = m.get('config', {})
            out['objective'] = (cfg.get('loop') or {}).get('objective', '?')
            out['per_backbone'] = (cfg.get('loop') or {}).get('per_backbone')
        except Exception:
            pass

    casc_f = run / 'results' / 'cascade' / 'cascade_metrics.csv'
    out['cascade'] = numeric(pd.read_csv(casc_f, comment='#')) if casc_f.is_file() else None
    out['n_pool'] = len(out['cascade']) if out['cascade'] is not None else np.nan

    sl_f = run / 'results' / 'final' / 'shortlist.csv'
    if sl_f.is_file():
        d = numeric(mr.load(run))
        out['short'] = d
        out['n_shortlist'] = len(d)
        out['survival'] = len(d) / out['n_pool'] if out['n_pool'] else np.nan
        if 'buns_score_paper' in d:
            out['buns_score_paper_best'] = d.buns_score_paper.min()
            out['buns_score_paper_med'] = d.buns_score_paper.median()
        if 'buns_ligand' in d:
            out['lig_buns_zero'] = (d.buns_ligand == 0).mean()
        for c in ('buried_frac', 'buried_frac_core', 'buried_frac_apolar', 'packstat',
                  'sasa_mg_A2', 'his_mg', 'ligand_plddt', 'interface_pae',
                  'axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop'):
            if c in d and d[c].notna().any():
                out[c] = d[c].median()
    else:
        out['short'] = None
        out['n_shortlist'] = np.nan

    iters = run / 'input' / 'sampling_dataframes'
    out['iterations'] = len(list(iters.glob('iter_*_data.pkl'))) if iters.is_dir() else np.nan

    odir = run / 'results' / 'final' / 'ORDER'
    orders = sorted(odir.glob('order*.csv')) if odir.is_dir() else []
    o = pd.read_csv(orders[0]) if orders else None

    # Campaigns whose cascade predates the geometry columns have none in their shortlist.
    # Measure the ORDER structures directly instead, so every campaign is comparable on the
    # one set that matters -- the designs you would actually synthesise.
    if o is not None and 'his_lonepair_angle' not in o.columns:
        mg = _load_module('metal_geometry')
        geo = {}
        for pdb in sorted(odir.glob('order*.pdb')):
            g = mg.from_pdb(str(pdb))
            if g:
                geo[pdb.name.split('_', 1)[1][:-4]] = g
        if geo:
            for col in ('axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop',
                        'mg_out_of_plane', 'n4_planarity'):
                o[col] = o['name'].map({k: v.get(col) for k, v in geo.items()})
            out['geometry_measured'] = True
    out['order'] = o
    return out


def funnel(c: dict) -> list[tuple[str, int]]:
    d = c['cascade']
    if d is None or not len(d):
        return []
    cum, rows = d.copy(), []
    for label, fn in CASCADE_STAGES:
        try:
            mask = fn(cum)
        except AttributeError:
            continue
        cum = cum[mask.reindex(cum.index, fill_value=False)]
        rows.append((label, len(cum)))
    return rows


def best_index(key: str, vals: list[float]) -> int | None:
    """Which column wins this row. None where 'better' is not defined."""
    if key in NO_DIRECTION:
        return None
    finite = [(i, v) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    pick = min if key in LOWER_IS_BETTER else max
    return pick(finite, key=lambda t: t[1])[0]


def esc(x) -> str:
    return html.escape(str(x))


def build(camps: list[dict]) -> str:
    n = len(camps)
    heads = ''.join(f'<th class=n>{esc(c["name"])}</th>' for c in camps)

    # ---- setup ----------------------------------------------------------------
    setup = ''
    for label, key in (('Fold engine', 'engine'), ('Cross-check', 'cross'),
                       ('NISE objective', 'objective'),
                       ('Iterations completed', 'iterations'),
                       ('Sequences per backbone', 'per_backbone')):
        cells = ''
        for c in camps:
            v = c.get(key)
            v = '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
            v = f'{v:.0f}' if isinstance(v, float) else v
            cells += f'<td class=n>{esc(v)}</td>'
        setup += f'<tr><td>{esc(label)}</td>{cells}</tr>'

    # ---- headline measures ----------------------------------------------------
    body = ''
    for key, label, fmt in ROWS:
        vals = [c.get(key) for c in camps]
        vals = [v if (v is not None and isinstance(v, (int, float, np.floating))
                      and np.isfinite(v)) else None for v in vals]
        if all(v is None for v in vals):
            continue
        win = best_index(key, vals)
        cells = ''
        for i, v in enumerate(vals):
            txt = '-' if v is None else fmt.format(v)
            cls = 'n good' if (win is not None and i == win) else 'n'
            cells += f'<td class="{cls}">{txt}</td>'
        body += f'<tr><td>{esc(label)}</td>{cells}</tr>'

    # ---- funnels --------------------------------------------------------------
    fns = [funnel(c) for c in camps]
    labels, fun = [], ''
    for f in fns:
        for lab, _ in f:
            if lab not in labels:
                labels.append(lab)
    for lab in labels:
        cells = ''
        for i, f in enumerate(fns):
            m = dict(f)
            v = m.get(lab)
            pool = camps[i].get('n_pool')
            if v is None:
                cells += '<td class=n>-</td>'
            else:
                pc = f' <span style="opacity:.5">({100*v/pool:.1f}%)</span>' if pool else ''
                cells += f'<td class=n>{v:,}{pc}</td>'
        fun += f'<tr><td>{esc(lab)}</td>{cells}</tr>'

    # ---- the BUNs / burial tension -------------------------------------------
    from scipy.stats import spearmanr
    corr = ''
    for c in camps:
        d = c.get('short')
        if d is None or 'buns_ligand' not in d or 'buried_frac_core' not in d:
            corr += f'<tr><td>{esc(c["name"])}</td><td colspan=3>no data</td></tr>'
            continue
        nun = d.buns_ligand.nunique()
        if nun < 2:
            corr += (f'<tr><td>{esc(c["name"])}</td><td class=n>{d.buns_ligand.iloc[0]:.0f}</td>'
                     f'<td class=n>-</td><td class=why>ligand BUNs is constant across all '
                     f'{len(d)} designs, so there is nothing to correlate</td></tr>')
            continue
        r = spearmanr(d.buried_frac_core, d.buns_ligand, nan_policy='omit')
        verdict = ('supports it' if r.statistic > 0.2 and r.pvalue < 0.05
                   else 'contradicts it' if r.statistic < -0.2 and r.pvalue < 0.05
                   else 'no signal')
        corr += (f'<tr><td>{esc(c["name"])}</td><td class=n>{d.buns_ligand.median():.0f}</td>'
                 f'<td class=n>{r.statistic:+.2f} (p={r.pvalue:.1g})</td>'
                 f'<td class=why>{verdict}</td></tr>')

    # ---- order sets -----------------------------------------------------------
    orders = ''
    for c in camps:
        o = c.get('order')
        if o is None or not len(o):
            continue
        cols = [x for x in ('name', 'buns_score_paper', 'buried_frac', 'buried_frac_core',
                            'packstat', 'interface_pae', 'af3_mono_plddt', 'axial_angle',
                            'his_lonepair_angle', 'mg_imidazole_oop',
                            'n4_planarity') if x in o.columns]
        th = ''.join(f'<th class={"n" if x != "name" else ""}>{esc(x)}</th>' for x in cols)
        tr = ''
        for _, r in o.iterrows():
            tds = ''
            for x in cols:
                v = r[x]
                if x == 'name':
                    tds += f'<td><code>{esc(str(v).replace("_model_0", ""))}</code></td>'
                else:
                    tds += f'<td class=n>{"-" if pd.isna(v) else f"{float(v):.2f}"}</td>'
            tr += f'<tr>{tds}</tr>'
        note = (' <span style="font-weight:400;color:var(--mut);font-size:.8rem">'
                '(geometry measured from the ORDER structures: this campaign\'s cascade '
                'predates those columns)</span>') if c.get('geometry_measured') else ''
        orders += (f'<h3 style="font-size:.95rem;margin:18px 0 6px">{esc(c["name"])}{note}</h3>'
                   f'<div class=scroll><table><thead><tr>{th}</tr></thead>'
                   f'<tbody>{tr}</tbody></table></div>')

    gen = datetime.now().strftime('%Y-%m-%d %H:%M')
    names = ', '.join(c['name'] for c in camps)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Campaign comparison</title><style>
:root{{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--card:#fff;--acc:#1f6f6b}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--fg:#e9e9e6;--mut:#9a9a95;--line:#2c2c33;--card:#1d1d22;--acc:#5fd3cb}}}}
:root[data-theme=dark]{{--bg:#16161a;--fg:#e9e9e6;--mut:#9a9a95;--line:#2c2c33;--card:#1d1d22;--acc:#5fd3cb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:0 16px 64px}}
.w{{max-width:1180px;margin:0 auto}}h1{{font-size:1.6rem;margin:28px 0 4px}}
h2{{font-size:1.1rem;margin:34px 0 10px;color:var(--acc)}}
p.sub{{color:var(--mut);margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:var(--card);font-weight:600;border-bottom:2px solid var(--line)}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:hover{{background:rgba(127,127,127,.09)}}
.scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
td.good{{color:var(--acc);font-weight:650}}
td.why{{white-space:normal;color:var(--mut);font-size:12px;max-width:40ch}}
code{{font-size:.9em;background:rgba(127,127,127,.12);padding:1px 5px;border-radius:4px}}
small.note{{color:var(--mut);display:block;margin-top:8px;white-space:normal;max-width:78ch}}
.callout{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:8px;padding:12px 16px;margin:14px 0}}
.callout p{{margin:.4em 0;white-space:normal}}
</style></head><body><div class=w>
<h1>Campaign comparison</h1>
<p class=sub>{esc(names)} &middot; generated {esc(gen)} by <code>tools/compare_campaigns.py</code>.
Every number is read from each campaign's own shortlist and cascade metrics.</p>

<h2>Setup</h2>
<div class=scroll><table><thead><tr><th>Parameter</th>{heads}</tr></thead>
<tbody>{setup}</tbody></table></div>

<h2>Headline measures</h2>
<div class=scroll><table><thead><tr><th>Measure (shortlist)</th>{heads}</tr></thead>
<tbody>{body}</tbody></table></div>
<small class=note>Highlighted = best in that row. No winner is marked for His-Mg (2.0&ndash;2.6&nbsp;&#8491;
is a window, not a direction), for counts, or for Mg out-of-plane and N4 planarity, which are
fixed by the Rosetta ligand params rather than by the design &mdash; see
<code>docs/CALIBRATION.md</code>.</small>

<h2>Terminal cascade funnel</h2>
<div class=scroll><table><thead><tr><th>Stage (cumulative)</th>{heads}</tr></thead>
<tbody>{fun}</tbody></table></div>
<small class=note>Cumulative survivors, so each row shows what that filter costs on top of
everything above it.</small>

<h2>Does BUNs disagree with burial?</h2>
<div class=callout>
<p>BUNs counts <em>buried</em> unsatisfied polar atoms. A design that buries the ligand less
has fewer buried polar atoms to leave unsatisfied, so a low BUNs score can mean good H-bonding
<em>or</em> an under-buried ligand &mdash; the measure cannot tell them apart.</p>
<p>If the second reading is the right one here, core burial and ligand BUNs should correlate
<em>positively</em> within a campaign: bury more, strand more. That is testable, and the test
is below.</p>
</div>
<div class=scroll><table>
<thead><tr><th>Campaign</th><th class=n>Ligand BUNs, median</th>
<th class=n>rho: core burial vs ligand BUNs</th><th>Reading</th></tr></thead>
<tbody>{corr}</tbody></table></div>
<small class=note>Spearman over the shortlist. This is evidence about how to read the ranking,
not a result about binding &mdash; nothing here has been measured in the lab.</small>

<h2>Suggested order sets</h2>
{orders or '<p class=sub>No campaign has an ORDER directory.</p>'}
<small class=note>Produced by <code>tools/order_set.py</code>: shortlist &rarr; dimer pAE
&gt; 14 and cross-fold ligand pLDDT &ge; 0.80 where those stages ran &rarr; rank by BUNs Score
&rarr; drop near-duplicates above 90% sequence identity.</small>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, action='append', required=True,
                    help='campaign run directory; repeat, in the order to display')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(HERE.parent / 'src'))
    mr = _load_module('make_report')

    camps = []
    for r in a.run:
        if not r.is_dir():
            print(f'  skipping {r}: not a directory', file=sys.stderr)
            continue
        c = collect(r, mr)
        camps.append(c)
        print(f'  {c["name"]}: pool {c.get("n_pool", float("nan")):,.0f}, '
              f'shortlist {c.get("n_shortlist", float("nan")):,.0f}, engine {c["engine"]}')
    if not camps:
        print('no campaigns to compare', file=sys.stderr)
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(camps))
    print(f'\n-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
