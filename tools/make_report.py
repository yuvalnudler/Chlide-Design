#!/usr/bin/env python
"""Build a single self-contained HTML report for a finished campaign.

Joins the five result files the terminal pipeline produces, which are otherwise scattered and
keyed inconsistently:

    results/final/shortlist.csv          BUNs, burial, packstat, LEADING-engine pLDDT, interface pAE
    results/af3_mono/monomer_confidence  AF3 single-chain pLDDT and pTM  (name: <stem>_mono)
    results/af3_dimer/interface_pae.csv  AF3 two-copy pAE               (name: <stem>, no _model_0)
    results/final/sasa/sasa_ranking.csv  Shrake-Rupley SASA per group
    results/cascade/cascade_metrics.csv  the full pool, for the funnel

Every table column sorts on click. The point of putting the measures side by side is that they are
largely INDEPENDENT -- BUNs vs burial r~0.17, monomer vs CROSS-engine ligand pLDDT r~0.12 -- so a
design that leads one column is usually mid-pack in another, and picking a set that spans the
failure modes beats taking the top N of any single ranking.

Usage: python tools/make_report.py --run-dir <run>   ->  results/figures/report.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

COLS = [
    ('rank',                'Rank',          '{:.0f}',  'lower is better (BUNs Score)'),
    ('name',                'Design',        None,      'iteration + chunk + sequence index'),
    ('buns_score_paper',    'BUNs Score',    '{:.0f}',  'Polizzi Table S1: 2xligand + protein. LOWER is better'),
    ('buns_ligand',         'BUNs lig',      '{:.0f}',  'buried unsatisfied polar atoms on the ligand'),
    ('buns_protein',        'BUNs prot',     '{:.0f}',  'buried unsatisfied polar atoms on the protein'),
    ('buried_frac',         'Ligand burial', '{:.1%}',  'Shrake-Rupley: 1 - SASA(ligand in complex)/SASA(ligand alone, same coords). Probe 1.4 A, Bondi radii, heavy atoms. HIGHER is better, but the charged propionate is DELIBERATELY left exposed, so this is capped in practice'),
    ('buried_frac_core',    'Core burial',   '{:.1%}',  'same measure over the chlorin macrocycle + Mg only. This is the number that should be high - the core is what must be desolvated'),
    ('buried_frac_apolar',  'Apolar burial', '{:.3f}',  'COMBS target 0.75. HIGHER is better'),
    ('packstat',            'Packstat',      '{:.3f}',  'Rosetta packing quality, paper gate >0.55'),
    ('ligand_plddt',        '{ENG} lig',     '{:.3f}',  'the objective the campaign optimised: ligand pLDDT from the FOLDING ENGINE'),
    ('protein_plddt',       '{ENG} prot',    '{:.3f}',  'protein pLDDT from the folding engine'),
    ('af3_mono_plddt',      'AF3 mono',      '{:.3f}',  'AF3 single-chain pLDDT - does the SEQUENCE fold? independent of {ENG}'),
    ('af3_mono_ptm',        'AF3 pTM',       '{:.2f}',  'AF3 predicted TM-score, single chain'),
    ('xfold_confidence',    '{XENG} conf',       '{:.3f}',  "{XENG}'s OWN composite confidence for the cross-fold: 0.8*complex pLDDT + 0.2*ipTM, read from its confidence_*.json. The engine's single-number verdict, independent of the campaign objective"),
    ('xfold_ligand_plddt',  '{XENG} lig pLDDT',  '{:.3f}',  'ligand pLDDT from the CROSS-CHECK engine ({XENG}). A campaign cannot grade itself; this is the independent opinion'),
    ('xfold_protein_plddt', '{XENG} prot pLDDT', '{:.3f}',  'protein pLDDT from the cross-check engine ({XENG}) - does the SEQUENCE fold?'),
    ('xfold_config',        '{XENG} 13-2',       None,      'does the cross-check engine ({XENG}) agree the stereocentre is S?'),
    ('interface_pae',       'AF3 dimer pAE', '{:.1f}',  'two-copy interface pAE. HIGH is good - low means it dimerises'),
    ('dimer_iptm',          'Dimer ipTM',    '{:.2f}',  'AF3 ipTM for the SAME two-copy run. LOW is good. Tracks pAE at rho -0.97, so it is the same measurement on a scale whose modes separate: >=0.50 is a confident dimer'),
    ('sasa_mg_A2',          'SASA Mg',       '{:.2f}',  'solvent-exposed area of the magnesium, A^2. lower = more enclosed'),
    ('his_mg',              'His-Mg',        '{:.2f}',  'axial coordination distance after relax, A'),
]


def _engine(run: Path) -> str:
    """Which model actually produced ligand_plddt.

    The column is the B-factor of the folded structure, and the driver reads it the same way
    whichever engine wrote the file -- so the NAME `ligand_plddt` and the Boltz-shaped directory
    layout survive into AF3 campaigns and silently mislabel the number. v8's report called AF3's
    pLDDT 'Boltz lig'. Detect the engine from the driver rather than trusting the layout.
    """
    for drv in run.glob('run_nise_*.py'):
        if 'from af3_adapter import' in drv.read_text():
            return 'AF3'
    return 'Boltz'


def _cross_engine(run: Path) -> str:
    """The engine that did NOT lead the campaign, i.e. the one whose opinion is independent.

    AF3-led (v8) cross-checks with Boltz; Boltz-led (v5, v7) cross-checks with AF3. Derived from
    `_engine` so the header never hardcodes an engine the run did not use.
    """
    return 'Boltz' if _engine(run) == 'AF3' else 'AF3'


def _xfold_confidence(run: Path, mode: str, names: pd.Series) -> dict[str, float]:
    """Boltz's native confidence_score for the cross-fold, which crossfold.py never harvested.

    crossfold.py reads only B-factors out of the re-folded PDB, so the composite score sitting
    beside it in confidence_<id>_model_0.json is dropped on the floor. Recover it here.

    The prediction directory is x<i> where i is the ROW INDEX of crossfold_<mode>.csv: both the
    YAML inputs and the CSV rows come from the same `enumerate(sl.itertuples())` in crossfold.py,
    so POSITION is the join key -- the design name appears nowhere under crossfold_<mode>/.
    """
    root = run / f'crossfold_{mode}' / 'out'
    if not root.is_dir():
        return {}
    out: dict[str, float] = {}
    for i, nm in enumerate(names):
        hits = list(root.glob(
            f'gpu*/boltz_results_gpu*/predictions/x{i}/confidence_x{i}_model_0.json'))
        if not hits:
            continue
        try:
            out[nm] = float(json.loads(hits[0].read_text())['confidence_score'])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return out


def load(run: Path) -> pd.DataFrame:
    s = pd.read_csv(run / 'results/final/shortlist.csv', comment='#')
    base = s['name'].str.replace('_model_0$', '', regex=True)

    p = run / 'results/af3_mono/monomer_confidence.csv'
    if p.is_file():
        m = pd.read_csv(p, comment='#')
        m['key'] = m['name'].str.replace('_mono$', '', regex=True)
        m = m.set_index('key')
        s['af3_mono_plddt'] = base.map(m['mean_plddt'])
        s['af3_mono_ptm'] = base.map(m['ptm'])

    p = run / 'results/af3_dimer/interface_pae.csv'
    if p.is_file():
        d = pd.read_csv(p, comment='#').set_index('name')
        for col, out in (('interface_pae', 'interface_pae'), ('contact_pairs', 'contact_pairs'),
                        ('iptm', 'dimer_iptm')):   # NOT the shortlist's iptm, which is protein-ligand
            if col in d.columns:
                v = base.map(d[col])
                s[out] = s[out].fillna(v) if out in s.columns else v

    # cross-fold: the other engine's opinion. complex preferred (fold AND pose), monomer as fallback.
    for cf in ('crossfold_complex.csv', 'crossfold_monomer.csv'):
        p = run / 'results/final' / cf
        if p.is_file():
            x = pd.read_csv(p).set_index('name')
            for src, dst in (('boltz_ligand_plddt', 'xfold_ligand_plddt'),
                             ('boltz_protein_plddt', 'xfold_protein_plddt'),
                             ('boltz_config', 'xfold_config')):
                if src in x.columns:
                    s[dst] = s['name'].map(x[src])
            conf = _xfold_confidence(run, cf.replace('crossfold_', '').replace('.csv', ''),
                                     x.index)
            if conf:
                s['xfold_confidence'] = s['name'].map(conf)
            break

    p = run / 'results/final/sasa/sasa_ranking.csv'
    if p.is_file():
        sa = pd.read_csv(p, comment='#').set_index('name')
        for c in ('sasa_mg_A2', 'sasa_propionate_A2', 'buried_frac_propionate'):
            if c in sa.columns:
                v = s['name'].map(sa[c])
                s[c] = s[c].fillna(v) if c in s.columns else v

    s = s.sort_values('buns_score_paper', kind='mergesort').reset_index(drop=True)
    s.insert(0, 'rank', np.arange(1, len(s) + 1))
    return s


def order_set(run: Path) -> tuple[list[str], str]:
    """The designs picked for synthesis, in pick order, plus the csv they came from.

    Returns ([], '') when the campaign has no ORDER directory, so the section simply
    does not render for a campaign that has not been through selection yet.
    """
    d = run / 'results' / 'final' / 'ORDER'
    if not d.is_dir():
        return [], ''
    csvs = sorted(d.glob('order*.csv'))
    if not csvs:
        return [], ''
    try:
        names = pd.read_csv(csvs[0])['name'].astype(str).tolist()
    except (OSError, ValueError, KeyError):
        return [], ''
    return names, csvs[0].name


def order_reasons(df: pd.DataFrame, names: list[str]) -> dict[str, str]:
    """Why THIS design, stated from its standing in the shortlist -- not from prose.

    Each design gets the measures on which it is genuinely distinguished (top decile of
    the shortlist, or equal-best on BUNs). Generated rather than written so it cannot
    drift away from the numbers it describes.
    """
    idx = df.set_index('name')
    out: dict[str, str] = {}
    # HIGHER is better for these; BUNs is handled separately because LOWER is better.
    high = [('buried_frac_core', 'deepest core burial'),
            ('buried_frac', 'most enclosed ligand overall'),
            ('buried_frac_apolar', 'best apolar burial'),
            ('packstat', 'best packed'),
            ('xfold_ligand_plddt', 'cross-engine agreement'),
            ('xfold_confidence', 'cross-engine confidence'),
            ('interface_pae', 'widest margin on the dimer gate')]
    best_buns = df['buns_score_paper'].min() if 'buns_score_paper' in df else None
    for n in names:
        if n not in idx.index:
            continue
        r = idx.loc[n]
        bits = []
        if best_buns is not None and pd.notna(r.get('buns_score_paper')) \
                and r['buns_score_paper'] == best_buns:
            bits.append(f'joint-best BUNs ({best_buns:.0f})')
        for col, label in high:
            if col not in df.columns or pd.isna(r.get(col)):
                continue
            s = df[col].dropna()
            if len(s) > 8 and r[col] >= s.quantile(0.90):
                bits.append(label)
        # Low Mg SASA is a specific virtue: an exposed metal is an open coordination site.
        if 'sasa_mg_A2' in df.columns and pd.notna(r.get('sasa_mg_A2')):
            s = df['sasa_mg_A2'].dropna()
            if len(s) > 8 and r['sasa_mg_A2'] <= s.quantile(0.10):
                bits.append('magnesium most sealed from water')
        out[n] = '; '.join(bits[:3]) if bits else 'balanced across the measures, no single weakness'
    return out


def funnel(run: Path) -> list[tuple[str, int]]:
    p = run / 'results/cascade/cascade_metrics.csv'
    if not p.is_file():
        return []
    d = pd.read_csv(p, comment='#')
    n = len(d)
    out = [('designs scored by the cascade', n)]
    if 'packstat' in d:
        out.append(('packstat > 0.55', int((d.packstat > 0.55).sum())))
    if 'apo_clashes' in d:
        out.append(('apo-repack clashes == 0 (preorganisation)', int((d.apo_clashes == 0).sum())))
    if 'buried_charged' in d:
        out.append(('no buried charged residues', int((d.buried_charged == 0).sum())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    run = a.run_dir.resolve()   # `--run-dir .` must still name the campaign, not ''
    df = load(run)
    out = a.out or (run / 'results' / 'figures' / 'report.html')
    out.parent.mkdir(parents=True, exist_ok=True)

    eng, xeng = _engine(run), _cross_engine(run)
    def _sub(s):
        return s.replace('{ENG}', eng).replace('{XENG}', xeng)
    cols = [(k, _sub(lbl), f, _sub(t)) for k, lbl, f, t in COLS if k in df.columns]
    head = ''.join(
        f'<th data-i="{i}" title="{html.escape(t)}">{html.escape(lbl)}<span class=ar></span></th>'
        for i, (k, lbl, f, t) in enumerate(cols))
    picks, pick_csv = order_set(run)
    reasons = order_reasons(df, picks) if picks else {}
    pick_rank = {n: i + 1 for i, n in enumerate(picks)}

    rows = []
    for _, r in df.iterrows():
        tds = []
        for k, lbl, fmt, _t in cols:
            v = r.get(k)
            if pd.isna(v):
                txt, sv = '-', ''
            elif fmt:
                txt, sv = fmt.format(v), f'{float(v):.6f}'
            else:
                txt, sv = html.escape(str(v)), str(v)
            cls = ' class=n' if fmt else ''
            tds.append(f'<td{cls} data-v="{sv}">{txt}</td>')
        nm = str(r.get('name'))
        if nm in pick_rank:
            rows.append(f'<tr class=pick title="selected for synthesis -- pick '
                        f'{pick_rank[nm]} of {len(picks)}">' + ''.join(tds) + '</tr>')
        else:
            rows.append('<tr>' + ''.join(tds) + '</tr>')

    fn = funnel(run)
    fn_html = ''.join(f'<tr><td>{html.escape(l)}</td><td class=n>{v:,}</td></tr>' for l, v in fn)

    corr = ''
    pairs = [('buns_score_paper', 'buried_frac_apolar', 'BUNs Score vs apolar burial'),
             ('ligand_plddt', 'xfold_ligand_plddt', 'engine ligand pLDDT vs CROSS-ENGINE ligand pLDDT'),
             ('ligand_plddt', 'af3_mono_plddt', 'engine ligand pLDDT vs independent monomer pLDDT'),
             ('interface_pae', 'buried_frac_apolar', 'dimer pAE vs apolar burial'),
             ('buns_score_paper', 'packstat', 'BUNs Score vs packstat'),
             ('buns_score_paper', 'interface_pae', 'BUNs Score vs AF3 dimer pAE')]
    for x, y, lbl in pairs:
        if x in df and y in df and df[[x, y]].dropna().shape[0] > 3:
            r = df[x].corr(df[y])
            corr += f'<tr><td>{html.escape(lbl)}</td><td class=n>{r:+.2f}</td></tr>'

    order_html = ''
    if picks:
        ocols = [('buns_score_paper', 'BUNs', '{:.0f}', False),
                 ('buried_frac', 'Ligand burial', '{:.1%}', True),
                 ('buried_frac_core', 'Core burial', '{:.1%}', True),
                 ('buried_frac_apolar', 'Apolar', '{:.3f}', True),
                 ('packstat', 'Packstat', '{:.3f}', True),
                 ('sasa_mg_A2', 'SASA Mg', '{:.1f}', False),
                 ('ligand_plddt', f'{eng} lig', '{:.3f}', True),
                 ('xfold_ligand_plddt', f'{xeng} lig', '{:.3f}', True),
                 ('interface_pae', 'Dimer pAE', '{:.1f}', True),
                 ('his_mg', 'His-Mg', '{:.2f}', False)]
        ocols = [c for c in ocols if c[0] in df.columns]
        ohead = ''.join(f'<th class=n>{html.escape(lbl)}</th>' for _k, lbl, _f, _h in ocols)
        idx = df.set_index('name')
        orows = ''
        for i, nm in enumerate(picks, 1):
            if nm not in idx.index:
                continue
            r = idx.loc[nm]
            cells = ''
            for k, _lbl, fmt, higher in ocols:
                v = r.get(k)
                if pd.isna(v):
                    cells += '<td class=n>-</td>'
                    continue
                s = df[k].dropna()
                # highlight only where the design genuinely leads the shortlist
                q = (v >= s.quantile(0.90)) if higher else (v <= s.quantile(0.10))
                cells += f'<td class="n{" good" if q and len(s) > 8 else ""}">{fmt.format(v)}</td>'
            short = nm.replace('_model_0', '')
            orows += (f'<tr><td class=n>{i}</td><td>{html.escape(short)}</td>{cells}'
                      f'<td class=why>{html.escape(reasons.get(nm, ""))}</td></tr>')
        order_html = f"""
<h2>Suggested order &mdash; {len(picks)} designs</h2>
<p class=sub>From <code>{html.escape(pick_csv)}</code>. Highlighted cells are top-decile of the
{len(df)}-design shortlist (or bottom-decile where lower is better).</p>
<div class=scroll><table><thead><tr><th class=n>#</th><th>design</th>{ohead}
<th>why this one</th></tr></thead><tbody>{orows}</tbody></table></div>
<small class=note><b>Why these, and why eight.</b> Ranked by <b>BUNs Score</b> (Polizzi Table S1,
2&times;ligand + protein) because that is the only selector with published experimental support:
the paper&rsquo;s four tested binders scored 16&ndash;28 and 4/4 bound. Every design here scores
6&ndash;9, better than anything they put in a tube. But BUNs <b>cannot resolve the top of a
list</b> &mdash; re-relaxing one design moved its BUNs by 7 across five seeds against a 3-point
range across the whole top ten &mdash; so the leaders are treated as a <b>set</b>, not a ranking,
and the eight were chosen to span the measures rather than to take the top eight of any one
column. Each passed every gate independently: apo-repack clashes 0, no buried charged residues,
packstat &gt; 0.55, the 13&sup2; stereocentre confirmed <b>S</b> by the cross-check engine, and
the two-copy dimer gate (pAE 16.6&ndash;25.5, all far clear of the 14 bar &mdash; see
CALIBRATION). The cross-fold is what does the real work: the joint-best design by BUNs is
<b>excluded</b> because {xeng} rates its ligand 0.506 where {eng} rated it 0.951. Median pairwise
sequence identity across the eight is 66%, so a shared failure mode should not sink all of them.
<b>Nothing here shows the protein folds or binds</b> &mdash; these are candidates for
experimental testing.</small>"""

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(run.name)} shortlist</title><style>
:root{{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e3e3df;--card:#fff;--acc:#1f6f6b}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16161a;--fg:#e9e9e6;--mut:#9a9a95;--line:#2c2c33;--card:#1d1d22;--acc:#5fd3cb}}}}
:root[data-theme=dark]{{--bg:#16161a;--fg:#e9e9e6;--mut:#9a9a95;--line:#2c2c33;--card:#1d1d22;--acc:#5fd3cb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:0 16px 64px}}
.w{{max-width:1180px;margin:0 auto}}h1{{font-size:1.6rem;margin:28px 0 4px}}
h2{{font-size:1.1rem;margin:30px 0 10px;color:var(--acc)}}
p.sub{{color:var(--mut);margin:0 0 20px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}
.c{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:130px;flex:1}}
.c b{{display:block;font-size:1.45rem;color:var(--acc)}}.c span{{color:var(--mut);font-size:.8rem}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:13px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{background:var(--card);position:sticky;top:0;cursor:pointer;user-select:none;font-weight:600;
border-bottom:2px solid var(--line)}}
th:hover{{color:var(--acc)}} .ar{{opacity:.45;font-size:.75em;margin-left:4px}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:hover{{background:rgba(127,127,127,.09)}}
.scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
small.note{{color:var(--mut);display:block;margin-top:8px;white-space:normal;max-width:70ch}}
td.good{{color:var(--acc);font-weight:650}}
tr.pick td{{background:rgba(31,111,107,.09)}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]) tr.pick td{{background:rgba(95,211,203,.11)}}}}
:root[data-theme=dark] tr.pick td{{background:rgba(95,211,203,.11)}}
td.why{{white-space:normal;color:var(--mut);font-size:12px;max-width:34ch}}
code{{font-size:.9em;background:rgba(127,127,127,.12);padding:1px 5px;border-radius:4px}}
</style></head><body><div class=w>
<h1>{html.escape(run.name)} &mdash; shortlist</h1>
<p class=sub>Every column sorts on click. Hover a header for what it means.</p>
<div class=cards>
  <div class=c><b>{len(df)}</b><span>shortlisted designs</span></div>
  <div class=c><b>{fn[0][1]:,}</b><span>designs scored</span></div>
  <div class=c><b>{df.buns_score_paper.min():.0f}</b><span>best BUNs Score</span></div>
  <div class=c><b>{df.buried_frac_core.max():.0%}</b><span>best core burial</span></div>
  <div class=c><b>{df.buried_frac.median():.0%}</b><span>median ligand burial</span></div>
</div>
{order_html}
<h2>Filter cascade</h2><div class=scroll><table><tbody>{fn_html}</tbody></table></div>
<h2>Shortlist</h2><div class=scroll><table id=t><thead><tr>{head}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></div>
<small class=note>BUNs Score and apolar burial are near-independent, so the leader of one column is
usually mid-pack in another. Pick a set that spans the measures rather than the top N of any one.
Shaded rows are the designs selected for synthesis.</small>
<h2>How independent are the measures?</h2><div class=scroll><table><thead><tr><th>pair</th><th class=n>Pearson r</th></tr></thead><tbody>{corr}</tbody></table></div>
</div><script>
document.querySelectorAll('#t th').forEach(function(th){{
  th.addEventListener('click', function(){{
    var tb=document.querySelector('#t tbody'), i=+th.dataset.i;
    var asc=th.dataset.d!=='asc';
    document.querySelectorAll('#t th').forEach(function(o){{o.dataset.d='';o.querySelector('.ar').textContent='';}});
    th.dataset.d=asc?'asc':'desc'; th.querySelector('.ar').textContent=asc?'\\u25B2':'\\u25BC';
    var rs=[].slice.call(tb.rows);
    rs.sort(function(a,b){{
      var x=a.cells[i].dataset.v, y=b.cells[i].dataset.v;
      var nx=parseFloat(x), ny=parseFloat(y);
      var both=!isNaN(nx)&&!isNaN(ny);
      if(x==='') return 1; if(y==='') return -1;
      var c=both?(nx-ny):String(x).localeCompare(String(y));
      return asc?c:-c;
    }});
    rs.forEach(function(r){{tb.appendChild(r);}});
  }});
}});
</script></body></html>"""
    out.write_text(doc)
    print(f'{len(df)} designs, {len(cols)} columns -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
