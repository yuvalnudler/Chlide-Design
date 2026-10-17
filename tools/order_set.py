#!/usr/bin/env python
"""Pick the designs to synthesise, and write results/final/ORDER/.

Until now this was done by hand: v8's ORDER directory was assembled manually, which is why
nothing downstream could regenerate it and why a campaign that had not been through selection
simply showed no order section. The rule below is the one applied to v8, written down.

Selection, in order:

  1. Start from results/final/shortlist.csv -- already past the terminal cascade.
  2. Apply the checks that exist for this campaign, and ONLY those. Each is skipped silently
     when its column is absent, so a campaign whose GPU stages have not run still produces a
     list rather than an empty one:
       * two-copy interface pAE > 14. The aggregation check. v8's top-ranked design by BUNs
         scored 10.7 here and had to be dropped -- it dimerises -- so this is not cosmetic.
       * the cross-check engine's ligand pLDDT >= 0.80, when it was run. A campaign cannot
         grade itself.
  3. Rank by BUNs Score (2*ligand + protein buried unsatisfied polars), the paper's Table S1
     criterion, with the composite as the tie-break.
  4. Walk that ranking and take designs that are not near-duplicates of one already taken
     (sequence identity < --max-identity, default 0.90). Ordering eight variations on one
     sequence spends the whole experiment on a single hypothesis; v8's picks came from eight
     different iterations.

The coordination-geometry columns are deliberately NOT part of the rule. They are reported
beside each pick so they can be read, but nothing shows they predict binding yet -- see
docs/CALIBRATION.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PAE_MIN = 14.0
XFOLD_LIG_MIN = 0.80
REPORT_COLS = ['buns_score_paper', 'buns_ligand', 'buns_protein', 'buried_frac',
               'buried_frac_core', 'buried_frac_apolar', 'packstat', 'sasa_mg_A2',
               'ligand_plddt', 'protein_plddt', 'xfold_ligand_plddt', 'xfold_confidence',
               'interface_pae', 'his_mg', 'axial_angle', 'his_lonepair_angle',
               'mg_imidazole_oop', 'composite']


def identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def pick(df: pd.DataFrame, n: int, max_identity: float) -> tuple[pd.DataFrame, list[str]]:
    """Returns the chosen rows and a human-readable log of what each filter cost."""
    log = [f'shortlist: {len(df)} designs']
    d = df.copy()
    if 'interface_pae' in d.columns and d['interface_pae'].notna().any():
        before = len(d)
        d = d[d['interface_pae'].isna() | (d['interface_pae'] > PAE_MIN)]
        log.append(f'two-copy interface pAE > {PAE_MIN:g}: {len(d)} kept ({before - len(d)} dropped)')
    else:
        log.append('two-copy interface pAE: NOT RUN for this campaign, not applied')
    if 'xfold_ligand_plddt' in d.columns and d['xfold_ligand_plddt'].notna().any():
        before = len(d)
        d = d[d['xfold_ligand_plddt'].isna() | (d['xfold_ligand_plddt'] >= XFOLD_LIG_MIN)]
        log.append(f'cross-fold ligand pLDDT >= {XFOLD_LIG_MIN}: {len(d)} kept ({before - len(d)} dropped)')
    else:
        log.append('cross-fold: NOT RUN for this campaign, not applied')

    sort_cols, asc = ['buns_score_paper'], [True]
    if 'composite' in d.columns:
        sort_cols.append('composite'); asc.append(False)
    d = d.sort_values(sort_cols, ascending=asc, kind='mergesort')

    chosen, seqs = [], []
    for _, r in d.iterrows():
        s = str(r.get('sequence', '') or '')
        if any(identity(s, t) >= max_identity for t in seqs):
            continue
        chosen.append(r)
        seqs.append(s)
        if len(chosen) >= n:
            break
    log.append(f'after de-duplication at {max_identity:.0%} identity: {len(chosen)} picked')
    return (pd.DataFrame(chosen) if chosen else d.head(0)), log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path,
                    default=Path(os.environ.get('CHLIDE_RUN_DIR')
                                 or Path(__file__).resolve().parent.parent))
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--max-identity', type=float, default=0.90)
    ap.add_argument('--force', action='store_true',
                    help='replace an ORDER directory this tool did not generate')
    a = ap.parse_args()

    R = a.run_dir
    sl = R / 'results' / 'final' / 'shortlist.csv'
    if not sl.is_file():
        print(f'no shortlist at {sl}', file=sys.stderr)
        return 1
    df = pd.read_csv(sl, comment='#')
    sel, log = pick(df, a.n, a.max_identity)
    for line in log:
        print('  ' + line)
    if sel.empty:
        print('nothing survived selection', file=sys.stderr)
        return 1

    out = R / 'results' / 'final' / 'ORDER'
    marker = out / '.generated_by_order_set'
    # An ORDER directory can be hand-curated -- v8's was, for months -- and the picks in it may
    # encode a judgement this rule does not. Refuse to clobber one we did not write.
    if out.is_dir() and any(out.glob('order*')) and not marker.exists() and not a.force:
        print(f'{out} already holds an order set this tool did not generate.', file=sys.stderr)
        print('Refusing to overwrite it. Move it aside, or pass --force if you mean to replace it.',
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('order*'):        # same "clear stale output first" idiom as the rest
        stale.unlink()
    marker.write_text('written by tools/order_set.py; safe to regenerate\n')

    final = R / 'results' / 'final'
    by_name = {}
    for p in final.glob('top*.pdb'):
        by_name[p.name.split('_', 1)[1][:-4]] = p

    rows, missing = [], []
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        name = str(r['name'])
        src = by_name.get(name)
        if src is None:
            missing.append(name)
        else:
            shutil.copy2(src, out / f'order{i:02d}_{name}.pdb')
        rows.append(r)

    cols = ['name'] + [c for c in REPORT_COLS if c in sel.columns] + ['sequence']
    csv_path = out / f'order{len(rows)}_{R.name}.csv'
    pd.DataFrame(rows)[cols].to_csv(csv_path, index=False)
    with (out / f'order{len(rows)}_{R.name}.fasta').open('w') as f:
        for i, r in enumerate(rows, 1):
            f.write(f'>order{i:02d}_{r["name"]}\n{r.get("sequence", "")}\n')

    print(f'\nwrote {csv_path}')
    print(f'  {len(rows)} designs, {len(rows) - len(missing)} with structures')
    if missing:
        print(f'  WARNING: no top*.pdb found for: {", ".join(missing)}')
    for i, r in enumerate(rows, 1):
        print(f'  order{i:02d}  {r["name"][:38]:38s} BUNs {r.get("buns_score_paper", float("nan")):.0f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
