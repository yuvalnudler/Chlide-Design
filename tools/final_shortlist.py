#!/usr/bin/env python
"""Assemble the final shortlist: every measure for every surviving design, in one table.

Joins the three sources a campaign produces:
  results/global_top/metrics.csv        in-loop measures (pLDDT, iptm, self-consistency RMSDs)
  results/cascade/cascade_metrics.csv   Rosetta stage (relax, packstat, apo clashes, BUNs, ...)
  results/af3_dimer/interface_pae.csv   AF3 two-copy aggregation proxy

and writes results/final/ with the surviving structures, shortlist.csv, and a README that says
which stages actually ran -- so the directory is still interpretable months later without this
conversation.

Ranking is by combined protein+ligand BUNs, which is the paper's reranking criterion. Note it
disagrees with ligand pLDDT: on the first 94 designs tested here the best by BUNs was rank 20 by
pLDDT, so the in-loop objective is not a substitute for this table.

Usage: python tools/final_shortlist.py --run-dir <run> [--top 25]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

PACKSTAT_MIN, RMSD_MAX, PAE_MIN = 0.55, 1.5, 14.0
COORD_LO, COORD_HI = 2.0, 2.6



def leading_engine(run: Path) -> str:
    """Which model folded this campaign, read from the driver's import.

    The directory layout cannot tell you: af3_adapter.to_boltz_layout writes AF3 output into
    a boltz_results_* tree on purpose. Only the driver's import is evidence.
    """
    for drv in run.glob('run_nise_*.py'):
        try:
            if 'from af3_adapter import' in drv.read_text():
                return 'AF3'
        except OSError:
            continue
    return 'Boltz'


def coordination_source(engine: str) -> str:
    """Who actually formed the His-Mg bond.

    This README used to assert "in-loop (patched Boltz) + Rosetta relax" unconditionally, and
    v8 shipped one saying so while folding with AF3 -- which has NO metal restraint at all.
    Rosetta relax alone closes it there (4.22 -> 2.50 A).
    """
    return ('Rosetta relax only (AF3 has no metal restraint)' if engine == 'AF3'
            else 'in-loop (patched Boltz) + Rosetta relax')


def read_csv(p: Path):
    if not p.is_file():
        return None
    with p.open() as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith('#')))
    return pd.DataFrame(rows) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--cascade-csv', type=Path, default=None,
                    help='default results/cascade/cascade_metrics.csv; use the corrected subset '
                         'when the full run had colliding output names')
    a = ap.parse_args()
    R = a.run_dir
    out = R / 'results' / 'final'
    out.mkdir(parents=True, exist_ok=True)

    casc = read_csv(a.cascade_csv or (R / 'results' / 'cascade' / 'cascade_metrics.csv'))
    if casc is None:
        print('no cascade_metrics.csv - run polizzi_cascade.py first'); return 1
    inloop = read_csv(R / 'results' / 'global_top' / 'metrics.csv')
    pae = read_csv(R / 'results' / 'af3_dimer' / 'interface_pae.csv')

    num = ['his_mg', 'his_mg_before', 'score', 'ca_rmsd_to_min', 'lig_rmsd_to_min', 'packstat',
           'apo_clashes', 'buried_charged', 'buns_ligand', 'buns_protein', 'buns_combined']
    for c in num:
        if c in casc:
            casc[c] = pd.to_numeric(casc[c], errors='coerce')

    # the cascade names a design by its file stem; the in-loop table by its prediction dir stem
    casc['key'] = casc['name'].str.replace(r'_(model_0|min)$', '', regex=True)
    if inloop is not None:
        inloop['key'] = inloop['stem'] if 'stem' in inloop else inloop['name']
        if 'stem' in casc:
            casc['inloop_key'] = casc['stem'].str.replace(r'_model_0$', '', regex=True)
        keep = [c for c in ('key', 'iteration', 'ligand_plddt', 'protein_plddt', 'iptm',
                            'ligand_rmsd', 'protein_rmsd', 'laser_bs_nll', 'sequence') if c in inloop]
        lk = 'inloop_key' if 'inloop_key' in casc else 'key'
        casc = casc.merge(inloop[keep].drop_duplicates('key'), left_on=lk, right_on='key',
                          how='left', suffixes=('', '_il'))
    if pae is not None and 'name' in pae:
        pae['key'] = pae['name'].str.replace(r'_(model_0|min)$', '', regex=True)
        pae['interface_pae'] = pd.to_numeric(pae['interface_pae'], errors='coerce')
        casc = casc.merge(pae[['key', 'interface_pae']].drop_duplicates('key'), on='key', how='left')

    stages = {
        'minimised': casc['his_mg'].notna(),
        f'His-Mg {COORD_LO}-{COORD_HI} A': casc['his_mg'].between(COORD_LO, COORD_HI),
        f'protein RMSD->min < {RMSD_MAX}': casc['ca_rmsd_to_min'] < RMSD_MAX,
        f'ligand RMSD->min < {RMSD_MAX}': casc['lig_rmsd_to_min'] < RMSD_MAX,
        f'packstat > {PACKSTAT_MIN}': casc['packstat'] > PACKSTAT_MIN,
        'apo clashes = 0': casc['apo_clashes'] == 0,
        'no buried charges': casc['buried_charged'] == 0,
    }
    mask = pd.Series(True, index=casc.index)
    print('cascade funnel:')
    for label, m in stages.items():
        mask &= m.fillna(False)
        print('  %-28s %6d alone  %6d cumulative' % (label, int(m.fillna(False).sum()), int(mask.sum())))
    surv = casc[mask].copy()

    has_pae = 'interface_pae' in surv and surv['interface_pae'].notna().any()
    if has_pae:
        before = len(surv)
        surv = surv[surv['interface_pae'].isna() | (surv['interface_pae'] > PAE_MIN)]
        print('  %-28s %6d cumulative  (AF3, keep pAE > %.0f: high pAE = unlikely to aggregate)'
              % ('interface pAE', len(surv), PAE_MIN))
        print(f'     ({before - len(surv)} dropped as aggregation-prone)')
    else:
        print('  interface pAE              not run - shortlist is provisional on that stage')

    # ---- the paper's BUNs Score -------------------------------------------------------------
    # Table S1 of Polizzi et al. states the reranking formula explicitly:
    #     BUNs Score = 2 x ligand BUNs + 1 x protein BUNs
    # v4 ranked on an unweighted 1:1 sum. On v4's 38 survivors the weighting keeps 9 of the same
    # 10 designs but reorders them substantially -- the design at rank 1 fell to 4, and rank 4 rose
    # to 1. It matters because it is what puts their own EPIC design 1st: EPIC had the MOST ligand
    # BUNs of the four they synthesised (4), so ligand BUNs alone would have ranked the best binder
    # (Kd 0.12 uM) last.
    surv['buns_score_paper'] = 2 * surv['buns_ligand'] + surv['buns_protein']

    # ---- composite -------------------------------------------------------------------------
    # Burial is scored here, never filtered. A hard burial gate would have discarded
    # iter7_chunk_2_seq_9 from v4 -- the best-enclosed design in that set by SASA (78% buried,
    # 95% core, Mg fully sealed) -- because it failed the alpha=50 in-loop test.
    # These components are close to independent on v4 data (SASA burial vs BUNs Score r = -0.001),
    # so the composite is a deliberate blend of disagreeing evidence, not a consensus. Every
    # component stays in the table, and each measure's own rank is reported beside it.
    COMPONENTS = [('buns_score_paper', False), ('buried_frac', True),
                  ('buried_frac_core', True), ('packstat', True)]
    present = [(c, hi) for c, hi in COMPONENTS if c in surv and surv[c].notna().any()]
    if present:
        pct = {}
        for c, higher_is_better in present:
            v = pd.to_numeric(surv[c], errors='coerce')
            r = v.rank(pct=True, ascending=higher_is_better)
            pct[c] = r
            surv[f'rank_{c}'] = v.rank(ascending=not higher_is_better, method='min')
        surv['composite'] = sum(pct.values()) / len(pct)
        print('  composite over %d components: %s' % (len(present), ', '.join(c for c, _ in present)))
    else:
        surv['composite'] = float('nan')

    # Rank on the paper's criterion so the table stays comparable to v4 and to Table S1; the
    # composite and every component rank sit beside it for the reader to weigh.
    surv = surv.sort_values('buns_score_paper')
    cols = [c for c in ('name', 'iteration', 'buns_score_paper', 'composite',
                        'buns_combined', 'buns_ligand', 'buns_protein',
                        'buried_frac', 'buried_frac_core', 'buried_frac_apolar', 'sasa_mg_A2',
                        'frac_apolar_hull', 'mg_in_a9', 'ligand_is_buried',
                        'packstat', 'apo_clashes', 'buried_charged', 'his_mg', 'his_mg_before',
                        'score', 'ca_rmsd_to_min', 'lig_rmsd_to_min', 'ligand_plddt',
                        'protein_plddt', 'iptm', 'ligand_rmsd', 'protein_rmsd', 'laser_bs_nll',
                        'interface_pae', 'rank_buns_score_paper', 'rank_buried_frac',
                        'rank_buried_frac_core', 'rank_packstat',
                        'sequence', 'output') if c in surv]
    sl = out / 'shortlist.csv'
    with sl.open('w', newline='') as f:
        f.write('# Final shortlist, ranked by BUNs Score = 2*ligand + 1*protein buried unsatisfied\n'
                '# polar atoms (Polizzi et al. Table S1), NOT by ligand pLDDT -- the two disagree.\n')
        f.write('# buns_combined is the unweighted 1:1 sum v4 used, kept for comparison.\n')
        f.write('# composite averages percentile ranks of BUNs Score, SASA burial (whole ligand and\n'
                '# core) and packstat. Those measures are close to independent -- on v4, SASA burial\n'
                '# and BUNs Score correlated at r = -0.001 -- so the composite BLENDS disagreeing\n'
                '# evidence and is a choice, not a truth. Use the per-measure rank_* columns too.\n')
        f.write('# Burial is scored, never filtered: a hard burial gate would have dropped v4\'s\n'
                '# best-enclosed design. mg_in_a9 and sasa_mg_A2 are flags worth reading directly --\n'
                '# a solvent-exposed magnesium is a specific defect, not something to average away.\n')
        f.write('# CAUTION: BUNs cannot resolve the top of this list. Re-relaxing one v4 design moved\n'
                '# its BUNs by 7 across five seeds (median spread 0.9) against a 3-point range across\n'
                '# the whole top ten. Treat the leaders as a set.\n')
        f.write('# Every column is computational. Nothing here shows the protein folds or binds; '
                'these are candidates for experimental testing.\n')
        surv[cols].to_csv(f, index=False, float_format='%.4f')

    # clear any previous ranking first: the files are named topNN_<design>, so a re-run with a
    # changed ranking leaves the old set behind and the folder ends up with two of everything.
    for old_pdb in out.glob('top*.pdb'):
        old_pdb.unlink()
    # polizzi_cascade writes `output` as a path RELATIVE to the run directory, so resolving it
    # against the process CWD silently copies nothing whenever this is invoked from anywhere else.
    def _resolve(v) -> Path | None:
        if not isinstance(v, str) or not v:
            return None
        p = Path(v)
        for cand in ((p,) if p.is_absolute() else (R / p, p)):
            if cand.is_file():
                return cand
        return None

    n = missing = 0
    for i, (_, r) in enumerate(surv.head(a.top).iterrows(), 1):
        src = _resolve(r.get('output'))
        if src is None:
            missing += 1
            continue
        shutil.copy(src, out / f'top{i:02d}_{r["name"]}.pdb'); n += 1
    if missing:
        print(f'  WARNING: {missing} of the top {a.top} had no minimised structure on disk')

    ENG = leading_engine(R)
    COORD_SOURCE = coordination_source(ENG)
    (out / 'README.md').write_text(f"""# Final shortlist - {R.name}

{len(surv)} designs survive every computable stage, ranked by BUNs Score
(2 x ligand + 1 x protein buried unsatisfied polar atoms, Polizzi et al. Table S1). Top {n} structures copied here as `topNN_*.pdb`; full table in
`shortlist.csv`.

**These structures are Rosetta-minimised** (FastRelax, ref2015, with a flat-bottom restraint on
the axial His NE2-Mg at 2.15 A). They are protonated and energy-minimised. The raw {ENG}
predictions live in `../global_top/` and are *not* minimised - use these for inspection.

## Stages that ran

| stage | source |
| --- | --- |
| self-consistency RMSDs, ligand/protein pLDDT | in-loop, every iteration |
| burial of Mg + the four pyrrole N | in-loop |
| His-Mg coordination {COORD_LO}-{COORD_HI} A | {COORD_SOURCE} |
| Rosetta minimisation, prediction->minimised RMSD | `tools/polizzi_cascade.py` |
| packstat > {PACKSTAT_MIN} | `tools/polizzi_cascade.py` |
| apo-repack clashes = 0 (preorganisation) | `tools/polizzi_cascade.py` |
| buried charged residues | `tools/polizzi_cascade.py` |
| BUNs, ligand / protein / combined | `tools/polizzi_cascade.py` |
| AF3 two-copy interface pAE > {PAE_MIN:.0f} | `tools/af3_interface_pae.py` {'(ran)' if has_pae else '(NOT RUN)'} |

## Not done

- Cysteine exclusion is enforced during sampling (LASErMPNN), not re-checked here.
- No experimental validation of any kind.

## Caveat worth keeping

Ranking by combined BUNs disagrees with ranking by ligand pLDDT, which is what the design loop
optimises: on the first 94 designs the best by BUNs was rank 20 by pLDDT. Neither ranking is
authoritative on its own.
""")
    print(f'\n{len(surv)} survivors -> {sl}')
    print(f'{n} structures copied to {out}')
    if len(surv):
        print('\ntop 5 by BUNs Score (2*ligand + protein):')
        for _, r in surv.head(5).iterrows():
            print('  %-32s score %3.0f (lig %2.0f/prot %2.0f)  packstat %.3f  His-Mg %.2f  pAE %s'
                  % (str(r['name'])[:32], r.buns_score_paper, r.buns_ligand, r.buns_protein,
                     r.packstat, r.his_mg,
                     f'{r.interface_pae:.1f}' if has_pae and pd.notna(r.get('interface_pae')) else '-'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
