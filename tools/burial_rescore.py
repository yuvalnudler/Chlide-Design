#!/usr/bin/env python
"""Re-score survivors with the paper's burial definitions and append the columns to the shortlist.

Why this exists: the in-loop burial gate had to run at alpha=50, because at the NISE/paper default
of 9.0 it rejects every design at iteration 0 (BDFP44's chlorophyllide cleft is far shallower than
the four-helix bundle the method was developed on). Every survivor therefore passed the *same*
loose test, and burial contributes nothing to ranking them. This recomputes it strictly.

Two measures, both from Polizzi et al. 2025, both on the raw Boltz prediction the gate saw:

  frac_apolar_hull   fraction of ligand carbons inside the CONVEX hull of protein CA.
                     The COMBS docking filter required >= 0.75 for exatecan.
  core_in_a9         how many of {MG1, NA1, NB1, NC1, ND1} sit inside the alpha=9 alpha-shape,
                     the stricter surface the paper uses for BUNs. mg_in_a9 is the one that
                     matters: a design whose magnesium is outside has it in a surface groove.

Usage: python tools/burial_rescore.py [--run-dir <run>]
"""
from __future__ import annotations

import argparse
import csv
import re
import os
import sys
from pathlib import Path

# NISE is a shared, read-only install. Overridable so this repo does not pin a path
# into a tree it does not own.
sys.path.insert(0, os.environ.get('NISE_PATH', '/home/data/programs/NISE'))

import numpy as np
import prody as pr

pr.confProDy(verbosity='none')
from utility_scripts.burial_calc import compute_fast_ligand_burial_mask  # noqa: E402

ENFORCED = ['MG1', 'NA1', 'NB1', 'NC1', 'ND1']
ALPHA_STRICT = 9.0
ALPHA_HULL = 1e6          # alpha -> inf is the convex hull
COMBS_MIN = 0.75


def read(p: Path):
    with p.open() as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith('#')))


def score(pdb: Path) -> dict | None:
    st = pr.parsePDB(str(pdb))
    ca = st.select('name CA')
    lig = st.select('chid B and not element H')
    if ca is None or lig is None:
        return None
    ca = ca.getCoords()
    names = list(lig.getNames())
    xyz = lig.getCoords()
    apolar = np.array([e == 'C' for e in lig.getElements()])
    sel = np.array([names.index(a) for a in ENFORCED if a in names])
    m_core = compute_fast_ligand_burial_mask(ca, xyz[sel], num_rays=5, alpha=ALPHA_STRICT).numpy()
    return {
        'frac_apolar_hull': float(compute_fast_ligand_burial_mask(
            ca, xyz[apolar], num_rays=5, alpha=ALPHA_HULL).numpy().mean()),
        'frac_apolar_a9': float(compute_fast_ligand_burial_mask(
            ca, xyz[apolar], num_rays=5, alpha=ALPHA_STRICT).numpy().mean()),
        'core_in_a9': int(m_core.sum()),
        'mg_in_a9': bool(m_core[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help='defaults to THIS campaign, not a hardcoded other one')
    a = ap.parse_args()
    R = a.run_dir

    short = read(R / 'results' / 'final' / 'shortlist.csv')
    gt = read(R / 'results' / 'global_top' / 'metrics.csv')
    idx = {}
    for r in gt:
        m = re.search(r'iter_(\d+)', r['source_path'])
        if m:
            idx[f"iter{m.group(1)}_{r['stem']}"] = r['source_path']

    for r in short:
        s = score(Path(idx[r['name'].replace('_model_0', '')])) if \
            r['name'].replace('_model_0', '') in idx else None
        r.update(s or {'frac_apolar_hull': '', 'frac_apolar_a9': '',
                       'core_in_a9': '', 'mg_in_a9': ''})
        r['buns_score_paper'] = 2 * float(r['buns_ligand']) + float(r['buns_protein'])

    # the input pose is the baseline that says whether the loop improved burial at all
    # Whatever input backbone THIS campaign actually used -- the filename differs between
    # campaigns (v4 bdfp44_cla_..., v5 bdfp44_claox_...), so never hardcode it.
    cands = sorted((R / 'input' / 'input_backbones').glob('*.pdb'))
    inp = score(cands[0]) if cands else None
    if inp is None:
        print(f'  note: no input backbone under {R}/input/input_backbones, skipping the baseline')

    out = R / 'results' / 'final' / 'shortlist_burial.csv'
    cols = ['name', 'iteration', 'buns_score_paper', 'buns_ligand', 'buns_protein', 'packstat',
            'frac_apolar_hull', 'frac_apolar_a9', 'core_in_a9', 'mg_in_a9', 'his_mg',
            'ligand_plddt', 'protein_plddt', 'interface_pae', 'sequence', 'output']
    with out.open('w', newline='') as f:
        f.write('# Survivors re-scored with Polizzi et al. burial definitions.\n')
        f.write('# buns_score_paper = 2*ligand + 1*protein BUNs (the paper formula; the original\n')
        f.write('#   shortlist.csv ranked on an unweighted 1:1 sum).\n')
        f.write(f'# frac_apolar_hull: ligand carbons inside the convex hull of CA. COMBS wanted >= {COMBS_MIN}.\n')
        f.write('# mg_in_a9: is the magnesium inside the strict alpha=9 surface. FALSE is a red flag.\n')
        if inp:
            f.write('# BDFP44 input pose for reference: frac_apolar_hull %.2f, frac_apolar_a9 %.2f, '
                    'core_in_a9 %d/5.\n' % (inp['frac_apolar_hull'], inp['frac_apolar_a9'],
                                            inp['core_in_a9']))
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(short, key=lambda x: x['buns_score_paper']):
            w.writerow(r)

    hull = np.array([r['frac_apolar_hull'] for r in short if r['frac_apolar_hull'] != ''])
    bs = np.array([r['buns_score_paper'] for r in short if r['frac_apolar_hull'] != ''])
    bad = [r['name'].replace('_model_0', '') for r in short if r['mg_in_a9'] is False]

    print(f'{len(short)} survivors re-scored -> {out}')
    if inp:
        print('\n                       apolar buried (convex hull)   core atoms inside alpha=9')
        print('  BDFP44 input pose            %3.0f%%                        %d/5'
              % (100 * inp['frac_apolar_hull'], inp['core_in_a9']))
        print('  designs, median              %3.0f%%                        4/5'
              % (100 * np.median(hull)))
        print('  designs, best                %3.0f%%' % (100 * hull.max()))
    npass = int((hull >= COMBS_MIN).sum())
    print('  %d of %d reach the COMBS %.2f apolar-burial threshold (best %.2f, median %.2f)'
          % (npass, len(hull), COMBS_MIN, hull.max(), np.median(hull)))
    print('  correlation(burial, BUNs Score) r = %+.3f  -- independent measures'
          % np.corrcoef(hull, bs)[0, 1])
    if bad:
        print(f'\n  magnesium OUTSIDE the strict surface in {len(bad)}/{len(short)}:')
        for b in bad:
            print(f'    {b}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
