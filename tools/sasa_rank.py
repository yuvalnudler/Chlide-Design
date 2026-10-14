#!/usr/bin/env python
"""Rank the shortlisted designs by how much of the chlorophyllide is actually desolvated.

Why this is worth doing separately: every burial number in this campaign so far comes from NISE's
alpha-shape test, which casts a ray at a mesh built through the CA trace and asks whether the
crossing count is odd. That answers "is this atom inside the polypeptide envelope" -- it knows
nothing about sidechains and nothing about solvent, so an atom sitting in a water-filled cavity
counts as buried. SASA asks the physical question instead: can a water molecule touch it.

Measure: Shrake-Rupley, implemented here rather than imported so the radii and probe are explicit.

    buried_fraction = 1 - SASA(ligand in complex) / SASA(ligand alone, same conformer)

Computed over heavy atoms only. The ligand-alone reference uses the identical coordinates, so the
ratio isolates occlusion by the protein and is not affected by the ligand's own conformation.

Reported per region, because they mean different things for this ligand:
  chlorin core   the macrocycle + Mg. Should be as buried as possible.
  propionate     CGA1/OA1/OA2. Should stay solvent-exposed -- it is charged, and we enforced that
                 during design. A design that buries it has paid a desolvation penalty for nothing.
  Mg             the metal itself. Nonzero SASA means the coordination site is open to water.

Usage: python tools/sasa_rank.py [--run-dir <run>] [--top 25]
"""
from __future__ import annotations

import os
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

# Bondi van der Waals radii, the standard set for SASA. Mg is not in Bondi's table; 1.73 A is the
# value in common use for Mg(II) and is what CHARMM/AMBER-derived sets carry.
RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'MG': 1.73, 'FE': 1.80}
PROBE = 1.40
N_POINTS = 960

PROPIONATE = {'CGA1', 'OA1', 'OA2'}
CORE = ({'MG1', 'NA1', 'NB1', 'NC1', 'ND1'}
        | {f'C{r}{i}' for r in 'ABCD' for i in '1234'}
        | {'CHA1', 'CHB1', 'CHC1', 'CHD1'})


def sphere_points(n: int) -> np.ndarray:
    """n roughly-equidistant points on the unit sphere (golden spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


UNIT = sphere_points(N_POINTS)


def sasa(target_xyz, target_r, context_xyz, context_r):
    """Per-atom SASA of the target atoms, occluded by the context atoms (which include them)."""
    out = np.empty(len(target_xyz))
    ext_ctx = context_r + PROBE
    for i, (c, r) in enumerate(zip(target_xyz, target_r)):
        ext = r + PROBE
        pts = c + UNIT * ext
        # only atoms that can possibly reach these points matter
        d = np.linalg.norm(context_xyz - c, axis=1)
        near = (d < ext + ext_ctx.max()) & (d > 1e-6)
        if near.any():
            dist = np.linalg.norm(pts[:, None, :] - context_xyz[near][None, :, :], axis=2)
            buried = (dist < ext_ctx[near][None, :]).any(axis=1)
        else:
            buried = np.zeros(len(pts), dtype=bool)
        out[i] = 4 * np.pi * ext ** 2 * (~buried).mean()
    return out


def parse(pdb: Path):
    """Heavy atoms from a PDB: names, elements, coords, and whether each is the ligand."""
    names, els, xyz, islig = [], [], [], []
    for line in pdb.read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        el = line[76:78].strip().upper()
        if el == 'H' or not el:
            continue
        names.append(line[12:16].strip())
        els.append(el)
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        islig.append(line.startswith('HETATM') and line[17:20].strip() == 'CLA')
    return (np.array(names), np.array(els), np.array(xyz, dtype=float), np.array(islig))


def score(pdb: Path) -> dict | None:
    names, els, xyz, islig = parse(pdb)
    if not islig.any():
        return None
    r = np.array([RADII.get(e, 1.70) for e in els])
    lx, lr, ln = xyz[islig], r[islig], names[islig]

    in_complex = sasa(lx, lr, xyz, r)          # occluded by protein + itself
    alone = sasa(lx, lr, lx, lr)               # occluded by itself only

    def frac(mask):
        a = alone[mask].sum()
        return float(1 - in_complex[mask].sum() / a) if a > 1e-9 else float('nan')

    apolar = np.array([e == 'C' for e in els[islig]])
    core = np.array([n in CORE for n in ln])
    prop = np.array([n in PROPIONATE for n in ln])
    mg = np.array([n == 'MG1' for n in ln])

    return {
        'buried_frac': frac(np.ones(len(ln), dtype=bool)),
        'buried_frac_apolar': frac(apolar),
        'buried_frac_core': frac(core),
        'buried_frac_propionate': frac(prop),
        'sasa_complex_A2': float(in_complex.sum()),
        'sasa_free_A2': float(alone.sum()),
        'sasa_mg_A2': float(in_complex[mg].sum()),
        'sasa_propionate_A2': float(in_complex[prop].sum()),
        'n_heavy': int(islig.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(os.environ.get('CHLIDE_RUN_DIR') or Path(__file__).resolve().parent.parent))
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()

    final = a.run_dir / 'results' / 'final'
    out = final / 'sasa'
    out.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(final.glob('top*.pdb'))[:a.top]
    if not pdbs:
        print(f'no top*.pdb in {final}')
        return 1

    # carry the other measures across so this table stands on its own
    with (final / 'shortlist.csv').open() as fh:
        short = {r['name']: r for r in csv.DictReader(l for l in fh if not l.startswith('#'))}
    burial_csv = final / 'shortlist_burial.csv'
    hull = {}
    if burial_csv.is_file():
        with burial_csv.open() as fh:
            hull = {r['name']: r for r in csv.DictReader(l for l in fh if not l.startswith('#'))}

    rows = []
    print(f'{len(pdbs)} structures, Shrake-Rupley probe {PROBE} A, {N_POINTS} points/atom\n')
    for p in pdbs:
        name = p.name.split('_', 1)[1].replace('.pdb', '')
        s = score(p)
        if s is None:
            print(f'  {p.name}: no CLA ligand'); continue
        s['name'] = name
        s['bunspdb_rank'] = int(p.name[3:5])
        sh = short.get(name, {})
        hu = hull.get(name, {})
        s['buns_score_paper'] = (2 * float(sh['buns_ligand']) + float(sh['buns_protein'])
                                 if sh else '')
        s['packstat'] = sh.get('packstat', '')
        s['ligand_plddt'] = sh.get('ligand_plddt', '')
        s['frac_apolar_hull'] = hu.get('frac_apolar_hull', '')
        s['mg_in_a9'] = hu.get('mg_in_a9', '')
        rows.append(s)

    rows.sort(key=lambda d: -d['buried_frac'])
    print('%-30s %7s %7s %7s %7s %6s %7s' % ('design', 'buried', 'core', 'apolar',
                                             'Mg SASA', 'prop', 'BUNsSc'))
    print('%-30s %7s %7s %7s %7s %6s %7s' % ('', '%', '%', '%', 'A2', 'A2', ''))
    for i, d in enumerate(rows, 1):
        print('%-30s %6.1f%% %6.1f%% %6.1f%% %7.2f %6.1f %7s' % (
            d['name'].replace('_model_0', '')[:30], 100 * d['buried_frac'],
            100 * d['buried_frac_core'], 100 * d['buried_frac_apolar'],
            d['sasa_mg_A2'], d['sasa_propionate_A2'],
            f"{d['buns_score_paper']:.0f}" if d['buns_score_paper'] != '' else '-'))

    cols = ['name', 'buried_frac', 'buried_frac_core', 'buried_frac_apolar',
            'buried_frac_propionate', 'sasa_complex_A2', 'sasa_free_A2', 'sasa_mg_A2',
            'sasa_propionate_A2', 'n_heavy', 'buns_score_paper', 'packstat', 'ligand_plddt',
            'frac_apolar_hull', 'mg_in_a9', 'bunspdb_rank']
    csv_path = out / 'sasa_ranking.csv'
    with csv_path.open('w', newline='') as f:
        f.write(f'# Shrake-Rupley SASA, probe {PROBE} A, {N_POINTS} points/atom, heavy atoms only.\n')
        f.write('# buried_frac = 1 - SASA(ligand in complex) / SASA(ligand alone, same coords).\n')
        f.write('# Ranked by buried_frac. Structures are the Rosetta-minimised poses in ../.\n')
        f.write('# buried_frac_propionate is reported but NOT ranked on: the propionate is meant\n')
        f.write('#   to stay solvent-exposed, so a HIGH value there is a defect, not a merit.\n')
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for d in rows:
            w.writerow(d)

    for old in out.glob('sasa*.pdb'):
        old.unlink()
    for i, d in enumerate(rows, 1):
        src = next((p for p in pdbs if d['name'] in p.name), None)
        if src:
            shutil.copy(src, out / f'sasa{i:02d}_{d["name"]}.pdb')

    b = np.array([d['buried_frac'] for d in rows])
    print('\nburied fraction: median %.1f%%  range %.1f-%.1f%%' % (100*np.median(b), 100*b.min(), 100*b.max()))
    if rows and rows[0]['frac_apolar_hull'] != '':
        h = np.array([float(d['frac_apolar_hull']) for d in rows if d['frac_apolar_hull'] != ''])
        ba = np.array([d['buried_frac_apolar'] for d in rows if d['frac_apolar_hull'] != ''])
        print('correlation(SASA apolar buried, alpha-hull apolar buried) r = %+.3f' % np.corrcoef(ba, h)[0, 1])
    bs = [d['buns_score_paper'] for d in rows if d['buns_score_paper'] != '']
    if len(bs) == len(rows):
        print('correlation(SASA buried, BUNs Score)                    r = %+.3f'
              % np.corrcoef(b, np.array(bs, dtype=float))[0, 1])
    print(f'\n-> {csv_path}')
    print(f'-> {len(rows)} structures copied as sasaNN_*.pdb')
    return 0


if __name__ == '__main__':
    sys.exit(main())
