#!/usr/bin/env python
"""Extract the CLA-OXY component from chlides_fixed.mmcif and repair it for use as a ligand.

CLA-OXY is chlorophyllide a carrying the 13(2)-hydroxyl that v4's CLA-IDE was missing: the sole
heavy-atom difference is OCD, bonded to CBD (the ring-E C13(2) carbon) at 1.40 A, with the C13(1)
ketone CAD=OBD retained. That part of the file is correct and is what we want.

The block also carries a hand-edit defect that would break any conformer generator, and this script
repairs it. Diagnosis, from the coordinates themselves:

  - CED (the methyl ester carbon) is bonded to O2D, HED1, HED2 and HED3, but HED3 sits 7.92 A away
    from CED -- and 0.97 A from O2A, at an O2A-H bond length with a CGA-O2A-HED3 angle of 106.2
    degrees. That is not a displaced methyl hydrogen; it is a carboxylic-acid proton wearing the
    wrong label. CLA-IDE has an H21 bonded to O2A and CLA-OXY does not, so what happened is clear:
    H21 was deleted and HED3 was pointed at its position.

  So the repair is not "move HED3 back" -- it is to restore the two atoms that were conflated:

  1. Rename the misplaced HED3 to H21, retype it H, and rebond it O2A-H21.
  2. Build a genuine HED3 on CED at the fourth tetrahedral position from O2D/HED1/HED2.
  3. Set O2A charge back to 0, since it now carries a hydrogen again.

Choosing the protonated (neutral COOH) form is deliberate: it is what v4 used, so the only chemical
difference between the two campaigns is the 13(2)-OH. That isolates the variable the comparison is
about. A deprotonated carboxylate would be the more realistic species at pH 7 and is a defensible
alternative, but it would confound the v4 -> v5 comparison.

Also corrected: _chem_comp.formula and formula_weight, which still carry the fully phytylated
chlorophyll a values (C55 H72 Mg N4 O6 / 906.489) rather than this chlorophyllide methyl ester.

Result: 80 atoms, 46 heavy (Mg 1, C 35, N 4, O 6) and 34 H -- i.e. exactly CLA-IDE plus the
hydroxyl, with every atom correctly typed. Note CLA-IDE itself types its H21 as element C; this
component does not inherit that bug.

Usage:
    python tools/repair_cla_oxy.py \
        --source /home/data/yuvaln/main_pipeline/input/chlides_fixed.mmcif \
        --out input/cla_oxy_userccd.cif [--ccd-code CLX]
"""
from __future__ import annotations

import os
import argparse
import shlex
import sys
from pathlib import Path

import numpy as np

ATOM_COLS = [
    'comp_id', 'atom_id', 'alt_atom_id', 'type_symbol', 'charge', 'pdbx_align',
    'pdbx_aromatic_flag', 'pdbx_leaving_atom_flag', 'pdbx_stereo_config',
    'model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z',
    'pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 'pdbx_model_Cartn_z_ideal',
    'pdbx_component_atom_id', 'pdbx_component_comp_id', 'pdbx_ordinal',
]
BOND_COLS = ['comp_id', 'atom_id_1', 'atom_id_2', 'value_order',
             'pdbx_aromatic_flag', 'pdbx_stereo_config', 'pdbx_ordinal']

CH_BOND = 1.09      # C-H
ELEMENT_WEIGHT = {'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999, 'MG': 24.305}


def quote(v: str) -> str:
    """Re-quote a CIF value if it contains whitespace (alt_atom_id values like `N A`)."""
    return f'"{v}"' if (' ' in v or '\t' in v) else v


def read_block(source: Path, comp: str):
    """Pull one data_ block's atom and bond rows out of a multi-component CCD file."""
    lines = source.read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == f'data_{comp}')
    except StopIteration:
        sys.exit(f'{comp} not found in {source}')
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('data_')), len(lines))
    block = lines[start:end]

    header, atoms, bonds = [], [], []
    for l in block:
        if l.startswith('_chem_comp.'):
            header.append(l)
            continue
        if not l.startswith(comp + ' '):
            continue
        # shlex, not split(): alt_atom_id holds quoted values with internal spaces ("N A")
        f = shlex.split(l)
        if len(f) == len(ATOM_COLS):
            atoms.append(dict(zip(ATOM_COLS, f)))
        elif len(f) == len(BOND_COLS):
            bonds.append(dict(zip(BOND_COLS, f)))
    return header, atoms, bonds


def xyz(a: dict) -> np.ndarray:
    return np.array([float(a['model_Cartn_x']), float(a['model_Cartn_y']), float(a['model_Cartn_z'])])


def set_xyz(a: dict, v: np.ndarray) -> None:
    a['model_Cartn_x'], a['model_Cartn_y'], a['model_Cartn_z'] = (f'{c:.3f}' for c in v)
    a['pdbx_model_Cartn_x_ideal'] = a['model_Cartn_x']
    a['pdbx_model_Cartn_y_ideal'] = a['model_Cartn_y']
    a['pdbx_model_Cartn_z_ideal'] = a['model_Cartn_z']


def fourth_tetrahedral(centre: np.ndarray, others: list[np.ndarray], bond: float) -> np.ndarray:
    """The remaining substituent position on a centre that already has three."""
    u = sum((o - centre) / np.linalg.norm(o - centre) for o in others)
    return centre - bond * u / np.linalg.norm(u)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path,
                    default=Path(os.environ.get(
                        'CHLIDES_MMCIF',
                        '/home/data/yuvaln/main_pipeline/input/chlides_fixed.mmcif')))
    ap.add_argument('--comp', default='CLA-OXY')
    ap.add_argument('--ccd-code', default='CLX',
                    help='code written into the repaired file; must not collide with CLA')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--deprotonated', action='store_true',
                    help='leave the C17 propionate as the CARBOXYLATE (-COO-), the species that '
                         'actually exists at pH 7. v4/v5/v6b used the neutral -COOH to keep the '
                         'v4->v5 comparison single-variable; that choice is no longer wanted.')
    a = ap.parse_args()

    header, atoms, bonds = read_block(a.source, a.comp)
    by = {x['atom_id']: x for x in atoms}
    print(f'{a.comp}: {len(atoms)} atoms, {len(bonds)} bonds')

    # --- repair 1: the misplaced HED3 is really O2A's proton -------------------------------
    hed3, o2a, ced = by.get('HED3'), by.get('O2A'), by.get('CED')
    if hed3 is None or o2a is None or ced is None:
        sys.exit('expected HED3, O2A and CED to be present')
    d_ced = np.linalg.norm(xyz(hed3) - xyz(ced))
    d_o2a = np.linalg.norm(xyz(hed3) - xyz(o2a))
    print(f'  HED3 as found: {d_ced:.2f} A from CED, {d_o2a:.2f} A from O2A')
    if d_ced < 1.3:
        print('  HED3 already bonded correctly -- nothing to repair, aborting to avoid damage')
        return 1
    if not (0.85 < d_o2a < 1.10):
        sys.exit(f'HED3 is displaced but not at an O-H distance from O2A ({d_o2a:.2f} A); '
                 'the defect is not the one this script knows how to fix')

    hed3['atom_id'] = hed3['alt_atom_id'] = hed3['pdbx_component_atom_id'] = 'H21'
    hed3['type_symbol'] = 'H'
    for b in bonds:
        if {b['atom_id_1'], b['atom_id_2']} == {'CED', 'HED3'}:
            b['atom_id_1'], b['atom_id_2'] = 'O2A', 'H21'
    print('  renamed HED3 -> H21, rebonded O2A-H21')

    # --- repair 2: build the real HED3 on CED ----------------------------------------------
    new = fourth_tetrahedral(xyz(ced), [xyz(by[n]) for n in ('O2D', 'HED1', 'HED2')], CH_BOND)
    fresh = dict(hed3)
    fresh.update(atom_id='HED3', alt_atom_id='HED3', pdbx_component_atom_id='HED3',
                 type_symbol='H', charge='0', pdbx_aromatic_flag='N',
                 pdbx_leaving_atom_flag='N', pdbx_stereo_config='N')
    set_xyz(fresh, new)
    atoms.append(fresh)
    bonds.append(dict(comp_id=a.comp, atom_id_1='CED', atom_id_2='HED3', value_order='SING',
                      pdbx_aromatic_flag='N', pdbx_stereo_config='N', pdbx_ordinal='0'))
    clash = min(np.linalg.norm(new - xyz(x)) for x in atoms if x['atom_id'] != 'HED3'
                and x['atom_id'] != 'CED')
    print(f'  rebuilt HED3 at {np.round(new, 3)} (CED-HED3 {CH_BOND:.2f} A, '
          f'nearest other atom {clash:.2f} A)')
    if clash < 1.4:
        sys.exit(f'rebuilt HED3 clashes at {clash:.2f} A -- refusing to write')

    # --- repair 3: the C17 propionate protonation state --------------------------------------
    # The misplaced HED3 really was O2A's proton, so by default we restore it and O2A goes
    # neutral. But a propionic acid has pKa ~4.8: at pH 7 it is >99% DEPROTONATED. The neutral
    # -COOH was a deliberate v4-comparability choice, and it has a real cost -- the designer sees
    # no charge on the ligand's longest, most solvent-exposed substituent, so it places nothing
    # there. Measured on v6b's top 25: the COOH proton donates an H-bond in 1/23 designs and not
    # one design puts an Arg or Lys near the carboxylate. --deprotonated asks the designer to
    # satisfy a real anion instead, which is the one unused specificity handle on this ligand.
    if a.deprotonated:
        atoms[:] = [x for x in atoms if x['atom_id'] != 'H21']
        bonds[:] = [b for b in bonds if 'H21' not in (b['atom_id_1'], b['atom_id_2'])]
        o2a['charge'] = '-1'
        print('  DEPROTONATED: removed H21, O2A charge -> -1 (carboxylate, the pH 7 species)')
    elif o2a['charge'] != '0':
        print(f"  O2A charge {o2a['charge']} -> 0 (now carries H21)")
        o2a['charge'] = '0'

    # --- repair 5: the magnesium is not bonded to anything -----------------------------------
    # This is the campaign-breaking one. The CLA-OXY block carries no MG-N bonds at all, so the
    # metal is a disconnected fragment -- which is precisely the state that made Boltz scatter the
    # Mg 27 A out of the ring before v4 switched to a custom component. The working v4 component
    # (main_pipeline/input/cla_ide_userccd.cif) declares four MG-N SING bonds and gives MG and all
    # four pyrrole N a formal charge of 0. Match that convention exactly: build_boltz_ccd.py's
    # DATIVE promotion is tuned to it (with N at charge 0 and three ring bonds, a fourth SING bond
    # to Mg overflows the valence, which is what triggers the promotion).
    mg_bonds = [b for b in bonds if 'MG' in (b['atom_id_1'], b['atom_id_2'])]
    if not mg_bonds:
        for n in ('NA', 'NB', 'NC', 'ND'):
            if n not in by:
                sys.exit(f'cannot add MG-{n}: atom missing')
            bonds.append(dict(comp_id=a.comp, atom_id_1='MG', atom_id_2=n, value_order='SING',
                              pdbx_aromatic_flag='N', pdbx_stereo_config='N', pdbx_ordinal='0'))
        print('  added 4 MG-N SING bonds (component had none -- Mg was disconnected)')
    for n, want in (('MG', '0'), ('NA', '0'), ('NB', '0'), ('NC', '0'), ('ND', '0')):
        if by[n]['charge'] != want:
            print(f"  {n} charge {by[n]['charge']} -> {want} (v4 convention)")
            by[n]['charge'] = want

    # --- repair 6: the C13(2) stereo label is stale ------------------------------------------
    # The block inherits pdbx_stereo_config = R at CBD from CLA-IDE, but that label no longer
    # applies. Verified from the conformers: the signed volume of (CHA, CAD, CGD) about CBD and
    # the side the fourth substituent sits on are identical in CLA-IDE and CLA-OXY, so the OH
    # occupies exactly the position the H did -- retention of configuration. The descriptor
    # nevertheless flips R -> S, because O outranks all three carbons whereas H ranked below them.
    # RDKit assigns S from the 3D coordinates; the text field says R. The geometry is the truth.
    cbd = by.get('CBD')
    if cbd is not None and cbd['pdbx_stereo_config'] == 'R':
        cbd['pdbx_stereo_config'] = 'S'
        print('  CBD stereo label R -> S (CIP relabelling; geometry unchanged, OH replaces H)')

    # --- repair 4: formula and weight --------------------------------------------------------
    counts: dict[str, int] = {}
    for x in atoms:
        counts[x['type_symbol'].upper()] = counts.get(x['type_symbol'].upper(), 0) + 1
    # CCD writes element symbols in title case: "C35 H34 Mg N4 O6", carbon and hydrogen first
    formula = ' '.join(f'{e.title()}{counts[e]}' if counts[e] > 1 else e.title()
                       for e in sorted(counts, key=lambda e: (e != 'C', e != 'H', e)))
    weight = sum(ELEMENT_WEIGHT[e] * n for e, n in counts.items())
    print(f'  composition: {counts}  ->  formula "{formula}", weight {weight:.3f}')

    # --- repair 7: heavy atoms must come before hydrogens ------------------------------------
    # THE ONE THAT ACTUALLY BREAKS PREDICTION. CCD components list every heavy atom before any
    # hydrogen, and Boltz relies on it. The source block appended OCD by hand at ordinal 78, i.e.
    # in the middle of the hydrogen block, so a component built straight from it has its hydroxyl
    # oxygen sitting at index 77 among the H's. Boltz then reads that atom as a hydrogen and the
    # whole ligand is mis-assigned: measured over 3 replicates against the chemically-equivalent
    # v4 molecule, ligand pLDDT fell 0.837 -> 0.49, predicted C=O stretched 1.23 -> 1.52 A, and the
    # Mg drifted 0.04 -> 0.75 A off the pyrrole-N centroid. Sorting heavy-first fixes it.
    # Stable sort: relative order within each group is preserved, so this is the minimal change.
    before = [x['atom_id'] for x in atoms]
    atoms.sort(key=lambda x: x['type_symbol'].upper() == 'H')
    if [x['atom_id'] for x in atoms] != before:
        first_h = next(i for i, x in enumerate(atoms) if x['type_symbol'].upper() == 'H')
        moved = [x['atom_id'] for x in atoms[:first_h] if x['atom_id'] not in before[:first_h]]
        print(f'  reordered heavy-atoms-first (moved {moved} ahead of the hydrogen block)')

    # --- renumber and emit -------------------------------------------------------------------
    for i, x in enumerate(atoms, 1):
        x['pdbx_ordinal'] = str(i)
        x['comp_id'] = x['pdbx_component_comp_id'] = a.ccd_code
    for i, b in enumerate(bonds, 1):
        b['pdbx_ordinal'] = str(i)
        b['comp_id'] = a.ccd_code

    out = [f'data_{a.ccd_code}']
    replaced = {
        '_chem_comp.id': a.ccd_code,
        '_chem_comp.three_letter_code': a.ccd_code,
        '_chem_comp.name': '"CHLOROPHYLLIDE A 13(2)-HYDROXY"',
        '_chem_comp.formula': f'"{formula}"',
        '_chem_comp.formula_weight': f'{weight:.3f}',
        '_chem_comp.pdbx_ideal_coordinates_details': '"rebuilt HED3; see tools/repair_cla_oxy.py"',
    }
    for l in header:
        key = l.split(None, 1)[0]
        out.append(f'{key} {replaced[key]}' if key in replaced else l)
    out += ['#', 'loop_'] + [f'_chem_comp_atom.{c}' for c in ATOM_COLS]
    out += [' '.join(quote(x[c]) for c in ATOM_COLS) for x in atoms]
    out += ['#', 'loop_'] + [f'_chem_comp_bond.{c}' for c in BOND_COLS]
    out += [' '.join(quote(b[c]) for c in BOND_COLS) for b in bonds]
    out.append('#')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text('\n'.join(out) + '\n')
    heavy = sum(n for e, n in counts.items() if e != 'H')
    print(f'\n-> {a.out}\n   {len(atoms)} atoms ({heavy} heavy), {len(bonds)} bonds, '
          f'code {a.ccd_code}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
