#!/usr/bin/env python
"""
Build a Boltz CCD component pickle from a PDB-format chem_comp CIF.

Why this exists: Boltz ships no CCD entry for chlorophyll-family ligands (CLA, CHL and
BCL are all absent from its complete 45228-component snapshot, while HEM and B12 are
present) -- almost certainly because RDKit ETKDG cannot embed an Mg-chlorin. Passing the
ligand as SMILES therefore forces the Mg to be a disconnected [Mg+2] ion, and Boltz then
parks it tens of angstroms from the macrocycle.

Boltz's get_mol() falls back to <cache>/mols/<CODE>.pkl for any code it does not know, and
parse_ccd_residue() reads the stored conformer rather than embedding one. So a component
built here is used verbatim, metal bonds and all.

Mg-N bond orders follow the HEM convention already in Boltz's cache: a coordinate bond is
written DATIVE whenever a plain SINGLE bond would overflow the nitrogen's valence, which is
what lets RDKit sanitize a metalloporphyrin at all.

Usage: build_boltz_ccd.py <component.cif> <out.pkl> [--conformer ideal|model]
"""
import sys, shlex, pickle, argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

BOND_ORDER = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
              'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}


def read_loop(lines, prefix):
    """Return (column names, rows) for the _<prefix>. loop in a chem_comp CIF.

    Strict: locates the loop_ whose header block belongs to <prefix>, reads header lines
    until the first non-underscore line, then reads body rows until the block terminator.
    """
    tag = '_' + prefix + '.'
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == 'loop_' and i + 1 < n and lines[i + 1].strip().startswith(tag):
            i += 1
            cols = []
            while i < n and lines[i].strip().startswith('_'):
                cols.append(lines[i].strip().split('.', 1)[1].split()[0]); i += 1
            rows = []
            while i < n:
                s = lines[i].strip()
                if s in ('#', '') or s == 'loop_' or s.startswith('_') or s.startswith('data_'):
                    break
                # CIF quotes values containing spaces, e.g. alt_atom_id 'N A' on the
                # pyrrole nitrogens; a naive split() would shred exactly those rows.
                rows.append(shlex.split(s)); i += 1
            return cols, rows
        i += 1
    raise ValueError('loop for %s not found' % prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cif'); ap.add_argument('out')
    ap.add_argument('--conformer', choices=['ideal', 'model'], default='ideal')
    a = ap.parse_args()

    lines = open(a.cif).read().splitlines()
    acols, arows = read_loop(lines, 'chem_comp_atom')
    bcols, brows = read_loop(lines, 'chem_comp_bond')
    ai = {c: i for i, c in enumerate(acols)}
    bi = {c: i for i, c in enumerate(bcols)}

    xk = ('pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 'pdbx_model_Cartn_z_ideal') \
        if a.conformer == 'ideal' else ('model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z')

    mol = Chem.RWMol()
    idx_of, coords, names = {}, [], []
    for r in arows:
        nm = r[ai['atom_id']]
        at = Chem.Atom(r[ai['type_symbol']].capitalize())
        at.SetFormalCharge(int(float(r[ai['charge']])))
        at.SetProp('name', nm)
        at.SetProp('alt_name', r[ai['alt_atom_id']])
        at.SetProp('leaving_atom', '1' if r[ai['pdbx_leaving_atom_flag']] == 'Y' else '0')
        at.SetNoImplicit(True)
        idx_of[nm] = mol.AddAtom(at)
        coords.append([float(r[ai[k]]) for k in xk]); names.append(nm)

    # Non-metal bonds first so nitrogen valences are known before the metal is attached.
    metal_bonds = []
    for r in brows:
        n1, n2, order = r[bi['atom_id_1']], r[bi['atom_id_2']], r[bi['value_order']]
        a1, a2 = mol.GetAtomWithIdx(idx_of[n1]), mol.GetAtomWithIdx(idx_of[n2])
        if a1.GetAtomicNum() in METALS or a2.GetAtomicNum() in METALS:
            metal_bonds.append((n1, n2, order)); continue
        mol.AddBond(idx_of[n1], idx_of[n2], BOND_ORDER[order])
        if r[bi['pdbx_aromatic_flag']] == 'Y':
            a1.SetIsAromatic(True); a2.SetIsAromatic(True)
            mol.GetBondBetweenAtoms(idx_of[n1], idx_of[n2]).SetIsAromatic(True)

    MAXVAL = {7: 3, 8: 2, 16: 2}
    for n1, n2, order in metal_bonds:
        i1, i2 = idx_of[n1], idx_of[n2]
        metal, donor = (i1, i2) if mol.GetAtomWithIdx(i1).GetAtomicNum() in METALS else (i2, i1)
        d = mol.GetAtomWithIdx(donor)
        used = sum(b.GetBondTypeAsDouble() for b in d.GetBonds())
        # DATIVE does not consume donor valence; use it wherever SINGLE would overflow.
        if used + 1 > MAXVAL.get(d.GetAtomicNum(), 4) - d.GetFormalCharge():
            mol.AddBond(donor, metal, Chem.BondType.DATIVE)
        else:
            mol.AddBond(donor, metal, BOND_ORDER[order])

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*c))
    conf.SetProp('name', 'Ideal' if a.conformer == 'ideal' else 'Computed')
    mol.AddConformer(conf, assignId=True)

    m = mol.GetMol()
    Chem.SanitizeMol(m)
    m.SetProp('MOL_NAME', a.out.split('/')[-1].rsplit('.', 1)[0])
    # Automorphisms, used for symmetry-corrected scoring. Optional but cheap.
    try:
        sym = m.GetSubstructMatches(m, uniquify=False, useChirality=True, maxMatches=1000)
        m.SetProp('symmetries', pickle.dumps(np.array(sym, dtype=np.int64)).hex())
    except Exception as e:
        print('warning: could not compute symmetries:', e)

    # RDKit drops atom/bond properties on pickling unless this is set globally, and Boltz
    # reads atom.GetProp('name') for every atom -- without it the component loads but
    # parsing dies with KeyError: 'name'. Boltz's own components are pickled this way.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(a.out, 'wb') as f:
        pickle.dump(m, f)

    heavy = sum(1 for at in m.GetAtoms() if at.GetAtomicNum() > 1)
    mets = [(at.GetProp('name'), at.GetDegree(),
             [str(b.GetBondType()) for b in at.GetBonds()])
            for at in m.GetAtoms() if at.GetAtomicNum() in METALS]
    print('wrote %s: %d atoms (%d heavy), %d bonds, %d conformer(s)'
          % (a.out, m.GetNumAtoms(), heavy, m.GetNumBonds(), m.GetNumConformers()))
    for nm, deg, bts in mets:
        print('  metal %s: degree %d, bonds %s' % (nm, deg, bts))


METALS = {12, 26, 27, 25, 30, 29, 42, 20, 11, 19}
if __name__ == '__main__':
    main()
