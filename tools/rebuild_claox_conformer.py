#!/usr/bin/env python
"""Rebuild the CLAOX component on v4's clean reference conformer instead of the crystal frame.

Why: Boltz reads the CCD component's stored conformer as its structural reference. v4's CLA-IDE
carries Corina-generated *ideal* coordinates (pdbx_ideal_coordinates_details = Corina). The
CLA-OXY block does not -- its model_Cartn_* and *_ideal columns are byte-identical, both the 1JB0
crystal frame. A 2.5 A crystal geometry is a noisier reference than an idealised one, and the
first CLAOX folds came out with inflated internal bonds (C=O 1.52 A predicted against 1.21 in the
component) and ligand pLDDT 0.49, against 0.84 for the chemically-equivalent v4 molecule folded
under the same conditions.

RDKit cannot simply embed an Mg-chlorin from scratch (ETKDG fails on the metal, which is the whole
reason this project ships a custom component). So instead of generating a conformer, this
transplants: take v4's Corina conformer, which covers 45 of the 46 heavy atoms, and place the one
new atom -- the 13(2) hydroxyl -- on it using the same local ring-E frame fit that
build_input_pose.py uses for the binding pose.

The result is the CLAOX topology on a Corina-quality conformer, differing from v4's reference by
exactly the hydroxyl.

Usage:
    python tools/rebuild_claox_conformer.py \
        --topology boltz_cache/mols/CLAOX.pkl \
        --reference boltz_cache/mols/CLA.pkl \
        --out boltz_cache/mols/CLAOX.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

# RDKit drops atom properties on pickle unless told otherwise, and Boltz keys every ligand atom
# by its 'name' property -- without this the component loads with 80 nameless atoms and Boltz
# fails with a bare KeyError: 'name'. build_boltz_ccd.py sets the same flag for the same reason.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

FRAME = ('CBD', 'CAD', 'CGD', 'CHA')   # local ring-E frame, shared by both molecules
NEW = ('OCD', 'HOD')                   # what only the oxy form has


def kabsch(P: np.ndarray, Q: np.ndarray):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def named(mol):
    return {a.GetPropsAsDict().get('name'): a.GetIdx() for a in mol.GetAtoms()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', required=True, help='component with the CLAOX topology')
    ap.add_argument('--reference', required=True, help='component whose conformer to adopt (v4 CLA)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    topo = pickle.load(open(a.topology, 'rb'))
    ref = pickle.load(open(a.reference, 'rb'))
    ti, ri = named(topo), named(ref)
    tc, rc = topo.GetConformer(), ref.GetConformer()

    shared = [n for n in ti if n in ri]
    missing = [n for n in ti if n not in ri and n not in NEW]
    if missing:
        sys.exit(f'topology has atoms the reference lacks beyond {NEW}: {missing}')
    print(f'{len(shared)} shared atoms, {len(ti) - len(shared)} new ({sorted(set(ti) - set(ri))})')

    # 1. copy the reference conformer onto every shared atom
    for n in shared:
        p = rc.GetAtomPosition(ri[n])
        tc.SetAtomPosition(ti[n], Point3D(p.x, p.y, p.z))

    # 2. place the new atoms by fitting the ring-E frame from the OLD topology conformer
    #    (which does have them) onto the reference frame
    old = pickle.load(open(a.topology, 'rb')).GetConformer()
    P = np.array([[*old.GetAtomPosition(ti[n])] for n in FRAME])
    Q = np.array([[*rc.GetAtomPosition(ri[n])] for n in FRAME])
    R, t = kabsch(P, Q)
    rms = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
    print(f'ring-E frame fit RMSD: {rms:.3f} A')
    if rms > 0.6:
        sys.exit(f'frame fit RMSD {rms:.2f} A too large to trust')
    for n in NEW:
        if n not in ti:
            continue
        v = (R @ np.array([*old.GetAtomPosition(ti[n])])) + t
        tc.SetAtomPosition(ti[n], Point3D(*(float(x) for x in v)))

    # 3. verify the rebuilt geometry
    P_ = lambda n: np.array([*tc.GetAtomPosition(ti[n])])
    checks = [('CBD-OCD', np.linalg.norm(P_('CBD') - P_('OCD')), 1.30, 1.55),
              ('OCD-HOD', np.linalg.norm(P_('OCD') - P_('HOD')), 0.85, 1.10)]
    mg = P_('MG')
    off = np.linalg.norm(mg - np.mean([P_(n) for n in ('NA', 'NB', 'NC', 'ND')], axis=0))
    checks.append(('Mg->N4 centroid', off, 0.0, 0.15))
    bad = []
    for name, v, lo, hi in checks:
        ok = lo <= v <= hi
        print(f'  {name:<16} {v:.3f} A  {"ok" if ok else "OUT OF RANGE"}')
        if not ok:
            bad.append(name)
    nearest = min(np.linalg.norm(P_('OCD') - P_(n)) for n in ti if n not in ('OCD', 'CBD', 'HOD'))
    print(f'  {"OCD nearest":<16} {nearest:.3f} A')
    if nearest < 1.9:
        bad.append('OCD clash')
    if bad:
        sys.exit(f'rebuilt conformer failed: {bad}')

    # ---- chirality tags: without these Boltz leaves the stereocentre free --------------------
    # Boltz ships a ChiralAtomPotential (model/potentials/potentials.py:538, resampling_weight 1.0,
    # on by default) but compute_chiral_atom_constraints (data/parse/schema.py:341) is gated on
    #     if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()])
    # Our v5 component carried no CIP tags, so Boltz built ZERO chiral constraints and the 13(2)
    # centre floated: 65% of v5 predictions came out as the wrong epimer, including every design
    # on the order list. The correct configuration is S, confirmed against PDB 5HPZ
    # (13'2-hydroxyl-Chlorophyll a, two independent copies, both S).
    #
    # The metal tag MUST be cleared. RDKit reports Mg as SP3 with four N neighbours and Boltz would
    # constrain it as a tetrahedral stereocentre, fighting the square-planar coordination. Measured:
    # 16 constraints with the metal left in (incl. MG), 12 with it cleared (C2A, C3A, CBD only).
    Chem.AssignStereochemistryFrom3D(topo)
    Chem.AssignStereochemistry(topo, cleanIt=True, force=True)
    for at in topo.GetAtoms():
        if at.GetSymbol() in ('Mg', 'Fe', 'Zn', 'Mn', 'Co', 'Ni', 'Cu'):
            at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.AssignStereochemistry(topo, cleanIt=False, force=False)

    centres = Chem.FindMolChiralCenters(topo, includeUnassigned=False)
    byname = {at.GetIdx(): at.GetPropsAsDict().get('name') for at in topo.GetAtoms()}
    tagged = {byname[i]: cfg for i, cfg in centres}
    print(f'  stereocentres tagged: {tagged}')
    if not all(at.HasProp('_CIPRank') for at in topo.GetAtoms()):
        sys.exit('not every atom carries _CIPRank -- Boltz will build no chiral constraints')
    if any(byname[i] in ('MG',) for i, _ in centres):
        sys.exit('the magnesium is still tagged as a stereocentre -- clearing failed')
    if tagged.get('CBD') != 'S':
        sys.exit(f"CBD is {tagged.get('CBD')}, expected S (verified against PDB 5HPZ)")

    with open(a.out, 'wb') as f:
        pickle.dump(topo, f)
    print(f'\n-> {a.out} (CLAOX topology on the v4 reference conformer)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
