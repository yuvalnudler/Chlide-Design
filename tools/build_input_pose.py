#!/usr/bin/env python
"""Convert the v4 BDFP44 input pose from CLA-IDE to CLAOX by adding the 13(2)-hydroxyl.

The point of v5 is to change one thing: the ligand. So this does NOT re-dock, and it does not
superimpose the whole CCD conformer onto the placed ligand -- either would move the binding pose
and confound the v4/v5 comparison. Instead it keeps every placed atom exactly where it is and
performs the minimal edit:

    CBD1-H31  ->  CBD1-OCD1-HOD1

OCD1 and HOD1 are positioned by superimposing the repaired CCD component onto the pose over the
local ring-E frame (CBD, CAD, CGD, CHA) and reading the hydroxyl coordinates out of that fit. That
gives correct local geometry -- C-O 1.40 A, O-H 0.97 A, proper angles, and the configuration at
C13(2) retained -- without touching the macrocycle, the Mg, the His85 coordination or the
propionate.

Atom naming follows the campaign's `+1` suffix convention (MG1, CBD1, ...), so the new atoms are
OCD1 and HOD1. H31 is dropped; the remaining hydrogens keep their names, which means H32-H34 are
NOT renumbered. That is deliberate: the Rosetta .params is regenerated from the SMILES anyway, and
renumbering would make the v4/v5 pose diff unreadable.

Usage:
    python tools/build_input_pose.py \
        --in  input/input_backbones/bdfp44_cla_his85_protonated.pdb \
        --ccd boltz_cache/mols/CLAOX.pkl \
        --out input/input_backbones/bdfp44_claox_his85_protonated.pdb
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# pose name (with the +1 suffix) -> CCD component name
FRAME = {'CBD1': 'CBD', 'CAD1': 'CAD', 'CGD1': 'CGD', 'CHA1': 'CHA'}
DROP = 'H31'          # the C13(2) hydrogen the hydroxyl replaces
ADD = {'OCD1': 'OCD', 'HOD1': 'HOD'}
HIS_RESNUM = 85


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation taking P onto Q (both N x 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def pdb_atoms(lines):
    for i, l in enumerate(lines):
        if l.startswith(('ATOM', 'HETATM')):
            yield i, l[12:16].strip(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])


def fmt(template: str, name: str, xyz: np.ndarray, element: str, serial: int) -> str:
    """Build a HETATM line from an existing one, replacing name, element and coordinates."""
    n = f' {name:<3s}' if len(name) < 4 else name
    return (template[:6] + f'{serial:5d}' + template[11:12] + n + template[16:30]
            + f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}' + template[54:76]
            + f'{element:>2s}' + template[78:].rstrip('\n'))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', type=Path, required=True)
    ap.add_argument('--ccd', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ligand-resname', default='CLA')
    a = ap.parse_args()

    lines = a.src.read_text().splitlines(keepends=True)
    het = {name: (i, xyz) for i, name, xyz in pdb_atoms(lines) if lines[i].startswith('HETATM')}
    prot = {(int(lines[i][22:26]), name): xyz for i, name, xyz in pdb_atoms(lines)
            if lines[i].startswith('ATOM')}
    print(f'{a.src.name}: {len(prot)} protein atoms, {len(het)} ligand atoms')

    missing = [n for n in list(FRAME) + [DROP] if n not in het]
    if missing:
        sys.exit(f'pose is missing {missing} -- is this the CLA-IDE input?')
    if any(n in het for n in ADD):
        sys.exit('pose already contains OCD1/HOD1; nothing to do')

    mol = pickle.load(open(a.ccd, 'rb'))
    conf = mol.GetConformer()
    ccd = {at.GetPropsAsDict().get('name'): np.array(conf.GetAtomPosition(at.GetIdx()))
           for at in mol.GetAtoms()}
    for n in list(FRAME.values()) + list(ADD.values()):
        if n not in ccd:
            sys.exit(f'CCD component has no atom {n}')

    # local fit over ring E only -- a global fit would move the binding pose
    P = np.array([ccd[FRAME[k]] for k in FRAME])
    Q = np.array([het[k][1] for k in FRAME])
    R, t = kabsch(P, Q)
    rms = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
    print(f'  ring-E frame fit over {list(FRAME)}: RMSD {rms:.3f} A')
    if rms > 0.5:
        sys.exit(f'ring-E fit RMSD {rms:.2f} A is too large to trust')

    new = {k: (R @ ccd[v]) + t for k, v in ADD.items()}

    cbd = het['CBD1'][1]
    d_oc = np.linalg.norm(new['OCD1'] - cbd)
    d_oh = np.linalg.norm(new['HOD1'] - new['OCD1'])
    print(f'  placed OCD1: CBD1-OCD1 {d_oc:.2f} A, OCD1-HOD1 {d_oh:.2f} A')
    if not (1.30 < d_oc < 1.50):
        sys.exit(f'CBD1-OCD1 {d_oc:.2f} A is not a C-O bond')

    # Clash check against protein and the rest of the ligand. CBD1 is OCD1's own bonding partner,
    # so it must be excluded or it reports itself as the worst contact at the C-O bond length.
    bonded = {'CBD1'}
    others = {**{f'{r}:{n}': v for (r, n), v in prot.items()},
              **{n: v for n, (_, v) in het.items() if n != DROP and n not in bonded}}
    worst = min(((np.linalg.norm(new['OCD1'] - v), k) for k, v in others.items()),
                key=lambda x: x[0])
    print(f'  nearest non-bonded atom to OCD1: {worst[1]} at {worst[0]:.2f} A')
    if worst[0] < 2.2:
        print(f'  WARNING: OCD1 is {worst[0]:.2f} A from {worst[1]} -- inspect before use')

    # report the coordination we must not have disturbed
    ne2 = prot.get((HIS_RESNUM, 'NE2'))
    if ne2 is not None:
        print(f'  His{HIS_RESNUM} NE2-Mg: {np.linalg.norm(ne2 - het["MG1"][1]):.2f} A (unchanged)')

    drop_i = het[DROP][0]
    last_het_i = max(i for i, _ in het.values())
    tail_serial = int(lines[last_het_i][6:11])

    out, serial_shift = [], 0
    for i, l in enumerate(lines):
        if i == drop_i:
            continue                                  # H31 becomes the hydroxyl
        out.append(l)
        if i == last_het_i:
            for k, (name, el) in enumerate((('OCD1', 'O'), ('HOD1', 'H')), 1):
                out.append(fmt(lines[last_het_i], name, new[name], el, tail_serial + k) + '\n')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(out))
    n_het = sum(1 for l in out if l.startswith('HETATM'))
    print(f'\n-> {a.out}\n   ligand now {n_het} atoms (was {len(het)}): -{DROP}, +OCD1, +HOD1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
