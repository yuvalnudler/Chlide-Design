#!/usr/bin/env python
"""Metal-restrained Rosetta FastRelax for NISE designs.

Why this exists: Boltz cannot model the axial His-Mg bond. Left alone it floors the contact at
2.939 A (measured identically across 3,186 designs on two unrelated scaffolds, r=0.039 against
ligand pLDDT, so the objective carries no information about coordination). With a covalent bond
constraint it overshoots to ~1.58 A. Neither is a physical coordination distance.

Rosetta settles it. A FlatHarmonic restraint at 2.15 A over ref2015 pulls both cases into the
physical window in ~25 s, without distorting the ligand:

    no constraint   2.94 -> 2.54 A    score -266 -> -483   ligand moved 0.20 A
    bond constraint 1.58 -> 2.29 A    score -176 -> -484   ligand moved 0.22 A

The relaxed structure is what the next NISE iteration should be designed against, and the
post-relax distance -- how close the pocket lets the His settle, and at what strain -- is the
discriminating measurement that the Boltz number is not.

MUST run on NISE's re-protonated, renamed designs. Raw Boltz output has no hydrogens and keeps CCD
atom names (MG, CHA); Rosetta rejects those with "too many tries in fill_missing_atoms". Promoted
designs already carry LASErMPNN names (MG1, CHA1) and hydrogens.

Runs in the Rosetta env: /home/yigall/mambaforge/envs/Rosetta/bin/python

Usage:
    rosetta_relax.py --params <lig.params> --out-dir DIR [--workers N]
                     [--his-index 85] [--r0 2.15] [--tol 0.25] [--repeats 1] PDB [PDB ...]
Writes <out-dir>/<stem>_relaxed.pdb per input and prints one JSON object per line to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np

_INIT = False


def _init(params: str):
    global _INIT
    if _INIT:
        return
    import pyrosetta
    pyrosetta.init(f'-mute all -extra_res_fa {params} -ex1 -ex2', silent=True)
    _INIT = True


def _ca(pose, n):
    return np.array([[pose.residue(i).xyz('CA').x, pose.residue(i).xyz('CA').y,
                      pose.residue(i).xyz('CA').z] for i in range(1, n + 1)])


def _lig_heavy(pose):
    i = pose.total_residue()
    r = pose.residue(i)
    return np.array([[r.xyz(j).x, r.xyz(j).y, r.xyz(j).z] for j in range(1, r.natoms() + 1)
                     if r.atom_type(j).element().strip() != 'H'])


def relax_one(pdb: str, params: str, out_dir: str, his_index: int,
              r0: float, tol: float, repeats: int) -> dict:
    """Relax one design. Never raises: a failure returns {'ok': False, 'error': ...} so a bad
    design cannot take down an iteration."""
    rec = {'input': pdb, 'ok': False}
    try:
        _init(params)
        from pyrosetta import pose_from_file
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType
        from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
        from pyrosetta.rosetta.core.scoring.func import FlatHarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID

        t0 = time.time()
        pose = pose_from_file(pdb)
        lig_i = pose.total_residue()
        lig = pose.residue(lig_i)
        n_prot = lig_i - 1

        mg = [i for i in range(1, lig.natoms() + 1) if lig.atom_name(i).strip() == 'MG1']
        if not mg:
            rec['error'] = 'no MG1 in ligand'
            return rec
        mg = mg[0]
        if pose.residue(his_index).name3() != 'HIS':
            rec['error'] = f'residue {his_index} is {pose.residue(his_index).name3()}, not HIS'
            return rec
        ne2 = pose.residue(his_index).atom_index('NE2')

        def dist(p):
            return float((p.residue(his_index).xyz('NE2') - p.residue(lig_i).xyz(mg)).norm())

        def ring_offset(p):
            r = p.residue(lig_i)
            n4 = np.array([[r.xyz(j).x, r.xyz(j).y, r.xyz(j).z]
                           for j in range(1, r.natoms() + 1)
                           if r.atom_name(j).strip() in ('NA1', 'NB1', 'NC1', 'ND1')])
            m = np.array([r.xyz(mg).x, r.xyz(mg).y, r.xyz(mg).z])
            return float(np.linalg.norm(m - n4.mean(0))) if len(n4) == 4 else float('nan')

        sf = ScoreFunctionFactory.create_score_function('ref2015')
        sf.set_weight(ScoreType.atom_pair_constraint, 1.0)
        # Flat-bottomed at the physical coordination distance: no penalty within r0 +/- tol,
        # harmonic outside. Matches metal_r0_angstrom=2.15 / "bounded" used elsewhere in the project.
        pose.add_constraint(AtomPairConstraint(AtomID(ne2, his_index), AtomID(mg, lig_i),
                                               FlatHarmonicFunc(r0, 0.2, tol)))
        rec['his_mg_before'] = dist(pose)
        rec['mg_ring_offset_before'] = ring_offset(pose)
        rec['score_before'] = float(sf(pose))
        ca0, lig0 = _ca(pose, n_prot), _lig_heavy(pose)

        fr = FastRelax(sf, repeats)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)

        rec['his_mg'] = dist(pose)
        rec['mg_ring_offset'] = ring_offset(pose)
        rec['score'] = float(sf(pose))
        rec['ca_rmsd'] = float(np.sqrt(((ca0 - _ca(pose, n_prot)) ** 2).sum(1).mean()))
        rec['ligand_rmsd'] = float(np.sqrt(((lig0 - _lig_heavy(pose)) ** 2).sum(1).mean()))
        rec['seconds'] = round(time.time() - t0, 1)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Strip any accumulated suffix: a design relaxed once per iteration would otherwise
        # become ..._relaxed_relaxed_relaxed_relaxed.pdb over a campaign.
        stem = re.sub(r'(_relaxed)+$', '', Path(pdb).stem)
        out = str(Path(out_dir) / (stem + '_relaxed.pdb'))
        pose.dump_pdb(out)
        rec['output'] = out
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001 - a failed design must not kill the batch
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def _worker(args):
    return relax_one(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('pdbs', nargs='+')
    ap.add_argument('--params', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument('--his-index', type=int, default=85,
                    help='1-based residue index of the axial His (resindex 84 + 1)')
    ap.add_argument('--r0', type=float, default=2.15)
    ap.add_argument('--tol', type=float, default=0.25)
    ap.add_argument('--repeats', type=int, default=1)
    a = ap.parse_args()

    jobs = [(p, a.params, a.out_dir, a.his_index, a.r0, a.tol, a.repeats) for p in a.pdbs]
    if a.workers > 1 and len(jobs) > 1:
        # 'spawn' so each worker runs its own pyrosetta.init; forking an initialised
        # Rosetta is not safe.
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for rec in pool.imap_unordered(_worker, jobs):
                print(json.dumps(rec), flush=True)
    else:
        for j in jobs:
            print(json.dumps(_worker(j)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
