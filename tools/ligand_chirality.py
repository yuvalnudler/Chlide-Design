#!/usr/bin/env python
"""Read the configuration of the ligand's 13-2 stereocentre out of a predicted pose.

WHY NOT `rmsd_use_chirality = True`
-----------------------------------
The obvious way to reject inverted epimers is to let calc_rmsd pass useChirality=True to
GetSubstructMatches, so a mirrored centre finds no atom match and scores inf. Measured on v5's
iteration-15 populations, that rejects EVERYTHING -- the correct epimer included:

    centre    reference pose   popA (inverted)   popB (correct)
    CBD1      S                R                 S
    MG1       R                R                 R        <- the metal, not a real centre
    CA2       (not perceived)  S                 S        <- perceived in predictions only

GetSubstructMatches requires EVERY specified centre in the query to agree. MG1 is a spurious
centre RDKit perceives on a 4-coordinate magnesium, and CA2 is perceived in the predictions but
not in the reference, so BOTH populations return 0 matches, every ligand_rmsd becomes inf, the
backbone queue empties at iteration 0 and the campaign freezes -- the v4 funnel-collapse mode.

So we check the one centre that is actually at issue, directly, and leave the substructure match
chirality-blind. That also keeps ligand_rmsd numerically identical in meaning to v5, which is what
makes v6b a single-variable comparison against it.

WHY THE TOPOLOGY COMES FROM THE COMPONENT, NOT FROM THE PDB
-----------------------------------------------------------
The first version of this module built the molecule with Chem.MolFromPDBBlock and then
AssignBondOrdersFromTemplate. MolFromPDBBlock infers HETATM bonds by interatomic distance, and on
a strained pose that guess goes wrong: 18 of 50 smoke-test predictions failed with
"Explicit valence for atom # 38 C, 5, is greater than permitted" and returned None. Since the
driver treats None as a gate failure, that alone would have rejected a third of every iteration's
designs for a reason having nothing to do with chemistry.

There is no need to guess. We build the component ourselves (tools/build_ligand.sh), so the
correct topology is already on disk in boltz_cache/mols/<CCD>.pkl. This module takes the bonds
from there and only the COORDINATES from the pose, which is the one thing the pose actually knows.

The correct configuration is S, verified against PDB 5HPZ (13'2-hydroxy-chlorophyll a, the
structure that defines CCD code 68G; both independent copies are S).
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import prody as pr
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog('rdApp.*')

CENTRE_NAME = 'CBD'          # CCD name of the 13-2 carbon; 'CBD1' after protonation
DEFAULT_CCD = 'CLAOX'
_CACHE: dict[str, tuple] = {}


def _protonated(name: str, seen: dict[str, int]) -> str:
    """The name protonate_and_add_conect_records.py gives this atom.

    It strips the digits off the CCD name to get a prefix, then numbers per prefix in order:
    MG -> MG1, CBD -> CBD1, OCD -> OCD1, O1A -> OA1, O2A -> OA2, CAA -> CAA1.
    """
    prefix = ''.join(c for c in name if not c.isdigit())
    seen[prefix] = seen.get(prefix, 0) + 1
    return f'{prefix}{seen[prefix]}'


def _template(component: Path) -> tuple:
    """Heavy-atom template from the component, with both naming conventions indexed."""
    key = str(component)
    if key not in _CACHE:
        mol = pickle.load(open(component, 'rb'))
        rw = Chem.RWMol(mol)
        for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'H'),
                          reverse=True):
            rw.RemoveAtom(idx)
        heavy = rw.GetMol()
        ccd = [a.GetPropsAsDict().get('name') for a in heavy.GetAtoms()]
        seen: dict[str, int] = {}
        prot = [_protonated(n, seen) for n in ccd]
        _CACHE[key] = (heavy, ccd, prot)
    return _CACHE[key]


def ligand_config(atoms: pr.AtomGroup, smiles: str | None = None,
                  component: Path | str | None = None,
                  centre: str = CENTRE_NAME) -> str | None:
    """CIP code ('S' or 'R') at the 13-2 centre, or None if it could not be determined.

    None is NOT a pass: the caller must treat it as a failure to verify and say so loudly. A
    silent None that defaults to "accept" would reinstate exactly the defect this module exists
    to catch. `smiles` is accepted and ignored, so the driver's call site reads the same as it
    would for a SMILES-based implementation.
    """
    if component is None:
        # Search both conventional locations. v6a folds with AF3 and has no Boltz cache to speak
        # of, so a hard default of boltz_cache/mols/ would return None for every design -- and
        # None is a gate FAILURE, which would empty the backbone queue every iteration and
        # freeze the campaign for a reason that has nothing to do with chemistry.
        # The RUN directory, not this file's location. These tools used to live INSIDE the
        # campaign, so __file__.parent.parent was the run; now they live in the repo and that
        # guess resolves to the repo root, which has no boltz_cache/ and no input/ -- so this
        # returned None for every design, and None is a gate FAILURE. Every ligand_rmsd became
        # inf, which in production empties the backbone queue at iteration 0 and freezes the
        # campaign: the v4 funnel collapse, from a path bug rather than from chemistry.
        here = Path(os.environ.get('CHLIDE_RUN_DIR')
                    or Path(__file__).resolve().parent.parent)
        for cand in (here / 'boltz_cache' / 'mols' / f'{DEFAULT_CCD}.pkl',
                     here / 'input' / f'{DEFAULT_CCD}.pkl'):
            if cand.is_file():
                component = cand
                break
        else:
            return None
    component = Path(component)
    if not component.is_file():
        return None
    try:
        heavy, ccd_names, prot_names = _template(component)
    except Exception:                                      # noqa: BLE001
        return None

    lig = atoms.select('(not protein) and not element H')
    if lig is None:
        return None
    pose = {n: c for n, c in zip(lig.getNames(), lig.getCoords())}

    # the pose uses either raw CCD names (Boltz output) or protonated ones (post-reduce, relaxed)
    for names in (ccd_names, prot_names):
        if all(n in pose for n in names):
            mol = Chem.Mol(heavy)
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, n in enumerate(names):
                x, y, z = (float(v) for v in pose[n])
                conf.SetAtomPosition(i, Point3D(x, y, z))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            try:
                Chem.AssignStereochemistryFrom3D(mol)
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            except Exception:                              # noqa: BLE001
                return None
            want = names[ccd_names.index(centre)] if centre in ccd_names else centre
            idx = {n: i for i, n in enumerate(names)}.get(want)
            if idx is None:
                return None
            return dict(Chem.FindMolChiralCenters(
                mol, includeUnassigned=False, useLegacyImplementation=False)).get(idx)
    return None
