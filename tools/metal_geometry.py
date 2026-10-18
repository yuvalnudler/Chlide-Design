#!/usr/bin/env python
"""Axial coordination QUALITY for a Mg-chlorin bound by a histidine.

The campaign gates coordination on ONE scalar: the His NE2 - Mg distance, 2.0-2.6 A. By the
time a design reaches the shortlist that gate is saturated -- across v8's top 25 the distance
is 2.425 +/- 0.030 A, so every survivor looks identical on the only coordination measure we
have. A distance cannot tell a real axial bond from a histidine sitting 2.4 A away while
pointing the wrong way.

Two independent things go unmeasured, and they are only weakly related (rho 0.30), so neither
substitutes for the other:

  WHERE the histidine sits
    axial_angle         Mg->NE2 against the chlorin ring normal. 0 = dead-on perpendicular.
                        v8 top 25: 1.6-10.0 deg, sd 2.5.

  HOW it is turned
    his_lonepair_angle  NE2->Mg against the CE1-NE2-CD2 bisector. The NE2 lone pair is an sp2
                        orbital, so this is the orbital-overlap direction.
                        v8 top 25: 6.5-34.6 deg, sd 6.1.
    mg_imidazole_oop    NE2->Mg out of the imidazole plane, which is where that lone pair
                        lies. 34 deg out is poor overlap even at a perfect 2.4 A.
                        v8 top 25: 1.7-34.4 deg, sd 6.8.

  Diagnostics
    mg_out_of_plane     Mg doming toward the His, A. NEARLY CONSTANT across designs (sd 0.019)
                        because the Rosetta ligand params fix the chlorin's internal geometry:
                        the loop cannot move the Mg out of its ring plane. Experimental
                        5-coordinate Mg-chlorophyll domes 0.3-0.5 A toward the axial ligand;
                        our minimised poses sit at 0.09 and CANNOT reproduce that. Read it as
                        QC on the method, never as a finding about the designs.
    n4_planarity        rms deviation of the four N from their plane; flags a distorted chlorin.

All of these are orthogonal to every column already in the shortlist (|rho| <= 0.36 against
his_mg, buns_score_paper, packstat, buried_frac*, sasa_mg_A2, ligand_plddt, composite; none
significant at p < 0.05), which is why they are worth reporting -- and they are REPORTED AND
FLAGGED ONLY. They are deliberately absent from `composite`: nothing yet shows they predict
binding, and Polizzi et al. Fig 1c is a direct warning about adopting a principled-looking
geometric term as an objective (Rosetta ligand energy failed that way over 70 iterations).

The maths lives here once. polizzi_cascade.py imports `from_pose` for in-campaign scoring;
`--pdb` measures a reference structure outside any campaign, and only that path needs the
atom-name fallback, because the Rosetta params pin MG1/NA1/NB1/NC1/ND1 inside a run while
pre-protonation reference PDBs use unsuffixed MG/NA/NB/NC/ND.

Usage:
    python tools/metal_geometry.py --pdb structure.pdb [--pdb other.pdb ...]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

PYRROLE_NAMES = (('NA1', 'NB1', 'NC1', 'ND1'), ('NA', 'NB', 'NC', 'ND'))
MG_NAMES = ('MG1', 'MG')
KEYS = ('his_mg', 'axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop',
        'mg_out_of_plane', 'n4_planarity')


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _ang(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def geometry(n4, mg, ne2, ce1, cd2) -> dict:
    """The pure-maths core. All five arguments are 3-vectors; n4 is (4, 3)."""
    n4 = np.asarray(n4, dtype=float)
    mg, ne2, ce1, cd2 = (np.asarray(x, dtype=float) for x in (mg, ne2, ce1, cd2))
    c = n4.mean(0)
    nrm = _unit(np.linalg.svd(n4 - c)[2][2])
    if np.dot(ne2 - c, nrm) < 0:              # orient the normal toward the histidine
        nrm = -nrm
    w = _unit(mg - ne2)
    bis = _unit(-(_unit(ce1 - ne2) + _unit(cd2 - ne2)))      # sp2 lone-pair direction at NE2
    imn = _unit(np.cross(ce1 - ne2, cd2 - ne2))              # imidazole plane normal
    return {
        'his_mg': float(np.linalg.norm(ne2 - mg)),
        'axial_angle': _ang(_unit(ne2 - mg), nrm),
        'his_lonepair_angle': _ang(w, bis),
        'mg_imidazole_oop': float(np.degrees(np.arcsin(abs(np.clip(float(np.dot(w, imn)), -1.0, 1.0))))),
        'mg_out_of_plane': float(np.dot(mg - c, nrm)),
        'n4_planarity': float(np.sqrt((((n4 - c) @ nrm) ** 2).mean())),
    }


# ---------------------------------------------------------------- report-only measures
#
# Everything below is REPORTED, never gated. They exist so v12 can decide from data; see the
# v11 plan. All are heavy-atom only: Boltz and AF3 emit no protein hydrogens, and the driver's
# in-loop protonation covers the ligand alone.

LIGAND_POLAR_O = ('OBD1', 'OD1', 'OD2', 'OCD1', 'OA1', 'OA2')
PROPIONATE = ('CGA1', 'OA1', 'OA2')
# Side-chain and backbone donors. Heavy atoms only, so a "donor" here is an N or an -OH oxygen
# that carries a hydrogen in reality; we cannot see the hydrogen to confirm it.
PROTEIN_DONORS = frozenset({'N', 'OG', 'OG1', 'OH', 'ND1', 'ND2', 'NE', 'NE1', 'NE2',
                            'NH1', 'NH2', 'NZ', 'OW'})
HBOND_MAX = 3.2          # heavy-atom donor...acceptor
BURIED_SASA = 5.0        # A^2, per atom, below which we call a ligand oxygen buried
PROP_SASA_MIN = 15.0     # A^2 summed over the propionate before we stop worrying about it


def his_second_shell(his_atoms: dict, protein_xyz, protein_names, protein_resnames,
                     protein_resnums, his_resnum: int) -> dict:
    """Does the axial His ND1 have an acceptor behind it?

    The coordinating nitrogen is NE2; ND1 is the far one, and in a real site it donates its
    proton to a nearby carbonyl, which fixes the imidazole's rotation. Measured in our ordered
    designs: v7 8/8 and v9 8/8 (both residue 81 backbone O), v10 7/8, v8 0/8.

    REPORTED, NOT GATED -- it does not predict orientation. v8 lacks it in all eight designs
    and still has the best coordination geometry of any campaign.

    `his_atoms` needs ND1 and at least one of CG/CE1 (to fix the donor direction).
    Returns NaN distance and False flags when there is no partner, never raises.
    """
    out = {'his_nd1_partner_dist': float('nan'), 'his_nd1_partner_angle': float('nan'),
           'his_nd1_partner': '', 'his_nd1_ok': False, 'his_nd1_strained': False}
    nd1 = his_atoms.get('ND1')
    ref = his_atoms.get('CG', his_atoms.get('CE1'))
    if nd1 is None or ref is None:
        return out
    nd1 = np.asarray(nd1, dtype=float)
    ref = np.asarray(ref, dtype=float)
    pxyz = np.asarray(protein_xyz, dtype=float)
    # acceptors: any protein oxygen NOT in the histidine itself
    cand = [i for i, n in enumerate(protein_names)
            if n and n[0] == 'O' and int(protein_resnums[i]) != int(his_resnum)]
    if not cand:
        return out
    d = np.linalg.norm(pxyz[cand] - nd1, axis=1)
    j = int(d.argmin())
    if d[j] > HBOND_MAX:
        return out
    i = cand[j]
    out['his_nd1_partner_dist'] = float(d[j])
    out['his_nd1_partner_angle'] = _ang(_unit(ref - nd1), _unit(pxyz[i] - nd1))
    out['his_nd1_partner'] = f'{protein_resnames[i]}{protein_resnums[i]}.{protein_names[i]}'
    # 2.6 A floor because real designs sit at 2.47-2.53; a bare "within 3.2" calls those a pass
    out['his_nd1_strained'] = bool(d[j] < 2.6)
    out['his_nd1_ok'] = bool(2.6 <= d[j] <= HBOND_MAX and out['his_nd1_partner_angle'] >= 100.0)
    return out


def _nearest_partner(point, pxyz, pnames, presnames, presnums, donors_only=False):
    """Closest protein donor/acceptor heavy atom to `point` within HBOND_MAX."""
    idx = [i for i, n in enumerate(pnames)
           if n and ((n in PROTEIN_DONORS) if donors_only else (n[0] == 'O' or n in PROTEIN_DONORS))]
    if not idx:
        return float('nan'), ''
    d = np.linalg.norm(np.asarray(pxyz)[idx] - np.asarray(point, dtype=float), axis=1)
    j = int(d.argmin())
    if d[j] > HBOND_MAX:
        return float('nan'), ''
    i = idx[j]
    return float(d[j]), f'{presnames[i]}{presnums[i]}.{pnames[i]}'


def polar_contacts(lig_names, lig_xyz, lig_sasa_complex, lig_sasa_alone,
                   protein_xyz, protein_names, protein_resnames, protein_resnums) -> dict:
    """Per-ligand-oxygen burial and whether anything is there to satisfy it.

    Reports burial BOTH ways on purpose. An earlier analysis used a contact-count proxy and
    concluded OBD1 was buried in 24/24 ordered designs; real per-atom SASA puts it at 22-38
    A^2 -- solvent-facing in every campaign. The two definitions must not be conflated again.
    The cascade's existing `buried_frac*` columns are the RELATIVE measure, over atom groups.

    What the finished campaigns actually show: OD2 is buried (1-8 A^2) in all four and paired
    in none, and OCD1 separates the engines -- v9 buried 8/8 and paired 8/8 (ASN84.ND2), v8
    buried 6/8 and paired 0.
    """
    out, unpaired, props = {}, 0, []
    for k, name in enumerate(lig_names):
        if name in PROPIONATE:
            props.append(float(lig_sasa_complex[k]))
        if name not in LIGAND_POLAR_O:
            continue
        sc = float(lig_sasa_complex[k])
        alone = float(lig_sasa_alone[k]) if lig_sasa_alone is not None else float('nan')
        rel = float(1 - sc / alone) if alone and alone > 1e-9 else float('nan')
        dist, who = _nearest_partner(lig_xyz[k], protein_xyz, protein_names,
                                     protein_resnames, protein_resnums)
        buried = sc < BURIED_SASA
        out[f'sasa_{name}'] = sc
        out[f'relburial_{name}'] = rel
        out[f'partner_{name}'] = who
        out[f'partner_dist_{name}'] = dist
        if buried and not who:
            unpaired += 1
    out['unpaired_buried_count'] = unpaired

    # propionate: charged, deliberately solvent-exposed. A tripwire, not a target.
    prop_sasa = float(sum(props)) if props else float('nan')
    pd_, pwho = float('nan'), ''
    for k, name in enumerate(lig_names):
        if name in ('OA1', 'OA2'):
            d, w = _nearest_partner(lig_xyz[k], protein_xyz, protein_names,
                                    protein_resnames, protein_resnums, donors_only=True)
            if d == d and (pd_ != pd_ or d < pd_):
                pd_, pwho = d, w
    out['prop_sasa'] = prop_sasa
    out['prop_donor_dist'] = pd_
    out['prop_donor'] = pwho
    out['prop_ok'] = bool((prop_sasa == prop_sasa and prop_sasa >= PROP_SASA_MIN)
                          or pd_ == pd_)
    return out


def from_pose(pose, his_index: int, li: int, mg_i: int) -> dict:
    """PyRosetta adapter, for polizzi_cascade.py. Returns {} rather than raising if an atom is
    missing, so a cascade row is never lost to a geometry failure."""
    try:
        lig = pose.residue(li)
        names = {lig.atom_name(i).strip(): i for i in range(1, lig.natoms() + 1)}
        pyr = next((g for g in PYRROLE_NAMES if all(n in names for n in g)), None)
        if pyr is None:
            return {}
        v = lambda x: np.array([x.x, x.y, x.z])
        n4 = [v(lig.xyz(names[n])) for n in pyr]
        his = pose.residue(his_index)
        g = geometry(n4, v(lig.xyz(mg_i)), *(v(his.xyz(a)) for a in ('NE2', 'CE1', 'CD2')))
    except Exception:
        return {}
    g.pop('his_mg', None)      # the cascade measures this itself, under its own constraint
    return g


def from_pdb(path: str) -> dict | None:
    """prody adapter for a reference structure outside a campaign. Picks the histidine whose
    NE2 is nearest the Mg, so it works on crystal structures with many histidines."""
    import prody as pr
    pr.confProDy(verbosity='none')
    ag = pr.parsePDB(str(path))
    if ag is None:
        return None
    lig = ag.select('resname CLA CLAOX CLX CHL BCL LIG') or ag.select('not protein and not water')
    if lig is None:
        return None
    names = set(lig.getNames())
    pyr = next((g for g in PYRROLE_NAMES if all(n in names for n in g)), None)
    mgn = next((n for n in MG_NAMES if n in names), None)
    if pyr is None or mgn is None:
        return None
    n4 = [lig.select(f'name {n}').getCoords()[0] for n in pyr]
    mg = lig.select(f'name {mgn}').getCoords()[0]
    best = None
    for res in ag.select('protein').getHierView().iterResidues():
        ne2 = res.select('name NE2')
        if ne2 is None or res.select('name CE1') is None or res.select('name CD2') is None:
            continue
        d = float(np.linalg.norm(ne2.getCoords()[0] - mg))
        if best is None or d < best[0]:
            best = (d, res)
    if best is None:
        return None
    res = best[1]
    g = geometry(n4, mg, *(res.select(f'name {a}').getCoords()[0] for a in ('NE2', 'CE1', 'CD2')))
    g['his'] = f'{res.getResname()}{res.getResnum()}'
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdb', action='append', required=True, help='repeatable')
    a = ap.parse_args()
    print(f'{"structure":42s}{"His":>6s}{"axial":>8s}{"lonepair":>10s}'
          f'{"off-imid":>10s}{"dome":>8s}{"N4rms":>8s}  His')
    rc = 0
    for p in a.pdb:
        g = from_pdb(p)
        if g is None:
            print(f'{p[-42:]:42s}  could not measure (no Mg-chlorin, or no histidine)')
            rc = 1
            continue
        print(f'{p.split("/")[-1][:42]:42s}{g["his_mg"]:6.2f}{g["axial_angle"]:8.1f}'
              f'{g["his_lonepair_angle"]:10.1f}{g["mg_imidazole_oop"]:10.1f}'
              f'{g["mg_out_of_plane"]:+8.2f}{g["n4_planarity"]:8.3f}  {g["his"]}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
