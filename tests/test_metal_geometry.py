"""The coordination and polar-contact measures, pinned to the finished campaigns.

These are REPORT-ONLY measures (see the v11 plan): they must never change a gate outcome.
The reference values below were measured on the ordered designs of v7-v10 and are the whole
point of the file -- if the code stops reproducing them it is no longer measuring what the
analysis measured.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = {t: Path(f'/home/data/yuvaln/nise_bdfp44_{t}') for t in ('v7', 'v8', 'v9', 'v10')}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / 'tools' / f'{name}.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _measure(path: Path) -> dict:
    warnings.filterwarnings('ignore')
    import prody as pr
    pr.confProDy(verbosity='none')
    mg, sr = _load('metal_geometry'), _load('sasa_rank')
    names, els, xyz, islig = sr.parse(path)
    r = np.array([sr.RADII.get(e, 1.70) for e in els])
    lx, lr, ln = xyz[islig], r[islig], names[islig]
    inc, alone = sr.sasa(lx, lr, xyz, r), sr.sasa(lx, lr, lx, lr)
    ag = pr.parsePDB(str(path))
    prot = ag.select('protein and not hydrogen')
    mgx = lx[[i for i, n in enumerate(ln) if n in ('MG1', 'MG')][0]]
    ne2 = ag.select('protein and name NE2')
    j = int(np.linalg.norm(ne2.getCoords() - mgx, axis=1).argmin())
    hrn = int(ne2.getResnums()[j])
    h = ag.select(f'chain {ne2.getChids()[j]} and resnum {hrn}')
    hat = dict(zip(h.getNames(), h.getCoords()))
    out = mg.his_second_shell(hat, prot.getCoords(), prot.getNames(),
                              prot.getResnames(), prot.getResnums(), hrn)
    out.update(mg.polar_contacts(ln, lx, inc, alone, prot.getCoords(), prot.getNames(),
                                 prot.getResnames(), prot.getResnums()))
    return out


def _order_set(tag: str) -> list[Path]:
    d = CAMPAIGNS[tag] / 'results' / 'final' / 'ORDER'
    return sorted(d.glob('order*.pdb')) if d.is_dir() else []


@pytest.fixture(scope='module')
def measured():
    out = {}
    for tag in CAMPAIGNS:
        pdbs = _order_set(tag)
        if pdbs:
            out[tag] = [_measure(p) for p in pdbs]
    if not out:
        pytest.skip('no campaign archives present')
    return out


# --------------------------------------------------------------- His second shell

@pytest.mark.parametrize('tag,expected', [('v7', 5), ('v8', 0), ('v9', 8), ('v10', 3)])
def test_his_second_shell_strict_counts(measured, tag, expected):
    """2.6-3.2 A AND angle >= 100. v8 has no partner at all; v7 and v10 lose designs to the
    2.6 A floor, which is exactly why that floor exists."""
    if tag not in measured:
        pytest.skip(f'{tag} absent')
    assert sum(r['his_nd1_ok'] for r in measured[tag]) == expected


def test_strict_failures_are_strain_not_angle(measured):
    """Every failure in v7/v9/v10 is distance < 2.6 A; the angles are all 113-136 deg. If an
    angle failure ever appears, the geometry has changed and the threshold needs re-deriving."""
    for tag in ('v7', 'v9', 'v10'):
        if tag not in measured:
            continue
        for r in measured[tag]:
            a = r['his_nd1_partner_angle']
            if a == a:                                  # a partner was found
                assert a >= 100.0, f'{tag}: angle {a:.1f} below the floor'


def test_the_partner_is_residue_81(measured):
    """Across every campaign that has one, the acceptor is residue 81's backbone O."""
    for tag, rows in measured.items():
        for r in rows:
            if r['his_nd1_partner']:
                assert r['his_nd1_partner'].endswith('81.O'), r['his_nd1_partner']


# --------------------------------------------------------------- polar contacts

def test_obd1_is_not_buried(measured):
    """The correction that prompted these measures: an earlier contact-count proxy called
    OBD1 buried in 24/24 ordered designs. Real per-atom SASA puts it at 22-38 A^2."""
    for tag, rows in measured.items():
        v = [r['sasa_OBD1'] for r in rows]
        assert min(v) > 15.0, f'{tag}: OBD1 min SASA {min(v):.1f}'


def test_od2_is_buried_and_never_paired(measured):
    """The one universal gap: buried in all four campaigns, satisfied in none."""
    for tag, rows in measured.items():
        assert max(r['sasa_OD2'] for r in rows) < 10.0
        assert not any(r['partner_OD2'] for r in rows), f'{tag} paired OD2'


def test_ocd1_separates_v8_from_v9(measured):
    """v9 buries the 13-2 hydroxyl and satisfies it with Asn84; v8 buries it and does not."""
    if 'v9' in measured:
        assert all(r['sasa_OCD1'] < 1.0 for r in measured['v9'])
        assert all('84' in r['partner_OCD1'] for r in measured['v9'])
    if 'v8' in measured:
        assert all(r['sasa_OCD1'] <= 7.0 for r in measured['v8'])
        assert not any(r['partner_OCD1'] for r in measured['v8'])


def test_propionate_stays_solvent_exposed(measured):
    """Charged and deliberately exposed. A tripwire: if this starts failing, the loop has
    begun burying it."""
    for tag, rows in measured.items():
        assert all(r['prop_ok'] for r in rows), tag


def test_measures_never_raise_on_missing_atoms():
    """They run inside the loop on every design, so a missing atom must give NaN, not an
    exception that kills an iteration."""
    mg = _load('metal_geometry')
    assert mg.his_second_shell({}, np.zeros((0, 3)), [], [], [], 85)['his_nd1_ok'] is False
    out = mg.polar_contacts([], np.zeros((0, 3)), [], [], np.zeros((0, 3)), [], [], [])
    assert out['unpaired_buried_count'] == 0
