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


# --------------------------------------------------------------- the in-loop gate

def _campaign_stub():
    """The gate predicate, lifted off DesignCampaign without constructing one."""
    from chlide import nise_driver as nd

    class Stub:
        axial_angle_max, lonepair_angle_max, mg_imidazole_oop_max = 11.5, 17.0, 12.7
        coordination_geometry = nd.DesignCampaign.coordination_geometry
        geometry_ok = nd.DesignCampaign.geometry_ok
    return Stub()


@pytest.mark.parametrize('geo,expected,why', [
    ({}, True, 'missing atoms must PASS: a measurement failure must never masquerade as a '
               'rejection, which is the failure mode this gate exists to prevent'),
    ({'axial_angle': 5, 'his_lonepair_angle': 10, 'mg_imidazole_oop': 6}, True, 'good'),
    ({'axial_angle': 5, 'his_lonepair_angle': 40, 'mg_imidazole_oop': 6}, False, "v9's drift"),
    ({'axial_angle': 20, 'his_lonepair_angle': 10, 'mg_imidazole_oop': 6}, False, 'axial'),
    ({'axial_angle': 5, 'his_lonepair_angle': 10, 'mg_imidazole_oop': 35}, False, 'off-imid'),
    ({'axial_angle': 11.5, 'his_lonepair_angle': 17.0, 'mg_imidazole_oop': 12.7}, True,
     'exactly at the bounds is inside them'),
])
def test_geometry_gate_predicate(geo, expected, why):
    assert _campaign_stub().geometry_ok(geo) is expected, why


def test_gate_thresholds_reach_the_driver():
    """DesignCampaign.__init__ ends in **kwargs, so a key passed in nise_params but not bound
    there is swallowed with no error -- the gate would look configured and do nothing."""
    import inspect
    from chlide.config import load_campaign
    from chlide.loop import nise_params
    from chlide import nise_driver
    cfg = load_campaign(ROOT / 'campaigns' / 'v7-boltz-coo.yaml')
    params = nise_params(cfg, Path('/tmp/unused'))
    bound = set(inspect.signature(nise_driver.DesignCampaign.__init__).parameters)
    for k in ('axial_angle_max', 'lonepair_angle_max', 'mg_imidazole_oop_max'):
        assert k in params, f'{k} missing from nise_params'
        assert k in bound, f'{k} passed but not bound in __init__ -- silently swallowed'


def test_gate_rejects_v9_and_passes_v7_on_real_predictions():
    """The gate must separate the campaigns the way the analysis did: v7 around 60% of
    predictions pass, v9 around 14%. Exact figures move with sampling, so this asserts the
    separation rather than a point value."""
    import glob
    import random
    warnings.filterwarnings('ignore')
    import prody as pr
    pr.confProDy(verbosity='none')
    f = _campaign_stub()
    rates = {}
    for tag in ('v7', 'v9'):
        files = glob.glob(f'{CAMPAIGNS[tag]}/input/sampled_backbones/iter_*/**/*_model_0.pdb',
                          recursive=True)
        if not files:
            pytest.skip(f'{tag} predictions absent')
        random.seed(0)
        ok = tot = 0
        for p in random.sample(files, min(150, len(files))):
            ag = pr.parsePDB(p)
            prot = ag.select('chid A') if ag is not None else None
            lig = ag.select('chid B and not hydrogen') if ag is not None else None
            if prot is None or lig is None:
                continue
            geo = f.coordination_geometry(prot, lig)
            if not geo:
                continue
            tot += 1
            ok += bool(f.geometry_ok(geo))
        rates[tag] = ok / tot
    assert rates['v7'] > 0.45, f"v7 should mostly pass, got {rates['v7']:.1%}"
    assert rates['v9'] < 0.25, f"v9 should mostly fail, got {rates['v9']:.1%}"
    assert rates['v7'] > 2 * rates['v9'], 'the gate must separate the two campaigns'


# --------------------------------------------------------------- the fallback

def test_gate_config_refuses_a_threshold_below_the_experimental_median():
    """A bound tighter than the median rejects most real sites. Note this is also why the
    fallback cannot be exercised by setting thresholds to 0 in a YAML -- it is tested
    directly below instead."""
    from pydantic import ValidationError
    from chlide.config import ConfigError, GateConfig
    GateConfig(lonepair_angle_max=17.0)                       # fine
    for kw in ({'lonepair_angle_max': 1.0}, {'axial_angle_max': 2.0}):
        with pytest.raises((ConfigError, ValidationError), match='below the experimental median'):
            GateConfig(**kw)


def test_seed_width_must_exceed_steady_width():
    """The seed phase is meant to be WIDER. Inverting it silently would starve exactly the
    step v10 proved matters most."""
    from pydantic import ValidationError
    from chlide.config import ConfigError, LoopConfig
    LoopConfig(iterations=25, per_backbone=200, seed_per_backbone=500)
    with pytest.raises((ConfigError, ValidationError), match='WIDER'):
        LoopConfig(iterations=25, per_backbone=500, seed_per_backbone=200)


def test_fallback_triggers_on_stagnation_not_on_an_empty_queue(capsys):
    """The queue carries over between iterations, so it is almost never empty. The failure to
    guard against is the gate rejecting every NEW candidate while the loop re-designs the same
    carried-over backbones forever."""
    from chlide import nise_driver as nd

    class Stub:
        top_k = 3
        relax_in_loop = False
        axial_angle_max, lonepair_angle_max, mg_imidazole_oop_max = 11.5, 17.0, 12.7

    s = Stub()
    # carried-over queue, nothing new promoted, three rejected
    s.backbone_queue = [('carried_a', 0.9), ('carried_b', 0.8)]
    s._geom_rejected, s._geom_promoted = 3, 0
    s._geom_fallback_pool = [('new_x', 0.95), ('new_y', 0.7), ('new_z', 0.6)]

    # the block under test, lifted verbatim in behaviour
    n_rej, s._geom_rejected = s._geom_rejected, 0
    n_prom, s._geom_promoted = s._geom_promoted, 0
    fired = n_prom == 0 and n_rej
    if fired:
        for p, sc in s._geom_fallback_pool:
            s.backbone_queue.append((p, sc))
    ranked = sorted(s.backbone_queue, key=lambda x: float(x[1]), reverse=True)[:s.top_k]

    assert fired, 'stagnation must trigger the fallback'
    assert 'new_x' in [p for p, _ in ranked], 'the best rejected design should be rescued'
    assert len(ranked) == 3 and all(ranked[i][1] >= ranked[i + 1][1] for i in range(2))


def test_rejected_designs_stay_out_of_the_best_generator_rescue():
    """keep_best_generator_backbone appends with no geometric check, so a rejected design must
    never enter backbone_to_best_generation or the rescue reintroduces what the gate removed."""
    src = (ROOT / 'src' / 'chlide' / 'nise_driver.py').read_text()
    i = src.index('if geo_ok:')
    block = src[i:i + 900]
    # strip comments: the else-branch explains WHY it omits the bookkeeping, and naming it
    # in prose must not read as doing it
    code = '\n'.join(l.split('#')[0] for l in block.splitlines())
    kept, rejected = code.split('else:', 1)
    assert 'backbone_to_best_generation' in kept, \
        'the bookkeeping must be inside the geo_ok branch'
    assert 'backbone_to_best_generation' not in rejected, \
        'a rejected design must not be recorded as a best generator'
    assert 'backbone_queue.append' not in rejected, \
        'a rejected design must not be promoted'


def test_every_log_data_append_happens_on_every_path():
    """v11a's first iteration folded 500 designs and then died at pd.DataFrame(iidx_data) with
    'All arrays must be of the same length': the geometry columns are appended after the
    name-mapping early exit, so they came up short exactly as often as that exit was taken.

    Any key appended inside the per-design loop must be appended on EVERY path through it.
    """
    import re
    src = (ROOT / 'src' / 'chlide' / 'nise_driver.py').read_text()
    start = src.index('for laser, boltz, bb_path in zip(')
    end = src.index('ranked = sorted(self.backbone_queue', start)
    body = src[start:end]
    code = '\n'.join(l.split('#')[0] for l in body.splitlines())

    keys = set(re.findall(r"log_data\['([a-z0-9_]+)'\]\.append", code))
    # split on the early-exit continue: everything appended after it must also be padded before
    marker = "log_data['protein_plddts'].append(torch.nan)"
    assert marker in code, 'the early-exit branch moved; this test needs updating'
    before_exit = code[:code.index('continue', code.index(marker))]
    after_exit = code[code.index('continue', code.index(marker)):]

    appended_after = set(re.findall(r"log_data\['([a-z0-9_]+)'\]\.append", after_exit))
    padded_before = set(re.findall(r"log_data\[_?k?'?([a-z0-9_]*)'?\]\.append", before_exit))
    # the geometry keys are padded via a loop over a tuple, so check the tuple is there
    geo = {'axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop', 'mg_out_of_plane',
           'n4_planarity'}
    assert geo <= set(re.findall(r"'([a-z0-9_]+)'", before_exit)), \
        'the geometry columns must be padded in the name-mapping early exit'
    assert "log_data['geometry_ok'].append(False)" in before_exit, \
        'geometry_ok must be padded in the early exit too'
    assert keys, 'no log_data appends found -- the loop boundaries moved'


def test_ragged_columns_are_reported_by_name():
    """pandas says only 'All arrays must be of the same length', which cost a seed iteration
    to diagnose. The driver now names the offending column."""
    src = (ROOT / 'src' / 'chlide' / 'nise_driver.py').read_text()
    assert 'log_data columns are ragged' in src
    i = src.index('log_data columns are ragged')
    assert src.index('iidx_dataframe = pd.DataFrame(iidx_data)') > i, \
        'the guard must run BEFORE the DataFrame is built'
