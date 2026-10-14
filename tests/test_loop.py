"""Config -> NISE params, checked against the drivers that really ran the campaigns."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from chlide.config import ConfigError, load_campaign
from chlide.loop import LASER_SAMPLING_PARAMS, nise_params

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / 'campaigns'
DRIVERS = {
    'v7-boltz-coo.yaml': Path('/home/data/yuvaln/nise_bdfp44_v7/run_nise_bdfp44_v7.py'),
    'v8-af3-coo.yaml': Path('/home/data/yuvaln/nise_bdfp44_v8/run_nise_bdfp44_v8.py'),
}

# Params that are DERIVED from the run directory or the machine, so they cannot match a
# driver literal. Each is checked separately below.
DERIVED = {'input_dir', 'model_checkpoint', 'boltz2x_executable_path', 'relax_params',
           'relax_python', 'laser_inference_device', 'debug', 'use_wandb',
           'laser_sampling_params'}


def _driver_params(path: Path) -> dict:
    """Read `params = dict(...)` without importing the driver (it pulls in torch and boltz)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'params'
                and isinstance(node.value, ast.Call)):
            out = {}
            for kw in node.value.keywords:
                out[kw.arg] = _const(kw.value)
            return out
    raise AssertionError(f'no params dict in {path}')


_UNKNOWN = object()
_FLAGS = {'SMOKE_TEST': False, 'debug': False}


def _const(node):
    if isinstance(node, ast.NamedExpr):
        return _const(node.value)
    if isinstance(node, ast.IfExp):
        try:
            return _const(node.body if _truth(node.test) else node.orelse)
        except Exception:
            return _UNKNOWN
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _UNKNOWN


def _truth(node):
    if isinstance(node, ast.Name) and node.id in _FLAGS:
        return _FLAGS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _truth(node.operand)
    if isinstance(node, ast.Constant):
        return bool(node.value)
    raise ValueError(ast.dump(node))


@pytest.mark.parametrize('campaign', sorted(DRIVERS), ids=sorted(DRIVERS))
def test_generated_params_match_the_driver(campaign, tmp_path):
    """Every literal parameter in the driver must come back out of the config.

    tools/config_equivalence.py checks the same mapping in the other direction. Both exist
    because the drivers are the only record of what actually produced v7 and v8.
    """
    drv = DRIVERS[campaign]
    if not drv.is_file():
        pytest.skip(f'{drv} not present')
    theirs = _driver_params(drv)
    ours = nise_params(load_campaign(CAMPAIGNS / campaign), tmp_path)

    missing = [k for k in theirs if k not in ours]
    assert not missing, f'params the driver sets and we do not: {missing}'

    bad = []
    for k, want in theirs.items():
        if k in DERIVED or want is _UNKNOWN:
            continue
        got = ours[k]
        if isinstance(want, (set, list, tuple)) and isinstance(got, (set, list, tuple)):
            if set(map(str, want)) != set(map(str, got)):
                bad.append(f'{k}: driver={want!r} ours={got!r}')
        elif got != want:
            bad.append(f'{k}: driver={want!r} ours={got!r}')
    assert not bad, 'generated params disagree with the driver:\n  ' + '\n  '.join(bad)


@pytest.mark.parametrize('campaign', sorted(DRIVERS), ids=sorted(DRIVERS))
def test_laser_sampling_params_match(campaign):
    drv = DRIVERS[campaign]
    if not drv.is_file():
        pytest.skip('driver not present')
    tree = ast.parse(drv.read_text())
    theirs = next(
        ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and getattr(n.targets[0], 'id', None) == 'laser_sampling_params')
    assert theirs == LASER_SAMPLING_PARAMS


# ------------------------------------------------------------------ derived values

def test_run_dir_is_an_argument_not_config(tmp_path):
    """A copied campaign writing into its predecessor's tree is the bug this prevents."""
    cfg = load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml')
    a = nise_params(cfg, tmp_path / 'runA')
    b = nise_params(cfg, tmp_path / 'runB')
    assert a['input_dir'] != b['input_dir']
    assert a['input_dir'] == (tmp_path / 'runA').resolve() / 'input'
    assert a['relax_params'].startswith(str((tmp_path / 'runA').resolve()))


def test_boltz_executable_is_the_patched_one(tmp_path):
    """Unpatched Boltz collapses the His-Mg bond to ~1.52 A."""
    p = nise_params(load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml'), tmp_path)
    assert 'boltz_coord_patch.py' in p['boltz2x_executable_path']
    assert (ROOT / 'tools' / 'boltz_coord_patch.py').is_file()


def test_laser_device_is_the_first_configured_gpu(tmp_path):
    p = nise_params(load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml'), tmp_path)
    assert p['laser_inference_device'] == p['boltz_inference_devices'][0]


# ------------------------------------------------------------------ metal_r0

def test_metal_r0_is_the_physical_bond_not_the_window_midpoint(tmp_path):
    """2.15 A is the Mg-N(His) distance Rosetta restrains to; the 2.0-2.6 window is what the
    result is ACCEPTED against. Deriving one from the other gives 2.30 and moves the restraint
    0.15 A off the real bond."""
    cfg = load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml')
    assert cfg.gates.metal_r0 == 2.15
    assert nise_params(cfg, tmp_path)['relax_r0'] == 2.15
    midpoint = sum(cfg.gates.coordination_window) / 2
    assert cfg.gates.metal_r0 != midpoint


def test_metal_r0_must_lie_inside_the_window():
    """Otherwise the restraint pulls every design to a distance the gate then rejects."""
    import copy

    import yaml
    raw = yaml.safe_load((CAMPAIGNS / '_base.yaml').read_text())
    raw.update(yaml.safe_load((CAMPAIGNS / 'v8-af3-coo.yaml').read_text()))
    raw = copy.deepcopy(raw)
    raw['gates']['metal_r0'] = 3.0
    from chlide.config import CampaignConfig
    with pytest.raises((ConfigError, Exception), match='outside coordination_window'):
        CampaignConfig.model_validate(raw)


# ------------------------------------------------------------------ constants stay constant

def test_chirality_rmsd_stays_off(tmp_path):
    """useChirality=True empties the backbone queue at iteration 0 and freezes the campaign."""
    for c in DRIVERS:
        assert nise_params(load_campaign(CAMPAIGNS / c), tmp_path)['rmsd_use_chirality'] is False


def test_wandb_is_never_enabled(tmp_path):
    """Upstream defaults it True and then fails on a hardcoded entity."""
    for c in DRIVERS:
        assert nise_params(load_campaign(CAMPAIGNS / c), tmp_path)['use_wandb'] is False


def test_reduce_is_never_invoked(tmp_path):
    p = nise_params(load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml'), tmp_path)
    assert p['use_reduce_protonation'] is False
    assert p['reduce_executable_path'] is None


def test_affinity_prediction_follows_the_objective(tmp_path):
    """`boltz2_predict_affinity` is a ternary on the objective in the driver, so the AST reader
    cannot evaluate it. Assert the rule directly instead of leaving it uncovered."""
    cfg = load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml')
    assert cfg.loop.objective == 'ligand_plddt'
    assert nise_params(cfg, tmp_path)['boltz2_predict_affinity'] is False

    import copy
    raw = copy.deepcopy(cfg.model_dump())
    raw['loop']['objective'] = 'ligand_plddt_and_pbind'
    from chlide.config import CampaignConfig
    assert nise_params(CampaignConfig.model_validate(raw), tmp_path)['boltz2_predict_affinity']


# ------------------------------------------------------------------ declared inputs exist

@pytest.mark.parametrize('campaign', sorted(DRIVERS), ids=sorted(DRIVERS))
def test_declared_inputs_resolve_for_the_real_campaign(campaign):
    """A campaign file that names a file nobody has is not a reproduction of anything.

    This caught two: the scaffold is under input/input_backbones/, not input/, and the ligand
    mmCIF is a SHARED asset outside every run directory.
    """
    from chlide.config import resolve_input
    run = DRIVERS[campaign].parent
    if not run.is_dir():
        pytest.skip('campaign not present')
    cfg = load_campaign(CAMPAIGNS / campaign)
    assert resolve_input(run, cfg.scaffold.pdb) is not None, f'scaffold {cfg.scaffold.pdb}'
    assert resolve_input(
        run, cfg.ligand.source, 'CHLIDES_MMCIF',
        '/home/data/yuvaln/main_pipeline/input/chlides_fixed.mmcif') is not None


def test_resolver_prefers_the_run_directory(tmp_path):
    """A shared fallback must never shadow a campaign's own copy."""
    from chlide.config import resolve_input
    (tmp_path / 'input').mkdir()
    local = tmp_path / 'input' / 'x.cif'
    local.write_text('local')
    shared = tmp_path / 'shared.cif'
    shared.write_text('shared')
    assert resolve_input(tmp_path, Path('input/x.cif'), None, str(shared)) == local


def test_resolver_returns_none_rather_than_a_path_that_does_not_exist(tmp_path):
    from chlide.config import resolve_input
    assert resolve_input(tmp_path, Path('input/missing.cif'), None, '/nope') is None
