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
    cfg = load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml')
    assert cfg.loop.objective == 'ligand_plddt'
    assert nise_params(cfg, tmp_path)['boltz2_predict_affinity'] is False

    import copy
    raw = copy.deepcopy(cfg.model_dump())
    raw['loop']['objective'] = 'ligand_plddt_and_pbind'   # boltz-led, so this is allowed
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


# ------------------------------------------------------------------ v9 / P(bind)

def test_v9_differs_from_v7_only_in_the_objective(tmp_path):
    """v9 is the single-variable P(bind) test: v7 with the objective changed and nothing else.

    Running past the plateau is a SEPARATE experiment (v9b), so the two never confound.
    """
    v7 = load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml')
    v9 = load_campaign(CAMPAIGNS / 'v9-boltz-pbind.yaml')
    for field in ('ligand', 'scaffold', 'gates', 'stages', 'compute'):
        assert getattr(v7, field) == getattr(v9, field), f'{field} drifted between v7 and v9'
    assert v7.engine == v9.engine, 'v9 must use the same engine configuration as v7'
    assert v7.loop.objective == 'ligand_plddt'
    assert v9.loop.objective == 'ligand_plddt_and_pbind'
    # everything else about the loop is identical, including the auto-stop
    assert (v7.loop.iterations, v7.loop.per_backbone, v7.loop.top_backbones) == \
           (v9.loop.iterations, v9.loop.per_backbone, v9.loop.top_backbones)
    assert v7.loop.convergence == v9.loop.convergence


def test_pbind_objective_turns_boltz_affinity_on(tmp_path):
    """Setting the objective is the WHOLE change: it flips boltz2_predict_affinity, which makes
    the driver emit `properties: [affinity: {binder: B}]` into every Boltz YAML."""
    v9 = nise_params(load_campaign(CAMPAIGNS / 'v9-boltz-pbind.yaml'), tmp_path)
    v7 = nise_params(load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml'), tmp_path)
    assert v9['boltz2_predict_affinity'] is True
    assert v7['boltz2_predict_affinity'] is False
    diff = {k for k in v9 if v9[k] != v7[k]}
    assert diff == {'objective_function', 'boltz2_predict_affinity'}, (
        f'v9 should differ from v7 in exactly those two params, got {diff}')


def test_pbind_requires_boltz():
    """P(bind) is a Boltz-2 head. An AF3-led campaign asking for it would silently get nothing."""
    import copy

    from chlide.config import CampaignConfig
    raw = copy.deepcopy(load_campaign(CAMPAIGNS / 'v9-boltz-pbind.yaml').model_dump())
    raw['engine']['fold'] = 'af3'
    raw['engine']['constrain_his_mg'] = False
    with pytest.raises(Exception, match='Boltz-2 head|contribute nothing'):
        CampaignConfig.model_validate(raw)


def test_debug_is_actually_small(tmp_path):
    """--debug must be a SMOKE TEST. The first version left iterations at 25 and per_backbone
    at 500, so `--debug` would have launched a full production run: GPU-days spent proving the
    wiring works."""
    cfg = load_campaign(CAMPAIGNS / 'v9-boltz-pbind.yaml')
    prod = nise_params(cfg, tmp_path, debug=False)
    dbg = nise_params(cfg, tmp_path, debug=True)
    assert prod['num_iterations'] == cfg.loop.iterations
    assert prod['sequences_sampled_per_backbone'] == 500
    assert dbg['num_iterations'] == 2
    assert dbg['sequences_sampled_per_backbone'] == len(cfg.engine.devices)
    designs = (dbg['num_iterations'] * dbg['sequences_sampled_per_backbone']
               * dbg['num_top_backbones_per_round'])
    assert designs <= 50, f'a smoke test that folds {designs} designs is not a smoke test'
    assert dbg['debug'] is True


def test_continuation_runs_past_the_plateau():
    """The published protocol: 35 iterations, and do NOT stop at the plateau.

    "performing NISE iterations beyond the plateau in ligand pLDDT or score was important for
    downstream filtering... both EPIC and APEX came from later iterations" (Nature 656, 237).
    """
    v9b = load_campaign(CAMPAIGNS / 'v9-continue-35.yaml')
    assert v9b.loop.iterations == 35
    assert v9b.loop.convergence.enabled is False
    assert v9b.loop.per_backbone == 500     # paper: 1,500/iteration, 500 for the first round
    assert v9b.loop.per_iteration == 1500


def test_continuation_differs_from_v9_only_in_the_length(tmp_path):
    """The second single-variable test. v7->v9 isolates P(bind); v9->v9b isolates length.

    If anything else drifts, neither comparison means what it claims.
    """
    v9 = load_campaign(CAMPAIGNS / 'v9-boltz-pbind.yaml')
    v9b = load_campaign(CAMPAIGNS / 'v9-continue-35.yaml')
    for field in ('ligand', 'scaffold', 'gates', 'stages', 'compute', 'engine'):
        assert getattr(v9, field) == getattr(v9b, field), f'{field} drifted'
    assert v9.loop.objective == v9b.loop.objective == 'ligand_plddt_and_pbind'
    assert (v9.loop.per_backbone, v9.loop.top_backbones) == \
           (v9b.loop.per_backbone, v9b.loop.top_backbones)
    assert (v9.loop.iterations, v9b.loop.iterations) == (25, 35)
    # it RESUMES v9's directory rather than starting fresh -- the driver sets
    # start_iteration = last_complete + 1, so a new run dir would redo everything.
    assert v9.loop.convergence.enabled and not v9b.loop.convergence.enabled
    # in the resolved params the only difference is the budget; the auto-stop travels by env
    diff = {k for k in nise_params(v9, tmp_path)
            if nise_params(v9, tmp_path)[k] != nise_params(v9b, tmp_path)[k]}
    assert diff == {'num_iterations'}, diff


def test_autostop_flag_reaches_the_driver():
    src = (ROOT / 'src' / 'chlide' / 'loop.py').read_text()
    assert "CHLIDE_CONVERGENCE_AUTOSTOP" in src
    drv = (ROOT / 'src' / 'chlide' / 'nise_driver.py').read_text()
    assert "CONVERGENCE_AUTOSTOP = os.environ.get('CHLIDE_CONVERGENCE_AUTOSTOP', '1') == '1'" in drv
    # the metric must still be computed when the halt is suppressed
    assert 'return False, cs' in drv


# ------------------------------------------------------------------ relax pool watcher

def test_relax_watcher_script_exists_and_refuses_non_campaign_dirs(tmp_path):
    """The pool must be relaxed DURING the run. v8 did it that way and caught up in six
    minutes; v9 had no watcher and harvested 12,972 unrelaxed designs in one block.

    The repo has an input/ directory of its own, so an early version of the guard accepted
    the REPO as a run directory and started relaxing against it. The marker has to be
    something only a real campaign has.
    """
    import subprocess
    sh = Path(__file__).resolve().parents[1] / 'tools' / 'relax_watcher.sh'
    assert sh.is_file(), 'relax_watcher.sh was never migrated; that is the v9 regression'

    def run(*args):
        return subprocess.run(['bash', str(sh), *args], capture_output=True, text=True,
                              timeout=30)

    assert run().returncode != 0
    repo = sh.parent.parent
    r = run('--run-dir', str(repo))
    assert r.returncode != 0, 'the repo is not a campaign, even though it has input/'
    assert 'run_manifest' in r.stderr
    assert run('--run-dir', str(tmp_path / 'nope')).returncode != 0

    # ...but a REAL run directory must be accepted the moment it is created, before any
    # iteration has finished. The first guard tested input/sampling_dataframes/, which only
    # appears after iteration 0, so it refused at exactly the point the launcher starts it --
    # v10's watcher died on launch. run_manifest.json is written before the driver starts.
    fresh = tmp_path / 'fresh'
    (fresh / 'input').mkdir(parents=True)
    (fresh / 'run_manifest.json').write_text('{}')
    r = run('--run-dir', str(fresh), '--parent-pid', '1', '--interval', '1')
    assert 'not a campaign run directory' not in r.stderr, \
        'a freshly created run directory must be accepted'


def test_relax_watcher_exits_instead_of_spinning_when_relax_pool_keeps_failing(tmp_path):
    """The drain path skips its sleep on purpose, so a relax_pool that fails every time would
    busy-loop a core. Bounded at five consecutive failures."""
    import subprocess
    sh = Path(__file__).resolve().parents[1] / 'tools' / 'relax_watcher.sh'
    if not sh.is_file():
        pytest.skip('relax_watcher.sh not migrated')
    run_dir = tmp_path / 'camp'
    (run_dir / 'input' / 'sampling_dataframes').mkdir(parents=True)
    r = subprocess.run(
        ['bash', str(sh), '--run-dir', str(run_dir), '--parent-pid', '1', '--interval', '1'],
        capture_output=True, text=True, timeout=90)
    assert r.returncode != 0
    assert 'exiting rather than spinning' in r.stdout


def test_watcher_is_configurable_and_defaults_on():
    """Default on: the work happens either way, only the timing differs."""
    from chlide.config import load_campaign
    root = Path(__file__).resolve().parents[1]
    cfg = load_campaign(root / 'campaigns' / 'v9-boltz-pbind.yaml')
    assert cfg.compute.relax_pool_watcher is True
    assert cfg.compute.relax_pool_interval >= 30
    assert cfg.compute.relax_pool_workers >= 1


def test_disabled_watcher_starts_nothing():
    from types import SimpleNamespace
    from chlide.loop import _start_relax_watcher
    events = []
    log = SimpleNamespace(emit=lambda **kw: events.append(kw))
    cfg = SimpleNamespace(compute=SimpleNamespace(relax_pool_watcher=False))
    assert _start_relax_watcher(cfg, Path('/tmp'), log) is None
    assert events and events[0]['event'] == 'relax_watcher_disabled'
