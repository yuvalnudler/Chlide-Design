"""Engine abstraction. No GPU: nothing here folds anything."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chlide.config import load_campaign
from chlide.engines import AF3Folder, BoltzFolder, EngineError, Folder, get_folder, other_engine
from chlide.engines.base import LigandSpec, check_metal_restraint

CAMPAIGNS = Path(__file__).resolve().parents[1] / 'campaigns'
V7_CACHE = Path('/home/data/yuvaln/nise_bdfp44_v7/boltz_cache')


def _boltz():
    if not (V7_CACHE.is_dir() and any(V7_CACHE.glob('*.ckpt'))):
        pytest.skip('no Boltz cache with checkpoints available')
    return BoltzFolder(cache=V7_CACHE)


def _lig():
    return LigandSpec(code='CLAOX', user_ccd=Path('input/CLAOX.pkl'), axial_his_resnum=85)


# ------------------------------------------------------------------ registry / protocol

def test_both_engines_satisfy_the_protocol():
    assert isinstance(AF3Folder(), Folder)
    assert isinstance(_boltz(), Folder)


def test_registry_rejects_unknown_engines():
    with pytest.raises(EngineError, match='unknown engine'):
        get_folder('rosettafold')


def test_cross_check_auto_is_the_other_engine():
    assert other_engine('af3') == 'boltz'
    assert other_engine('boltz') == 'af3'


def test_config_cross_check_matches_the_registry():
    """The YAML's resolved cross-check and the engine registry must not drift apart."""
    for name in ('v7-boltz-coo.yaml', 'v8-af3-coo.yaml'):
        cfg = load_campaign(CAMPAIGNS / name)
        assert cfg.engine.resolved_cross_check == other_engine(cfg.engine.fold)


# ------------------------------------------------------------------ declared capabilities

def test_af3_declares_no_restraint_support():
    """Both are measured facts, not opinions -- see docs/ENGINES.md.

    AF3 honours pdbx_stereo_config only ~59% of the time on designed sequences (it never
    converged over v8's 17 iterations), and bondedAtomPairs has no distance window.
    """
    a = AF3Folder()
    assert a.supports_chirality_restraints is False
    assert a.supports_metal_distance_restraint is False


def test_boltz_declares_restraint_support():
    b = _boltz()
    assert b.supports_chirality_restraints is True
    assert b.supports_metal_distance_restraint is True


def test_af3_refuses_a_metal_restraint():
    """The structural version of the bug a self-test caught: af3_json defaulted
    bondedAtomPairs ON after the harm had already been measured."""
    with pytest.raises(EngineError, match='1.50|distance window'):
        check_metal_restraint(AF3Folder(), True)


def test_boltz_accepts_a_metal_restraint():
    check_metal_restraint(_boltz(), True)      # must not raise


def test_af3_write_input_refuses_a_metal_restraint(tmp_path):
    with pytest.raises(EngineError):
        AF3Folder().write_input(tmp_path, 'x', 'MAAA', _lig(), constrain_metal=True)


# ------------------------------------------------------------------ input generation

def test_boltz_input_is_valid_yaml_and_msa_free(tmp_path):
    import yaml
    p = _boltz().write_input(tmp_path, 'chunk_0_seq_0', 'MAAAKL', _lig())
    d = yaml.safe_load(p.read_text())
    assert p.suffix == '.yaml'
    prot = next(s['protein'] for s in d['sequences'] if 'protein' in s)
    assert prot['sequence'] == 'MAAAKL'
    assert prot['msa'] == 'empty', 'campaigns run MSA-free; an MSA changes the objective'
    assert 'constraints' not in d, 'no restraint unless asked for'


def test_boltz_metal_constraint_names_the_axial_his(tmp_path):
    import yaml
    p = _boltz().write_input(tmp_path, 's', 'MAAA', _lig(), constrain_metal=True)
    d = yaml.safe_load(p.read_text())
    bond = d['constraints'][0]['bond']
    assert bond['atom1'] == ['A', 85, 'NE2'] and bond['atom2'] == ['B', 1, 'MG']


def test_boltz_metal_constraint_needs_a_his_number(tmp_path):
    lig = LigandSpec(code='CLAOX', user_ccd=Path('x.pkl'), axial_his_resnum=None)
    with pytest.raises(EngineError, match='axial_his_resnum'):
        _boltz().write_input(tmp_path, 's', 'MAAA', lig, constrain_metal=True)


# ------------------------------------------------------------------ cache resolution

def test_boltz_cache_must_carry_checkpoints(tmp_path):
    """An empty directory is not a cache. v8's boltz_cache has one pickle and zero ckpts."""
    from chlide.engines.boltz import resolve_cache
    (tmp_path / 'boltz_cache').mkdir()
    with pytest.raises(EngineError, match='no Boltz cache'):
        resolve_cache(run=tmp_path, explicit=str(tmp_path / 'boltz_cache'))


def test_explicit_cache_wins():
    from chlide.engines.boltz import resolve_cache
    if not (V7_CACHE.is_dir() and any(V7_CACHE.glob('*.ckpt'))):
        pytest.skip('no Boltz cache available')
    assert resolve_cache(run=Path('/nonexistent'), explicit=str(V7_CACHE)) == V7_CACHE


# ------------------------------------------------------------------ engine-aware artifacts

def _final_shortlist():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        'fs', str(Path(__file__).resolve().parents[1] / 'tools' / 'final_shortlist.py'))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['x']
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return mod


def test_shortlist_readme_names_the_right_engine():
    """v8 shipped a README claiming patched-Boltz coordination while folding with AF3.

    AF3 has no metal restraint at all; Rosetta relax alone closes the bond there
    (4.22 -> 2.50 A). The text was hardcoded. Now it is derived.
    """
    fs = _final_shortlist()
    v7, v8 = (Path(f'/home/data/yuvaln/nise_bdfp44_v{n}') for n in (7, 8))
    if not v8.is_dir():
        pytest.skip('campaigns not present')
    assert fs.leading_engine(v8) == 'AF3'
    assert fs.leading_engine(v7) == 'Boltz'
    assert 'Rosetta relax only' in fs.coordination_source('AF3')
    assert 'patched Boltz' not in fs.coordination_source('AF3')
    assert 'patched Boltz' in fs.coordination_source('Boltz')


def test_engine_detection_reads_the_driver_not_the_layout():
    """af3_adapter.to_boltz_layout writes AF3 output into a boltz_results_* tree on purpose,
    so the directory name is not evidence. Only the driver's import is."""
    src = (Path(__file__).resolve().parents[1] / 'tools' / 'final_shortlist.py').read_text()
    assert "'from af3_adapter import' in drv.read_text()" in src


# ------------------------------------------------------------------ the merged driver

DRIVER = Path(__file__).resolve().parents[1] / 'src' / 'chlide' / 'nise_driver.py'


def test_driver_dispatches_input_format_as_well_as_the_fold_call():
    """Boltz reads .yaml, AF3 reads .json. BOTH sites must be engine-dependent.

    The merge originally swapped only predict_complex_structures, so a Boltz-led run wrote AF3
    JSONs, Boltz found no *.yaml to fold, and the driver died with "not all boltz predictions
    were written to disk" after 4.5 minutes of GPU time. A smoke test found it; this finds it
    in 20 ms.
    """
    src = DRIVER.read_text()
    assert "chunk_{idx}_seq_{seq_idx}.yaml" in src, 'no Boltz input path'
    assert "chunk_{idx}_seq_{seq_idx}.json" in src, 'no AF3 input path'
    # and the choice must be made, not hardcoded
    i_json = src.index("chunk_{idx}_seq_{seq_idx}.json")
    guard = src.rindex('FOLD_ENGINE', 0, i_json)
    assert i_json - guard < 400, 'the .json path is not guarded by a FOLD_ENGINE check'


def test_driver_has_both_fold_implementations_and_a_dispatcher():
    src = DRIVER.read_text()
    for needle in ('def _fold_boltz(', 'def _fold_af3(',
                   'def predict_complex_structures(', "if FOLD_ENGINE == 'af3':",
                   "if FOLD_ENGINE == 'boltz':"):
        assert needle in src, needle


def test_driver_defaults_are_the_safe_ones():
    """A mis-wired run must fold UNRESTRAINED rather than crush His-Mg to 1.50 A and flip 13-2."""
    src = DRIVER.read_text()
    assert "CONSTRAIN_HIS_MG_BOND = os.environ.get('CHLIDE_CONSTRAIN_HIS_MG', '0') == '1'" in src


def test_driver_separates_the_run_directory_from_the_repo_tools():
    """The per-campaign copies conflated these because the driver lived inside the run."""
    src = DRIVER.read_text()
    assert "NISE_RUN_DIR = os.environ.get('CHLIDE_RUN_DIR')" in src
    assert "TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'" in src
    # tools come from the repo
    for tool in ('rosetta_relax.py', 'finish_at_iter.sh', 'boltz_coord_patch.py'):
        assert f"TOOLS_DIR / '{tool}'" in src, f'{tool} should come from the repo'


def test_yaml_boilerplate_carries_affinity_and_the_metal_bond():
    """P(bind) needs `properties: affinity`; the metal bond needs the constraints block."""
    import importlib.util
    import os
    import sys
    os.environ['CHLIDE_CONSTRAIN_HIS_MG'] = '1'
    spec = importlib.util.spec_from_file_location('_drv', str(DRIVER))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['drv']
    try:
        spec.loader.exec_module(mod)
    except Exception:                       # torch/boltz may be absent in some envs
        pytest.skip('driver dependencies unavailable')
    finally:
        sys.argv = argv
        os.environ.pop('CHLIDE_CONSTRAIN_HIS_MG', None)
    y = mod.get_boltz_yaml_boilerplate('MAAA', 'C', True)
    assert 'properties:' in y and 'affinity:' in y and 'binder: B' in y
    assert 'constraints:' in y and 'NE2' in y and 'MG' in y
    assert 'msa: empty' in y
    assert mod.get_boltz_yaml_boilerplate('MAAA', 'C', False).count('affinity') == 0


def test_derived_fixbeta_backbones_are_not_counted_as_inputs(tmp_path):
    """sample_sequences writes `<stem>_fixbeta.pdb` NEXT TO the input backbone, so a cold
    start in a directory a previous attempt used sees two and dies with "More than one input
    backbone not currently supported". It only bites on a cold start, which is why eight
    campaigns never hit it -- until a smoke test restarted iteration 0 twice in five minutes.
    """
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('_drv2', str(DRIVER))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['drv']
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip('driver dependencies unavailable')
    finally:
        sys.argv = argv
    (tmp_path / 'scaffold.pdb').write_text('ATOM\n')
    (tmp_path / 'scaffold_fixbeta.pdb').write_text('ATOM\n')
    (tmp_path / 'notes.txt').write_text('x')
    got = [p.name for p in mod._input_backbones(tmp_path)]
    assert got == ['scaffold.pdb'], got


# ------------------------------------------------------------------ run-dir resolution

V9_SMOKE = Path('/home/data/yuvaln/v9_smoke')


def test_ligand_config_finds_the_component_via_the_run_directory():
    """ligand_config is IMPORTED by the driver, so there is no --run-dir to correct its guess.

    It used to derive the run directory from its own __file__, which worked while tools lived
    INSIDE the campaign. Now they live in the repo, so that guess resolved to the repo root --
    no boltz_cache/, no input/ -- and it returned None for every design. None is a gate
    FAILURE, so every ligand_rmsd became inf; in production that empties the backbone queue at
    iteration 0 and freezes the campaign. The v4 funnel collapse, caused by a path.
    """
    import os
    import sys
    if not (V9_SMOKE / 'boltz_cache' / 'mols' / 'CLAOX.pkl').is_file():
        pytest.skip('no seeded run directory available')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import warnings
    warnings.filterwarnings('ignore')
    import prody as pr
    pr.confProDy(verbosity='none')
    from ligand_chirality import ligand_config

    preds = sorted((V9_SMOKE / 'input' / 'sampled_backbones' / 'iter_0').glob(
        'boltz_results_boltz_inputs/predictions/*/*_model_0.pdb'))
    if not preds:
        pytest.skip('no predictions in the smoke run')

    old = os.environ.get('CHLIDE_RUN_DIR')
    try:
        os.environ['CHLIDE_RUN_DIR'] = str(V9_SMOKE)
        got = [ligand_config(pr.parsePDB(str(p))) for p in preds[:3]]
        assert all(g == 'S' for g in got), got

        # and with the run directory pointing nowhere it must return None -- a failure to
        # verify, never a silent pass
        os.environ['CHLIDE_RUN_DIR'] = '/nonexistent-run-dir'
        assert ligand_config(pr.parsePDB(str(preds[0]))) is None
    finally:
        if old is None:
            os.environ.pop('CHLIDE_RUN_DIR', None)
        else:
            os.environ['CHLIDE_RUN_DIR'] = old


def test_no_tool_silently_guesses_the_run_directory():
    """Every __file__-derived run directory must defer to $CHLIDE_RUN_DIR.

    Ten tools used it only as an argparse default, which the runner always overrides -- latent
    rather than live. ligand_chirality used it at call time with no override, which is what
    broke. Fixing only the live one would leave the same trap for the next tool imported as a
    library.
    """
    import re
    bad = []
    for f in sorted((Path(__file__).resolve().parents[1] / 'tools').glob('*.py')):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if 'Path(__file__).resolve().parent.parent' in line and 'CHLIDE_RUN_DIR' not in line:
                # the fix spans two lines in ligand_chirality; check the previous line too
                bad.append(f'{f.name}:{i}')
    # allow a match whose CHLIDE_RUN_DIR guard is on the preceding line
    real = []
    for loc in bad:
        name, ln = loc.split(':')
        lines = (Path(__file__).resolve().parents[1] / 'tools' / name).read_text().splitlines()
        ctx = ' '.join(lines[max(0, int(ln) - 2):int(ln)])
        if 'CHLIDE_RUN_DIR' not in ctx:
            real.append(loc)
    assert not real, f'tools guessing the run directory without $CHLIDE_RUN_DIR: {real}'


def test_finish_at_iter_detects_the_engine_from_the_manifest(tmp_path):
    """Which model led decides which one cross-checks. Get it wrong and the campaign grades
    itself -- the one thing the fold check exists to prevent.

    The old test grepped "$RUN"/run_nise_*.py, a per-campaign driver script that repo-based
    campaigns do not have, so it fell through to "Boltz" for all of them. v9 happened to be
    Boltz-led so it got the right stage by luck; an AF3-led campaign would have been
    cross-checked with AF3.
    """
    import json
    import subprocess
    sh = Path(__file__).resolve().parents[1] / 'tools' / 'finish_at_iter.sh'
    if not sh.is_file():
        pytest.skip('finish_at_iter.sh not migrated')
    body = sh.read_text()
    start = body.index('detect_engine() {')
    end = body.index('\n}', start) + 2
    fn = f'PY={sys.executable}\n' + body[start:end]

    def detect(d: Path) -> str:
        r = subprocess.run(['bash', '-c', fn + f'\ndetect_engine "{d}"'],
                           capture_output=True, text=True)
        return r.stdout.strip()

    af3 = tmp_path / 'af3led'; af3.mkdir()
    (af3 / 'run_manifest.json').write_text(json.dumps({'engine': {'fold': 'af3'}}))
    assert detect(af3) == 'AF3'

    boltz = tmp_path / 'boltzled'; boltz.mkdir()
    (boltz / 'run_manifest.json').write_text(json.dumps({'engine': {'fold': 'boltz'}}))
    assert detect(boltz) == 'Boltz'

    bare = tmp_path / 'bare'; bare.mkdir()
    assert detect(bare) == 'Boltz', 'no manifest and no driver must fall back, not crash'
