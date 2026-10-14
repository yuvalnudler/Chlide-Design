"""Engine abstraction. No GPU: nothing here folds anything."""

from __future__ import annotations

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
