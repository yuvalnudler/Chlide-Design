"""Config validation.

Each test names the incident it guards. These are not hypotheticals -- every one
of them cost this project campaign time at least once. See docs/LESSONS.md.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from chlide.config import CampaignConfig, ConfigError, load_campaign

CAMPAIGNS = Path(__file__).resolve().parents[1] / 'campaigns'


def _raw(name: str = 'v8-af3-coo.yaml') -> dict:
    """The fully merged dict for a shipped campaign, ready to mutate."""
    base = yaml.safe_load((CAMPAIGNS / '_base.yaml').read_text())
    over = yaml.safe_load((CAMPAIGNS / name).read_text())

    def merge(b, o):
        out = dict(b)
        for k, v in o.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return merge(base, over)


def _build(**mutations) -> CampaignConfig:
    raw = copy.deepcopy(_raw())
    for dotted, value in mutations.items():
        node = raw
        *path, leaf = dotted.split('__')
        for k in path:
            node = node[k]
        node[leaf] = value
    return CampaignConfig.model_validate(raw)


# --------------------------------------------------------------- the shipped files


@pytest.mark.parametrize('name', ['v7-boltz-coo.yaml', 'v8-af3-coo.yaml'])
def test_shipped_campaigns_parse(name):
    cfg = load_campaign(CAMPAIGNS / name)
    assert cfg.name.startswith('v')
    assert cfg.summary()


def test_v7_and_v8_differ_only_in_engine():
    """The whole point of the repo: an engine swap is config, not a forked driver."""
    v7 = load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml')
    v8 = load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml')
    for field in ('ligand', 'scaffold', 'loop', 'gates', 'stages', 'compute'):
        assert getattr(v7, field) == getattr(v8, field), f'{field} drifted between v7 and v8'
    assert (v7.engine.fold, v8.engine.fold) == ('boltz', 'af3')
    assert v7.engine.constrain_his_mg and not v8.engine.constrain_his_mg


def test_cross_check_auto_resolves_to_the_other_engine():
    assert load_campaign(CAMPAIGNS / 'v8-af3-coo.yaml').engine.resolved_cross_check == 'boltz'
    assert load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml').engine.resolved_cross_check == 'af3'


# --------------------------------------------------------------- ligand protonation


def test_deprotonated_requires_anionic_smiles():
    """INCIDENT (v7): component and pose were deprotonated, the SMILES was not.

    The SMILES builds the Rosetta params via AddHs, so a neutral C(=O)O put an acid
    proton into every relaxed structure. Caught only when the user saw it rendered.
    """
    neutral = _raw()['ligand']['smiles'].replace('C(=O)[O-]', 'C(=O)O')
    with pytest.raises((ConfigError, ValidationError), match='(?i)deprotonated|\\[O-\\]'):
        _build(ligand__smiles=neutral)


def test_neutral_rejects_anionic_smiles():
    with pytest.raises((ConfigError, ValidationError)):
        _build(ligand__deprotonated=False)


def test_bonded_magnesium_is_rejected():
    """INCIDENT: ETKDGv3 cannot embed a bonded Mg; Boltz fails 'Failed to compute 3D conformer'."""
    bonded = _raw()['ligand']['smiles'].replace('.[Mg+2]', '') + ''
    bonded = bonded.replace('[n-]2)', '[n-]2[Mg+2](-N)')
    with pytest.raises((ConfigError, ValidationError), match='(?i)magnesium|disconnected'):
        _build(ligand__smiles=bonded)


def test_missing_magnesium_is_rejected():
    # carries [O-] so the deprotonation rule is satisfied and the Mg rule is what fires
    with pytest.raises((ConfigError, ValidationError), match='(?i)magnesium'):
        _build(ligand__smiles='CCC(=O)[O-]')


def test_atom_cannot_be_both_buried_and_exposed():
    with pytest.raises((ConfigError, ValidationError), match='(?i)buried and exposed'):
        _build(ligand__atoms_enforce_exposed=['OA1', 'OA2', 'MG1'])


# --------------------------------------------------------------- residue addressing


def test_axial_his_is_zero_based_resindex():
    """INCIDENT: resnum-based selection moves when Boltz renumbers the output 1-N."""
    cfg = _build()
    assert cfg.scaffold.axial_his_resindex == 84          # HIS85, 0-based
    assert cfg.scaffold.fixed_identity_selection == 'resindex 84'
    assert 'resnum' not in cfg.scaffold.fixed_identity_selection


def test_negative_resindex_is_rejected():
    with pytest.raises((ConfigError, ValidationError)):
        _build(scaffold__axial_his_resindex=-1)


def test_resnum_key_is_rejected_outright():
    """extra='forbid': a config that says `resnum` must not parse and quietly ignore it."""
    raw = copy.deepcopy(_raw())
    raw['scaffold']['axial_his_resnum'] = 85
    with pytest.raises(ValidationError):
        CampaignConfig.model_validate(raw)


# --------------------------------------------------------------- engine coherence


def test_af3_cannot_constrain_his_mg():
    """INCIDENT: af3_json defaulted bondedAtomPairs ON after the harm was measured.

    AF3 has no distance window: it crushes His-Mg to 1.50 A and flips 13-2 to R.
    """
    with pytest.raises((ConfigError, ValidationError), match='(?i)constrain_his_mg|1.50'):
        _build(engine__constrain_his_mg=True)


def test_boltz_may_constrain_his_mg():
    cfg = load_campaign(CAMPAIGNS / 'v7-boltz-coo.yaml')
    assert cfg.engine.constrain_his_mg


def test_duplicate_gpus_are_rejected():
    with pytest.raises((ConfigError, ValidationError), match='(?i)duplicate'):
        _build(engine__devices=['cuda:0', 'cuda:0', 'cuda:1'])


def test_malformed_device_is_rejected():
    with pytest.raises((ConfigError, ValidationError), match='(?i)cuda'):
        _build(engine__devices=['gpu0'])


def test_empty_device_list_is_rejected():
    with pytest.raises((ConfigError, ValidationError)):
        _build(engine__devices=[])


def test_crossfold_against_self_is_rejected():
    """A cross-check against the engine that folded measures nothing."""
    with pytest.raises((ConfigError, ValidationError), match='(?i)same engine|measures nothing'):
        _build(engine__cross_check='af3')


# --------------------------------------------------------------- paths


def test_absolute_paths_are_rejected():
    """INCIDENT (five times): a copied campaign kept its predecessor's absolute paths.

    One driver's input_dir still pointed at v5's input tree and would have written
    new iterations into a closed archive.
    """
    with pytest.raises((ConfigError, ValidationError), match='(?i)absolute'):
        _build(scaffold__pdb='/home/data/yuvaln/nise_bdfp44_v5/input/pose.pdb')


def test_paths_may_not_escape_the_run_directory():
    with pytest.raises((ConfigError, ValidationError), match='(?i)escape'):
        _build(ligand__source='../nise_bdfp44_v5/input/chlides_fixed.mmcif')


# --------------------------------------------------------------- loop and gates


def test_chirality_rmsd_defaults_off():
    """INCIDENT (v4/v6b): useChirality=True empties the backbone queue at iteration 0.

    RDKit perceives MG1 and CA2 as centres, so BOTH populations return zero matches,
    every ligand_rmsd becomes inf and the funnel collapses.
    """
    assert _build().gates.rmsd_use_chirality is False


def test_convergence_cannot_outlast_the_run():
    with pytest.raises((ConfigError, ValidationError), match='(?i)never converge'):
        _build(loop__iterations=5)


def test_coordination_window_must_be_ordered():
    with pytest.raises((ConfigError, ValidationError), match='(?i)ordered'):
        _build(gates__coordination_window=[2.6, 2.0])


def test_grey_band_cannot_exceed_the_pass_bar():
    with pytest.raises((ConfigError, ValidationError), match='(?i)grey_band'):
        _build(gates__dimer_pae_grey_band=20.0)


def test_dimer_defaults_match_calibration_doc():
    """docs/CALIBRATION.md: keep the paper's 14, treat 9-14 as inspect-not-fail, 5.0 A contacts."""
    g = _build().gates
    assert (g.dimer_interface_pae, g.dimer_pae_grey_band,
            g.dimer_contact_cutoff) == (14.0, 9.0, 5.0)


def test_per_iteration_is_the_paper_production_scale():
    assert _build().loop.per_iteration == 1500


def test_generation_is_off_by_default():
    """Neither backbone generator is validated on chlorophyllide; it stays behind a flag."""
    assert _build().stages.generate.enabled is False


def test_config_is_frozen():
    cfg = _build()
    with pytest.raises(ValidationError):
        cfg.gates.packstat = 0.1
