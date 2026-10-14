"""Config -> NISE parameters.

The NISE iteration loop lives in the campaign driver's `main()`, which instantiates
`DesignCampaign(**kwargs)` from the shared NISE install. v7's and v8's drivers are 1200 lines
each and their `params` dicts are BYTE-IDENTICAL: the whole Boltz/AF3 difference is a module
import, a `CONSTRAIN_HIS_MG_BOND` flag, and a replaced `predict_complex_structures`.

This module is the translation half -- campaign YAML to the `params` dict that loop takes.
`tools/config_equivalence.py` checks the same mapping in reverse against the real drivers, so
the two directions are pinned to each other and to the code that actually ran the campaigns.

Values that are DERIVED rather than configured are built here, and each says why. A parameter
that is a constant in every campaign is not a knob, and promoting it to one invites someone to
turn it.
"""

from __future__ import annotations

import os
from pathlib import Path

from chlide.config import CampaignConfig

__all__ = ['nise_params', 'LASER_SAMPLING_PARAMS', 'nise_directory']

# Identical in every campaign to date. Cysteine is disabled because the paper's cascade drops
# any sequence containing one, and disable_charged_fs keeps D/E/K/R out of the buried shell
# around the ligand -- a buried charge is a cascade failure, so sampling them wastes the budget.
LASER_SAMPLING_PARAMS = {
    'sequence_temp': 0.5,
    'first_shell_sequence_temp': 0.7,
    'chi_temp': 1e-6,
    'seq_min_p': 0.0,
    'chi_min_p': 0.0,
    'disable_pbar': True,
    'disabled_residues_list': ['X', 'C'],
    'budget_residue_sele_string': None,
    'ala_budget': 4,
    'gly_budget': 0,
    'disable_charged_fs': True,
}


def nise_directory() -> Path:
    """The shared NISE install. Read-only: this repo never writes into it."""
    return Path(os.environ.get('NISE_PATH', '/home/data/programs/NISE'))


def nise_params(cfg: CampaignConfig, run_dir: Path, *, debug: bool = False) -> dict:
    """Everything `main()` takes, built from a validated campaign.

    `run_dir` is a CLI argument, never a property of the config -- that separation is what
    stops a copied campaign writing into its predecessor's tree.
    """
    run_dir = Path(run_dir).resolve()
    devices = list(cfg.engine.devices)
    laser = dict(LASER_SAMPLING_PARAMS)

    params: dict = dict(
        debug=debug,
        # Upstream defaults this to `True and not debug`, which silently turns W&B on for a
        # production run and then fails on a hardcoded entity. Off, always.
        use_wandb=False,

        # DERIVED from the run directory, never configured. A literal here is how one driver
        # came to point at a finished campaign's input tree.
        input_dir=run_dir / 'input',

        ligand_3lc=cfg.ligand.code,
        ligand_smiles=cfg.ligand.smiles,
        ligand_rmsd_mask_atoms=set(cfg.ligand.rmsd_mask_atoms),
        ligand_atoms_enforce_buried=set(cfg.ligand.atoms_enforce_buried),
        ligand_atoms_enforce_exposed=set(cfg.ligand.atoms_enforce_exposed),
        laser_sampling_params=laser,

        objective_function=cfg.loop.objective,
        drop_rmsd_mask_atoms_from_ligand_plddt_calc=True,   # constant in every campaign
        keep_input_backbone_in_queue=False,                 # constant
        keep_best_generator_backbone=True,                  # constant

        # STAYS FALSE, and this was measured. useChirality=True needs EVERY perceived centre
        # to agree; RDKit perceives MG1 (4-coordinate Mg, not a real centre) and CA2 alongside
        # the one centre we care about, so both the inverted AND the correct population return
        # zero matches, every ligand_rmsd becomes inf, the backbone queue empties at iteration
        # 0 and the campaign freezes. The 13-2 centre is gated directly instead.
        rmsd_use_chirality=cfg.gates.rmsd_use_chirality,

        self_consistency_ligand_rmsd_threshold=cfg.gates.self_consistency_ligand_rmsd,
        self_consistency_protein_rmsd_threshold=cfg.gates.self_consistency_protein_rmsd,
        align_on_binding_site=False,                        # constant

        # 0-BASED POSITIONAL. Boltz renumbers its output 1-N, so a resnum-based selection moves.
        fixed_identity_residue_indices=cfg.scaffold.fixed_identity_selection,

        # reduce is never invoked: reduce_executable_path is None, so RDKit protonates.
        use_reduce_protonation=False,
        reduce_hetdict_path=None,
        reduce_executable_path=None,

        model_checkpoint=(nise_directory() / 'LASErMPNN' /
                          'model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt'),

        num_iterations=cfg.loop.iterations,
        num_top_backbones_per_round=cfg.loop.top_backbones,
        sequences_sampled_at_once=cfg.loop.sequences_sampled_at_once,
        sequences_sampled_per_backbone=cfg.loop.per_backbone,

        # BDFP44's cleft is far shallower than the four-helix bundle NISE was developed on. At
        # the default alpha 9.0 only 10/45 ligand atoms read buried and the Mg reads EXPOSED,
        # so the burial gate would reject every design at iteration 0.
        burial_mask_alpha_hull_alpha=cfg.scaffold.burial_alpha,

        # DERIVED: the patch ships with this repo, never borrowed from a campaign.
        boltz2x_executable_path=_boltz_executable(),
        boltz_inference_devices=devices,
        use_boltz_conformer_potentials=True,                # Boltz-x mode; effectively constant
        boltz2_predict_affinity='pbind' in cfg.loop.objective,
        use_boltz_1x=False,                 # multi-device is bugged in boltz v2.1.1
        boltz2_disable_kernels=False,
        boltz2_disable_nccl_p2p=True,       # consumer cards have no P2P; NCCL hangs without it

        relax_in_loop=cfg.stages.relax.enabled,
        relax_python=os.environ.get('ROSETTA_PYTHON',
                                    '/home/yigall/mambaforge/envs/Rosetta/bin/python'),
        relax_params=str(run_dir / 'input' / 'lig_from_input_pdb.params'),
        relax_top_n=cfg.compute.relax_workers,
        relax_workers=cfg.compute.relax_workers,
        # The physical Mg-N(His) distance, NOT the midpoint of the acceptance window. Deriving
        # it from the window gives 2.30 and moves the restraint 0.15 A off the real bond; the
        # config validates that it lies INSIDE the window instead.
        relax_r0=cfg.gates.metal_r0,
        relax_timeout=cfg.compute.relax_timeout,
        coordination_window=tuple(cfg.gates.coordination_window),

        laser_inference_device=devices[0],
        laser_inference_dropout=True,                       # constant
    )
    return params


def _boltz_executable() -> str:
    """The patched Boltz entry point.

    Upstream's ConnectionsPotential has a ONE-SIDED flat bottom, so a declared bond collapses
    to ~1.52 A. The wrapper supplies a two-sided 2.0-2.6 A window. The shared boltz install is
    never modified.
    """
    patch = Path(__file__).resolve().parents[2] / 'tools' / 'boltz_coord_patch.py'
    py = os.environ.get('BOLTZ_PYTHON', '/home/yigall/mambaforge/envs/boltz/bin/python')
    return f'{py} {patch}'
