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
import time
from pathlib import Path

from chlide.config import CampaignConfig

__all__ = ['nise_params', 'run_campaign', 'LASER_SAMPLING_PARAMS', 'nise_directory']

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

        # debug is a SMOKE TEST, and it has to actually be small. Without these two
        # reductions `--debug` launches a full production run: 25 iterations x 1500
        # designs, which is GPU-days spent proving the wiring works.
        num_iterations=2 if debug else cfg.loop.iterations,
        num_top_backbones_per_round=cfg.loop.top_backbones,
        sequences_sampled_at_once=cfg.loop.sequences_sampled_at_once,
        sequences_sampled_per_backbone=(len(devices) if debug else cfg.loop.per_backbone),

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


def run_campaign(cfg: CampaignConfig, run_dir: Path, *, campaign_file: Path | None = None,
                 debug: bool = False, allow_foreign_gpu: bool = False,
                 dry_run: bool = False) -> int:
    """Launch a NISE campaign from a validated config.

    The driver is `chlide.nise_driver`, ONE module for both engines. v7 and v8 were two
    1200-line files whose only structural difference was which fold function existed; here
    `FOLD_ENGINE` selects at runtime. Nothing is copied into the run directory.
    """
    import uuid

    from chlide.eventlog import EventLog
    from chlide.gpu import GpuLock, assert_gpus_free
    from chlide.provenance import write_manifest

    run_dir = Path(run_dir).resolve()
    params = nise_params(cfg, run_dir, debug=debug)
    run_id = f'{cfg.name}-{time.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:6]}'
    log = EventLog(run_dir / 'logs' / 'events.jsonl', run_id)

    if dry_run:
        print(f'DRY RUN {run_id}')
        print(f'  engine {cfg.engine.fold}, constrain_his_mg={cfg.engine.constrain_his_mg}')
        print(f'  objective {cfg.loop.objective}')
        print(f'  would write {run_dir / "run_manifest.json"}')
        return 0

    # Refuse before writing anything, so a blocked launch leaves no half-made run directory.
    assert_gpus_free(list(cfg.engine.devices), allow_foreign=allow_foreign_gpu)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    # Boltz finds its checkpoints and its 45k-component farm through BOLTZ_CACHE. The old
    # launcher set it on the command line (`BOLTZ_CACHE=$RUN/boltz_cache nohup ...`); without
    # it Boltz falls back to ~/.boltz, which does not exist for this user, and tries to
    # DOWNLOAD the models mid-run.
    if cfg.engine.fold == 'boltz' or cfg.engine.resolved_cross_check == 'boltz':
        cache = run_dir / 'boltz_cache'
        if not any(cache.glob('*.ckpt')):
            raise RuntimeError(
                f'{cache} has no checkpoints. Run `chlide setup --seed-from <campaign>` first.')
        os.environ['BOLTZ_CACHE'] = str(cache)

    # The driver reads these at import time; they are the whole engine switch.
    os.environ['CHLIDE_RUN_DIR'] = str(run_dir)
    os.environ['CHLIDE_FOLD_ENGINE'] = cfg.engine.fold
    os.environ['CHLIDE_CONSTRAIN_HIS_MG'] = '1' if cfg.engine.constrain_his_mg else '0'
    os.environ['CHLIDE_CONVERGENCE_AUTOSTOP'] = '1' if cfg.loop.convergence.enabled else '0'

    # The manifest is written AFTER the environment is set, so it records BOLTZ_CACHE and
    # can hash the Boltz checkpoints. Written the other way round it recorded an empty
    # env_overrides and hashed nothing, which is a manifest that looks complete and is not.
    manifest = write_manifest(run_dir, cfg, params, campaign_file=campaign_file,
                              extra={'run_id': run_id})
    log.emit(event='run_start', campaign=cfg.name, engine=cfg.engine.fold,
             objective=cfg.loop.objective, manifest=str(manifest))
    print(f'{run_id}\n  manifest {manifest}\n  events   {log.path}')

    with GpuLock(list(cfg.engine.devices)):
        from chlide import nise_driver
        if nise_driver.FOLD_ENGINE != cfg.engine.fold:
            raise RuntimeError(
                f'driver imported with FOLD_ENGINE={nise_driver.FOLD_ENGINE!r} but the '
                f'campaign asks for {cfg.engine.fold!r}. The driver reads it at import time, '
                f'so it must not be imported before this point.')
        t0 = time.time()
        try:
            nise_driver.main(**params)
        except BaseException as e:               # includes KeyboardInterrupt / SIGTERM paths
            log.emit(event='run_failed', error=f'{type(e).__name__}: {e}',
                     seconds=round(time.time() - t0, 1))
            raise
    log.emit(event='run_end', seconds=round(time.time() - t0, 1))
    return 0
