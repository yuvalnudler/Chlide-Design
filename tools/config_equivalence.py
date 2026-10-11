#!/usr/bin/env python
"""Does a campaign YAML actually reproduce its original driver?

The Phase 1 gate. Reads the `params = dict(...)` literal out of a NISE driver by
AST (importing it would pull in torch and boltz) and diffs it against the resolved
CampaignConfig. Anything that differs must be EXPLAINED here, not silently dropped.

Usage:
  python tools/config_equivalence.py campaigns/v8-af3-coo.yaml \
      /home/data/yuvaln/nise_bdfp44_v8/run_nise_bdfp44_v8.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chlide.config import load_campaign  # noqa: E402

# Driver keys deliberately NOT represented in the config, with the reason.
# A key here is a decision; a key missing from both is a bug.
NOT_IN_CONFIG = {
    'debug': 'SMOKE_TEST is a launch flag, not campaign identity',
    'use_wandb': 'disabled everywhere; upstream default fails on a hardcoded entity',
    'input_dir': 'derived from the run directory, which is a CLI argument',
    'laser_sampling_params': 'LASErMPNN sampling block -- lifted verbatim, see stages/design.py',
    'objective_function': 'loop.objective',
    'model_checkpoint': 'resolved from LASER_PATH at runtime; hashed into the manifest',
    'relax_python': 'machine-specific interpreter path -> env_overrides',
    'relax_params': 'derived: <run>/input/lig_from_input_pdb.params',
    'boltz2x_executable_path': 'derived: <run>/tools/boltz_coord_patch.py',
    'laser_inference_device': 'derived: engine.devices[0]',
    'use_reduce_protonation': 'False everywhere; reduce is never invoked',
    'reduce_hetdict_path': 'unused when use_reduce_protonation is False',
    'reduce_executable_path': 'unused; /nfs/polizzi does not exist on this machine',
    'boltz2_predict_affinity': 'derived from the objective containing "pbind"',
    'use_boltz_1x': 'False; multi-device is bugged in boltz v2.1.1',
    'boltz2_disable_kernels': 'False; cuEquivariance kernels are fine',
    'drop_rmsd_mask_atoms_from_ligand_plddt_calc': 'always True -- a constant, not a knob',
    'keep_input_backbone_in_queue': 'always False -- a constant',
    'keep_best_generator_backbone': 'always True -- a constant',
    'align_on_binding_site': 'always False -- a constant',
    'relax_in_loop': 'stages.relax.enabled',
    'relax_r0': 'midpoint of gates.coordination_window; kept derived so they cannot drift',
    'use_boltz_conformer_potentials': 'always True (Boltz-x mode); implied by engine.fold',
    'boltz2_disable_nccl_p2p': 'always True on these consumer cards; an engine constant',
    'laser_inference_dropout': 'always True -- a constant',
}

# driver key -> how to read the same value out of the config
MAPPING = {
    'ligand_3lc':                             lambda c: c.ligand.code,
    'ligand_smiles':                          lambda c: c.ligand.smiles,
    'ligand_rmsd_mask_atoms':                 lambda c: set(c.ligand.rmsd_mask_atoms),
    'ligand_atoms_enforce_buried':            lambda c: set(c.ligand.atoms_enforce_buried),
    'ligand_atoms_enforce_exposed':           lambda c: set(c.ligand.atoms_enforce_exposed),
    'fixed_identity_residue_indices':         lambda c: c.scaffold.fixed_identity_selection,
    'burial_mask_alpha_hull_alpha':           lambda c: c.scaffold.burial_alpha,
    'num_iterations':                         lambda c: c.loop.iterations,
    'num_top_backbones_per_round':            lambda c: c.loop.top_backbones,
    'sequences_sampled_per_backbone':         lambda c: c.loop.per_backbone,
    'sequences_sampled_at_once':              lambda c: c.loop.sequences_sampled_at_once,
    'self_consistency_ligand_rmsd_threshold': lambda c: c.gates.self_consistency_ligand_rmsd,
    'self_consistency_protein_rmsd_threshold': lambda c: c.gates.self_consistency_protein_rmsd,
    'rmsd_use_chirality':                     lambda c: c.gates.rmsd_use_chirality,
    'coordination_window':                    lambda c: tuple(c.gates.coordination_window),
    'relax_top_n':                            lambda c: c.compute.relax_workers,
    'relax_workers':                          lambda c: c.compute.relax_workers,
    'relax_timeout':                          lambda c: c.compute.relax_timeout,
    'boltz_inference_devices':                lambda c: list(c.engine.devices),
}


def driver_params(path: Path) -> dict:
    """Pull `params = dict(...)` out of the driver without importing it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'params'
                and isinstance(node.value, ast.Call)):
            out = {}
            for kw in node.value.keywords:
                out[kw.arg] = _literal(kw.value)
            return out
    raise SystemExit(f'no `params = dict(...)` found in {path}')


_UNRESOLVED = object()


def _literal(node):
    """Evaluate what we can; everything else is a runtime-derived expression."""
    if isinstance(node, ast.NamedExpr):          # (x := value)
        return _literal(node.value)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        if isinstance(node, ast.IfExp):
            # `2 if SMOKE_TEST else 25` and `500 if not debug else ...`. Resolve the
            # PRODUCTION branch: SMOKE_TEST and debug are both False in a real run.
            # Taking .body blindly reads the smoke-test value and reports a false
            # mismatch -- which is exactly what it did the first time.
            try:
                taken = node.body if _truth(node.test) else node.orelse
            except _Unknown:
                return _UNRESOLVED
            try:
                return _literal(taken)
            except Exception:
                return _UNRESOLVED
        return _UNRESOLVED


class _Unknown(Exception):
    """The branch condition depends on something we cannot evaluate statically."""


# Launch-time flags, at their production values.
_FLAGS = {'SMOKE_TEST': False, 'debug': False}


def _truth(node) -> bool:
    if isinstance(node, ast.Name) and node.id in _FLAGS:
        return _FLAGS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _truth(node.operand)
    if isinstance(node, ast.Constant):
        return bool(node.value)
    raise _Unknown(ast.dump(node))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign', type=Path)
    ap.add_argument('driver', type=Path)
    a = ap.parse_args()

    cfg = load_campaign(a.campaign)
    params = driver_params(a.driver)

    matched, mismatched, unresolved, unexplained = [], [], [], []
    for key, value in params.items():
        if key in MAPPING:
            ours = MAPPING[key](cfg)
            if value is _UNRESOLVED:
                unresolved.append((key, ours))
            elif ours == value or (isinstance(value, (list, tuple, set))
                                   and set(map(str, value)) == set(map(str, ours))):
                matched.append(key)
            else:
                mismatched.append((key, value, ours))
        elif key not in NOT_IN_CONFIG:
            unexplained.append(key)

    print(f'=== {a.campaign.name}  vs  {a.driver.name}')
    print(f'  {len(matched)} driver params reproduced exactly')
    for key, ours in unresolved:
        print(f'  ~ {key}: driver value is a runtime expression; config says {ours!r}')
    for key, theirs, ours in mismatched:
        print(f'  MISMATCH {key}: driver={theirs!r}  config={ours!r}')
    for key in unexplained:
        print(f'  UNEXPLAINED {key}: present in the driver, absent from the config '
              f'AND from NOT_IN_CONFIG')
    missing = [k for k in MAPPING if k not in params]
    for key in missing:
        print(f'  NOT IN DRIVER {key}: the config models something the driver never had')

    bad = len(mismatched) + len(unexplained)
    print('  ->', 'EQUIVALENT' if not bad else f'{bad} unexplained difference(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
