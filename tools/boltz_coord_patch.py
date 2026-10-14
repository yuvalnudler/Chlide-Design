#!/usr/bin/env python
"""Boltz CLI wrapper that makes a declared bond hold a *coordination* distance.

Boltz applies `ConnectionsPotential` (boltz/model/potentials/potentials.py:410) to every bond
declared in a yaml `constraints:` block. It is a flat-bottom distance potential, and it is
one-sided:

    lower_bounds = None                                  # nothing resists collapse
    upper_bounds = torch.full((...,), parameters["buffer"])   # buffer = 2.0, hardcoded

So a declared His NE2 - Mg bond is pushed under 2.0 A and then left alone. Measured over 200
designs of the nise_bdfp44_relax campaign: 1.26-1.95 A, median 1.52 A. Real Mg-N(His)
coordination is 2.05-2.15 A. Without the constraint Boltz instead floors the contact at 2.939 A
(identical across 3,186 designs on two unrelated scaffolds), so neither default is usable.

This wrapper supplies the missing lower bound and widens the upper one, giving a flat bottom over
the physical window, then hands off to Boltz's own CLI. Two facts make it safe:

  * FlatBottomPotential.compute_function already honours a real lower_bounds -- it only
    substitutes -inf when the value is None (potentials.py:241).
  * connected_atom_index is populated ONLY from explicit yaml bond constraints
    (featurizerv2.py:2021), so this touches the declared His-Mg bond and nothing else. The
    ligand's own Mg-N bonds come from the CCD component and are unaffected.

`buffer` is not reachable from the CLI or from steering_args (booleans only), which is why this
is a monkey-patch. The shared install at /home/yigall/mambaforge/envs/boltz is never modified.

Usage: exactly like `boltz`, e.g.
    python boltz_coord_patch.py predict <dir> --devices 1 --out_dir <out> --use_potentials
Override the window with BOLTZ_COORD_MIN / BOLTZ_COORD_MAX (angstrom).
"""
from __future__ import annotations

import os
import sys

import torch

from boltz.model.potentials import potentials as _P

COORD_MIN = float(os.environ.get('BOLTZ_COORD_MIN', '2.0'))
COORD_MAX = float(os.environ.get('BOLTZ_COORD_MAX', '2.6'))


def _compute_args(self, feats, parameters):
    """Two-sided flat bottom over [COORD_MIN, COORD_MAX] for declared connections."""
    pair_index = feats['connected_atom_index'][0]
    n = pair_index.shape[1]
    dev = pair_index.device
    lo = float(parameters.get('lower_buffer', COORD_MIN))
    hi = float(parameters.get('buffer', COORD_MAX))
    # Boltz's own default buffer is 2.0; if the caller left it there, widen to COORD_MAX so the
    # flat bottom actually spans the coordination window instead of being a single point.
    if abs(hi - 2.0) < 1e-9:
        hi = COORD_MAX
    lower_bounds = torch.full((n,), lo, device=dev)
    upper_bounds = torch.full((n,), hi, device=dev)
    k = torch.ones_like(upper_bounds)
    return pair_index, (k, lower_bounds, upper_bounds), None, None, None


_P.ConnectionsPotential.compute_args = _compute_args
print(f'[boltz_coord_patch] ConnectionsPotential flat bottom set to '
      f'{COORD_MIN}-{COORD_MAX} A', file=sys.stderr, flush=True)

from boltz.main import cli  # noqa: E402  patch must land before the CLI builds any potential

if __name__ == '__main__':
    cli()
