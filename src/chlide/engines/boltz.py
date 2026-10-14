"""Boltz-2x behind the Folder protocol.

Two things make this not a plain `boltz predict` call, and both are load-bearing:

1. **The coordination patch.** Upstream's ConnectionsPotential has a ONE-SIDED flat bottom
   (`lower_bounds=None`, `buffer=2.0` hardcoded), so a declared bond collapses to ~1.52 A.
   tools/boltz_coord_patch.py supplies a two-sided 2.0-2.6 A window and measures 2.03 A with
   the Mg 0.12 A off the N4 centroid. The shared boltz install is never modified.

2. **No DDP.** Upstream issues one `boltz predict --devices N` and lets Lightning shard it.
   That deadlocks on these cards: all ranks park on a futex at 0% utilisation, with or without
   NCCL_P2P_DISABLE and NCCL_SHM_DISABLE (verified twice; they hang before allocating any GPU
   memory). The work is embarrassingly parallel -- one structure per input file -- so the
   inputs are sharded and one single-GPU process is run per card.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

from chlide.engines.base import EngineError, FoldResult, LigandSpec

_TOOLS = Path(__file__).resolve().parents[3] / 'tools'
BOLTZ_PYTHON = os.environ.get('BOLTZ_PYTHON', '/home/yigall/mambaforge/envs/boltz/bin/python')


def resolve_cache(run: Path | None = None, explicit: str | None = None) -> Path:
    """A Boltz cache that actually carries the model checkpoints.

    Same resolution as tools/crossfold.py, and for the same reason: an AF3-led campaign has no
    Boltz cache of its own, and the old answer was a hardcoded pointer at a sibling campaign.
    """
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    elif os.environ.get('BOLTZ_CACHE'):
        cands.append(Path(os.environ['BOLTZ_CACHE']).expanduser())
    if run is not None:
        cands.append(run / 'boltz_cache')
    cands.append(Path.home() / '.boltz')
    for c in cands:
        if c.is_dir() and any(c.glob('*.ckpt')):
            return c
    raise EngineError('no Boltz cache with checkpoints. Looked in:\n  '
                      + '\n  '.join(str(c) for c in cands))


class BoltzFolder:
    name = 'boltz'
    input_suffix = '.yaml'

    # 12 ChiralAtomPotential constraints on C2A/C3A/CBD, which require _CIPRank on EVERY atom
    # with the Mg tag cleared (boltz/data/parse/schema.py:341 gates on that).
    supports_chirality_restraints = True
    # Only with the patch below. Unpatched Boltz would collapse the bond to ~1.52 A.
    supports_metal_distance_restraint = True

    def __init__(self, cache: Path | None = None, run: Path | None = None):
        self.cache = cache or resolve_cache(run)
        self.patch = _TOOLS / 'boltz_coord_patch.py'
        if not self.patch.is_file():
            raise EngineError(f'boltz_coord_patch.py missing at {self.patch}; unpatched Boltz '
                              'collapses the His-Mg bond to ~1.52 A')

    def write_input(self, dest_dir: Path, stem: str, sequence: str,
                    ligand: LigandSpec, *, constrain_metal: bool = False) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        lines = ['version: 1', 'sequences:',
                 '  - protein:', '      id: A', f'      sequence: {sequence}',
                 '      msa: empty',
                 '  - ligand:', '      id: B', f'      ccd: {ligand.code}']
        if constrain_metal:
            if ligand.axial_his_resnum is None:
                raise EngineError('constrain_metal needs ligand.axial_his_resnum')
            lines += ['constraints:', '  - bond:',
                      f'      atom1: [A, {ligand.axial_his_resnum}, NE2]',
                      '      atom2: [B, 1, MG]']
        out = dest_dir / f'{stem}.yaml'
        out.write_text('\n'.join(lines) + '\n')
        return out

    def fold(self, inputs_dir: Path, out_dir: Path, devices: Sequence[str],
             ligand: LigandSpec, *, debug: bool = False) -> FoldResult:
        devices = list(devices)
        shards: list[Path] = []
        yamls = sorted(inputs_dir.glob('*.yaml'))
        if not yamls:
            return FoldResult(engine=self.name, root=out_dir, notes=['no inputs'])
        for i, d in enumerate(devices):
            s = inputs_dir.parent / f'boltz_inputs_gpu{d}'
            s.mkdir(parents=True, exist_ok=True)
            shards.append(s)
        for i, y in enumerate(yamls):
            (shards[i % len(shards)] / y.name).write_text(y.read_text())

        env = dict(os.environ, BOLTZ_CACHE=str(self.cache))
        procs = []
        t0 = time.time()
        for d, s in zip(devices, shards):
            if not any(s.glob('*.yaml')):
                continue
            cmd = (f'CUDA_VISIBLE_DEVICES={d} NCCL_P2P_DISABLE=1 {BOLTZ_PYTHON} {self.patch} '
                   f'predict {s} --devices 1 --out_dir {out_dir}/gpu{d} '
                   f'--output_format pdb --override --use_potentials')
            procs.append(subprocess.Popen(cmd, shell=True, env=env,
                                          stdout=subprocess.DEVNULL if not debug else None,
                                          stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        found = list(out_dir.glob('gpu*/boltz_results_*/predictions/*/*_model_0.pdb'))
        return FoldResult(engine=self.name, root=out_dir, n_ok=len(found),
                          n_failed=max(0, len(yamls) - len(found)),
                          seconds=time.time() - t0,
                          notes=[f'cache {self.cache}', 'patched two-sided 2.0-2.6 A window'])
