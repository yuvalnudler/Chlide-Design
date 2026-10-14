"""AlphaFold3 behind the Folder protocol.

Thin wrapper over tools/af3_adapter.py, which holds the five AF3 traps and the measurements
behind them. Nothing here re-implements the adapter; this only declares AF3's capabilities and
adapts the signatures.

AF3 is the faster engine here, which was a surprise: measured on 4 GPUs with --input_dir
batching, 10.4 s/design/GPU against Boltz's 22-25. The campaign that assumed AF3 cost 3-5x more
sized its iterations at 600 designs for nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from chlide.engines.base import FoldResult, LigandSpec, check_metal_restraint

_TOOLS = Path(__file__).resolve().parents[3] / 'tools'


def _adapter():
    """Load tools/af3_adapter.py by path -- it is a script, not an installed module."""
    f = _TOOLS / 'af3_adapter.py'
    if not f.is_file():
        raise FileNotFoundError(f'af3_adapter.py not found at {f}')
    spec = importlib.util.spec_from_file_location('_af3_adapter', str(f))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['af3_adapter']
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


class AF3Folder:
    name = 'af3'
    input_suffix = '.json'

    # AF3 honours pdbx_stereo_config from the user CCD, but only ~59% of the time on designed
    # sequences -- it never converged over v8's 17 iterations (59% -> 70%). That is NOT a
    # restraint mechanism, so it is declared False and the 13-2 gate does the work downstream.
    supports_chirality_restraints = False
    # bondedAtomPairs exists but has NO distance window. See base.check_metal_restraint.
    supports_metal_distance_restraint = False

    def write_input(self, dest_dir: Path, stem: str, sequence: str,
                    ligand: LigandSpec, *, constrain_metal: bool = False) -> Path:
        check_metal_restraint(self, constrain_metal)
        dest_dir.mkdir(parents=True, exist_ok=True)
        payload = _adapter().af3_json(
            stem, sequence, ligand.code, ligand.user_ccd,
            his_resnum=None)          # unrestrained, deliberately
        out = dest_dir / f'{stem}.json'
        out.write_text(json.dumps(payload, indent=2))
        return out

    def fold(self, inputs_dir: Path, out_dir: Path, devices: Sequence[str],
             ligand: LigandSpec, *, debug: bool = False) -> FoldResult:
        ad = _adapter()
        stems = [p.stem for p in sorted(inputs_dir.glob('*.json'))]
        t0 = time.time()
        ad.fold_shards(inputs_dir, out_dir, list(devices), ligand.code, debug=debug)
        ok, bad = ad.to_boltz_layout(out_dir, out_dir, stems)
        return FoldResult(engine=self.name, root=out_dir, n_ok=ok, n_failed=bad,
                          seconds=time.time() - t0,
                          notes=['unrestrained His-Mg; Rosetta relax forms the coordination'])
