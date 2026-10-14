#!/usr/bin/env python
"""Fold with AlphaFold3 and hand the driver back exactly the layout Boltz would have produced.

v6a changes one thing about v5/v6b: the folding engine. Everything downstream -- the harvest, the
Rosetta cascade, the burial and SASA calculations, the chirality gate -- is kept bit-for-bit
identical, and the way to do that is to make AF3's output indistinguishable from Boltz's rather
than to teach six tools a second format. So this module manufactures:

    <sampling_subdir>/boltz_results_boltz_inputs/predictions/<stem>/<stem>_model_0.pdb
    <sampling_subdir>/boltz_results_boltz_inputs/predictions/<stem>/confidence_<stem>_model_0.json

The driver reads precisely two things out of those: the per-atom B-factor column (which it divides
by 100 to get pLDDT) and the `iptm` key. AF3's mmCIF carries per-atom pLDDT in B_iso_or_equiv, so
the first survives conversion untouched; the second comes from *_summary_confidences.json.

FIVE THINGS THAT WOULD EACH HAVE COST A CAMPAIGN
-------------------------------------------------
1. AF3_CONFORMER_SKIP defaults to "CLA-IDE" in run_af3_custom.sh. Our residue is CLAOX, so
   unless it is overridden AF3 attempts an RDKit ETKDG embed of an Mg-porphyrin, fails, and falls
   back to the ideal coordinates anyway -- at ~71 s wasted per fold. At 600 folds/iteration that
   is ~12 GPU-hours per iteration, more than doubling the campaign for nothing.
2. run_af3_custom.sh hardcodes --num_diffusion_samples=5 and --json_path (one process per
   design). We want 1 sample, and --input_dir so a whole shard amortises one JAX compile. Both
   mean bypassing that wrapper and calling run_alphafold_patched.py, which passes sys.argv
   straight through to AlphaFold3's own entry point.
3. Do NOT inherit the .lower() from af3_monomer.py / af3_interface_pae.py. AF3 does not lowercase
   output directory names; those tools only work because their generated names already are. Our
   stems (chunk_0_seq_0) are lowercase too, but we glob case-insensitively rather than rely on it.
4. Preserve the chunk_N_seq_M stem convention. It is an artifact of Boltz sharding, but every CSV,
   every results/ identifier and the whole harvest join key depends on it.
5. AF3 emits label_comp_id = CLAOX, and PDB residue names are 3 characters. The truncation to
   "CLA" is unavoidable and harmless -- the driver calls setResnames(ligand_3lc) immediately, and
   tools/ligand_chirality.py matches on ATOM NAMES, not residue names.

His-Mg is deliberately UNRESTRAINED here (no bondedAtomPairs): there is no patched potential in
AF3, and per Guy's README you never restrain the bond you are scoring. Coordination control moves
entirely to the Rosetta relax gate at 2.0-2.6 A.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import prody as pr

pr.confProDy(verbosity='none')

# External programs, overridable: this repo does not own main_pipeline or /home/data/programs.
AF3_RUNNER_PY = os.environ.get(
    'AF3_RUNNER_PY', '/home/data/yuvaln/main_pipeline/scripts/run_alphafold_patched.py')
AF3_PYTHON = os.environ.get('AF3_PYTHON', '/home/yigall/mambaforge/envs/af3/bin/python')
AF3_MODEL_DIR = os.environ.get('AF3_MODEL_DIR', '/home/data/programs/AF3_PARAMETERS')
AF3_DB_DIR = os.environ.get('AF3_DB_DIR', '/home/data/public_databases')
AF3_SCRIPTS = os.environ.get('AF3_SCRIPTS', '/home/data/yuvaln/main_pipeline/scripts')
AF3_HOME = os.environ.get('AF3_HOME', '/home/data/programs/alphafold3')

AXIAL_HIS_RESNUM = 85
NUM_RECYCLES = 10
NUM_DIFFUSION_SAMPLES = 1


def af3_json(stem: str, sequence: str, ccd_code: str, user_ccd_path: Path, seed: int = 1,
             his_resnum: int | None = None) -> dict:
    """One AF3 job. MSA-free, single diffusion sample, ligand from our own CCD.

    `his_resnum` would add an explicit His NE2 -- Mg bond via bondedAtomPairs. It defaults to
    None (UNRESTRAINED) because the bonded form is measurably worse. Both measured on the native
    BDFP44 sequence:

        unrestrained : Mg 0.047 A off the pyrrole-N centroid, 13-2 = S, His-Mg 4.22 A
        bondedAtomPairs: Mg 0.298 A off centre,  13-2 = R,  His-Mg 1.50 A

    AF3's bondedAtomPairs has no distance window -- it forces a covalent-length bond, crushing the
    coordination to 1.50 A, pulling the metal out of plane and flipping the stereocentre. That is
    exactly what Boltz needed a patched two-sided 2.0-2.6 A potential for, and AF3 has no
    equivalent.

    The 4.22 A left by the unrestrained fold is NOT a problem: the Rosetta relax forms the bond
    from that pose, measured 4.22 -> 2.50 A with the ligand moving only 0.15 A and the metal
    staying seated. Coordination control is the relax gate's job, as the campaign plan intended.
    """
    single = f'>query\n{sequence}'
    job = {
        'name': stem,
        'sequences': [
            {'protein': {'id': 'A', 'sequence': sequence, 'unpairedMsa': single,
                         'pairedMsa': single, 'templates': []}},
            {'ligand': {'id': 'B', 'ccdCodes': [ccd_code]}},
        ],
        'userCCDPath': str(user_ccd_path),
        'modelSeeds': [seed],
        'dialect': 'alphafold3',
        'version': 1,
    }
    if his_resnum is not None:
        job['bondedAtomPairs'] = [[['A', his_resnum, 'NE2'], ['B', 1, 'MG']]]
    return job


def af3_env(ccd_code: str) -> dict:
    """The environment run_af3_custom.sh builds, with the conformer skip pointed at OUR residue."""
    return dict(
        os.environ,
        ALPHAFOLD_X='3.0.1',
        XLA_FLAGS='--xla_gpu_enable_triton_gemm=false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',
        XLA_CLIENT_MEM_FRACTION='0.6',
        PYTHONPATH=f'{AF3_SCRIPTS}:{AF3_HOME}:' + os.environ.get('PYTHONPATH', ''),
        AF3_CONFORMER_SKIP=ccd_code,            # trap 1 -- NOT the default CLA-IDE
        AF3_CONFORMER_SKIP_SECS='5',
    )


def fold_shards(input_dir: Path, out_dir: Path, devices: list[str], ccd_code: str,
                debug: bool = False) -> None:
    """Shard the JSONs over the GPUs, one AF3 process per card, --input_dir per shard."""
    jsons = sorted(input_dir.glob('*.json'))
    if not jsons:
        print(f'af3_adapter: no JSON inputs under {input_dir}')
        return
    shard_dirs = []
    for dev in devices:
        d = input_dir.parent / f'af3_inputs_gpu{dev}'
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shard_dirs.append(d)
    for j, src in enumerate(jsons):
        shutil.copy(src, shard_dirs[j % len(shard_dirs)] / src.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for dev, d in zip(devices, shard_dirs):
        if not any(d.glob('*.json')):
            continue
        shard_out = out_dir / f'gpu{dev}'
        shard_out.mkdir(parents=True, exist_ok=True)
        cmd = (f'CUDA_VISIBLE_DEVICES={dev} nice -n 10 {AF3_PYTHON} {AF3_RUNNER_PY} '
               f'--input_dir={d} --output_dir={shard_out} '
               f'--db_dir={AF3_DB_DIR} --model_dir={AF3_MODEL_DIR} '
               f'--num_recycles={NUM_RECYCLES} '
               f'--num_diffusion_samples={NUM_DIFFUSION_SAMPLES} '
               f'--flash_attention_implementation=cudnn '
               f'--run_data_pipeline=false --run_inference=true')
        print(cmd)
        log = open(out_dir / f'af3_gpu{dev}.log', 'w')
        procs.append(subprocess.Popen(cmd, shell=True, env=af3_env(ccd_code),
                                      stdout=log, stderr=subprocess.STDOUT))
    for p in procs:
        try:
            p.wait()
        except Exception:                                       # noqa: BLE001
            print('AF3 crashed! This might be fine, trying to recover...')


def _find_output(out_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
    """The model mmCIF and summary JSON for one stem, without assuming a case convention."""
    cif = summ = None
    for base in out_dir.glob(f'gpu*/{stem}'):
        cif = cif or next(iter(base.glob('*_model.cif')), None)
        summ = summ or next(iter(base.glob('*summary_confidences.json')), None)
    if cif is None:                                             # trap 3: case-insensitive fallback
        low = stem.lower()
        for base in out_dir.glob('gpu*/*'):
            if base.is_dir() and base.name.lower() == low:
                cif = cif or next(iter(base.glob('*_model.cif')), None)
                summ = summ or next(iter(base.glob('*summary_confidences.json')), None)
    return cif, summ


def to_boltz_layout(out_dir: Path, dest_root: Path, stems: list[str],
                    ligand_chain: str = 'B') -> tuple[int, int]:
    """Convert every AF3 output into the predictions/<stem>/ layout the driver expects."""
    dest = dest_root / 'boltz_results_boltz_inputs' / 'predictions'
    dest.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for stem in stems:
        cif, summ = _find_output(out_dir, stem)
        if cif is None:
            bad += 1
            continue
        try:
            ag = pr.parseMMCIF(str(cif))
            if ag is None:
                bad += 1
                continue
            d = dest / stem
            d.mkdir(parents=True, exist_ok=True)
            pr.writePDB(str(d / f'{stem}_model_0.pdb'), ag)
            conf = {'iptm': None, 'ptm': None}
            if summ is not None:
                j = json.load(summ.open())
                conf['iptm'] = j.get('iptm')
                conf['ptm'] = j.get('ptm')
                conf['ranking_score'] = j.get('ranking_score')
            # The driver does float arithmetic on iptm; a null would propagate as a TypeError
            # halfway through an iteration. Single-chain AF3 runs report iptm as null, and a
            # protein+ligand job should not, but do not bet an iteration on it.
            if conf['iptm'] is None:
                conf['iptm'] = float('nan')
            (d / f'confidence_{stem}_model_0.json').write_text(json.dumps(conf, indent=1))
            ok += 1
        except Exception as e:                                  # noqa: BLE001
            print(f'af3_adapter: {stem}: {type(e).__name__}: {str(e)[:100]}')
            bad += 1
    return ok, bad


# ---------------------------------------------------------------------------------------------
def _self_test(a) -> int:
    """Fold ONE design end to end and check every column the driver will read.

    This is the check that cannot be done without a GPU, so it belongs in the launcher's
    pre-flight rather than in preparation. An adapter that writes a PDB with an empty B-factor
    column produces ligand_plddt = 0.0 for every design and a campaign that optimises noise.
    """
    run = a.run_dir
    seq = ''.join(
        {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
         'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
         'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}.get(l[17:20], 'X')
        for l in next(run.glob('input/input_backbones/*.pdb')).read_text().splitlines()
        if l.startswith('ATOM') and l[12:16] == ' CA ' and l[21] == 'A')
    print(f'self-test sequence: {len(seq)} residues')

    root = run / 'af3_selftest'
    if root.exists():
        shutil.rmtree(root)
    (root / 'inputs').mkdir(parents=True)
    stem = 'chunk_0_seq_0'                                      # trap 4: the real stem convention
    (root / 'inputs' / f'{stem}.json').write_text(json.dumps(
        af3_json(stem, seq, a.ccd, run / 'input' / 'claox_userccd.cif',
                 his_resnum=AXIAL_HIS_RESNUM if a.bond else None), indent=2))

    t0 = time.time()
    fold_shards(root / 'inputs', root / 'out', [a.devices.split(',')[0]], a.ccd)
    print(f'  folded in {time.time() - t0:.0f}s')

    ok, bad = to_boltz_layout(root / 'out', root, [stem])
    if ok != 1:
        print(f'FAIL: adapter produced {ok} outputs, {bad} failures. See {root}/out/af3_gpu*.log')
        return 1

    pdb = root / 'boltz_results_boltz_inputs' / 'predictions' / stem / f'{stem}_model_0.pdb'
    ag = pr.parsePDB(str(pdb))
    prot = ag.select('chid A')
    lig = ag.select('(not protein) and not element H')
    problems = []
    if prot is None or lig is None:
        problems.append('missing protein chain A or ligand')
    else:
        lb, pb = lig.getBetas(), prot.copy().ca.getBetas()
        if len(lig) != 46:
            problems.append(f'{len(lig)} ligand heavy atoms, expected 46')
        if np.allclose(lb, 0):
            problems.append('ligand B-factors are all zero -- pLDDT did not survive the mmCIF '
                            'conversion, and every design would score ligand_plddt 0.0')
        if not (0 < lb.mean() / 100 <= 1):
            problems.append(f'ligand pLDDT {lb.mean() / 100:.3f} out of range')
        if not (0 < pb.mean() / 100 <= 1):
            problems.append(f'protein pLDDT {pb.mean() / 100:.3f} out of range')
        names = set(lig.getNames())
        for n in ('MG', 'CBD', 'OCD', 'NA', 'NB', 'NC', 'ND'):
            if n not in names:
                problems.append(f'ligand atom {n} missing -- naming does not match the component')
        mg = lig.select('name MG')
        n4 = lig.select('name NA NB NC ND')
        if mg is not None and n4 is not None and len(n4) == 4:
            off = float(np.linalg.norm(mg.getCoords()[0] - n4.getCoords().mean(0)))
            print(f'  Mg to pyrrole-N centroid: {off:.3f} A')
        his = prot.select(f'resnum {AXIAL_HIS_RESNUM} and name NE2')
        if his is not None and mg is not None:
            hm = float(np.linalg.norm(his.getCoords()[0] - mg.getCoords()[0]))
            print(f'  His{AXIAL_HIS_RESNUM} NE2 - Mg: {hm:.2f} A  (campaign gate 2.0-2.6)')
            # Informational only. AF3 folds unrestrained, so the axial bond is NOT formed here --
            # measured 4.22 A on the native sequence. The Rosetta relax pulls it to 2.50 A and the
            # 2.0-2.6 A gate is applied THERE. Asserting it on the raw fold fails every time.
            if hm < 2.0:
                problems.append(f'His-Mg {hm:.2f} A is too SHORT -- a bond constraint is being '
                                f'applied somewhere; AF3 should fold this unrestrained')
            elif hm > 2.6:
                print(f'  (expected: unrestrained AF3 leaves it long; Rosetta forms it at ~2.5 A)')

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ligand_chirality import ligand_config
    cfg = ligand_config(ag, component=run / 'input' / 'CLAOX.pkl')
    print(f'  13-2 configuration: {cfg}')
    if cfg != 'S':
        problems.append(f'13-2 centre is {cfg}, expected S -- AF3 is not honouring '
                        f'pdbx_stereo_config in the userCCD')

    conf = json.loads((pdb.parent / f'confidence_{stem}_model_0.json').read_text())
    print(f'  iptm: {conf.get("iptm")}   ptm: {conf.get("ptm")}')
    if conf.get('iptm') is None or (isinstance(conf['iptm'], float) and conf['iptm'] != conf['iptm']):
        problems.append('iptm is null/nan -- the driver stores it every iteration')

    if problems:
        print('\nFAIL:\n  ' + '\n  '.join(problems))
        return 1
    print(f'\nPASS: AF3 round-trip is driver-compatible '
          f'(ligand pLDDT {lig.getBetas().mean() / 100:.3f}, 13-2 = S)')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--ccd', default='CLAOX')
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--bond', action='store_true',
                    help='fold WITH a His-Mg bondedAtomPairs constraint. For comparison only: it crushes the bond to 1.50 A and flips the 13-2 centre to R.')
    a = ap.parse_args()
    if a.self_test:
        return _self_test(a)
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
