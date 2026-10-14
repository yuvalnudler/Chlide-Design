#!/usr/bin/env python
"""Fold each shortlisted design as a SINGLE chain with AF3 and record its fold confidence.

Why this is worth its own stage: the campaign's objective is Boltz ligand pLDDT, and the terminal
cascade is almost entirely Rosetta. Neither asks an independent model whether the protein folds at
all. AF3, run MSA-free on one chain, is that second opinion -- and on v5 it disagrees sharply:
designs with Boltz ligand pLDDT above 0.95 have come back with AF3 chain pLDDT as low as 0.44.

This is distinct from af3_interface_pae.py, which folds TWO copies to probe aggregation. Chain-A
pLDDT can be read out of those dimer runs, but it is confounded -- a badly packed dimer depresses
the per-chain confidence without the monomer itself being bad. For "does this fold", fold one
chain.

No ligand is included. The question is whether the designed sequence encodes the intended fold;
the ligand's contribution is already covered by Boltz and by the Rosetta stages.

Usage:
    python tools/af3_monomer.py --run-dir <run> --top 25 --devices 0,1,2,3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np

# The AF3 launcher is an external PROGRAM, not a campaign artifact. Overridable so
# this repo does not hardcode a path into a tree it does not own.
AF3_RUNNER = os.environ.get(
    'AF3_RUNNER', '/home/data/yuvaln/main_pipeline/scripts/run_af3_custom.sh')
AF3_PYTHON = '/home/yigall/mambaforge/envs/af3/bin/python'
AA3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
          'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def sequence_of(pdb: Path) -> str:
    return ''.join(AA3to1.get(l[17:20], 'X') for l in pdb.read_text().splitlines()
                   if l.startswith('ATOM') and l[12:16] == ' CA ' and l[21] == 'A')


def write_json(name: str, seq: str, out: Path) -> Path:
    single = f'>query\n{seq}'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'name': name,
        'sequences': [{'protein': {'id': 'A', 'sequence': seq, 'unpairedMsa': single,
                                   'pairedMsa': single, 'templates': []}}],
        'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 1}, indent=2))
    return out


def score(out_dir: Path, name: str) -> dict:
    """ptm and mean atom pLDDT from one AF3 output directory."""
    rec = {'name': name, 'ok': False}
    base = out_dir / name.lower()
    summ = next(iter(base.glob('*summary_confidences.json')), None)
    conf = next((f for f in base.glob('*confidences.json') if 'summary' not in f.name), None)
    if summ is None or conf is None:
        rec['error'] = f'missing AF3 output under {base}'
        return rec
    try:
        rec['ptm'] = json.load(summ.open()).get('ptm')
        rec['ranking_score'] = json.load(summ.open()).get('ranking_score')
        j = json.load(conf.open())
        if 'atom_plddts' in j:
            a = np.array(j['atom_plddts'], dtype=float)
            rec['mean_plddt'] = float(a.mean() / 100)
            rec['frac_above_70'] = float((a > 70).mean())
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {str(e)[:70]}'
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(os.environ.get('CHLIDE_RUN_DIR') or Path(__file__).resolve().parent.parent))
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--skip-fold', action='store_true', help='only score existing outputs')
    a = ap.parse_args()

    final = a.run_dir / 'results' / 'final'
    pdbs = sorted(final.glob('top*.pdb'))[:a.top]
    if not pdbs:
        print(f'no top*.pdb under {final} -- run final_shortlist.py first')
        return 1

    out_dir = a.run_dir / 'results' / 'af3_mono'
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in pdbs:
        name = p.name.split('_', 1)[1].replace('.pdb', '').replace('_model_0', '') + '_mono'
        seq = sequence_of(p)
        if len(seq) < 50:
            print(f'  skipping {p.name}: short/unreadable chain A')
            continue
        done = (out_dir / name.lower()).is_dir() and any((out_dir / name.lower()).glob('*_model.cif'))
        jobs.append((name, None if done else write_json(name, seq, out_dir / 'inputs' / f'{name}.json')))

    pending = [(n, j) for n, j in jobs if j is not None]
    print(f'{len(jobs)} designs; {len(jobs) - len(pending)} already folded, folding {len(pending)}')

    if pending and not a.skip_fold:
        devs = a.devices.split(',')
        procs, logs = [], []
        t0 = time.time()
        for i, (name, js) in enumerate(pending):
            # keep at most one fold per GPU in flight; AF3 saturates a card on its own
            while sum(1 for _, p in procs if p.poll() is None) >= len(devs):
                time.sleep(5)
            log = open(out_dir / f'{name}.log', 'w')
            cmd = ['bash', AF3_RUNNER, '-i', str(js), '-o', str(out_dir),
                   '-d', devs[i % len(devs)], '--run_data_pipeline=false']
            procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                 env=dict(os.environ, AF3_PYTHON=AF3_PYTHON))))
            logs.append(log)
        for _, p in procs:
            p.wait()
        for f in logs:
            f.close()
        print(f'  folded {len(pending)} in {time.time() - t0:.0f}s')

    recs = [score(out_dir, n) for n, _ in jobs]
    ok = [r for r in recs if r.get('ok')]
    csv_path = out_dir / 'monomer_confidence.csv'
    with csv_path.open('w', newline='') as f:
        f.write('# AF3 single-chain fold confidence, MSA-free, no ligand.\n')
        f.write('# An independent check that the SEQUENCE encodes the fold -- orthogonal to the\n')
        f.write('# Boltz ligand pLDDT the campaign optimises and to every Rosetta stage.\n')
        w = csv.DictWriter(f, fieldnames=['name', 'ptm', 'mean_plddt', 'frac_above_70',
                                          'ranking_score', 'ok', 'error'], extrasaction='ignore')
        w.writeheader()
        for r in sorted(recs, key=lambda x: -(x.get('mean_plddt') or -1)):
            w.writerow(r)
    print(f'\n{len(ok)}/{len(recs)} scored')
    if ok:
        pl = np.array([r['mean_plddt'] for r in ok])
        print('  mean pLDDT: median %.3f   range %.3f-%.3f' % (np.median(pl), pl.min(), pl.max()))
        print('  below 0.70 (AF3 doubts the fold): %d/%d' % (int((pl < 0.70).sum()), len(pl)))
        print('\n  %-30s %6s %9s' % ('design', 'ptm', 'meanPLDDT'))
        for r in sorted(ok, key=lambda x: -x['mean_plddt']):
            print('  %-30s %6.2f %9.3f' % (r['name'].replace('_mono', ''), r['ptm'], r['mean_plddt']))
    print(f'\n-> {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
