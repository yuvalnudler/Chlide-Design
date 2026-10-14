#!/usr/bin/env python
"""AF3 two-copy interface pAE: the paper's aggregation filter.

Polizzi et al. 2025, Methods:

    "Two copies of each designed protein sequence were predicted using AlphaFold and were
     visually inspected for domain swaps or low interface pAE (pAE < 14; residues within 5.0 A
     in predicted multimer), which we used as a proxy for aggregation potential."

So the filter KEEPS designs whose interface pAE is HIGH. A confident dimer interface means the
model thinks two copies bind each other, which predicts aggregation or domain swapping rather
than a clean monomer. Elsewhere the paper states the same filter as "average interfacial
pAE > 15". Getting the direction backwards would select precisely the designs to avoid.

The statistic is the mean pAE over cross-chain residue pairs that are actually in contact
(CB within 5.0 A) in the predicted dimer -- NOT `chain_pair_pae_min` from the summary file, which
is a single minimum over the whole chain pair and far laxer.

No ligand is included: the filter is about the protein's self-association.

Runs AF3 through the project's existing native runner
(main_pipeline/scripts/run_af3_custom.sh, af3 conda env, --run_data_pipeline=false for MSA-free
folding), sharding jobs across the given GPUs.

Usage:
    python tools/af3_interface_pae.py --run-dir <run> --top 50 --devices 0,1,2,3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np

# The AF3 launcher is an external PROGRAM, not a campaign artifact. Overridable so
# this repo does not hardcode a path into a tree it does not own.
AF3_RUNNER = os.environ.get(
    'AF3_RUNNER', '/home/data/yuvaln/main_pipeline/scripts/run_af3_custom.sh')
AF3_PYTHON = '/home/yigall/mambaforge/envs/af3/bin/python'
CONTACT_CUTOFF = 5.0     # CB-CB, per the paper's "residues within 5.0 A"
PAE_MIN = 14.0           # keep designs ABOVE this

AA3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
          'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def sequence_of(pdb: Path) -> str:
    seq = []
    for line in pdb.read_text().splitlines():
        if line.startswith('ATOM') and line[12:16] == ' CA ' and line[21] == 'A':
            seq.append(AA3to1.get(line[17:20], 'X'))
    return ''.join(seq)


def write_json(name: str, seq: str, out: Path) -> Path:
    """Two copies of the protein, no ligand, MSA- and template-free."""
    single = f'>query\n{seq}'
    fold = {
        'name': name,
        'sequences': [
            {'protein': {'id': 'A', 'sequence': seq, 'unpairedMsa': single,
                         'pairedMsa': single, 'templates': []}},
            {'protein': {'id': 'B', 'sequence': seq, 'unpairedMsa': single,
                         'pairedMsa': single, 'templates': []}},
        ],
        'modelSeeds': [1],
        'dialect': 'alphafold3',
        'version': 1,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fold, indent=2))
    return out


def _cb_coords(cif_path: Path):
    """CB (CA for glycine) per residue, with chain ids, from the AF3 mmCIF."""
    import prody as pr
    pr.confProDy(verbosity='none')
    st = pr.parseMMCIF(str(cif_path))
    sel = st.select('name CB or (resname GLY and name CA)')
    return sel.getChids(), sel.getResnums(), sel.getCoords()


def interface_pae(out_dir: Path, name: str) -> dict:
    """Mean pAE over cross-chain residue pairs in contact."""
    rec = {'name': name, 'ok': False}
    base = out_dir / name.lower()
    conf = next(iter(base.glob('*_confidences.json')), None)
    conf = next((f for f in base.glob('*_confidences.json')
                 if 'summary' not in f.name), conf)
    cif = next(iter(base.glob('*_model.cif')), None)
    if conf is None or cif is None:
        rec['error'] = f'missing AF3 output under {base}'
        return rec
    try:
        d = json.load(conf.open())
        pae = np.array(d['pae'], dtype=float)
        tok_chain = np.array(d['token_chain_ids'])
        tok_res = np.array(d['token_res_ids'], dtype=int)

        chids, resnums, xyz = _cb_coords(cif)
        chains = sorted(set(chids))
        if len(chains) != 2:
            rec['error'] = f'expected 2 chains, found {chains}'
            return rec
        a, b = chains
        ia, ib = np.where(chids == a)[0], np.where(chids == b)[0]
        dist = np.linalg.norm(xyz[ia][:, None, :] - xyz[ib][None, :, :], axis=-1)
        pa, pb = np.where(dist < CONTACT_CUTOFF)
        rec['contact_pairs'] = int(len(pa))
        if not len(pa):
            # no interface at all -- the ideal outcome; report a sentinel above any threshold
            rec.update(interface_pae=float('inf'), ok=True, note='no CB contacts within 5 A')
            return rec

        # map each contacting residue to its pae row/col
        def tok_of(ch, rn):
            m = np.where((tok_chain == ch) & (tok_res == rn))[0]
            return m[0] if len(m) else None
        vals = []
        for x, y in zip(pa, pb):
            ta = tok_of(a, resnums[ia][x])
            tb = tok_of(b, resnums[ib][y])
            if ta is None or tb is None:
                continue
            vals.append(pae[ta, tb])
            vals.append(pae[tb, ta])
        if not vals:
            rec['error'] = 'contacts found but no matching pae tokens'
            return rec
        rec['interface_pae'] = float(np.mean(vals))
        rec['interface_pae_min'] = float(np.min(vals))
        summ = next((f for f in base.glob('*_summary_confidences.json')), None)
        if summ:
            s = json.load(summ.open())
            rec['iptm'] = s.get('iptm')
            rec['chain_pair_pae_min'] = (np.array(s['chain_pair_pae_min'])[0, 1]
                                         if 'chain_pair_pae_min' in s else None)
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--devices', default='0,1,2,3')
    ap.add_argument('--cascade-csv', type=Path, default=None,
                    help='explicit cascade_metrics.csv; default picks the newest that exists')
    ap.add_argument('--skip-fold', action='store_true', help='only score existing AF3 outputs')
    a = ap.parse_args()

    # v4 wrote a corrected subset to cascade_fixed/; v5 writes straight to cascade/. Prefer an
    # explicit --cascade-csv, else take whichever exists, newest first, so this cannot silently
    # read a stale file from a previous campaign.
    cands = [a.cascade_csv] if a.cascade_csv else [
        a.run_dir / 'results' / 'cascade_fixed' / 'cascade_metrics.csv',
        a.run_dir / 'results' / 'cascade' / 'cascade_metrics.csv']
    cands = [c for c in cands if c and c.is_file()]
    if not cands:
        print(f'no cascade_metrics.csv under {a.run_dir}/results -- run polizzi_cascade.py first')
        return 1
    casc = max(cands, key=lambda c: c.stat().st_mtime)
    print(f'reading {casc}')
    with casc.open() as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith('#'))]

    def passes(r):
        try:
            return (float(r['packstat']) > 0.55 and int(r['apo_clashes']) == 0
                    and int(r.get('buried_charged', -1)) == 0
                    and float(r['ca_rmsd_to_min']) < 1.5 and float(r['lig_rmsd_to_min']) < 1.5)
        except Exception:
            return False
    surv = [r for r in rows if passes(r)]
    surv.sort(key=lambda r: float(r.get('buns_combined') or 9e9))
    surv = surv[:a.top]
    print(f'{len(rows)} cascade rows -> {len([r for r in rows if passes(r)])} survivors '
          f'-> folding top {len(surv)} by combined BUNs')
    if not surv:
        print('nothing to fold'); return 1

    af3_out = a.run_dir / 'results' / 'af3_dimer'
    json_dir = af3_out / 'inputs'
    af3_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r in surv:
        pdb = Path(r['output'])
        seq = sequence_of(pdb)
        if len(seq) < 50:
            print(f'  skipping {r["name"]}: short/unreadable chain A'); continue
        name = r['name'].replace('_model_0', '').replace('_min', '')
        if (af3_out / name.lower()).is_dir() and any((af3_out / name.lower()).glob('*_model.cif')):
            jobs.append((name, None))       # already folded; score it without re-running
            continue
        jobs.append((name, write_json(name, seq, json_dir / f'{name}.json')))

    if not a.skip_fold:
        devs = a.devices.split(',')
        procs, logs = [], []
        pending = [(n, j) for n, j in jobs if j is not None]
        print(f'  {len(jobs)-len(pending)} already folded; folding {len(pending)}')
        for i, (name, js) in enumerate(pending):
            dev = devs[i % len(devs)]
            log = open(af3_out / f'{name}.log', 'w')
            env = dict(os.environ, AF3_PYTHON=AF3_PYTHON)
            cmd = ['bash', AF3_RUNNER, '-i', str(js), '-o', str(af3_out), '-d', dev,
                   '--run_data_pipeline=false']
            procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
            logs.append(log)
            # keep at most len(devs) folds in flight
            while sum(1 for _, p in procs if p.poll() is None) >= len(devs):
                import time; time.sleep(10)
        for name, p in procs:
            p.wait()
        for f in logs:
            f.close()

    recs = [interface_pae(af3_out, name) for name, _ in jobs]
    ok = [r for r in recs if r.get('ok')]
    keep = [r for r in ok if r['interface_pae'] > PAE_MIN]
    csv_path = af3_out / 'interface_pae.csv'
    with csv_path.open('w', newline='') as f:
        f.write(f'# AF3 two-copy interface pAE (Polizzi 2025 aggregation proxy). KEEP pAE > {PAE_MIN}:\n')
        f.write('# a confident dimer interface predicts aggregation / domain swapping.\n')
        f.write(f'# statistic = mean pae over cross-chain residue pairs with CB within {CONTACT_CUTOFF} A.\n')
        w = csv.DictWriter(f, fieldnames=['name', 'interface_pae', 'interface_pae_min',
                                          'contact_pairs', 'iptm', 'chain_pair_pae_min',
                                          'ok', 'error', 'note'], extrasaction='ignore')
        w.writeheader()
        for r in sorted(recs, key=lambda x: -(x.get('interface_pae') or -1)):
            w.writerow(r)
    print(f'\n{len(ok)}/{len(recs)} scored; {len(keep)} pass pAE > {PAE_MIN}')
    for r in sorted(keep, key=lambda x: -x['interface_pae'])[:10]:
        print('  %-34s interface pAE %6.1f  contacts %4d' %
              (r['name'][:34], r['interface_pae'], r.get('contact_pairs', -1)))
    print(f'-> {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
