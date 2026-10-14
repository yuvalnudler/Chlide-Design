#!/usr/bin/env python
"""Minimise every harvest-passing design, incrementally, while the GPUs keep folding.

v4 relaxed only the top 12 candidates per iteration (the ones that could become next-round
backbones) and left the terminal cascade to minimise all 3,867 pooled designs in one multi-hour
block at the end. This moves that work into the campaign: each iteration's survivors are relaxed
during the *next* iteration's GPU phase, so by the time the run finishes the cascade's stage 5 is
already done and the harvest can rank on physical geometry rather than raw predictions.

Three things make this safe to run alongside a live campaign:

  Incremental. index.csv is keyed on source_path + a size:mtime_ns stamp. The driver rewrites each
  Boltz output in place twice (after atom renaming, and again after protonation), so a path alone
  is not a valid cache key -- a stale hit would silently serve geometry from a superseded file.
  Previous iterations' outputs never change, so steady-state work is only the newest iteration.

  Niced. The driver's own in-loop relax has a 1800 s timeout and falls back to the UNRELAXED
  ranking when it expires, changing the campaign trajectory with only a one-line warning. This
  pool must never be the reason that fires, so it runs at nice 19 / ionice idle with a worker cap.

  Locked. An O_EXCL pid lockfile with a liveness check, so a harvest invocation and a driver
  invocation cannot relax the same design twice.

Output layout matters and is not cosmetic:

    results/relaxed_pool/iter_12/iter12_chunk_0_seq_369_model_0_relaxed.pdb

polizzi_cascade.py and burial_rescore.py both recover the iteration by regex on `/iter_(\\d+)/`.
Without that directory component the cascade falls back to iteration -1 and unqualified names,
which is how an earlier run ended up with 3,612 of 3,867 designs sharing a stem. The `iter<N>_`
filename prefix then keeps the file self-describing once it is copied flat into results/final/.

Usage:
    python tools/relax_pool.py --run-dir <run>                  # relax whatever is pending
    python tools/relax_pool.py --run-dir <run> --iteration 12   # just that iteration
    python tools/relax_pool.py --run-dir <run> --all            # backfill everything
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROSETTA_PY = os.environ.get('ROSETTA_PYTHON',
                           '/home/yigall/mambaforge/envs/Rosetta/bin/python')
# Must match harvest_global_top.py's PAPER exactly. The point of this pool is to minimise what the
# harvest will rank, so a looser gate here just burns CPU on designs the harvest then drops. Note
# ligand_rmsd is 1.5, NOT the driver's in-loop self-consistency threshold of 2.5 -- those are
# different filters at different stages and it is easy to copy the wrong one.
PAPER = dict(protein_rmsd=1.5, ligand_rmsd=1.5, ligand_plddt=0.8, protein_plddt=0.8)
FIELDS = ['source_path', 'src_stamp', 'iteration', 'stem', 'name', 'relaxed_path', 'ok', 'error',
          'his_mg_before', 'his_mg', 'mg_ring_offset_before', 'mg_ring_offset', 'score',
          'ca_rmsd', 'ligand_rmsd', 'seconds', 'relax_version']
RELAX_VERSION = '1'          # bump to invalidate every cached row


def stamp(p: Path) -> str:
    try:
        st = p.stat()
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def load_index(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open() as fh:
        return {r['source_path']: r for r in csv.DictReader(fh)}


def valid(row: dict, src: Path) -> bool:
    """A cached row counts only if the source is unchanged AND the output still exists."""
    if row.get('relax_version') != RELAX_VERSION or row.get('src_stamp') != stamp(src):
        return False
    if row.get('ok') != 'True':
        return False
    rp = row.get('relaxed_path')
    return bool(rp) and Path(rp).is_file()


def pending(run: Path, index: dict, only_iter: int | None, do_all: bool) -> list[tuple[int, str, Path]]:
    """Designs that pass the harvest self-consistency gates and have no valid relaxed output."""
    out = []
    sdf = run / 'input' / 'sampling_dataframes'
    for pkl in sorted(sdf.glob('iter_*_data.pkl')):
        try:
            it = int(pkl.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        if only_iter is not None and it != only_iter:
            continue
        if not (run / 'input' / 'sampled_backbones' / f'iter_{it}' / 'backbone_queue.txt').exists():
            continue                                    # iteration still being written
        df = pd.read_pickle(pkl)
        for c in ('protein_rmsds', 'ligand_rmsds', 'ligand_plddts', 'protein_plddts'):
            df[c] = pd.to_numeric(df.get(c), errors='coerce')
        keep = ((df.protein_rmsds < PAPER['protein_rmsd']) & (df.ligand_rmsds < PAPER['ligand_rmsd'])
                & (df.ligand_plddts > PAPER['ligand_plddt'])
                & (df.protein_plddts > PAPER['protein_plddt'])).fillna(False)
        sub = df[keep]
        # The per-iteration pickle has no source_path column -- harvest_global_top.py:68-70 derives
        # it from `rosetta_paths` (which holds the Boltz model path) and takes the design stem from
        # that path's parent directory. Match that exactly, or the two tools disagree about which
        # file a design is.
        if 'rosetta_paths' not in sub:
            continue
        # highest ligand pLDDT first, so a --budget cut leaves the tail unrelaxed, not the head
        sub = sub.sort_values('ligand_plddts', ascending=False)
        for _, r in sub.iterrows():
            src = Path(str(r['rosetta_paths']))
            if not src.is_file():
                continue
            row = index.get(str(src))
            if row and valid(row, src) and not do_all:
                continue
            out.append((it, src.parent.name, src))
    return out


def acquire(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)          # raises if dead
            return False
        except (ValueError, ProcessLookupError, OSError):
            lock.unlink(missing_ok=True)      # stale: holder is gone
            return acquire(lock)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(os.environ.get('CHLIDE_RUN_DIR') or Path(__file__).resolve().parent.parent))
    ap.add_argument('--iteration', type=int, default=None)
    ap.add_argument('--all', action='store_true', help='re-relax even cached designs')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--budget', type=int, default=0, help='max designs this invocation (0 = no cap)')
    ap.add_argument('--nice', type=int, default=19)
    ap.add_argument('--params', type=Path, default=None)
    ap.add_argument('--his-index', type=int, default=85)
    ap.add_argument('--r0', type=float, default=2.15)
    a = ap.parse_args()

    run = a.run_dir
    root = run / 'results' / 'relaxed_pool'
    root.mkdir(parents=True, exist_ok=True)
    idx_path = root / 'index.csv'
    params = a.params or (run / 'input' / 'lig_from_input_pdb.params')
    if not params.is_file():
        print(f'no ligand params at {params} -- has the campaign written its helper files yet?')
        return 1

    index = load_index(idx_path)
    todo = pending(run, index, a.iteration, a.all)
    if a.budget:
        todo = todo[:a.budget]
    if not todo:
        print(f'nothing pending ({len(index)} designs already in the index)')
        return 0

    lock = root / '.relax.lock'
    if not acquire(lock):
        print('another relax_pool holds the lock; exiting')
        return 0

    try:
        by_iter: dict[int, list[Path]] = {}
        for it, _stem, src in todo:
            by_iter.setdefault(it, []).append(src)
        print(f'{len(todo)} design(s) pending across {len(by_iter)} iteration(s), '
              f'{a.workers} workers at nice {a.nice}')

        new = existing = 0
        write_header = not idx_path.is_file()
        with idx_path.open('a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
            if write_header:
                w.writeheader()
            for it in sorted(by_iter):
                paths = by_iter[it]
                out_dir = root / f'iter_{it}'      # the /iter_N/ component is an API, see docstring
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = (['nice', f'-n{a.nice}', 'ionice', '-c3', ROSETTA_PY,
                        str(run / 'tools' / 'rosetta_relax.py')]
                       + [str(p) for p in paths]
                       + ['--params', str(params), '--out-dir', str(out_dir),
                          '--workers', str(a.workers), '--his-index', str(a.his_index),
                          '--r0', str(a.r0)])
                t0 = time.time()
                # stream: rosetta_relax prints one JSON per design with flush=True, so a kill
                # partway through still leaves every completed design recorded
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line.startswith('{'):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = Path(rec.get('input', ''))
                    stem = re.sub(r'(_relaxed)+$', '', src.stem)
                    rec.update(source_path=str(src), src_stamp=stamp(src), iteration=it,
                               stem=stem, name=f'iter{it}_{stem}',
                               relaxed_path=rec.get('output', ''), relax_version=RELAX_VERSION)
                    w.writerow(rec)
                    fh.flush()
                    new += 1 if rec.get('ok') else 0
                proc.wait()
                ok = sum(1 for _ in paths)
                print(f'  iter {it}: {len(paths)} design(s) in {time.time() - t0:.0f}s '
                      f'-> {out_dir}')
        print(f'\n{new} relaxed successfully; index now {len(load_index(idx_path))} rows')
        print(f'-> {idx_path}')
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
