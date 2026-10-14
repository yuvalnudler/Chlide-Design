#!/usr/bin/env python
"""Collect the best designs of a NISE campaign into one place.

NISE keeps no cross-iteration record of its best designs. Each iteration writes a
`backbone_queue.txt` holding the poses that seed the *next* expansion -- a working queue, not a
ranking: it is overwritten every round, holds only top_k entries, and via
`keep_best_generator_backbone` can contain a lower-scoring pose. So after a campaign the good
designs are scattered across tens of thousands of prediction directories with nothing pointing
at them. This walks the whole run and writes a ranked, self-describing harvest.

Ranking follows Polizzi et al. 2025 (Fig. E7): keep designs passing protein Ca RMSD < 1.5 A,
ligand RMSD < 1.5 A, ligand pLDDT > 0.8 and protein pLDDT > 0.8, then order by ligand pLDDT --
the quantity NISE itself maximises.

Mg geometry is reported but deliberately NOT ranked on. Boltz models the axial His NE2-Mg contact
as non-bonded and floors it at ~2.94 A (real coordination is ~2.1 A); measured across 11
iterations of the 6HRK run that floor never moved, so ranking on it would be ranking on an
artifact. Mg-to-N4-centroid IS meaningful -- it says whether the metal stayed in the macrocycle.

Usage:
    python tools/harvest_global_top.py [--run-dir DIR] [--top 50] [--out results/global_top]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import prody as pr

pr.confProDy(verbosity='none')

# Polizzi et al. 2025 Fig. E7, the four stages computable from NISE output alone.
PAPER = {'protein_rmsd': 1.5, 'ligand_rmsd': 1.5, 'ligand_plddt': 0.8, 'protein_plddt': 0.8}
LIGAND_RMSD_STRICT = 1.0  # the main text's tighter value, reported as a flag column
COORD_LO, COORD_HI = 2.0, 2.6   # physical Mg-N(His) window, same as the in-loop relax gate


def load_relax_index(run_dir) -> dict:
    """Rows written by tools/relax_pool.py, keyed by the design's raw Boltz output path.

    Only rows whose relaxed structure is still on disk count -- a half-deleted pool must degrade
    to 'not relaxed yet' rather than pointing the cascade at a missing file.
    """
    import csv as _csv
    p = Path(run_dir) / 'results' / 'relaxed_pool' / 'index.csv'
    if not p.is_file():
        return {}
    out = {}
    with p.open() as fh:
        for r in _csv.DictReader(fh):
            if r.get('ok') == 'True' and r.get('relaxed_path') and Path(r['relaxed_path']).is_file():
                out[r['source_path']] = r
    return out


def _num(row, key, cast):
    try:
        return cast(row[key])
    except (KeyError, TypeError, ValueError):
        return None

# Axial His, 0-based positional index; stable across Boltz's renumbering to 1..N.
HIS_RESINDEX = 84
MG_NAMES = ('MG1', 'MG')
N4_SETS = (('NA1', 'NB1', 'NC1', 'ND1'), ('NA', 'NB', 'NC', 'ND'))

NOT_COMPUTED = ('Rosetta packstat>0.55, apo-repack clash count and AF2 interface pAE<14 are NOT '
                'computed here (no pyrosetta / AF2 in this environment), so this is four stages '
                'of the paper cascade, not all of them.')


def load_all(run_dir: Path) -> pd.DataFrame:
    """Every design across every iteration, with its structure path resolved."""
    frames = []
    pattern = str(run_dir / 'input' / 'sampling_dataframes' / 'iter_*_data.pkl')
    for f in sorted(glob.glob(pattern), key=lambda p: int(re.search(r'iter_(\d+)', p).group(1))):
        it = int(re.search(r'iter_(\d+)', f).group(1))
        try:
            d = pd.read_pickle(f)
        except Exception as e:
            print(f'  skipping {os.path.basename(f)}: {e}')
            continue
        d['iteration'] = it
        # `rosetta_paths` holds the Boltz model path; its parent directory name is the design stem.
        d['stem'] = [Path(str(p)).parent.name for p in d['rosetta_paths']]
        d['source_path'] = [str(p) for p in d['rosetta_paths']]
        frames.append(d)
    if not frames:
        raise SystemExit(f'no iteration dataframes under {run_dir}/input/sampling_dataframes')
    return pd.concat(frames, ignore_index=True)


def geometry(pdb_path: str) -> dict:
    """Mg placement in the chlorin, and the axial His contact."""
    out = {'mg_ring_offset': np.nan, 'his_mg': np.nan}
    try:
        st = pr.parsePDB(pdb_path)
        lig = st.select('chid B')
        if lig is None:
            return out
        mg = next((lig.select(f'name {n}') for n in MG_NAMES
                   if lig.select(f'name {n}') is not None), None)
        n4 = next((lig.select('name ' + ' '.join(s)) for s in N4_SETS
                   if lig.select('name ' + ' '.join(s)) is not None
                   and len(lig.select('name ' + ' '.join(s))) == 4), None)
        if mg is None or n4 is None:
            return out
        c = mg.getCoords()[0]
        out['mg_ring_offset'] = float(np.linalg.norm(c - n4.getCoords().mean(0)))
        ne2 = st.select(f'resindex {HIS_RESINDEX} and name NE2')
        if ne2 is not None:
            out['his_mg'] = float(np.linalg.norm(ne2.getCoords()[0] - c))
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--out', type=Path, default=None, help='default: <run-dir>/results/global_top')
    ap.add_argument('--top', type=int, default=50, help='structures to copy (default 50)')
    a = ap.parse_args()
    out = a.out or (a.run_dir / 'results' / 'global_top')
    out.mkdir(parents=True, exist_ok=True)

    print(f'reading {a.run_dir} ...')
    df = load_all(a.run_dir)
    print(f'  {len(df)} designs across {df.iteration.nunique()} iterations')

    passing = ((df['protein_rmsds'] < PAPER['protein_rmsd'])
               & (df['ligand_rmsds'] < PAPER['ligand_rmsd'])
               & (df['ligand_plddts'] > PAPER['ligand_plddt'])
               & (df['protein_plddts'] > PAPER['protein_plddt']))
    df['cascade_pass'] = passing.fillna(False)
    df['ligand_rmsd_strict'] = (df['ligand_rmsds'] < LIGAND_RMSD_STRICT).fillna(False)
    print(f'  {int(df.cascade_pass.sum())} pass the 4-stage cascade '
          f'({100 * df.cascade_pass.mean():.1f}%); '
          f'{int((df.cascade_pass & df.ligand_rmsd_strict).sum())} also under '
          f'{LIGAND_RMSD_STRICT} A ligand RMSD')

    # ---- relaxed pool -------------------------------------------------------------------------
    # relax_pool.py minimises every gate-passing design during the campaign (see that file). Where
    # a relaxed structure exists we gate on its geometry and report it, which moves paper stages 5
    # and 6 out of the terminal cascade and into the harvest.
    #
    # The ORDERING KEY stays ligand_plddt. That is deliberate and worth stating: ligand pLDDT is a
    # Boltz confidence carried in the B-factor column, and pose.dump_pdb does not preserve it, so
    # it cannot be recomputed on a relaxed structure. What the relax buys is a better *gate*, not a
    # better sort key.
    rl = load_relax_index(a.run_dir)
    if rl:
        idx = df['source_path'].map(lambda p: rl.get(str(p)))
        df['relaxed'] = idx.notna()
        for c, key, cast in (('relaxed_path', 'relaxed_path', str), ('relax_score', 'score', float),
                             ('his_mg_relaxed', 'his_mg', float),
                             ('mg_ring_offset_relaxed', 'mg_ring_offset', float),
                             ('ca_rmsd_to_min', 'ca_rmsd', float),
                             ('lig_rmsd_to_min', 'ligand_rmsd', float)):
            df[c] = idx.map(lambda r, k=key, f=cast: _num(r, k, f) if r else None)
        # Paper stages 5-6, now computable here: the minimisation must have succeeded, the metal
        # must still be coordinated after it, and the pose must not have moved far to get there.
        relax_pass = (df['relaxed']
                      & df['his_mg_relaxed'].between(COORD_LO, COORD_HI)
                      & (df['ca_rmsd_to_min'] < PAPER['protein_rmsd'])
                      & (df['lig_rmsd_to_min'] < PAPER['protein_rmsd'])).fillna(False)
        n_rel, n_pass = int(df['relaxed'].sum()), int(relax_pass.sum())
        print(f'  relaxed pool: {n_rel}/{len(df)} designs minimised, {n_pass} pass '
              f'His-Mg {COORD_LO}-{COORD_HI} A + RMSD-to-minimised gates')
        # Designs not yet relaxed keep cascade_pass so metrics.csv does not shrink between polls
        # while the pool catches up; they are simply excluded from the structure copy below.
        df['cascade_pass'] = df['cascade_pass'] & (relax_pass | ~df['relaxed'])
    else:
        df['relaxed'] = False
        print('  relaxed pool: no index yet (run tools/relax_pool.py) -- '
              'ranking on unrelaxed predictions')

    ranked = df[df.cascade_pass].sort_values('ligand_plddts', ascending=False).reset_index(drop=True)

    # Geometry only for what we rank and report -- parsing every design would dominate runtime.
    # Cached by source path across runs: this is re-run after every iteration, and without a cache
    # the cost grows linearly with the campaign (re-parsing thousands of unchanged structures).
    cache_file = out / '.geometry_cache.csv'
    cache = {}
    if cache_file.exists():
        try:
            c = pd.read_csv(cache_file)
            cache = {r.source_path: (r.mg_ring_offset, r.his_mg) for r in c.itertuples()}
        except Exception:
            cache = {}
    rows, fresh = [], 0
    for p in ranked['source_path']:
        if p in cache:
            rows.append({'mg_ring_offset': cache[p][0], 'his_mg': cache[p][1]})
        else:
            g = geometry(p)
            rows.append(g)
            cache[p] = (g['mg_ring_offset'], g['his_mg'])
            fresh += 1
    print(f'  Mg geometry for {len(ranked)} designs ({fresh} newly parsed, '
          f'{len(ranked) - fresh} from cache)')
    ranked = pd.concat([ranked, pd.DataFrame(rows)], axis=1)
    try:
        pd.DataFrame([{'source_path': k, 'mg_ring_offset': v[0], 'his_mg': v[1]}
                      for k, v in cache.items()]).to_csv(cache_file, index=False)
    except Exception as e:
        print(f'  warning: could not write geometry cache: {e}')

    cols = ['iteration', 'stem', 'ligand_plddts', 'protein_plddts', 'iptms',
            'ligand_rmsds', 'protein_rmsds', 'laser_nll', 'laser_bs_nll',
            'mg_ring_offset', 'his_mg', 'ligand_rmsd_strict', 'sequences', 'source_path',
            # post-relax columns from the pool; source_path still points at the raw prediction so
            # burial_rescore.py / final_shortlist.py / paper_figures.py keep working unchanged
            'relaxed', 'relaxed_path', 'relax_score', 'his_mg_relaxed',
            'mg_ring_offset_relaxed', 'ca_rmsd_to_min', 'lig_rmsd_to_min']
    cols = [c for c in cols if c in ranked.columns]
    rename = {'ligand_plddts': 'ligand_plddt', 'protein_plddts': 'protein_plddt',
              'iptms': 'iptm', 'ligand_rmsds': 'ligand_rmsd', 'protein_rmsds': 'protein_rmsd',
              'sequences': 'sequence'}

    metrics = out / 'metrics.csv'
    with metrics.open('w') as f:
        f.write(f'# NISE harvest of {a.run_dir.name}: {len(ranked)} designs passing the '
                f'Polizzi 2025 Fig. E7 cascade, ranked by ligand pLDDT.\n')
        f.write(f'# Cascade: protein RMSD < {PAPER["protein_rmsd"]}, ligand RMSD < '
                f'{PAPER["ligand_rmsd"]}, ligand pLDDT > {PAPER["ligand_plddt"]}, '
                f'protein pLDDT > {PAPER["protein_plddt"]}\n')
        f.write(f'# {NOT_COMPUTED}\n')
        f.write('# ORDERING is by ligand pLDDT (the campaign objective). It cannot be recomputed\n'
                '# on a minimised structure -- it is a Boltz confidence in the B-factor column and\n'
                '# pose.dump_pdb does not preserve it -- so the relax improves the GATES, not the\n'
                '# sort key. his_mg/mg_ring_offset are pre-relax; the _relaxed columns are post.\n')
        ranked[cols].rename(columns=rename).to_csv(f, index=False, float_format='%.4f')
    print(f'  wrote {metrics}')

    allcsv = out / 'all_designs.csv'
    acols = [c for c in ['iteration', 'stem', 'ligand_plddts', 'protein_plddts', 'iptms',
                         'ligand_rmsds', 'protein_rmsds', 'laser_nll', 'laser_bs_nll',
                         'ligand_is_buried', 'cascade_pass', 'sequences', 'source_path']
             if c in df.columns]
    df[acols].rename(columns=rename).to_csv(allcsv, index=False, float_format='%.4f')
    print(f'  wrote {allcsv} ({len(df)} rows)')

    # Clear the previous ranking first. Without this the directory accumulates every ranking the
    # campaign has ever produced -- v4 ended with 682 rank*.pdb files for 50 slots, which is how a
    # cascade run ended up consuming structures that were no longer in the top 50.
    # final_shortlist.py:123-124 already does this; match it.
    for old in out.glob('rank*.pdb'):
        old.unlink()

    n = n_rel = 0
    for i, r in ranked.head(a.top).iterrows():
        # prefer the minimised structure: it is protonated and energy-minimised, which is what the
        # cascade and any visual inspection actually want
        rp = r.get('relaxed_path')
        src = Path(rp) if isinstance(rp, str) and rp and Path(rp).is_file() else Path(r['source_path'])
        if not src.is_file():
            print(f'  WARNING missing structure: {src}')
            continue
        suffix = '_relaxed' if src != Path(r['source_path']) else ''
        dst = out / f'rank{i + 1:02d}_iter{int(r["iteration"])}_{r["stem"]}{suffix}.pdb'
        shutil.copy(src, dst)
        n += 1
        n_rel += 1 if suffix else 0
    print(f'  copied {n} structures to {out} ({n_rel} minimised, {n - n_rel} raw predictions)')

    if len(ranked):
        b = ranked.iloc[0]
        print(f'\nbest: iter {int(b.iteration)} {b.stem}  ligand pLDDT {b.ligand_plddts:.3f}  '
              f'protein pLDDT {b.protein_plddts:.3f}  ligand RMSD {b.ligand_rmsds:.2f}  '
              f'protein RMSD {b.protein_rmsds:.2f}')
        if np.isfinite(b.get('mg_ring_offset', np.nan)):
            print(f'      Mg {b.mg_ring_offset:.2f} A off the N4 centroid, '
                  f'His NE2-Mg {b.his_mg:.2f} A')
    return 0


if __name__ == '__main__':
    sys.exit(main())
