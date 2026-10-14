#!/usr/bin/env python
"""Terminal filter cascade for a NISE chlorophyllide campaign, following Polizzi et al. 2025.

The in-loop gates cover the first four stages of the paper's Fig. E7 cascade (self-consistency
protein/ligand RMSD, ligand and protein pLDDT). This tool adds the rest, which the paper applies
once to designs pooled across all iterations rather than per round:

  5. Rosetta energy minimisation (ref2015) with the axial Mg-N(His) restraint
  6. prediction -> minimised RMSD < 1.5 A, protein and ligand
  7. packstat > 0.55
  8. apo-repack clashes == 0  ("binding site preorganisation")
  9. buried unsatisfied polar atoms (BUNs) -- the paper's final ranking

Stage 8 follows the paper's description literally: minimise WITHOUT the ligand so binding-site
sidechains relax into the empty pocket, then put the ligand back by superimposing the bound
structure onto the unbound one over binding-site residues, and count sidechain heavy atoms within
2 A of any ligand heavy atom. A design whose pocket collapses when the ligand is absent is not
preorganised, however good its predicted complex looks.

The AF3 two-copy interface-pAE stage (aggregation proxy, keep pAE > 14) is separate and GPU-bound:
see af3_interface_pae.py. Run that last, on the survivors.

CPU only. Runs in the Rosetta env:
    /home/yigall/mambaforge/envs/Rosetta/bin/python tools/polizzi_cascade.py \
        --run-dir <run> [--limit N] [--workers 8]
Writes <run>/results/cascade/{relaxed/*.pdb, cascade_metrics.csv} and prints a stage-by-stage
survivor table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np

# Paper thresholds (Fig. E7 and Methods).
PACKSTAT_MIN = 0.55
RMSD_TO_MIN_MAX = 1.5      # prediction -> minimised, protein and ligand
APO_CLASH_MAX = 0          # sidechain heavy atoms within 2.0 A of a ligand heavy atom
APO_CLASH_DIST = 2.0
COORD_WINDOW = (2.0, 2.6)  # axial His NE2 - Mg
COORD_R0, COORD_TOL = 2.15, 0.25
HIS_INDEX = 85             # 1-based; resindex 84 + 1
BINDING_SITE_CUTOFF = 5.0

_INIT = False


def _init(params):
    global _INIT
    if _INIT:
        return
    import pyrosetta
    pyrosetta.init(f'-mute all -extra_res_fa {params} -ex1 -ex2', silent=True)
    _INIT = True


def _xyz(res, i):
    v = res.xyz(i)
    return np.array([v.x, v.y, v.z])


def _lig_heavy(pose, li):
    r = pose.residue(li)
    return np.array([_xyz(r, j) for j in range(1, r.natoms() + 1)
                     if r.atom_type(j).element().strip() != 'H'])


def _ca(pose, n):
    return np.array([_xyz(pose.residue(i), pose.residue(i).atom_index('CA')) for i in range(1, n + 1)])


def _kabsch(P, Q):
    """Rotation+translation putting P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def _sasa_of(pdb_path) -> dict:
    """Shrake-Rupley ligand burial, from tools/sasa_rank.py so the constants live in one place."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_sr', str(Path(__file__).parent / 'sasa_rank.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.score(Path(pdb_path))
    if not s:
        return {}
    return {k: s[k] for k in ('buried_frac', 'buried_frac_core', 'buried_frac_apolar',
                              'buried_frac_propionate', 'sasa_mg_A2', 'sasa_propionate_A2')
            if k in s}


def process(pdb: str, params: str, out_dir: str, pre: dict | None = None) -> dict:
    """Run stages 5-9 on one design. Never raises.

    `pre` is a row from results/relaxed_pool/index.csv. When given, `pdb` is already the minimised
    structure, so stage 5 (the holo FastRelax) is skipped and stages 5-6 are read from that row.
    Stage 6 MUST come from the index: prediction->minimised RMSD measured against an already
    minimised input degenerates to ~0 and stops meaning anything.
    """
    _it = re.search(r'/iter_(\d+)/', str(pdb))
    _stem = re.sub(r'(_relaxed)+$', '', Path(pdb).stem)
    rec = {'input': pdb, 'stem': _stem,
           'iteration': int(_it.group(1)) if _it else -1,
           'name': f'iter{_it.group(1)}_{_stem}' if _it else _stem, 'ok': False}
    try:
        _init(params)
        from pyrosetta import pose_from_file
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType
        from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
        from pyrosetta.rosetta.core.scoring.func import FlatHarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.core.scoring.packstat import compute_packing_score
        from pyrosetta.rosetta.protocols.simple_filters import BuriedUnsatHbondFilter
        from pyrosetta.rosetta.protocols.grafting import delete_region

        t0 = time.time()
        pose = pose_from_file(pdb)
        li = pose.total_residue()
        n_prot = li - 1
        lig = pose.residue(li)
        mg = next((i for i in range(1, lig.natoms() + 1)
                   if lig.atom_name(i).strip() == 'MG1'), None)
        if mg is None:
            rec['error'] = 'no MG1'
            return rec
        if pose.residue(HIS_INDEX).name3() != 'HIS':
            rec['error'] = f'residue {HIS_INDEX} is {pose.residue(HIS_INDEX).name3()}'
            return rec
        ne2 = pose.residue(HIS_INDEX).atom_index('NE2')

        def his_mg(p):
            return float((p.residue(HIS_INDEX).xyz('NE2') - p.residue(li).xyz(mg)).norm())

        sf = ScoreFunctionFactory.create_score_function('ref2015')
        sf.set_weight(ScoreType.atom_pair_constraint, 1.0)
        pose.add_constraint(AtomPairConstraint(AtomID(ne2, HIS_INDEX), AtomID(mg, li),
                                               FlatHarmonicFunc(COORD_R0, 0.2, COORD_TOL)))
        rec['his_mg_before'] = his_mg(pose)
        ca0, lig0 = _ca(pose, n_prot), _lig_heavy(pose, li)

        # --- stage 5: minimise the complex -------------------------------------------------
        if pre is None:
            fr = FastRelax(sf, 1)
            fr.constrain_relax_to_start_coords(True)
            fr.apply(pose)
            rec['his_mg'] = his_mg(pose)
            rec['score'] = float(sf(pose))
            # --- stage 6: how far the prediction moved -------------------------------------
            rec['ca_rmsd_to_min'] = float(np.sqrt(((ca0 - _ca(pose, n_prot)) ** 2).sum(1).mean()))
            rec['lig_rmsd_to_min'] = float(np.sqrt(((lig0 - _lig_heavy(pose, li)) ** 2).sum(1).mean()))
        else:
            # already minimised by relax_pool.py during the campaign; re-measuring stage 6 here
            # would compare a minimised pose against itself
            rec['pre_relaxed'] = True
            rec['his_mg'] = his_mg(pose)
            rec['score'] = float(sf(pose))
            for dst, key in (('ca_rmsd_to_min', 'ca_rmsd'), ('lig_rmsd_to_min', 'ligand_rmsd'),
                             ('his_mg_before', 'his_mg_before')):
                try:
                    rec[dst] = float(pre[key])
                except (KeyError, TypeError, ValueError):
                    rec[dst] = float('nan')

        # --- stage 7: core packing ----------------------------------------------------------
        try:
            rec['packstat'] = float(compute_packing_score(pose, 1))
        except Exception as e:
            rec['packstat'] = float('nan')
            rec['packstat_error'] = str(e)[:80]

        # --- stage 9: buried unsatisfied polar atoms ----------------------------------------
        # The paper's headline ranking is the LIGAND's buried unsatisfied polars, but its
        # reranking uses "a linear combination of the number of buried, non-H-bonded polar atoms
        # of the protein and ligand" -- which is what put their own EPIC design 1st. Report all
        # three; combined is an unweighted sum, the simplest reading of "linear combination".
        from pyrosetta.rosetta.core.select.residue_selector import (
            ResidueIndexSelector, ResidueSpanSelector, LayerSelector)
        try:
            rec['buns_total'] = float(BuriedUnsatHbondFilter().compute(pose))
        except Exception as e:
            rec['buns_total'] = float('nan')
            rec['buns_error'] = f'{type(e).__name__}: {str(e)[:60]}'
        for key, sel in (('buns_ligand', ResidueIndexSelector(str(li))),
                         ('buns_protein', ResidueSpanSelector(1, n_prot))):
            try:
                f = BuriedUnsatHbondFilter()
                f.set_residue_selector(sel)
                rec[key] = float(f.compute(pose))
            except Exception as e:
                rec[key] = float('nan')
                rec.setdefault('buns_error', f'{key}: {type(e).__name__}')
        if np.isfinite(rec.get('buns_ligand', np.nan)) and np.isfinite(rec.get('buns_protein', np.nan)):
            rec['buns_combined'] = rec['buns_ligand'] + rec['buns_protein']
        else:
            rec['buns_combined'] = rec.get('buns_total', float('nan'))
        # keep the old column name so earlier outputs stay comparable
        rec['buns'] = rec.get('buns_ligand', rec.get('buns_total', float('nan')))

        # --- SASA: how much of the ligand is actually desolvated -------------------------------
        # New in v5. Every other burial number in this project comes from NISE's alpha-shape test,
        # which casts a ray at a mesh through the CA trace -- it knows nothing about sidechains and
        # nothing about solvent, so an atom in a water-filled cavity counts as buried, and it is
        # binary per atom so it cannot rank. SASA asks the physical question instead.
        # On v4's 25 shortlisted designs SASA burial and BUNs Score were uncorrelated (r = -0.001),
        # so this is independent evidence, not a restatement of the ranking.
        try:
            rec.update(_sasa_of(rec.get('output') or pdb))
        except Exception as e:
            rec['sasa_error'] = f'{type(e).__name__}: {str(e)[:60]}'

        # --- buried charged residues ---------------------------------------------------------
        # The paper removed designs containing them outright. In-loop we only block D/E/K/R in the
        # buried FIRST SHELL (disable_charged_fs), which is narrower, so this catches the rest.
        try:
            layer = LayerSelector()
            layer.set_layers(True, False, False)          # core only
            core = layer.apply(pose)
            charged = [i for i in range(1, n_prot + 1)
                       if core[i] and pose.residue(i).name3() in ('ASP', 'GLU', 'LYS', 'ARG')]
            rec['buried_charged'] = len(charged)
            rec['buried_charged_residues'] = ','.join(
                f'{pose.residue(i).name3()}{i}' for i in charged)
        except Exception as e:
            rec['buried_charged'] = -1
            rec['buried_charged_error'] = f'{type(e).__name__}: {str(e)[:60]}'

        # --- stage 8: apo repack, then put the ligand back ----------------------------------
        # Paper: minimise WITHOUT the ligand, superimpose bound onto unbound over binding-site
        # residues, reintroduce the ligand, count sidechain atoms within 2 A of it.
        try:
            lig_xyz = _lig_heavy(pose, li)
            bs = []
            for i in range(1, n_prot + 1):
                r = pose.residue(i)
                for j in range(1, r.natoms() + 1):
                    if r.atom_type(j).element().strip() == 'H':
                        continue
                    if np.min(np.linalg.norm(lig_xyz - _xyz(r, j), axis=1)) < BINDING_SITE_CUTOFF:
                        bs.append(i)
                        break
            apo = pose.clone()
            delete_region(apo, li, li)                 # remove the ligand
            apo_sf = ScoreFunctionFactory.create_score_function('ref2015')
            fr2 = FastRelax(apo_sf, 1)
            fr2.constrain_relax_to_start_coords(True)
            fr2.apply(apo)
            # superimpose bound -> unbound over binding-site CA, then measure in apo frame
            B = np.array([_xyz(pose.residue(i), pose.residue(i).atom_index('CA')) for i in bs])
            A = np.array([_xyz(apo.residue(i), apo.residue(i).atom_index('CA')) for i in bs])
            R, t = _kabsch(B, A)
            lig_in_apo = (R @ lig_xyz.T).T + t
            clashes = 0
            for i in bs:
                r = apo.residue(i)
                for j in range(1, r.natoms() + 1):
                    if r.atom_type(j).element().strip() == 'H' or j <= r.last_backbone_atom():
                        continue
                    if np.min(np.linalg.norm(lig_in_apo - _xyz(r, j), axis=1)) < APO_CLASH_DIST:
                        clashes += 1
            rec['apo_clashes'] = int(clashes)
            rec['binding_site_residues'] = len(bs)
        except Exception as e:
            rec['apo_clashes'] = -1
            rec['apo_error'] = f'{type(e).__name__}: {str(e)[:60]}'

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stem = re.sub(r'(_relaxed)+$', '', Path(pdb).stem)
        it = re.search(r'/iter_(\d+)/', str(pdb))
        if it:
            stem = f'iter{it.group(1)}_{stem}'
        out = str(Path(out_dir) / (stem + '_min.pdb'))
        pose.dump_pdb(out)
        rec['output'] = out
        rec['seconds'] = round(time.time() - t0, 1)
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001 - one bad design must not kill the batch
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def _worker(a):
    return process(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--params', type=Path, default=None)
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--from-metrics', action='store_true',
                    help='take every design in results/global_top/metrics.csv (the full '
                         'cascade-passing population) instead of the rank*.pdb snapshots, which '
                         'are an artefact of the harvest refreshing top-50 each iteration')
    ap.add_argument('--out-root', type=Path, default=None,
                    help='default <run>/results/cascade; point elsewhere to keep a corrected '
                         'subset separate from the full-population metrics')
    ap.add_argument('--stems-from', type=Path, default=None,
                    help='file of design stems (one per line); process only source designs whose '
                         'stem matches, used to re-do a subset with correct naming')
    ap.add_argument('--workers', type=int, default=8,
                    help='keep modest while a campaign is running')
    ap.add_argument('--pre-relaxed', action='store_true',
                    help='reuse results/relaxed_pool minimised structures; skips the holo FastRelax')
    ap.add_argument('--no-resume', action='store_true',
                    help='ignore cascade_partial.jsonl and re-score every design from scratch')
    a = ap.parse_args()

    params = a.params or (a.run_dir / 'input' / 'lig_from_input_pdb.params')
    if a.from_metrics:
        import csv as _csv
        mfile = a.run_dir / 'results' / 'global_top' / 'metrics.csv'
        with mfile.open() as fh:
            rows = list(_csv.DictReader(l for l in fh if not l.startswith('#')))
        seen, src = set(), []
        for r in rows:
            sp = r.get('source_path')
            if sp and sp not in seen and Path(sp).is_file():
                seen.add(sp); src.append(Path(sp))
        print(f'{len(src)} designs from {mfile.name} ({len(rows)} rows)')
    else:
        src = sorted((a.run_dir / 'results' / 'global_top').glob('rank*.pdb'))
    if a.stems_from and a.stems_from.is_file():
        want = {l.strip() for l in a.stems_from.read_text().splitlines() if l.strip()}
        src = [p for p in src if re.sub(r'(_relaxed)+$', '', p.stem) in want]
        print(f'  filtered to {len(src)} designs matching {len(want)} stems')
    if a.limit:
        src = src[:a.limit]
    if not src:
        print(f'no designs in {a.run_dir}/results/global_top'); return 1
    out_root = a.out_root or (a.run_dir / 'results' / 'cascade')
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'{len(src)} designs, {a.workers} workers, params {params}')

    # --pre-relaxed: consume relax_pool.py's minimised structures instead of minimising again.
    # Saves the holo FastRelax on every design -- roughly 40% of this tool's CPU -- and, more
    # usefully, means stage 5 was already amortised across the campaign instead of blocking here
    # for hours at the end. The apo minimisation in stage 8 is unavoidable and becomes the cost.
    pre_by_src = {}
    if a.pre_relaxed:
        import csv as _csv
        ip = a.run_dir / 'results' / 'relaxed_pool' / 'index.csv'
        if ip.is_file():
            with ip.open() as fh:
                for r in _csv.DictReader(fh):
                    if r.get('ok') == 'True' and Path(r.get('relaxed_path', '')).is_file():
                        pre_by_src[r['source_path']] = r
            hit = sum(1 for p in src if str(p) in pre_by_src)
            print(f'  pre-relaxed: {hit}/{len(src)} designs have a minimised structure')
            # swap in the minimised file where we have one
            src = [Path(pre_by_src[str(p)]['relaxed_path']) if str(p) in pre_by_src else p
                   for p in src]
        else:
            print(f'  --pre-relaxed given but no {ip}; minimising from scratch')

    rev = {v['relaxed_path']: v for v in pre_by_src.values()}
    jobs = [(str(p), str(params), str(out_root / 'minimized'), rev.get(str(p))) for p in src]

    # ---- streaming partial results, so an interruption costs only the in-flight batch --------
    # This run accumulated every scored row in memory and wrote the CSV once at the end. When the
    # cascade was SIGKILLed at 2770/11201 on 6 Oct, all 2770 rows died with the process -- the
    # 2774 PDBs left in minimized/ are outputs, not scores, and re-deriving the scores costs the
    # same as redoing the designs. Each row is now appended as it completes and re-read on
    # restart, so only designs not already scored are re-run.
    partial = out_root / 'cascade_partial.jsonl'
    done_by_input = {}
    if partial.is_file() and not a.no_resume:
        with partial.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:          # noqa: BLE001  a torn last line after a kill
                    continue
                if r.get('input'):
                    done_by_input[r['input']] = r
        if done_by_input:
            print(f'  RESUMING: {len(done_by_input)} designs already scored in {partial.name}')
    recs = list(done_by_input.values())
    jobs = [j for j in jobs if j[0] not in done_by_input]
    if not jobs:
        print('  every design already scored; writing the final CSV from the partial')

    pf = partial.open('a')
    try:
        if a.workers > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context('spawn').Pool(a.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                    recs.append(r)
                    pf.write(json.dumps(r, default=str) + '\n'); pf.flush()
                    if i % 10 == 0 or i == len(jobs):
                        print(f'  {i}/{len(jobs)}', flush=True)
        else:
            for j in jobs:
                r = _worker(j)
                recs.append(r)
                pf.write(json.dumps(r, default=str) + '\n'); pf.flush()
    finally:
        pf.close()

    ok = [r for r in recs if r.get('ok')]
    cols = ['name', 'iteration', 'stem', 'input', 'his_mg_before', 'his_mg', 'score', 'ca_rmsd_to_min', 'lig_rmsd_to_min',
            'packstat', 'apo_clashes', 'buried_charged', 'buns_ligand', 'buns_protein',
            'buried_frac', 'buried_frac_core', 'buried_frac_apolar', 'sasa_mg_A2',
            'buns_combined', 'buns_total', 'buried_charged_residues', 'binding_site_residues',
            'seconds', 'output']
    csv_path = out_root / 'cascade_metrics.csv'
    with csv_path.open('w', newline='') as f:
        f.write(f'# Polizzi et al. 2025 terminal cascade, stages 5-9. Thresholds: packstat > {PACKSTAT_MIN}, '
                f'prediction->minimised RMSD < {RMSD_TO_MIN_MAX} A (protein and ligand), '
                f'apo-repack clashes == {APO_CLASH_MAX}, His-Mg {COORD_WINDOW[0]}-{COORD_WINDOW[1]} A.\n')
        f.write('# Ranked by buried unsatisfied polar atoms (BUNs), the paper\'s final ranking. '
                'AF3 two-copy interface pAE > 14 is applied separately by af3_interface_pae.py.\n')
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(ok, key=lambda x: (x.get('buns_combined', 9e9))):
            w.writerow(r)

    def n(pred):
        return sum(1 for r in ok if pred(r))
    print(f'\nstage survivors (of {len(recs)} attempted, {len(ok)} relaxed successfully):')
    stages = [
        ('minimised ok', lambda r: True),
        (f'His-Mg {COORD_WINDOW[0]}-{COORD_WINDOW[1]} A', lambda r: COORD_WINDOW[0] <= r.get('his_mg', 9) <= COORD_WINDOW[1]),
        (f'protein RMSD->min < {RMSD_TO_MIN_MAX}', lambda r: r.get('ca_rmsd_to_min', 9) < RMSD_TO_MIN_MAX),
        (f'ligand RMSD->min < {RMSD_TO_MIN_MAX}', lambda r: r.get('lig_rmsd_to_min', 9) < RMSD_TO_MIN_MAX),
        (f'packstat > {PACKSTAT_MIN}', lambda r: r.get('packstat', 0) > PACKSTAT_MIN),
        (f'apo clashes == {APO_CLASH_MAX}', lambda r: r.get('apo_clashes', -1) == APO_CLASH_MAX),
        ('no buried charged residues', lambda r: r.get('buried_charged', -1) == 0),
    ]
    cum = list(ok)
    for label, pred in stages:
        cum = [r for r in cum if pred(r)]
        print(f'  {label:34s} {n(pred):5d} alone   {len(cum):5d} cumulative')
    if cum:
        best = sorted(cum, key=lambda x: x.get('buns_combined', 9e9))[:5]
        print('\ntop survivors by BUNs (lower is better):')
        for r in best:
            print('  %-34s BUNs(lig/prot/comb) %4.0f/%4.0f/%4.0f  packstat %.3f  His-Mg %.2f  score %.0f'
                  % (r['name'][:34], r.get('buns_ligand', float('nan')),
                     r.get('buns_protein', float('nan')), r.get('buns_combined', float('nan')),
                     r.get('packstat', float('nan')), r.get('his_mg', float('nan')),
                     r.get('score', float('nan'))))
        with (out_root / 'survivors.txt').open('w') as f:
            for r in sorted(cum, key=lambda x: x.get('buns_combined', 9e9)):
                f.write(r['output'] + '\n')
        print(f'\n{len(cum)} survivors listed in {out_root}/survivors.txt '
              f'-> feed to af3_interface_pae.py')
    print(f'metrics: {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
