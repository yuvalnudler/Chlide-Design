# Lessons

Every entry cost real time or a real campaign. Each states the symptom, the cause, the fix, and
the measurement that settles it. Ordered by how expensive they were.

Source campaigns: `nise_6hrk_his117` (run 1) → `nise_bdfp44_his85` (v2) → `nise_bdfp44_relax` (v3)
→ `nise_bdfp44_coord` (v4) → `nise_bdfp44_v5` → `nise_bdfp44_v6b` → `nise_bdfp44_v7` (Boltz tip)
and `nise_bdfp44_v8` (AF3, internally "v6a" — a fork, not a successor).

---

## 1. Hardcoded run directories are the project's single most expensive failure mode

**Five incidents.** A campaign directory is made by copying the previous one, so every tool arrives
with the *previous* campaign's absolute path baked in.

- `burial_rescore.py` defaulted to v4 while run from v5
- `af3_interface_pae.py` read v4's `cascade_fixed/`
- `build_ligand.sh` had `RUN=.../nise_bdfp44_v5` hardcoded and **rebuilt v5's ligand component
  while being run from v6b**, overwriting a finished campaign's archived artefact
- the v6b driver was copied with `input_dir = .../nise_bdfp44_v5/input`. NISE writes
  `sampled_backbones/` and `sampling_dataframes/` *under* `input_dir`, so v6b would have written
  its iterations into v5's closed archive and folded them against v5's untagged component
- `v7/tools/finish_at_iter.sh` carried `pkill -f run_nise_bdfp44_v6b.py` — running it would have
  killed the **live** v6b campaign

**Fix.** Derive, never hardcode:
```python
run_dir = Path(__file__).resolve().parent.parent
```
```bash
RUN=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DRIVER=$(cd "$RUN" && ls run_nise_*.py | head -1)   # a literal here pkills the WRONG campaign
```
Test it **from a different cwd** — that is the case a literal survives. The launcher greps the
driver for the derived form as a pre-flight check.

**In this repo this class of bug is structurally impossible:** code lives once, the run directory
is a CLI argument.

---

## 2. Boltz builds ZERO chiral constraints unless every atom carries `_CIPRank`

`compute_chiral_atom_constraints` is gated on `all(atom.HasProp("_CIPRank"))`
(`boltz/data/parse/schema.py:341`). v5's component had no CIP tags, so Boltz built **no** chiral
constraints and the 13² centre floated free.

**Measured:** over a seeded random sample of 400 v5 iteration-15 predictions, **only 32% had the
correct S configuration**. All 8 designs on v5's order list were **R** — the wrong epimer.

Worse, the selection *enriched* the error: the pool was 32% S but the shortlist only 21% and the
top 25 just 12%. A chirality-blind RMSD gives an inverted ligand a **lower** score (0.31 Å vs
1.52 Å, because a mirrored atom match fits more closely), so the cascade preferred wrong epimers.

**Fix.** Tag the component before pickling, and **clear the metal's tag**:
```python
Chem.AssignStereochemistryFrom3D(mol)
Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
for a in mol.GetAtoms():
    if a.GetSymbol() in ('Mg','Fe','Zn','Mn','Co','Ni','Cu'):
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
```
RDKit perceives a 4-coordinate Mg as an SP3 tetrahedral centre: **16 constraints with the metal
in, 12 with it cleared.** Constraining it fights the square-planar coordination.

**Result:** v6b, v7 and v8-with-Boltz all reached **100% S at every iteration**.

---

## 3. `rmsd_use_chirality = True` freezes the campaign — it does not filter epimers

The obvious fix for #2 is to make the self-consistency RMSD chirality-aware. **It rejects the
correct epimer too.**

Measured on v5's two iteration-15 populations:

| centre | reference pose | popA (inverted) | popB (correct) |
|---|---|---|---|
| CBD1 (13²) | S | **R** | S |
| MG1 | R | R | R | ← a 4-coordinate metal, not a real centre |
| CA2 | *not perceived* | S | S | ← perceived in predictions, absent from the reference |

`GetSubstructMatches(useChirality=True)` requires **every** perceived centre to agree. Both
populations return **zero** matches, every `ligand_rmsd` becomes `inf`, the backbone queue empties
at iteration 0, and the campaign freezes — the v4 funnel-collapse mode.

**Fix.** Keep the substructure match chirality-blind and gate the *one* centre directly
(`ligand_chirality.py`). This also keeps `ligand_rmsd` numerically comparable across campaigns.

---

## 4. Build the molecule from the component, never from PDB distances

`ligand_chirality.py` v1 used `Chem.MolFromPDBBlock` + `AssignBondOrdersFromTemplate`.
`MolFromPDBBlock` infers HETATM bonds by **interatomic distance**, and on a strained pose that
guess fails: **18 of 50** smoke-test predictions died with
`Explicit valence for atom # 38 C, 5, is greater than permitted` and returned `None`.

Since `None` is a gate *failure*, that alone would have rejected a third of every iteration for a
reason with nothing to do with chemistry.

**Fix.** We build the component ourselves, so the correct topology is already on disk. Take bonds
from `boltz_cache/mols/<CCD>.pkl` and only the **coordinates** from the pose — the one thing the
pose actually knows. Result: 50/50 readable.

---

## 5. R/S labels are meaningless without naming the substituent

Natural chlorophyllide is **13²R**; 5HPZ's 13²-hydroxy form is **13²S**. These are the *same
spatial arrangement*. Verified by swapping OH→H in place, with every atom in the same position:

| at C13² | CIP label |
|---|---|
| OH (13²-hydroxy) | **S** |
| H (plain chlorophyllide) | **R** |

The descriptor flips purely because OH outranks carbon while H ranks last. Never quote R/S without
stating what is attached at 13². (yigall's AMBER files label `68G` as R; 5HPZ says S for both
copies — reconcile by asking which substituent each refers to.)

---

## 6. The ligand SMILES builds the Rosetta params — it is not cosmetic

Deprotonating the C17 propionate requires **three** changes, and missing any one leaves an acid
proton in the output:

1. the CCD component (what Boltz folds) — `O2A` charge −1, no `H21`
2. the input backbone pose — strip the `H13` atom
3. **`ligand_smiles`** — `CCC(=O)[O-]`, because `construct_helper_files` builds
   `lig_from_input_pdb.params` from it via `AddHs`

With only (1) and (2) fixed, every *relaxed* structure still carried `H13` on `OA2` — and relaxed
structures are what a reviewer opens. Verified after the fix: 79 atoms / 33 H,
`CHARGE OA2 FORMAL -1`, no bond from `OA2` to any H, C–O distances 1.25/1.26 Å (delocalised).

**Corollary.** Stripping `H13` from the input pose without renumbering the remaining hydrogens
creates a *name collision* with the regenerated params (pose `H14` ↔ params `H13`), which
superimposes two atoms. Strip **all** ligand hydrogens from the input pose instead and let Rosetta
rebuild them from the ICOOR table — which is exactly why Boltz designs (heavy-atom-only output)
always relax cleanly.

---

## 7. Deprotonating the carboxylate cost the campaign its packing gradient

Attributed with a three-way control, which is why v6b was worth cascading even though it was
stopped as a design campaign:

| | chirality | ligand | packstat r(iteration) |
|---|---|---|---|
| v5 | no | neutral COOH | **+0.365** (0.588 → 0.653) |
| v6b | **yes** | neutral COOH | **+0.183** (0.611 → 0.617) |
| v7 | yes | **COO⁻** | **−0.002** (0.624 → 0.589) |

v6b still climbs with chirality enforced, so **chirality is not the cause — the anion is.** The
propionate's own burial is unchanged (0.525 vs 0.534), so it is not being pushed out; the whole
pocket is looser. Plausibly the designer places polar residues near the charge and nothing in the
objective rewards packing.

Caveat: v6b is 2,934 designs over 8 iterations against v5's 11,428 over 17 — direction clear,
magnitude less so.

---

## 8. A campaign climbs through its own gates — do not loosen them early

| iteration | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| v7 % passing all four gates | **0.0** | 0.9 | 9.9 | 46.1 | **77.7** | ~68 |
| v8 p90 ligand pLDDT | 0.786 | 0.789 | 0.837 | 0.913 | 0.937 | → **0.964** |

**v7's iteration 0 passed literally nothing** and the funnel widened ~78× by iteration 4. A gate
that looks starving at iteration 0 is the selection pressure, not a bug.

This matters when porting a gate between engines: calibrating on a sample pooled across iterations
understates what a campaign achieves, and calibrating on designs optimised for a *different* engine
understates it further. v8's AF3 gates were left at Boltz's 0.80 on this reasoning — a calibration
said only 10% would pass, and v8 reached 0.964.

Distinguish from a **real** funnel collapse (v4), where the queue genuinely emptied and never
recovered.

---

## 9. Never grade the exam with the model that sat it

A campaign optimises its folding model's confidence, so that model cannot also be the check.

- Boltz-led → check with AF3 (`af3_monomer.py`). On v5 it rejected **7 of 29** finalists that Boltz
  rated 0.94+.
- AF3-led → check with Boltz (`crossfold.py`). On v8's 25 finalists: Boltz protein pLDDT median
  0.966 (min 0.665), ligand 0.894 (min 0.506) — and **Boltz independently confirmed 13² = S in
  25/25**.

The engines correlate at **r = +0.13** on ligand pLDDT, so the check is genuinely independent.

The **dimer** check is always AF3 — it is an aggregation proxy on two copies, and AF3 is the
reference for that regardless of what folded the campaign.

---

## 10. Gates tuned on one model's confidence do not transfer

On the *same native sequence*: AF3 ligand pLDDT **0.64**, Boltz **0.83**. On 30 of v7's own designs
re-folded: AF3 median 0.753 vs Boltz 0.865, correlation **r = +0.18**.

Port a gate by the **fraction it admits**, not the number — Boltz's 0.80 admits 79%, the AF3
equivalent is ~0.69. But see #8: with the campaign climbing, the stricter number may be correct
anyway.

---

## 11. Long terminal jobs must stream partial results

The cascade accumulated every scored row in memory and wrote the CSV once at the end. A SIGKILL at
**2,770 of 11,201** took all 2,770 rows with it. The 2,774 PDBs left on disk are *outputs*, not
scores — re-deriving the scores costs the same as redoing the designs.

**Fix.** Append each row to `cascade_partial.jsonl` and flush; re-read on restart. Verified: a kill
and restart preserved 105 rows and skipped them.

Related: a **path alone is not a valid cache key**. The driver rewrites each Boltz output in place
twice (atom renaming, then protonation), so `relax_pool.py` keys on `source_path + size:mtime_ns`.

---

## 12. Stage order was wrong and only worked by accident

`af3_monomer.py` reads `results/final/top*.pdb` and `burial_rescore.py` reads `shortlist.csv` —
both produced by `final_shortlist.py`, which ran **after** them. v5 only worked because an earlier
run had left those files on disk; v7's first clean run failed both stages.

**Correct order:** shortlist → fold check → burial → **rebuild shortlist** (so the new columns
enter the ranking) → SASA → report.

---

## 13. Report the truth, not the aspiration

`finish_at_iter.sh` printed `DONE. shortlist: .../shortlist.csv` unconditionally. After the cascade
was SIGKILLed, the run still said DONE and the figures stage still sent a Telegram — so a failed
harvest looked like a success, and `results/final/` was empty.

**Fix.** Check the artefact exists and is non-empty; otherwise print `INCOMPLETE` and point at the
WARNING lines. Same principle: label columns by the engine that produced them — v8's report called
AF3's pLDDT "Boltz lig" because the column name is Boltz-shaped.

---

## 14. Shared-machine etiquette is a correctness issue

- **GPU:** the launcher must **hard-fail**, not warn, on any foreign compute process. v5 only
  warned; "a warning scrolls past and four GPUs get double-booked."
- **CPU:** size the worker pool from **other users'** load only. Counting our own collapsed it to 4
  workers and a 14-hour cascade. `RESERVE=16` cores.
- **Kills:** scope every `pkill` to the run directory. A bare `pkill -f nise_telegram_report.py`
  matches any campaign's reporter, and the Boltz line would have killed another campaign's four
  folding processes outright.
- **`pkill -f` self-matches.** A pattern that appears in your own command line kills your own
  shell — this happened three times in one session, once with `-9`. Build the pattern from
  concatenated variables so the literal never exists in the command line, or kill by PID.

Measured facts worth encoding: 1 Boltz process per GPU is optimal (2 is 10–25% **slower**);
LigandMPNN at ~1.8 GiB fits two per card; AF3 sets `XLA_CLIENT_MEM_FRACTION=0.6` and cannot share.

---

## 15. A converged campaign must harvest itself

v7 converged at 01:37 and sat idle until 10:45 — **nine hours of dead GPUs on a shared box** —
because the driver's convergence check only breaks the loop and exits, and `finish_at_iter.sh` is a
separate script nobody invoked.

**Fix.** Chain into the terminal pipeline when the loop ends (either exit path), spawned detached
via `setsid` so it outlives the driver, with a short delay so the driver is gone before the
script's `pkill` runs. `NISE_NO_AUTO_HARVEST=1` suppresses it.

---

## 16. BUNs cannot resolve the top of a shortlist

Re-relaxing one v4 design moved its BUNs by **7 across five seeds** (median spread 0.9) against a
**3-point range across the whole top ten**. Treat the leaders as a set.

The measures are near-independent — BUNs vs apolar burial **r = +0.12**, vs packstat **+0.11**, vs
ligand pLDDT **+0.13** — so the leader of one column is mid-pack in another. Select a set that
spans the measures, with hard gates applied *before* ranking.

The paper's own data supports this: its best binder (EPIC, Kd 0.12 µM) had the **most** ligand BUNs
of the four synthesised, so ligand BUNs alone would have ranked the best binder last.

---

## 17. Burial is scored, never filtered

A hard burial gate would have dropped v4's best-enclosed design. Report it; rank on it; do not gate
on it. Likewise `OCD1` is deliberately **excluded** from `ligand_atoms_enforce_buried`: that set is
an all-or-nothing gate, and a hydroxyl can legitimately face solvent.

---

## 18. Upstream bugs worth remembering

- **`ConnectionsPotential` is one-sided.** `lower_bounds = None`, hardcoded `buffer = 2.0`.
  Declared bonds collapse to 1.26–1.95 Å (median 1.52); unconstrained Boltz floors the contact at
  **2.939 Å, identical across 3,186 designs on two unrelated scaffolds**. The patch supplies a
  two-sided 2.0–2.6 Å window → measured 2.03 Å with Mg 0.12 Å off centre.
- **AF3's `bondedAtomPairs` has no distance window.** It forces a covalent-length bond: His–Mg
  crushed to **1.50 Å**, metal pulled 0.298 Å out of plane, and the 13² centre **flipped to R**.
  Fold unrestrained (His–Mg 4.22 Å) and let Rosetta form the bond — measured 4.22 → **2.50 Å** with
  the ligand moving 0.15 Å.
- **`AF3_CONFORMER_SKIP` defaults to `CLA-IDE`.** Unless set to your residue, AF3 attempts an RDKit
  ETKDG embed of an Mg-porphyrin, fails, and falls back anyway — ~71 s wasted per fold.
- **The CCD conformer is the structural reference.** The CLA-OXY block carries the 1JB0 *crystal*
  frame in both its model and ideal columns. Folding against it gives pLDDT 0.490 and 1.52 Å
  carbonyls; v4's Corina frame gives 0.774 and 1.22 Å. Atom reordering was tested and was **not**
  the cause (ordering-only fix left pLDDT at 0.478).
- **`CLX` is a real PDB component.** Registering a custom molecule under a taken code makes Boltz
  fold a different ligand without complaint.
