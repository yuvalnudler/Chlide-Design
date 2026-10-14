# The published paper vs our pipeline

Fry, Slaw & Polizzi, **Nature** (2026), doi:10.1038/s41586-026-10670-w — the peer-reviewed
version of the bioRxiv preprint our pipeline was modelled on. Read alongside
`41586_2026_10670_MOESM1_ESM.pdf` (Supplementary Information, 99 pp) and
`MOESM3` (Peer Review File, 35 pp), both in `/home/data/yuvaln/`.

The published version is **not** the preprint we built against. Three things changed.

---

## 1. They switched structure predictors: RFAA → Boltz-2

The preprint's NISE loop used RoseTTAFold-All-Atom. The published one uses **Boltz-2**, and
the SI is explicit about why:

> "RFAA did not recover the designed binding orientation, with no prediction achieving a
> ligand heavy-atom RMSD < 5.0 Å relative to the Boltz-2 design model... improvements in
> protein–ligand co-structure prediction models, as implemented in Boltz-2, enabled us to
> identify designs that **we would likely have discarded** if we had used RFAA self-consistency."

They also solved an X-ray structure of EPIC and found RFAA **warped the ligand conformer**;
Boltz-1 and AF3 modelled it more accurately. Since RFAA's output is the *input to the next
LASErMPNN round*, a badly modelled ligand corrupts the whole trajectory.

**For us:** this validates the engine-swap capability as a first-class feature rather than a
convenience, which is what `src/chlide/engines/` exists for. It does not tell us AF3 beats
Boltz-2 — they never ran that comparison for design.

## 2. The NISE objective is no longer ligand pLDDT alone

The APEX trajectories maximised **ligand pLDDT + P(bind)** (`affinity_probability_binary`,
Boltz-2's binding head), not ligand pLDDT alone.

The justification is orthogonality, and it is measured:

- ligand pLDDT **is** correlated with ipAE and ipTM (Extended Data Fig. 8) — so those add nothing
- P(bind) is **uncorrelated** with ligand pLDDT — so it does
- high P(bind) → better Rosetta **shape complementarity** (Mann-Whitney p < 1e-12) and lower
  **void volume** (p = 0.022), with **no** difference in ligand SASA (p = 0.724)
- P(bind) discerns binders from non-binders but **not** stronger from weaker

**For us this is the single most actionable finding.** Shape complementarity and core void
volume are exactly where our designs are weakest: v8 packstat 0.633 and apolar burial 0.690
against the COMBS target of 0.75. P(bind) is the one published signal that targets that
specific deficiency, and **no campaign of ours has ever computed it** — v5, v7 and v8 all ran
`objective_function = 'ligand_plddt'`, which sets `boltz2_predict_affinity = False`.

Two corrections to our own record:

- `main_pipeline/scripts/boltz2_gate.py` was right all along. Its `Spearman = -0.058` result is
  **Boltz `confidence_score` vs AF3 `ranking_score`** — a cross-engine ranking disagreement —
  and the file explicitly says that result "says nothing about affinity". It even names
  `affinity_probability_binary` as a different head.
- The *plan* listed "boltz2_gate.py — disabled on evidence (Spearman −0.058)" under
  deliberately-not-taken. That conflated the confidence gate with the affinity head. The
  affinity head was never evaluated, not rejected.

**The catch:** P(bind) is a **Boltz-2** output. AF3 has no equivalent. v8's move to AF3 forfeits
it. Using it means a Boltz-led trajectory (v7's configuration) or a hybrid.

NISE supports it directly: `objective_function` accepts `ligand_plddt_and_pbind`, which flips
`boltz2_predict_affinity` on. In our config that is **one line**: `loop.objective`.

## 3. The funnel is much stricter than ours

| | Published (APEX) | Ours (v8) |
|---|---|---|
| ligand pLDDT | **> 0.9** | > 0.80 |
| ligand heavy-atom RMSD (selection) | < 1.5 Å | < 1.5 Å |
| protein Cα RMSD | < 1.5 Å | < 1.5 Å |
| P(bind) | **> 0.5** (0.8 if > 3,000 pass) | not computed |
| apo-repack clashes | 0 | 0 |
| ranking | BUNs, tie-break ligand pLDDT | BUNs Score |
| cross-check | **AF2-Multimer *and* RF3** | one engine |
| aggregation proxy | AF2-Multimer interface pAE | AF3 two-copy pAE |
| surface Ala / Gly budget | **2 / 0** | 4 / 0 |
| final selection | top-2 per trajectory | top 8 overall |

Gate values do **not** transfer across engines (our own measurement: Boltz ligand pLDDT median
0.865 vs AF3 0.753 on the same designs, r = +0.13 to +0.18), so "they use 0.9, we use 0.8" is
not directly a looseness. But v8's p90 ligand pLDDT is **0.964**, so a 0.9 gate is affordable on
the AF3 scale and our 0.80 is almost certainly not doing selection work.

**CORRECTION.** An earlier draft of this file claimed our ligand RMSD gate is 2.5 Å against
their 1.5 Å. That was wrong — it compared two different gates. We have **three**:

| where | value | what it gates |
|---|---|---|
| in-loop self-consistency (`run_nise_*.py`) | ligand **2.5**, protein 1.5 | which predictions become the next iteration's backbones — deliberately permissive, this is the EXPANSION step and a tight gate here starves the trajectory |
| harvest (`harvest_global_top.py`, `PAPER` dict) | ligand **1.5**, protein 1.5 | what reaches the shortlist — the paper's own values |
| cascade (`polizzi_cascade.py`, `RMSD_TO_MIN_MAX`) | 1.5 | prediction vs Rosetta-minimised — a different quantity entirely |

Our **selection** gate has been 1.5 Å since the first campaign, matching the paper. The 2.5 is
the in-loop expansion threshold and has no counterpart in the published filtering description.
So on ligand RMSD we are not looser than the paper.

---

## Results, for calibration of expectations

| campaign | tested | bound | best Kd |
|---|---|---|---|
| **APEX** (apixaban, NTF2, NISE) | 6 | **5** | **80 pM** — tightest fully computational binder reported |
| **EPIC** (exatecan, 4-helix, NISE) | 4 | **4** | 0.12 µM → **1.2 nM** after neural proofreading |
| Baker lab, same NTF2 backbones, LigandMPNN + Rosetta | ~10,000 | 1 | 700 nM |
| COMBS + Rosetta (traditional) | 9,024 | 4 | 0.68 µM |

83% hit rate against ~0.01%. The comparison is on the **same backbones** for apixaban, so it is
a method comparison, not a target comparison.

## What the referees said that applies to us

- **Referee #3:** "the main limitation of this method is that a realistic ligand–protein
  backbone pose must be given as input... one could argue this is more of an *affinity
  maturation* method rather than a true zero-shot design." **We are squarely in that regime** —
  we start from BDFP44 with chlorophyllide already placed. We should describe our work that way
  rather than as zero-shot design.
- **Referee #5** (code review): pin dependencies, bundle a runnable example, and add "a small
  test suite — even a smoke test that... checks output format... to protect against regressions
  as the code evolves." This is an independent reviewer asking for precisely what
  `chlide-design` is: pinned config, campaign YAMLs, 90 GPU-free tests.
- **Referee #2** noted the work's "partial abandonment of confidence metrics such as ipTM,
  ipAE... apart from ligand pLDDT." Extended Data Fig. 8 is the justification: those *are*
  correlated with ligand pLDDT, so they are redundant — unlike P(bind).

## Where our problem is genuinely harder

Neither exatecan nor apixaban has any of these:

- a **coordinated metal** (Mg) with an axial His whose bond no engine restrains correctly
  without patching
- a **stereocentre we must enforce** — AF3 honours `pdbx_stereo_config` only ~59% of the time
  on designed sequences, and never converged over v8's 17 iterations
- a **charged substituent we deliberately keep solvent-exposed**, which caps whole-ligand burial

So our lower packing numbers are not simply worse execution; they are partly a harder ligand.
That is an explanation, not an excuse — P(bind) still targets the gap.
