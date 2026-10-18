# Gate calibration

Every gate value here is paired with the measurement that set it. A gate without
evidence is a guess that has survived long enough to look like a rule.

Numbers taken from Polizzi et al. (zero-shot ligand-binder design) are marked
**[paper]**; numbers measured on our own campaigns are marked **[ours]** with the
campaign that produced them. When the two disagree, both are recorded.

---

## Two-copy interface pAE — the aggregation proxy

**Gate: keep designs with interface pAE > 14.** Low pAE means AlphaFold is
*confident* about a dimer interface, which the paper uses as a proxy for
aggregation and domain swapping.

### What the paper actually says **[paper]**

Two passages, two campaigns, two numbers — and both come from **AlphaFold 2**:

> "Two copies of each designed protein sequence were predicted using AlphaFold and
> were visually inspected for domain swaps or low interface pAE (pAE < 14; residues
> within 5.0 Å in predicted multimer), which we used as a proxy for aggregation
> potential."

> "...and average interfacial pAE > 15 (**AF2 multimer**, two copies of the sequence)."

Three things follow, and all three matter:

1. The contact cutoff is **5.0 Å**, not 5.5. `af3_interface_pae.py` uses 5.0 — correct.
2. The statistic is the **mean pAE over cross-chain residue pairs in contact**, not
   `chain_pair_pae_min` from AF3's summary file. Those are different numbers.
3. The threshold was **never the sole gate** — it was paired with *visual inspection*
   for domain swaps. We have no visual-inspection step, so our pAE gate is carrying
   weight the paper never asked it to carry alone.

**The transfer problem:** 14 and 15 were calibrated on AF2 multimer. We fold with
AF3 (v8) or Boltz-2x (v5, v7). Nothing guarantees the scale carries over.

### What our data says **[ours: v8, n=142]**

The two-copy pAE distribution is **bimodal for AF3-folded designs** — a confident-dimer
mode below ~7 and a monomer mode above ~16, with a genuine valley between:

```
 pAE   0- 2  ######                                  6
       2- 4  ##############################         26   <- confident dimer
       4- 6  ###########                            11
       6- 8  ############                           12
       8-10  #####                                   5   <- the valley
      10-12  ######                                  6
      12-14  ######                                  6
      14-16  #####                                   5
      16-18  ##########                             10
      18-20  #################                      17   <- monomer
      20-22  ################                       16
      22-24  ###############                        15
      24-26  ######                                  6
```

**v5 and v7 (Boltz-led) are flat — no valley at all.** For those campaigns no
threshold is defensible from the distribution; 14 is an inherited convention.
The bimodality is a property of the AF3-folded set.

### AF3's ipTM locates the valley

AF3 reports `iptm` for the same two-copy prediction. It is **not a second opinion** —
it tracks interface pAE at **Spearman −0.97** (v8; −0.97 v5, −0.92 v7). It is the
same measurement on a scale where the modes separate cleanly, with an *empty*
histogram bin at **ipTM 0.50–0.55**. That empty bin is the engine's own answer to
"where does a confident dimer begin", and it maps to:

> **interface pAE ≈ 9**

### Verdict

**The paper's 14 is conservative for AF3, but keep it.**

- 14 sits on the *rising flank* of the monomer mode, not in the valley. The
  data-driven cut for AF3 is **9**.
- At 14, **72/142** designs fail. **17 of those 72** have ipTM 0.23–0.47 — AF3 itself
  does not believe those dimers. Those 17 are rejected by the averaging, not by the
  model: interface pAE is a mean over a contact set whose size varies (4 to 35 pairs
  here), so a small plausible contact patch drags the mean down without AF3 ever
  committing to the interface.
- Disagreement is **strictly one-directional**: there is no design that ipTM fails and
  pAE passes. `pAE > 14` is a strict subset of `ipTM < 0.50`. Tightening costs recall
  only; it never lets a dimer through.

**It cost us nothing on v8.** All 8 ordered designs sit at pAE 16.6–25.5 / ipTM
0.07–0.26 — deep in the monomer mode, nowhere near either threshold. The one
high-quality design 14 rejects (`iter11_chunk_3_seq_77`, BUNs 6 — the joint best)
independently fails the Boltz cross-fold at ligand pLDDT 0.506, so two unrelated
lines of evidence drop it. That agreement is reassuring about both.

### How to use it

| | |
|---|---|
| `pAE > 14` | pass — keep the paper's bar as the default |
| `9 < pAE ≤ 14` | **grey band — inspect, do not auto-fail.** Check ipTM; if < 0.50 the design is a candidate when the shortlist is thin |
| `pAE ≤ 9` | fail — both the paper and AF3's own ipTM agree |
| no CB contacts | pass (reported as `inf`) — AF3 did not build an interface at all |

Report **ipTM alongside pAE** (`make_report.py` column *Dimer ipTM*). It is free —
already in the summary file — and its scale separates where pAE's does not.

**Do not** substitute ipTM for pAE. At ρ = −0.97 it adds no independent information,
and pAE is what the paper validated against 4/4 binders that bound.

**Watch out:** the shortlist's own `iptm` column is the **monomer protein–ligand**
ipTM from the folding run — a completely different quantity. Merging the two tables
on a bare `iptm` key silently collides them. Dimer ipTM is `dimer_iptm`.

### Open

Our proxy has never been tested against an experiment. The paper's 4/4 binders is the
only evidence that this gate selects for anything real, and it is evidence for
AF2-at-14, not AF3-at-9. Until we have SEC or SEC-MALS on a batch, treat the whole
gate as inherited, not validated — and prefer the conservative end, since designs are
cheap and tube time is not.

---

## Gates carried from the Polizzi cascade **[paper]**

| gate | value | note |
|---|---|---|
| packstat | > 0.55 | |
| apo-repack clashes | == 0 | |
| buried charged residues | none | |
| ligand / protein RMSD to minimised | < 1.5 Å | |
| BUNs Score | 2×ligand + 1×protein | the ranking statistic, not a gate |
| cysteine | disallowed | |

**BUNs cannot resolve the top of the list [ours: v4].** Re-relaxing one design moved
its BUNs by 7 across five seeds (median spread 0.9) against a 3-point range across the
whole top ten. Treat the leaders as a set, not a ranking.

Paper's tested binders had BUNs Score **16–28** and Kd 0.12–16.8 µM, 4/4 bound. Our
v8 best is **6** — far better on this axis than anything they put in a tube, which
says more about BUNs as a selector than about our designs.

## Ligand burial — a measure, never a gate **[ours]**

```
buried_fraction = 1 - SASA(ligand in complex) / SASA(ligand alone, SAME coords)
```

Shrake–Rupley, **probe 1.4 Å**, Bondi radii (Mg 1.73 Å, absent from Bondi's table), 960
points/atom, heavy atoms only. Implemented in `sasa_rank.py` rather than imported, so the
radii and probe are explicit and auditable — **FreeSASA is not installed on this machine**,
and the hand-rolled version is the same algorithm FreeSASA runs by default. Using the
identical conformer for the free-ligand reference means the ratio isolates occlusion by
protein and is unaffected by the ligand's own conformation. The cascade imports this module
directly, so the whole shortlist and the top-25 detail table share one set of constants.

Why it exists separately from the in-loop burial test: NISE's alpha-shape gate casts a ray at
a mesh built through the **CA trace** and counts crossings. That answers "is this atom inside
the polypeptide envelope" — it knows nothing about sidechains and nothing about solvent, so an
atom in a water-filled cavity counts as buried. SASA asks whether a water can touch it.

**v8, n=142 shortlist:**

| region | median | why it matters |
|---|---|---|
| whole ligand | ~62% | **capped by design** — see below |
| chlorin core (macrocycle + Mg) | **82%** | the number that should be high; this is what must be desolvated |
| apolar | 68.5% | COMBS docking filter target was ≥ 0.75 |
| propionate | 43% | *should* stay exposed |
| Mg SASA | 3.8 Å² (range 1.2–5.5) | ~0 means the coordination site is sealed from water |

**Do not read whole-ligand burial as a quality score.** The C17 propionate is charged and we
*enforced* its exposure during sampling (`atoms_enforce_exposed: [OA1, OA2]`), so it drags the
whole-ligand figure down on purpose. A design that buries it has paid a desolvation penalty
for nothing. Rank on **core** burial; read whole-ligand only alongside the propionate column.

**Burial is scored, never filtered.** A hard burial gate would have dropped v4's best-enclosed
design, and the in-loop gate already had to run at alpha=50 because the paper's default of 9.0
rejects every BDFP44 design at iteration 0 — the chlorophyllide cleft is far shallower than the
four-helix bundle the method was developed on.

**Open:** v8's apolar burial is 0.690 against v5's 0.884. That gap is the cost of the correct
(deprotonated, 13²-S) ligand, not a regression in design quality — v5 was burying a neutral
acid. Neither reaches the COMBS 0.75 target.

## P(bind) — the signal we have never measured **[paper, 2026]**

Boltz-2's `affinity_probability_binary`. The published paper's best campaign (APEX, 80 pM)
maximised **ligand pLDDT + P(bind)** as the NISE objective, not ligand pLDDT alone.

Why it is not redundant, measured by them:
- ligand pLDDT **is** correlated with ipAE and ipTM, so those add nothing
- P(bind) is **uncorrelated** with ligand pLDDT
- high P(bind) → better shape complementarity (p < 1e-12) and lower void volume (p = 0.022);
  no difference in ligand SASA (p = 0.724)
- it separates binders from non-binders, **not** strong from weak

**No campaign of ours has computed it.** v5, v7 and v8 all set
`objective_function = 'ligand_plddt'`, which leaves `boltz2_predict_affinity = False`.
It was never evaluated — not rejected. (The `Spearman = -0.058` result in
`main_pipeline/scripts/boltz2_gate.py` is Boltz `confidence_score` vs AF3 `ranking_score`,
a different quantity; that file says so itself.)

It targets our known weakness: v8 packstat 0.633, apolar burial 0.690 vs the COMBS target 0.75.

**Constraint:** P(bind) is a Boltz-2 head. AF3 has no equivalent, so an AF3-led campaign
forfeits it. See `docs/PAPER_2026.md`.

## Engine-dependent gates **[ours]**

Ligand pLDDT does **not** transfer between engines — Boltz median 0.865 vs AF3 0.753
on the same designs, and the two correlate at only r = +0.13 to +0.18. A gate tuned on
one engine is meaningless on the other. See `ENGINES.md`.

Cross-fold gate (the non-leading engine's opinion): ligand pLDDT ≥ 0.80, protein
pLDDT ≥ 0.80, 13² stereocentre must agree. On v8's top 25 the cross-fold dropped
designs the leading engine rated 0.95 — it is the single highest-yield check we have.

## Axial coordination geometry — reported, never ranked **[ours]**

The campaign gates coordination on one scalar, His NE2–Mg ∈ [2.0, 2.6] Å. By the shortlist
that gate is **saturated** — v8's top 25 sit at 2.425 ± 0.030 Å — so every survivor looks
identical on the only coordination measure we had. A distance cannot distinguish a real axial
bond from a histidine sitting 2.4 Å away while pointing the wrong way.

`tools/metal_geometry.py` adds five columns, computed in the cascade from the same pose that
already yields `his_mg`. They split into **where the His sits** and **how it is turned**, and
the two are only weakly related (ρ = 0.30), so neither substitutes for the other. All are
orthogonal to every existing shortlist column (|ρ| ≤ 0.36 against `his_mg`,
`buns_score_paper`, `packstat`, `buried_frac*`, `sasa_mg_A2`, `ligand_plddt`, `composite`;
none significant at p < 0.05).

### Replicate noise — read this before interpreting any single design

Four designs, five independent FastRelax replicates of the same input:

| measure | within-design sd | between-design sd | ratio |
|---|---|---|---|
| `axial_angle` | 0.47° | 3.77° | **0.12** |
| `his_lonepair_angle` | 1.86° | 5.38° | 0.35 |
| `mg_imidazole_oop` | 3.12° | 6.02° | **0.52** |
| `mg_out_of_plane` | 0.000 Å | 0.011 Å | 0.00 |
| `his_mg` (existing) | 0.008 Å | 0.028 Å | 0.28 |
| `packstat` (existing) | 0.015 | 0.018 | 0.85 |

`axial_angle` is the cleanest measure in this table — signal 8× the noise, better than
`packstat`, which it sits beside. **`mg_imidazole_oop` is the weakest**: half its spread is
relax noise, so a difference below ~6° between two designs means nothing. Do not repeat the
packstat mistake of reading a single-measurement gap smaller than the replicate sd.

### Reference values

| structure | His–Mg (Å) | axial (°) | lone-pair (°) | Mg off-imid (°) | dome (Å) |
|---|---|---|---|---|---|
| v8 top 25 (minimised) | 2.425 ± 0.030 | 5.6 ± 2.5 | 16.0 ± 6.1 | 14.5 ± 6.8 | +0.092 ± 0.019 |
| v8 top01 | 2.43 | 2.9 | 9.6 | 9.6 | +0.08 |
| BDFP44 input pose (Yigal) | 2.45 | 11.4 | 7.8 | 5.5 | +0.22 |
| 6HRK + His117 swap | 2.99 | 29.9 | 4.7 | 2.6 | −0.38 |

**The loop trades placement for orientation.** Against the input pose it improves the axial
angle (11.4° → 5.6° mean) while making the imidazole orientation markedly worse
(off-imid 5.5° → 14.5°, lone-pair 7.8° → 16.0°). Both shifts are far larger than the
replicate noise above. The restraint is a *distance* restraint, so the loop is free to satisfy
it by moving the histidine into place while letting the rotamer drift off the lone-pair axis.
This is a hypothesis about the restraint, not a measured cause.

### Why these are not in `composite`

Nothing shows they predict binding. Polizzi et al. Fig 1c is a direct warning: Rosetta ligand
energy, an equally principled geometric/energetic term, was tested as the loop objective over
70 iterations and *"the designs failed to optimize"*. Promoting an unvalidated term into
`composite` would silently change which designs get ordered. They are reported, and
`make_report.py` flags `his_lonepair_angle` or `mg_imidazole_oop` above 25° and `axial_angle`
above 8° — thresholds ~4× and ~8× the respective replicate noise, so a flag is meaningful.

### `mg_out_of_plane` is QC on the method, not a design measure

It is effectively constant (replicate sd 0.000 Å, between-design sd 0.011 Å) because the
Rosetta ligand params fix the chlorin's internal geometry — the loop **cannot** move the Mg
out of its ring plane. Experimental 5-coordinate Mg-chlorophyll domes 0.3–0.5 Å toward the
axial ligand; our minimised poses sit at 0.09 Å and cannot reproduce that. Report it as a
check that the ligand is intact; never read it as a finding about the designs.

### Flag thresholds come from experimental structures, not from our own campaigns

The advisory thresholds in `make_report.GEOMETRY_FLAGS` were first set from v8's own
distribution, which put the axial bar at 8°. The experimental p90 is **11.5°**, so that bar
fired inside the normal range and flagged **all eight** of v7's picks — the campaign with the
most realistic coordination we have. Calibrating a threshold against your own output tells you
only which of your runs differ from each other.

They are now the 98th percentile of the 57 His-coordinated Mg-chlorophylls measured in
`main_pipeline_v6/input/reference_structures/` (1M3X, 7COU), so a flag means *outside what
nature does*:

| measure | experimental median | p90 | p98 → **flag above** | max |
|---|---|---|---|---|
| `axial_angle` | 6.7° | 11.5 | **17** | 24.4 |
| `his_lonepair_angle` | 10.9° | 17.1 | **19** | 52.9 |
| `mg_imidazole_oop` | 6.0° | 12.7 | **17** | 17.9 |

Effect on the order sets: v7 8→3 flagged, v8 1→1, v9 8→8. The thresholds now separate the
campaigns the way the experimental data does, rather than the way v8 happened to.

`tools/compare_campaigns.py --run ...` regenerates the experimental reference from the crystal
structures, so these numbers can be rechecked rather than trusted.
