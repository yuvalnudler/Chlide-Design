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

## Engine-dependent gates **[ours]**

Ligand pLDDT does **not** transfer between engines — Boltz median 0.865 vs AF3 0.753
on the same designs, and the two correlate at only r = +0.13 to +0.18. A gate tuned on
one engine is meaningless on the other. See `ENGINES.md`.

Cross-fold gate (the non-leading engine's opinion): ligand pLDDT ≥ 0.80, protein
pLDDT ≥ 0.80, 13² stereocentre must agree. On v8's top 25 the cross-fold dropped
designs the leading engine rated 0.95 — it is the single highest-yield check we have.
