# Folding engines

Both engines satisfy the same `Folder` protocol and emit the same on-disk layout, so no downstream
stage knows which ran. Their *quirks are documented here, not normalised away* — the differences
are real and sometimes scientifically meaningful.

| | Boltz-2x | AlphaFold3 |
|---|---|---|
| speed | 22–25 s/design/GPU | **10.4** s/design/GPU batched · 46 s unbatched |
| ligand pLDDT scale | median 0.865 | median 0.753 |
| chirality | 12 constraints, **100% S** | `pdbx_stereo_config` only, **~59% S** |
| His–Mg | patched 2.0–2.6 Å window | unrestrained, Rosetta forms it |
| memory | 2,811 MiB/process | 7,373 MiB (`XLA_CLIENT_MEM_FRACTION=0.6`) |
| per card | 1 (2 is 10–25% slower) | 1 — cannot share |

## Boltz-2x

- **Requires the coordination patch.** Upstream `ConnectionsPotential` has `lower_bounds = None`
  and a hardcoded `buffer = 2.0`, so declared bonds collapse to a 1.52 Å median. Unconstrained it
  floors at 2.939 Å. The patch supplies a two-sided 2.0–2.6 Å window → 2.03 Å measured.
- **Chirality needs `_CIPRank` on every atom** or it builds **zero** constraints. Clear the metal's
  tag or it is constrained as a tetrahedral centre (16 constraints vs 12).
- The component's **conformer is the structural reference** — a crystal frame gives pLDDT 0.490,
  a Corina frame 0.774.
- `ligand_smiles` must keep Mg **disconnected** (`.[Mg+2]`) or RDKit ETKDG cannot embed it and
  Boltz hard-fails with "Failed to compute 3D conformer". The **CCD** keeps its four Mg–N bonds.

## AlphaFold3

- **`AF3_CONFORMER_SKIP` must name our residue** (default is `CLA-IDE`). Otherwise AF3 attempts an
  ETKDG embed of an Mg-porphyrin, fails, and falls back anyway — ~71 s wasted per fold.
- **Use `--input_dir`, not `--json_path`.** Batching amortises the JAX compile across a shard:
  46 s → 10.4 s per fold. Bypass `run_af3_custom.sh`, which hardcodes `--json_path` and
  `--num_diffusion_samples=5`.
- **Set `--jax_compilation_cache_dir`.** 148 s → 32 s warm, and makes repeat runs bit-identical.
- **Do NOT use `bondedAtomPairs`.** No distance window: His–Mg crushed to 1.50 Å, metal 0.298 Å out
  of plane, 13² flipped to R. Fold unrestrained; Rosetta forms the bond (4.22 → 2.50 Å).
- **No `.lower()` on output names.** AF3 does not lowercase; the older tools only work because
  their generated names already are. Glob case-insensitively.
- `label_comp_id` is 5 characters (`CLAOX`) and PDB truncates to 3 (`CLA`). Harmless — downstream
  matches on **atom** names.

## Cross-checking

`cross_check: auto` picks the engine that did **not** lead. A campaign optimises its own model's
confidence, so that model cannot also be the check. The engines correlate at **r = +0.13** on
ligand pLDDT, which is what makes the check worth running.

The **dimer** check is always AF3 — an aggregation proxy on two copies, where AF3 is the reference
regardless of what folded the campaign.
