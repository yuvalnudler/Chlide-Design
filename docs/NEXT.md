# Where this is, and what to do next

Updated 2026-10-11. Read this first after a break; it is the handoff note.

## State

**Phase 2 in progress.** Phases 0 and 1 are done and verified.

| | |
|---|---|
| 0 bootstrap, pyproject, CI | done |
| 1 `config.py` + campaign YAMLs | done — both proven equivalent to their drivers (19 params, 0 unexplained, `tools/config_equivalence.py`) |
| **2 engines + stages + tools** | **9 of 22 tools migrated** |
| 3 provenance, logging, watchdog, GPU | not started |
| 4 backbone generation (default-off) | not started |
| 5 docs, golden tests, data migration | `CALIBRATION.md` and this file landed early |

37 tests pass, no GPU needed. `pytest -q` is the gate.

## What is migrated

`sasa_rank.py`, `make_report.py`, `paper_figures.py`, `crossfold.py`, `af3_interface_pae.py`,
`af3_monomer.py`, `polizzi_cascade.py`, `boltz_coord_patch.py`, plus `src/chlide/notify.py`
(telegram, vendored verbatim). **No cross-campaign paths remain in `tools/`** — enforced by
`tests/test_report.py::test_no_cross_campaign_paths_in_tools`.

**Cross-folding an AF3-led campaign now needs an explicit cache.** v8's `boltz_cache` holds one
component pickle and ZERO checkpoints, so Boltz cannot run from it. `crossfold.py` used to reach
into v7 via a hardcoded `BOLTZ_DONOR`; it now resolves `--boltz-cache` / `$BOLTZ_CACHE` →
`<run>/boltz_cache` (only if it has checkpoints) → `~/.boltz`, and **refuses with the list of
places it looked** rather than folding against an empty cache. For v8 pass
`--boltz-cache /home/data/yuvaln/nise_bdfp44_v7/boltz_cache`. Note the checkpoints physically live
in the boltz owner's home (`~yigall/.boltz`) and v7's cache is a tree of symlinks into it, so
`~/.boltz` is not a reliable fallback for every user. These are canonical **here**;
`tools/sync_to_campaign.sh` pushes them outward one-way and records the commit in the
campaign's `TOOLS_PROVENANCE`. Both v7 and v8 are synced and show zero divergence.

**Never edit a tool inside a campaign directory.** That is how the same script came to
exist in eight versions and how two stale engine labels shipped (`BUGS.md`, 2025-10-11).

## Next, in order

1. **`af3_adapter.py`** (329, 2 hardcoded paths) and the engine adapters →
   `src/chlide/engines/{base,boltz,af3}.py` behind the `Folder` protocol. This is the
   remaining structural work; everything above it is file movement.
2. **`harvest_global_top.py`** (289), **`relax_pool.py`** (236), **`final_shortlist.py`** (244),
   **`rosetta_relax.py`** (177), **`burial_rescore.py`** (143) — the in-loop pipeline.
3. The ligand build chain: `repair_cla_oxy.py` (278, 2 hardcoded), `build_boltz_ccd.py`,
   `rebuild_claox_conformer.py`, `build_input_pose.py`, `ligand_chirality.py`.
4. `calibrate_af3_gates.py`, `recalibrate_gates.py`, `replay_convergence.py`,
   `relax_replicates.py` — still carry hardcoded paths.
5. Then the stage modules and `chlide run`.

**Invariants to preserve while moving these** (each has already cost a bug):
- `crossfold.py`'s join to the folded output is **positional** (`x{i}` ↔ CSV row *i*; the design
  name appears nowhere under `crossfold_*/`). Safe only because `rows.append` sits OUTSIDE
  `if hits:` in the row loop — a skipped row would shift every design silently.
- `af3_interface_pae.py`'s contact cutoff is **5.0 Å**, matching the paper. Do not "fix" to 5.5.
- `polizzi_cascade.py` imports `sasa_rank.py` by `__file__`-relative path so the SASA radii and
  probe exist ONCE. Tested by `test_cascade_loads_sasa_from_its_own_directory`.

## Fixes owed on migration (from the plan, still outstanding)

- `build_ligand.sh` does not pass `--deprotonated` — re-running it silently regresses the
  ligand to the neutral acid. The current CCDs came from a manual invocation.
- `final_shortlist.py`'s generated README is hardcoded Boltz text; v8 ships a README
  claiming patched-Boltz coordination it never used. Make it engine-aware.
- Remaining hardcoded cross-campaign paths: `calibrate_af3_gates.py`, `recalibrate_gates.py`,
  `smoke_chirality.py`, `af3_monomer.py`, `repair_cla_oxy.py`.
- Move `make_report`/`sasa_rank`'s `CORE` and `PROPIONATE` atom sets into `LigandConfig` —
  they are ligand-specific and are module constants today.

## Known gaps, not bugs

- **ruff has never run.** It is not installed in any env on this machine, so CI's first run
  may surface lint fixes. Long lines in the three migrated tools are under a narrow
  per-file `E501` ignore, explained in `pyproject.toml`.
- `chlide run` does not exist. The repo validates campaigns; it executes nothing.
- The driver-equivalence and report tests **skip** in CI, since the campaign archives live
  outside the repo.

## Campaigns

v7 (Boltz-led) and v8 (AF3-led) are both **complete**; nothing is running and all four GPUs
are idle. v8 is the best: 17 iterations, 23,200 designs, shortlist 142, best BUNs 6. Its 8
selected designs are in `results/final/ORDER/`. **No wet-lab step exists yet** — every gate
here is computationally calibrated and experimentally untested.
