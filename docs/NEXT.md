# Where this is, and what to do next

Updated 2026-10-11. Read this first after a break; it is the handoff note.

## State

**Phase 2 in progress.** Phases 0 and 1 are done and verified.

| | |
|---|---|
| 0 bootstrap, pyproject, CI | done |
| 1 `config.py` + campaign YAMLs | done — both proven equivalent to their drivers (19 params, 0 unexplained, `tools/config_equivalence.py`) |
| **2 engines + stages + tools** | **3 of 22 tools migrated** |
| 3 provenance, logging, watchdog, GPU | not started |
| 4 backbone generation (default-off) | not started |
| 5 docs, golden tests, data migration | `CALIBRATION.md` and this file landed early |

37 tests pass, no GPU needed. `pytest -q` is the gate.

## What is migrated

`tools/sasa_rank.py`, `tools/make_report.py`, `tools/paper_figures.py`, plus
`src/chlide/notify.py` (telegram, vendored verbatim). These are canonical **here**;
`tools/sync_to_campaign.sh` pushes them outward one-way and records the commit in the
campaign's `TOOLS_PROVENANCE`. Both v7 and v8 are synced and show zero divergence.

**Never edit a tool inside a campaign directory.** That is how the same script came to
exist in eight versions and how two stale engine labels shipped (`BUGS.md`, 2025-10-11).

## Next, in order

1. **`crossfold.py`** (165 lines) — `BOLTZ_DONOR` is hardcoded to v7 *from inside v8*.
   De-hardcode to `engine.cross_check` resolution. Note its join to the folded output is
   **positional** (`x{i}` ↔ CSV row *i*, the design name appears nowhere in the output
   tree); that is safe only because `rows.append` sits outside `if hits:` in the row loop.
   Preserve that property or the join silently breaks.
2. **`af3_interface_pae.py`** (251) — 1 hardcoded path. Contact cutoff is **5.0 Å**, which
   matches the paper; do not "fix" it to 5.5.
3. **`polizzi_cascade.py`** (468) — imports `sasa_rank.py` directly so the SASA radii and
   probe live in ONE place. Keep that on migration; it is why the 142-design shortlist and
   the top-25 detail table cannot drift apart.
4. **`af3_adapter.py`** (329, 2 hardcoded paths) and the engine adapters →
   `src/chlide/engines/{base,boltz,af3}.py` behind the `Folder` protocol.
5. Remaining ~15 tools, then the stage modules.

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
