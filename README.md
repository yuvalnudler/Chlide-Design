# chlide-design

Pipeline for designing **chlorophyllide-binding proteins** on the BDFP44 scaffold, consolidating
eight NISE campaigns and the earlier Snakemake pipeline into one versioned codebase.

## Why this repo exists

Campaigns used to be **directory copies**. The same tool existed in up to eight divergent versions,
nothing was under version control, and in one week that model caused five incidents — a tool
rebuilding a finished campaign's ligand, a driver pointed at another campaign's input tree, a
campaign inheriting a crash from the branch it was copied from, and `pkill` patterns that would
have killed a live run's folding processes.

Here, **code lives once and a campaign is a YAML file.** The run directory is an argument.

## Status

Phase 1 of 5. The config layer is real and tested; the stages are not written yet, so
**campaigns still run from their own directories** — this repo does not execute anything
end to end. What works today:

```bash
pip install -e ".[dev]"
chlide validate campaigns/*.yaml     # parse + cross-field checks
pytest -q                            # 30 tests, no GPU
```

`v7-boltz-coo.yaml` and `v8-af3-coo.yaml` are verified to reproduce their original
drivers exactly (`tools/config_equivalence.py`, 19 parameters each, zero unexplained
differences). They differ from each other in **two lines**; in the old model that same
difference required forking the whole 1200-line driver, and the two `params` dicts were
byte-identical.

Not yet built: `chlide run`, the engine adapters, the stage modules, provenance and the
watchdog. See the phase table in the plan.

### What the campaigns produce today

v8 (AF3-led, the best campaign: 17 iterations, 23,200 designs, shortlist 142, best BUNs 6)
and v7 (Boltz-led) both render a full HTML report from one shared, engine-aware script —
Shrake–Rupley ligand burial (whole / chlorin core / apolar), both engines' pLDDT, the
cross-check engine's own composite confidence, the two-copy dimer pair, and for v8 a
**Suggested order** table whose rationale is derived from each design's standing in the
shortlist. Those scripts still live in the campaign directories; migrating them here as
`stages/report.py` is the first Phase 2 increment.

**This is the open risk.** Every edit to `make_report.py` or `paper_figures.py` is still
hand-copied between `nise_bdfp44_v7/tools/` and `nise_bdfp44_v8/tools/` — the exact practice
this repo exists to end. It has already produced one stale-label bug per copy
(`docs/BUGS.md`, 2025-10-11).

## Layout

| path | what |
|---|---|
| `campaigns/` | one YAML per campaign — the only thing that changes between runs |
| `src/chlide/config.py` | pydantic campaign schema — **built** |
| `src/chlide/engines/` | `boltz.py`, `af3.py` behind one `Folder` protocol — *not yet written* |
| `src/chlide/stages/` | ligand · scaffold · design · fold · relax+gate · cascade · score · report — *not yet written* |
| `tools/` | standalone CLI scripts; today only `config_equivalence.py` |
| `docs/` | **read these first** |

## Documentation

- **[LESSONS.md](docs/LESSONS.md)** — 18 hard-won lessons with the measurements that settle them.
  Read before changing anything about chirality, ligand protonation, gates or kill patterns.
- **[BUGS.md](docs/BUGS.md)** — append-only incident log.
- **[ENGINES.md](docs/ENGINES.md)** — per-engine quirks and traps. Neither engine is a drop-in for
  the other; the differences are documented, not hidden.
- **[NEXT.md](docs/NEXT.md)** — handoff note: state, next steps, outstanding fixes. Read first.
- **[CALIBRATION.md](docs/CALIBRATION.md)** — every gate value with the measurement that
  set it, including why the paper's dimer pAE 14 (an AF2 number) is kept for AF3 even
  though AF3's own ipTM puts the separation at 9.

## Shared-machine etiquette

This runs on a shared box. The launcher **hard-fails** if any GPU carries a foreign process,
reserves CPU cores for other users (sized from *their* load, not ours), and scopes every `pkill` to
its own run directory. These are correctness requirements, not politeness — see LESSONS #14.
