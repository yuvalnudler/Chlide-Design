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

## Quickstart

```bash
pip install -e ".[dev]"
chlide validate campaigns/v8-af3-coo.yaml
chlide run campaigns/v8-af3-coo.yaml --out runs/v9
```

## Layout

| path | what |
|---|---|
| `campaigns/` | one YAML per campaign — the only thing that changes between runs |
| `src/chlide/engines/` | `boltz.py`, `af3.py` behind one `Folder` protocol |
| `src/chlide/stages/` | ligand · scaffold · design · fold · relax+gate · cascade · score · report |
| `tools/` | standalone CLI scripts (cascade, shortlist, reports, calibration) |
| `docs/` | **read these first** |

## Documentation

- **[LESSONS.md](docs/LESSONS.md)** — 18 hard-won lessons with the measurements that settle them.
  Read before changing anything about chirality, ligand protonation, gates or kill patterns.
- **[BUGS.md](docs/BUGS.md)** — append-only incident log.
- **[ENGINES.md](docs/ENGINES.md)** — per-engine quirks and traps. Neither engine is a drop-in for
  the other; the differences are documented, not hidden.
- **CALIBRATION.md** — gate values and the evidence behind them.

## Shared-machine etiquette

This runs on a shared box. The launcher **hard-fails** if any GPU carries a foreign process,
reserves CPU cores for other users (sized from *their* load, not ours), and scopes every `pkill` to
its own run directory. These are correctness requirements, not politeness — see LESSONS #14.
