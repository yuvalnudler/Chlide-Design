# Incident log

Append-only. Newest first. One entry per incident that cost time or a run.
Format: date · what broke · root cause · fix · how we'd catch it now.

---

## 2026-10-11 · v8 harvest skipped both AF3 scoring stages
`finish_at_iter.sh` guards the GPU stages behind `NISE_ALLOW_GPU=1`; the auto-harvest spawned it
without that variable, so `af3_interface_pae` and the fold check silently skipped and the shortlist
shipped without dimer pAE or cross-fold columns.
**Fix:** run them manually; auto-harvest should pass the flag when it has verified the GPUs itself.
**Catch:** the completion message should list which stages ran, not just whether a file exists.

## 2026-10-11 · report labelled AF3's pLDDT as "Boltz"
`ligand_plddt` is the B-factor of whatever folded the design, but the column name and the report
header are Boltz-shaped. v8's report called AF3's number "Boltz lig".
**Fix:** `make_report.py` detects the engine from the driver and labels accordingly.
**Catch:** never hardcode an engine name in a label; derive it.

## 2026-10-09 · v8 crashed at iteration 0 with the convergence bug v6b had already fixed
v8 branched from v6b *before* the `None`-formatting fix landed, so it inherited the crash.
**Root cause:** copy-per-campaign means fixes do not propagate.
**Fix:** ported the fix. **Catch:** one repo, code lives once. This is why `chlide-design` exists.

## 2026-10-09 · AF3 folded with a bond constraint that flipped the stereocentre
`af3_json` defaulted to emitting `bondedAtomPairs` after I added support for it, despite having
measured that the bonded form crushes His–Mg to 1.50 Å and flips 13² to R. The self-test also
asserted the 2.0–2.6 Å gate against the *raw fold*, which unrestrained AF3 can never satisfy.
**Fix:** default `his_resnum=None`; the gate is checked after relax, not on the fold.
**Catch:** the round-trip self-test — it caught this before the campaign spent two days.

## 2026-10-06 · cascade SIGKILLed at 2770/11201, all scores lost
Rows accumulated in memory; the CSV was written once at the end.
**Fix:** stream to `cascade_partial.jsonl`, re-read on restart. **Catch:** verified by killing and
restarting a 6-design run — second run skipped all 6.

## 2026-10-06 · three cascades ran simultaneously
`pkill` had not taken effect before I relaunched, so each restart stacked another 24-worker pool.
Throughput *fell* with more processes. One cleanup command used `pkill -9 -f 'finish_at_iter.sh'`,
whose pattern appeared in the very command running it — killing my own shell mid-cleanup.
**Fix:** build kill patterns from concatenated variables so the literal never appears in the
command line; verify the singleton after starting.

## 2026-10-06 · trumpeldor rebooted twice (11:00, 11:06), killing the harvest
No OOM entries, 496 GB free — not memory. Cause undiagnosable: `journalctl -b -1` was empty.
**Catch:** persistent journald would make the next one diagnosable.

## 2026-10-06 · v7 converged at 01:37, sat idle until 10:45
Nine hours of dead GPUs. The driver exits on convergence; the terminal pipeline is a separate
script nobody invoked. **Fix:** chain into it on loop exit, detached. See LESSONS #15.

## 2026-10-04 · deprotonation incomplete — relaxed structures kept the acid proton
The CCD and the input pose were both correct, but `ligand_smiles` still said `C(=O)O` and
`construct_helper_files` builds the Rosetta params from it via `AddHs`.
**Fix:** `CCC(=O)[O-]`. **Catch:** config validation rejects a neutral SMILES when
`deprotonated: true`. See LESSONS #6.

## 2026-10-03 · build_ligand.sh rebuilt a *finished* campaign's ligand
`RUN=/home/data/yuvaln/nise_bdfp44_v5` hardcoded; run from v6b it overwrote v5's archived
component. **Fix:** derive RUN from `BASH_SOURCE`. See LESSONS #1.

## 2026-10-03 · v6b driver pointed at v5's input tree
`input_dir = .../nise_bdfp44_v5/input`, and NISE writes its iterations under `input_dir` — v6b
would have written into v5's closed archive and folded against v5's untagged component.
**Caught by:** a pre-launch sweep for hardcoded paths. Eleven files carried them.

---

## 2025-10-11 — `iptm` column collision between monomer and dimer tables

**Symptom:** merging `results/af3_dimer/interface_pae.csv` into `results/final/shortlist.csv`
on a bare `iptm` key produced `iptm_x`/`iptm_y` and a `KeyError`, or worse, silently
selected the wrong one.

**Cause:** both tables carry a column named `iptm` and they are different quantities.
The shortlist's is the **monomer protein–ligand** ipTM from the folding run. The dimer
table's is the **two-copy** ipTM. Nothing in either name says which.

**Fix:** `make_report.py` maps the dimer file's `iptm` to `dimer_iptm` on merge, with an
inline comment. Any new merge of these two tables must do the same.

**Lesson:** two tables in the same results tree that use one name for two quantities is a
latent wrong-number bug, not just a merge nuisance. Rename at the boundary, not at the
point of use.

---

## 2025-10-11 — hardcoded engine names in shared scripts (recurring class)

**Symptom:** v8's figure 4 legend read "Boltz prediction" while the histogram plotted AF3's
output. The report's cross-fold columns said "X-fold", hiding which engine they came from.
Then the newly added figure 6 asserted "the distribution is bimodal" in its title — true for
v8 (AF3-led), **false for v7**, whose distribution is flat.

**Cause:** the same script serves campaigns with different leading engines, and every label
that names an engine — or asserts a property of the data — was written for whichever campaign
it was first run on. The directory tree is no help: `af3_adapter.to_boltz_layout` writes AF3
output into a `boltz_results_*` tree on purpose, so the *layout* proves nothing. Only the
driver's import does.

**Fix:** `paper_figures.engine()` and `make_report._engine()` / `_cross_engine()` detect the
leading and cross-check engines by grepping the driver for `from af3_adapter import`. Labels
are `{ENG}`/`{XENG}` placeholders substituted at render time. Figure 6's bimodality claim is
now *measured* (trough < 0.6 × the smaller of the two modal peaks) and the title states
"unimodal: no valley" when there isn't one.

**Lesson:** a label that names an engine, or asserts a shape in the data, is a claim. In a
script shared across campaigns it must be derived, not typed. Two of these shipped before
anyone noticed; the third was caught only because the fix for the first two made us re-run v7.

---

## The repo's launch path never starts the relax-pool watcher (2026-10-17, v9)

**Symptom:** v9's harvest had **12,972 designs pending relaxation** and spent hours on them.
v8's harvest caught its pool up in six minutes.

**Cause:** the in-loop relax is not the pool relax, and only ever touches `relax_top_n=12`
designs per iteration — just enough to gate which backbones get promoted. Both campaigns did
exactly that. The difference is a *separate* background watcher running `tools/relax_pool.py`
alongside the campaign:

```
v8: logs/relax_watcher.log -> "relax watcher started: workers=12 interval=600s"
    ran 2026-10-09 14:49 to 2026-10-10 06:49, pool at 4,983 by harvest
v9: no relax_watcher.log, no watcher, pool at 0
```

`grep -rn 'relax_pool|relax_watcher' src/ campaigns/` finds **nothing but a comment**. v8 was
launched from a per-campaign script that started the watcher by hand; `chlide run` never did,
and nobody noticed because the harvest does the same work — just all at once, at the end,
while someone is waiting for it.

**Lesson:** when work is normally absorbed by a background process, its absence does not fail,
it just moves the cost to the worst possible moment. The migration checked that every *tool*
came across; it did not check that everything the old launcher *started* came across.

**Fix (not yet applied):** start the pool watcher from `src/chlide/loop.py` when
`relax.pool_watcher` is enabled, defaulting on, and have the harvest report how much of the
pool was already done so a silent regression is visible in the log. Must land before v9b/v10.
