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
