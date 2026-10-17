#!/usr/bin/env bash
# Stop the campaign cleanly at a chosen iteration, then run the whole terminal pipeline unattended.
#
# Ordering matters and is not arbitrary:
#
#   1. Wait for iteration N to write BOTH its artifacts. The driver's resume logic treats an
#      iteration as complete only when sampled_backbones/iter_N/backbone_queue.txt AND
#      sampling_dataframes/iter_N_data.pkl both exist, so stopping on that boundary means nothing
#      is lost and the run can still be resumed to 25 later if you change your mind.
#   2. Stop the DRIVER first, so it cannot start iteration N+1 while we work.
#   3. Only then catch the relax pool up on iteration N -- it must not race the driver's own
#      in-loop relax for the pool lock.
#   4. The cascade is CPU-only. GPUs are free from step 2 onward; AF3 in step 6 is the only thing
#      that needs them back.
#
# Usage: nohup bash tools/finish_at_iter.sh 16 --run-dir /path/to/campaign > finish.log 2>&1 &
#        (or export CHLIDE_RUN_DIR). This script lives in the SHARED repo, so it cannot infer the
#        campaign from its own location -- it used to, which silently pointed it at the repo and
#        left it waiting forever on a path that could never appear.
set -u
PY=/home/yigall/mambaforge/envs/lasermpnn/bin/python
RPY=/home/yigall/mambaforge/envs/Rosetta/bin/python
STOP_AT=${1:?usage: finish_at_iter.sh <iteration> [--run-dir DIR]}
shift
RUN=${CHLIDE_RUN_DIR:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir) RUN=${2:?--run-dir needs a path}; shift 2 ;;
    *) echo "[fatal] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN" ] || { echo "[fatal] no run directory: pass --run-dir DIR or set CHLIDE_RUN_DIR" >&2; exit 2; }
RUN=$(cd "$RUN" 2>/dev/null && pwd) || { echo "[fatal] run dir does not exist: $RUN" >&2; exit 2; }
# Fail fast instead of sleeping forever on an iteration file that cannot appear.
[ -d "$RUN/input/sampling_dataframes" ] || {
  echo "[fatal] $RUN is not a campaign run directory (no input/sampling_dataframes/)" >&2; exit 2; }
# One harvest per run dir. Two watchers running the terminal pipeline at once corrupts results/.
exec 9>"$RUN/.harvest.lock" || { echo "[fatal] cannot open $RUN/.harvest.lock" >&2; exit 2; }
flock -n 9 || { echo "[fatal] another harvest already holds $RUN/.harvest.lock -- refusing" >&2; exit 3; }
# Scoped to THIS campaign. An unscoped -- or empty -- pattern pkills every other run on the box.
DRIVER="chlide\.cli run .*--out ${RUN}([[:space:]]|/|$)"
pgrep -f "$DRIVER" >/dev/null 2>&1 || echo "[warn] no live driver for this campaign; harvesting what is on disk"
RESERVE=${RESERVE:-16}   # cores always left free for other users on this shared box
if [ -z "${WORKERS:-}" ]; then
  # Size from OTHER users' load, not total: counting our own collapses this to the floor.
  _others=$(ps -eo user,pcpu --no-headers | awk -v me="$USER" '$1!=me {s+=$2} END {printf "%d", s/100}')
  _free=$(( $(nproc) - _others - RESERVE ))
  # Hard ceiling 32: Yigal's standing limit on this box, independent of how idle it looks.
  # The old cap was 48, which on a quiet box asked for 35 and would have broken it.
  WORKERS=$(( _free < 4 ? 4 : (_free > 32 ? 32 : _free) ))
  echo "[cpu] $(nproc) cores, other users using ${_others}, reserving ${RESERVE}, taking ${WORKERS} workers"
fi
AF3_TOP=${AF3_TOP:-200}
cd "$RUN" || exit 1
say(){ printf '\n[%s] %s\n' "$(date +%F' '%H:%M:%S)" "$*"; }

# GPU stages run BY DEFAULT. v8's harvest skipped both because NISE_ALLOW_GPU was unset, and
# they had to be run by hand a day later -- the shortlist sat without its dimer pAE until then.
# The gate is now on whether the cards are actually free, not on remembering a variable.
#   NISE_ALLOW_GPU=0  force skip        NISE_ALLOW_GPU=1  force run (skip the check)
#   GPU_FREE_MIB      per-card idle allowance, default 2000
gpus_free() {
  case "${NISE_ALLOW_GPU:-auto}" in
    0) say "GPU stage SKIPPED -- NISE_ALLOW_GPU=0"; return 1 ;;
    1) say "GPU stage FORCED on -- NISE_ALLOW_GPU=1, availability check bypassed"; return 0 ;;
  esac
  command -v nvidia-smi >/dev/null 2>&1 || { say "GPU stage SKIPPED -- no nvidia-smi"; return 1; }
  # Anyone else's compute process means the cards are theirs. Never contend: this box is shared
  # and has been taken over mid-run before.
  local foreign busy
  foreign=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
            while read -r pid; do
              [ -n "$pid" ] || continue
              [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" != "$USER" ] && echo x
            done | wc -l)
  if [ "${foreign:-0}" -gt 0 ]; then
    say "GPU stage SKIPPED -- ${foreign} compute process(es) from another user on the cards"
    return 1
  fi
  # nvidia-smi cannot always resolve other users' PIDs, so also require the cards to be idle.
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
         awk -v lim="${GPU_FREE_MIB:-2000}" '$1 > lim {n++} END {print n+0}')
  if [ "${busy:-1}" -gt 0 ]; then
    say "GPU stage SKIPPED -- ${busy} card(s) still hold >${GPU_FREE_MIB:-2000} MiB"
    return 1
  fi
  say "GPUs idle and unowned -- running GPU stage"
  return 0
}

PKL="input/sampling_dataframes/iter_${STOP_AT}_data.pkl"
QUE="input/sampled_backbones/iter_${STOP_AT}/backbone_queue.txt"

say "waiting for iteration ${STOP_AT} to complete (both artifacts on disk)"
while [ ! -f "$PKL" ] || [ ! -f "$QUE" ]; do
  if ! pgrep -f "$DRIVER" >/dev/null; then
    say "driver exited before iteration ${STOP_AT} finished -- continuing with what is on disk"
    break
  fi
  sleep 60
done
say "iteration ${STOP_AT} banked ($(ls input/sampling_dataframes/iter_*_data.pkl | wc -l) complete)"

# --- 2. stop the loop; GPUs are free from here -------------------------------------------------
say "stopping driver and reporter"
pkill -f "$DRIVER" 2>/dev/null
# Scope every pkill to THIS run directory. A bare `pkill -f nise_telegram_report.py` matches
# the reporter of any campaign on the box, and the Boltz line below would have killed v6b's
# four folding processes outright.
pkill -f "nise_telegram_report.py --run-dir $RUN" 2>/dev/null
sleep 10
pkill -f "run_alphafold_patched.py --input_dir=$RUN" 2>/dev/null   # AF3 stragglers, this run only
sleep 5
say "GPUs released: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

# --- 3. catch the relax pool up, then retire the watcher ---------------------------------------
say "stopping the relax watcher, then catching the pool up on the final iteration"
pkill -f "$RUN/tools/relax_watcher.sh" 2>/dev/null
sleep 3
while pgrep -f relax_pool.py >/dev/null; do sleep 20; done      # let any in-flight pass finish
"$PY" tools/relax_pool.py --run-dir "$RUN" --workers "$WORKERS"
say "relaxed pool: $(( $(wc -l < results/relaxed_pool/index.csv) - 1 )) designs"

# --- 4. refresh the harvest so metrics.csv covers every iteration -------------------------------
say "harvest"
"$PY" tools/harvest_global_top.py --run-dir "$RUN" --top 50

# --- 5. terminal cascade, CPU only --------------------------------------------------------------
say "cascade (--pre-relaxed: holo minimisation already done during the run)"
nice -n5 "$RPY" tools/polizzi_cascade.py --run-dir "$RUN" --from-metrics --pre-relaxed \
     --workers "$WORKERS" || say "WARNING: cascade returned non-zero"

# --- 6. AF3 aggregation proxy, the only GPU step ------------------------------------------------
say "AF3 two-copy interface pAE, top ${AF3_TOP} by BUNs (GPU, ~1.2 h)"
if gpus_free; then
  "$PY" tools/af3_interface_pae.py --run-dir "$RUN" --top "$AF3_TOP" --devices 0,1,2,3 \
     || say "WARNING: AF3 stage returned non-zero"
else
  say "  run it later with:  $PY tools/af3_interface_pae.py --run-dir $RUN --top $AF3_TOP --devices 0,1,2,3"
  say "  then re-run the shortlist rebuild so interface_pae enters the ranking."
fi
say "GPUs now free for good: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

# --- 7. scoring passes and the shortlist ---------------------------------------------------------
# AF3 single-chain fold confidence -- the only independent check that the SEQUENCE encodes the
# fold. Orthogonal to Boltz ligand pLDDT and to every Rosetta stage, and on v5 it disagreed
# sharply with both: designs at Boltz 0.95+ came back as low as AF3 0.44. Must run before the
# shortlist so the ranking can see it. This is the LAST GPU step.
# ---- STAGE ORDER FIXED ------------------------------------------------------------------
# af3_monomer.py reads results/final/top*.pdb and burial_rescore.py reads shortlist.csv --
# BOTH are produced by final_shortlist.py. They used to run before it, so on a first run they
# failed with "no top*.pdb ... run final_shortlist.py first" and FileNotFoundError. v5 only
# worked because an earlier run had left those files on disk. Correct order: build the
# shortlist, score it, then rebuild it so the AF3 and burial columns land in the ranking.
say "final shortlist"
"$PY" tools/final_shortlist.py --run-dir "$RUN" --top 25 || say "WARNING: shortlist returned non-zero"
# ---- ENGINE-DEPENDENT fold check ---------------------------------------------------------
# The two GPU scoring stages answer different questions and are NOT interchangeable:
#   dimer  (af3_interface_pae.py)  ALWAYS AF3 -- it is an aggregation proxy, two copies of the
#          design, and AF3 is the reference for that regardless of what folded the campaign.
#   fold   engine-DEPENDENT. A campaign optimises its own folder's confidence, so that folder
#          cannot also be the check. Boltz-led -> check with AF3. AF3-led -> check with Boltz.
#          Running af3_monomer.py on an AF3 campaign grades the exam with the model that sat it.
_ENGINE=$(grep -lq 'from af3_adapter import' "$RUN"/run_nise_*.py 2>/dev/null && echo AF3 || echo Boltz)
say "fold check: campaign engine is ${_ENGINE}, cross-checking with the other model"
if [ "$_ENGINE" = "AF3" ]; then
  "$PY" tools/crossfold.py --run-dir "$RUN" --top 25 --mode complex --devices 0,1,2,3 \
       || say "WARNING: Boltz cross-fold returned non-zero"
else
say "AF3 monomer fold confidence, top 25 (GPU, ~5 min)"
if gpus_free; then
  "$PY" -u tools/af3_monomer.py --run-dir "$RUN" --top 25 --devices 0,1,2,3 \
     || say "WARNING: AF3 monomer stage returned non-zero"
else
  say "  run it later with:  $PY -u tools/af3_monomer.py --run-dir $RUN --top 25 --devices 0,1,2,3"
fi
fi

say "strict burial rescore (alpha=9)"
"$PY" tools/burial_rescore.py --run-dir "$RUN" || say "WARNING: burial rescore returned non-zero"
say "rebuilding the shortlist so AF3-monomer and burial columns enter the ranking"
"$PY" tools/final_shortlist.py --run-dir "$RUN" --top 25 || say "WARNING: shortlist rebuild returned non-zero"
say "SASA ranking"
"$PY" tools/sasa_rank.py --run-dir "$RUN" --top 25 || say "WARNING: sasa returned non-zero"

# --- 8. figures and the final Telegram report -----------------------------------------------
# Easy to forget, and the reporter is already dead by this point (step 2 kills it before any
# result exists), so without this the campaign finishes silently and produces no plots.
say "joined HTML report (all measures, sortable)"
"$PY" tools/make_report.py --run-dir "$RUN" || say "WARNING: report returned non-zero"
say "figures + final Telegram report"
"$PY" tools/paper_figures.py --run-dir "$RUN" --telegram || say "WARNING: figures/telegram returned non-zero"

# Report the TRUTH, not an aspiration. This line used to print the shortlist path
# unconditionally; after the cascade was SIGKILLed at 2770/11201 it still said "DONE ...
# shortlist.csv" and a Telegram went out from the figures stage (which runs regardless),
# so the run looked finished when nothing downstream had produced a single row.
if [ -s "$RUN/results/final/shortlist.csv" ]; then
  say "DONE. shortlist: $RUN/results/final/shortlist.csv ($(( $(wc -l < "$RUN/results/final/shortlist.csv") - 1 )) designs)"
else
  say "INCOMPLETE -- no shortlist was produced. Check the WARNING lines above;"
  say "  the cascade and the scoring passes must succeed before results/final/shortlist.csv exists."
fi
ls -1 results/final/*.csv results/final/sasa/*.csv 2>/dev/null | sed 's/^/  /'
