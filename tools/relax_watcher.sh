#!/usr/bin/env bash
# Keep the relaxed pool caught up WHILE the campaign runs, instead of at the end.
#
# The in-loop relax is not this. It touches only relax_top_n (12) designs per iteration, just
# enough to gate which backbones get promoted. Everything else has to be minimised before the
# terminal cascade can score it, and that work has to happen somewhere.
#
# v8 had this watcher running beside it and its harvest caught up in six minutes. v9 did not,
# because the migration moved every tool but never moved what the old launcher STARTED, so
# 12,972 designs were relaxed in one block at the end while someone waited. Absorbing the cost
# during the run is the entire point. See docs/BUGS.md.
#
# relax_pool.py is incremental and locked, so overlapping invocations are harmless and a missed
# poll costs nothing. nice + a worker cap keep it from starving the driver's in-loop relax,
# which has a 1800 s timeout and silently falls back to the UNRELAXED ranking if it expires --
# that would change the campaign trajectory for the sake of this background job.
#
# Usage: bash tools/relax_watcher.sh --run-dir DIR [--parent-pid PID] [--workers N] [--interval S]
#        Normally started for you by `chlide run`; see src/chlide/loop.py.
set -u
# Code lives beside this script; DATA comes from --run-dir. Conflating the two is what broke
# both the v9 harvest and the watcher this file replaces.
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-/home/yigall/mambaforge/envs/lasermpnn/bin/python}
RUN=${CHLIDE_RUN_DIR:-}
PARENT=
WORKERS=${WORKERS:-12}
INTERVAL=${INTERVAL:-600}
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir)    RUN=${2:?--run-dir needs a path}; shift 2 ;;
    --parent-pid) PARENT=${2:?--parent-pid needs a pid}; shift 2 ;;
    --workers)    WORKERS=${2:?--workers needs a number}; shift 2 ;;
    --interval)   INTERVAL=${2:?--interval needs seconds}; shift 2 ;;
    *) echo "[fatal] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN" ] || { echo "[fatal] no run directory: pass --run-dir or set CHLIDE_RUN_DIR" >&2; exit 2; }
RUN=$(cd "$RUN" 2>/dev/null && pwd) || { echo "[fatal] run dir does not exist: $RUN" >&2; exit 2; }
[ -d "$RUN/input/sampling_dataframes" ] || {
  echo "[fatal] $RUN is not a campaign run directory (no input/sampling_dataframes/)" >&2
  exit 2; }

# One watcher per run. relax_pool locks internally, but two watchers double the CPU this is
# supposed to be gentle about.
exec 9>"$RUN/.relax_watcher.lock" || { echo "[fatal] cannot open lock" >&2; exit 2; }
flock -n 9 || { echo "[fatal] another relax watcher holds $RUN/.relax_watcher.lock" >&2; exit 3; }

say() { printf '[%s] %s\n' "$(date +%F' '%H:%M:%S)" "$*"; }

# Is the campaign still going? An explicit parent pid is exact; the pattern is the fallback for
# a watcher started by hand. The OLD watcher grepped "$RUN"/run_nise_*.py, which repo campaigns
# do not have -- that left DRIVER empty, so `pgrep -f ""` matched everything and it could never
# decide the campaign had ended.
campaign_alive() {
  if [ -n "$PARENT" ]; then
    kill -0 "$PARENT" 2>/dev/null
  else
    pgrep -f "chlide\.cli run .*--out ${RUN}([[:space:]]|/|$)" >/dev/null 2>&1
  fi
}

say "relax watcher started: workers=$WORKERS interval=${INTERVAL}s parent=${PARENT:-<pattern>} run=$RUN"
trap 'say "relax watcher stopping (signal)"; exit 0' TERM INT

fails=0
while true; do
  if OUT=$("$PY" "$TOOLS"/relax_pool.py --run-dir "$RUN" --workers "$WORKERS" 2>&1); then
    fails=0
  else
    # Never spin. The drain path below deliberately skips the sleep, so a relax_pool that
    # fails every time would otherwise busy-loop a core until somebody noticed.
    fails=$((fails + 1))
    say "relax_pool failed (consecutive failures: $fails)"
    printf '%s\n' "$OUT" | tail -3 | sed 's/^/    /'
    if [ "$fails" -ge 5 ]; then
      say "relax_pool failed $fails times in a row -- exiting rather than spinning"
      exit 1
    fi
    sleep "$INTERVAL"
    continue
  fi

  if ! printf '%s' "$OUT" | grep -q 'nothing pending'; then
    printf '%s\n' "$OUT" | grep -vE '^\s*$' | sed 's/^/    /'
  fi

  if ! campaign_alive; then
    # Leave only once the campaign is over AND nothing is left, so the harvest never inherits
    # a backlog. The pass above already ran, so this tests the current state.
    if printf '%s' "$OUT" | grep -q 'nothing pending'; then
      say "campaign finished and pool is caught up -- watcher exiting"
      exit 0
    fi
    continue          # drain without sleeping: nobody is competing for the cores now
  fi
  sleep "$INTERVAL"
done
