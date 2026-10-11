#!/usr/bin/env bash
# Push this repo's tools into a campaign directory, ONE WAY, with the commit recorded.
#
# The campaign directories are not going away: v7 and v8 are finished archives and their
# drivers still run from there. What has to stop is editing a tool inside a campaign and
# hand-copying it to the next one -- that is how the same script came to exist in eight
# divergent versions, and how two stale engine labels shipped (docs/BUGS.md, 2025-10-11).
#
# So: the repo is the only place a tool is edited. This script carries it outward and drops
# a TOOLS_PROVENANCE file naming the commit, so a campaign can always say which version it ran.
# It refuses to run from a dirty tree, because a provenance file naming a commit that does not
# contain the code is worse than none.
#
# Usage: bash tools/sync_to_campaign.sh /home/data/yuvaln/nise_bdfp44_v8 [tool.py ...]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:?usage: sync_to_campaign.sh <campaign-dir> [tool.py ...]}
shift || true
[ -d "$DEST/tools" ] || { echo "no $DEST/tools -- is that a campaign directory?" >&2; exit 1; }

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "repo has uncommitted changes; commit first so the provenance is meaningful" >&2
    git status --short >&2
    exit 1
fi
SHA=$(git rev-parse HEAD)

TOOLS=("$@")
[ ${#TOOLS[@]} -eq 0 ] && TOOLS=(sasa_rank.py make_report.py paper_figures.py)

for t in "${TOOLS[@]}"; do
    [ -f "$REPO/tools/$t" ] || { echo "no such tool: $t" >&2; exit 1; }
    cp "$REPO/tools/$t" "$DEST/tools/$t"
    echo "  -> $DEST/tools/$t"
done

{
    echo "# Tools copied from the chlide-design repo. DO NOT EDIT THEM HERE."
    echo "# Edit in the repo and re-run tools/sync_to_campaign.sh, or the divergence starts again."
    echo "repo    = github.com/yuvalnudler/Chlide-Design"
    echo "commit  = $SHA"
    echo "synced  = $(date -Iseconds)"
    echo "tools   = ${TOOLS[*]}"
} > "$DEST/TOOLS_PROVENANCE"
echo "recorded $DEST/TOOLS_PROVENANCE @ ${SHA:0:8}"
