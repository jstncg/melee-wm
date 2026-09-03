#!/usr/bin/env bash
# Render sorted-list indices [START, END) with N parallel Dolphins. Same log format as render_all.sh.
# Usage: START=250 END=700 RENDER_PARALLEL=24 bash render_slice.sh
set -u
ROOT=${MELEE_ROOT:-/workspace/melee}; N=${RENDER_PARALLEL:-24}
cd "$ROOT"; mkdir -p video render/logs
find slp -name "*.slp" | sort | sed -n "$((START + 1)),${END}p" | { if [ -n "${REVERSE:-}" ]; then tac; else cat; fi; } | tr '\n' '\0' | xargs -0 -P "$N" -I{} bash -c '
  f="$1"; b=$(basename "$f" .slp)
  [ -s "video/$b.json" ] && exit 0
  python3 render/render_one.py "$f" "video/$b.mp4" > "render/logs/$b.log" 2>&1 \
    && echo "ok   $b $(python3 -c "import json;d=json.load(open(\"video/$b.json\"));print(d[\"mp4_frames\"],d[\"end_frame\"],d[\"seconds\"])")" \
    || echo "FAIL $b"
' _ {} >> render/render_all.log 2>&1
echo "render_slice done" >> render/render_all.log
