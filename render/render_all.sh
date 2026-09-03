#!/usr/bin/env bash
# Render every .slp in $ROOT/slp to $ROOT/video with N parallel Dolphins. Skips games already done.
# Usage: RENDER_PARALLEL=24 bash render_all.sh
set -u
ROOT=${MELEE_ROOT:-/workspace/melee}
N=${RENDER_PARALLEL:-24}
cd "$ROOT"
mkdir -p video render/logs
find slp -name '*.slp' -print0 | sort -z | xargs -0 -P "$N" -I{} bash -c '
  f="$1"; b=$(basename "$f" .slp)
  [ -s "video/$b.json" ] && exit 0
  python3 render/render_one.py "$f" "video/$b.mp4" > "render/logs/$b.log" 2>&1 \
    && echo "ok   $b $(python3 -c "import json;d=json.load(open(\"video/$b.json\"));print(d[\"mp4_frames\"],d[\"end_frame\"],d[\"seconds\"])")" \
    || echo "FAIL $b"
' _ {} >> render/render_all.log 2>&1
echo "render_all done: $(grep -c "^ok" render/render_all.log) ok, $(grep -c "^FAIL" render/render_all.log) failed" >> render/render_all.log
