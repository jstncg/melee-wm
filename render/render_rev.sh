#!/usr/bin/env bash
# Extra renderer: same as render_all.sh but walks the list in reverse so it does not collide with it.
set -u
ROOT=${MELEE_ROOT:-/workspace/melee}; N=${RENDER_PARALLEL:-8}
cd "$ROOT"; mkdir -p video render/logs
find slp -name '*.slp' -print0 | sort -z -r | xargs -0 -P "$N" -I{} bash -c '
  f="$1"; b=$(basename "$f" .slp)
  [ -s "video/$b.json" ] && exit 0
  python3 render/render_one.py "$f" "video/$b.mp4" > "render/logs/$b.log" 2>&1 \
    && echo "ok   $b $(python3 -c "import json;d=json.load(open(\"video/$b.json\"));print(d[\"mp4_frames\"],d[\"end_frame\"],d[\"seconds\"])")" \
    || echo "FAIL $b"
' _ {} >> render/render_all.log 2>&1
