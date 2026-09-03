#!/usr/bin/env bash
# Finish the dataset on pod 1: render the last games (32 parallel) -> package shards -> upload shards to HF.
# Idempotent. Logs to /workspace/melee/render/finish.log. Run: nohup bash render/finish.sh &
set -u
ROOT=/workspace/melee
LOG=$ROOT/render/finish.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.local/bin:$HOME/.pixi/bin:$PATH
cd "$ROOT"

log "STAGE render start: $(ls video/*.mp4 2>/dev/null | wc -l)/889 present"
RENDER_PARALLEL=${RENDER_PARALLEL:-32} bash render/render_all.sh
RENDER_PARALLEL=24 bash render/render_all.sh   # retry pass for startup races
log "STAGE render done: $(ls video/*.mp4 | wc -l)/889"

if [ ! -f shards/train/index.json ]; then
  rm -rf video_split && mkdir -p video_split/train video_split/test
  ls video/*.mp4 | sort > /tmp/all_mp4.txt
  n=$(wc -l < /tmp/all_mp4.txt)
  head -n $((n - 40)) /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/train/$(basename "$f")"; done
  tail -n 40 /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/test/$(basename "$f")"; done
  log "STAGE package start: $((n - 40)) train / 40 test"
  PACKAGE_WORKERS=32 uv run python package.py video_split/train npz shards/train --games-per-shard 20 > render/package_train.log 2>&1
  PACKAGE_WORKERS=32 uv run python package.py video_split/test npz shards/test --games-per-shard 20 > render/package_test.log 2>&1
  log "STAGE package done: $(du -sh shards | cut -f1)"
fi

set -a; . "$ROOT/.env"; set +a   # HF_TOKEN read at upload time so a write token can be dropped in during the render
export HF_REPO=${HF_REPO:-justincg/melee-fox-falcon-bf}
log "STAGE upload start repo=$HF_REPO"
uv run --with huggingface_hub python - <<'PY' >> "$LOG" 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path="/workspace/melee/shards")
print("upload ok", repo)
PY
log "STAGE upload exit=$?"
log "FINISH done"
runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG" 2>&1   # self-stop: billing ends even if nobody is watching
