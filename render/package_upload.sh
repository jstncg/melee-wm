#!/usr/bin/env bash
# On the data pod: package the games that have frame data -> shards, upload shards + raw data to HF, then stop the pod.
# Idempotent. Logs to /workspace/melee/render/package_upload.log. Run: nohup bash render/package_upload.sh &
set -u
ROOT=/workspace/melee
LOG=$ROOT/render/package_upload.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.local/bin:$PATH
cd "$ROOT"
set -a; . "$ROOT/.env"; set +a
export HF_REPO=${HF_REPO:-justincg/melee-fox-falcon-bf}

if [ ! -f shards/train/index.json ]; then
  rm -rf video_split && mkdir -p video_split/train video_split/test
  for f in video/*.mp4; do [ -s "${f%.mp4}.json" ] && echo "$f"; done | sort > /tmp/all_mp4.txt   # only games with frame data
  n=$(wc -l < /tmp/all_mp4.txt)
  head -n $((n - 40)) /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/train/$(basename "$f")"; ln -s "$PWD/${f%.mp4}.json" "video_split/train/$(basename "${f%.mp4}.json")"; done
  tail -n 40 /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/test/$(basename "$f")"; ln -s "$PWD/${f%.mp4}.json" "video_split/test/$(basename "${f%.mp4}.json")"; done
  log "STAGE package start: $((n - 40)) train / 40 test"
  PACKAGE_WORKERS=12 uv run python package.py video_split/train npz shards/train --games-per-shard 20 > render/package_train.log 2>&1
  PACKAGE_WORKERS=12 uv run python package.py video_split/test npz shards/test --games-per-shard 20 > render/package_test.log 2>&1
  log "STAGE package done: train=$(tail -1 render/package_train.log) | test=$(tail -1 render/package_test.log) | $(du -sh shards | cut -f1)"
fi

log "STAGE upload start repo=$HF_REPO"
uv run --with huggingface_hub python - <<'PY' >> "$LOG" 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
# upload_large_folder has no path_in_repo; upload the parent with allow_patterns so subdirs keep their names
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path="/workspace/melee",
                        allow_patterns=["shards/**", "npz/**", "slp/**", "video/**"])
print("uploaded shards npz slp video", flush=True)
PY
log "STAGE upload exit=$?"
log "DONE"
runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG" 2>&1
