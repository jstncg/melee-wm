#!/usr/bin/env bash
# Export pod 1's data to a private HF dataset so any pod (community cloud) can pull it. No ISO (copyrighted; lives on the Mac).
# Needs HF_TOKEN and HF_REPO in env. Idempotent (upload_large_folder resumes). Logs to render/export.log.
set -u
ROOT=/workspace/melee
LOG=$ROOT/render/export.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.local/bin:$PATH
cd "$ROOT"
log "START video=$(ls video/*.mp4 | wc -l) npz=$(ls npz/*.npz 2>/dev/null | wc -l) slp=$(ls slp/*.slp | wc -l) size=$(du -shc video npz slp | tail -1 | cut -f1)"
uv run --with huggingface_hub python - <<'PY' >> "$LOG" 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for d in ["npz", "slp", "video"]:
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=f"/workspace/melee/{d}", path_in_repo=d)  # ponytail: path_in_repo unsupported in old hub versions; verify on pod
    print("uploaded", d, flush=True)
PY
log "EXIT $?"
