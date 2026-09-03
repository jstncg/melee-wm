#!/usr/bin/env bash
# Smoke run: 1000 codec steps on the full train shards, then reconstruct one held-out clip and upload the PNG.
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
until grep -q "data download done" $LOG; do sleep 15; done
cd /workspace/mira
stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
log "SMOKE codec 1001 steps start"
pixi run python scripts/train_codec.py model=raev2_codec_melee dataset=melee wandb.mode=disabled run.compile=false \
  run.steps=1001 run.batch_size=2 run.log_every=5% validation.val_first=false validation.val_every=50% validation.val_n_samples=16 \
  run.checkpoint_every=100% optim.scheduler.warmup_steps=100 optim.scheduler.decay_steps=900 dataloader.num_workers=8 \
  run.output_dir=/workspace/smoke_codec 2>&1 | stamp > /workspace/smoke_codec.log
log "SMOKE train exit; steps logged: $(grep -c 'Step [0-9]*:' /workspace/smoke_codec.log); first/last: $(grep -E 'Step [0-9]+:' /workspace/smoke_codec.log | sed -n '1p;$p' | cut -d' ' -f2- | tr '\n' ' ')"
grep -E "val/|validation" /workspace/smoke_codec.log | tail -2 | cut -c1-200 | while read -r l; do log "VAL $l"; done
CK=$(ls -d /workspace/smoke_codec/checkpoint* | tail -1); log "SMOKE checkpoint=$CK"
pixi run python /workspace/recon.py "$CK" /workspace/smoke_recon.png >> $LOG 2>&1
pixi run python - <<'PY' >> $LOG 2>&1
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/workspace/smoke_recon.png", path_in_repo="bench/smoke/recon.png", repo_id="justincg/melee-fox-falcon-bf", repo_type="dataset"); print("png uploaded")
PY
log "SMOKE DONE"
