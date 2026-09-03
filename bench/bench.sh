#!/usr/bin/env bash
# Benchmark codec + world-model step time on this pod. Needs HF_TOKEN in env. Logs to /workspace/bench.log.
# Usage: nohup env HF_TOKEN=... bash bench.sh > /dev/null 2>&1 &
set -u
LOG=/workspace/bench.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); log "GPU=$GPU vcpu=$(nproc)"

# 1. env
cd /workspace
[ -x $HOME/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash >/dev/null 2>&1
[ -d mira ] || git clone -q https://github.com/mira-wm/mira.git
cd mira
[ -d .pixi/envs/default ] || { log "STAGE pixi setup start"; pixi run setup > /workspace/pixi_setup.log 2>&1; log "STAGE pixi setup exit=$?"; }
cp -r /workspace/melee_cfg/actions/. configs/actions/; cp -r /workspace/melee_cfg/dataset/. configs/dataset/
cp /workspace/melee_cfg/model/raev2_codec_melee.yaml configs/model/; mkdir -p configs/model/latent_world_model && cp /workspace/melee_cfg/model/latent_world_model/200m.yaml configs/model/latent_world_model/

# 2. data: shards + dino weights from HF
log "STAGE data download start"
pixi run python - <<'PY' >> /workspace/bench.log 2>&1
import os
from huggingface_hub import snapshot_download
snapshot_download("justincg/melee-fox-falcon-bf", repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir="/workspace/melee", allow_patterns=["shards/**", "weights/**"])
print("data ok")
PY
export RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
log "STAGE data download done: $(ls /workspace/melee/shards/train | wc -l) train files, weights=$(ls /workspace/melee/weights)"

# 3. codec: 60 steps at increasing batch until OOM; timestamp every step log
stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
COMMON="model=raev2_codec_melee dataset=melee wandb.mode=disabled run.compile=false run.steps=61 run.log_every=1 validation.val_first=false validation.val_every=100% validation.val_n_samples=8 run.checkpoint_every=100% optim.scheduler.warmup_steps=10 optim.scheduler.decay_steps=0 dataloader.num_workers=6"
for bs in 2 4 8 16; do
  log "CODEC bs=$bs start"
  pixi run python scripts/train_codec.py $COMMON run.batch_size=$bs run.output_dir=/workspace/codec_bs$bs 2>&1 | stamp > /workspace/codec_bs$bs.log
  if grep -q "OutOfMemoryError" /workspace/codec_bs$bs.log; then log "CODEC bs=$bs OOM"; break; fi
  spt=$(grep -E "Step [0-9]+:" /workspace/codec_bs$bs.log | awk '/Step 20:/{t0=$1} /Step 60:/{t1=$1} END{if(t0&&t1) printf "%.3f", (t1-t0)/40}')
  log "CODEC bs=$bs sec/step=${spt:-NA} maxmem=$(grep -oE 'max_memory[^,]*' /workspace/codec_bs$bs.log | tail -1)"
  [ -z "$spt" ] && { log "CODEC bs=$bs no Step lines, tail: $(grep -vE '^\s*$' /workspace/codec_bs$bs.log | tail -2 | cut -c1-200)"; break; }
done

# 4. world model: 40 steps with the bs=2 codec checkpoint (untrained; speed only)
CK=$(ls -d /workspace/codec_bs2/checkpoint* 2>/dev/null | tail -1)
log "WM codec_checkpoint=$CK"
WCOMMON="dataset=melee wandb.mode=disabled run.compile=false run.steps=41 run.log_every=1 validation.val_first=false validation.val_every=100% validation.val_n_samples=8 validation.downstream_val_every=100% run.checkpoint_every=100% optim.scheduler.warmup_steps=10 dataloader.num_workers=6 model/latent_world_model@model.architecture.config=200m model.architecture.config.video.width=384 model.architecture.config.video.height=288 model.architecture.config.codec_checkpoint=$CK"
for bs in 1 2 4 8; do
  log "WM bs=$bs start"
  pixi run python scripts/train_world_model.py $WCOMMON run.batch_size=$bs run.output_dir=/workspace/wm_bs$bs 2>&1 | stamp > /workspace/wm_bs$bs.log
  if grep -q "OutOfMemoryError" /workspace/wm_bs$bs.log; then log "WM bs=$bs OOM"; break; fi
  spt=$(grep -E "Step [0-9]+:" /workspace/wm_bs$bs.log | awk '/Step 10:/{t0=$1} /Step 40:/{t1=$1} END{if(t0&&t1) printf "%.3f", (t1-t0)/30}')
  log "WM bs=$bs sec/step=${spt:-NA}"
  [ -z "$spt" ] && { log "WM bs=$bs no Step lines, tail: $(grep -vE '^\s*$' /workspace/wm_bs$bs.log | tail -2 | cut -c1-200)"; break; }
done
log "BENCH DONE"
