#!/usr/bin/env bash
# Overnight pipeline on the GPU pod: wait for render_all -> package shards (train/test) -> train codec.
# Logs to /workspace/melee/render/pipeline.log. Idempotent: re-run resumes at the first unfinished stage.
set -u
ROOT=/workspace/melee
LOG=$ROOT/render/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.local/bin:$HOME/.pixi/bin:$PATH

# 1. wait for the render
until grep -q "render_all done" "$ROOT/render/render_all.log" 2>/dev/null; do sleep 120; done
until [ "$(pgrep -fc "AppRun[.]wrapped")" = "0" ]; do sleep 60; done   # extra renderers (render_rev.sh) may still be finishing
# a second pod renders a slice and syncs results here; wait until (nearly) every game has arrived, then a grace period
until [ "$(ls $ROOT/video/*.mp4 2>/dev/null | wc -l)" -ge "${MIN_GAMES:-880}" ]; do sleep 120; done
sleep 600
# retry pass: render_all skips finished games, so this only re-runs failures (e.g. Dolphin startup races)
log "STAGE retry pass: $(ls $ROOT/video/*.mp4 | wc -l) games present"
RENDER_PARALLEL=24 bash "$ROOT/render/render_all.sh"
log "STAGE render done: $(tail -1 "$ROOT/render/render_all.log")"

# 2. package into MIRA shards, last 40 games held out as test
cd "$ROOT"
if [ ! -f shards/train/index.json ]; then
  rm -rf video_split && mkdir -p video_split/train video_split/test
  ls video/*.mp4 | sort > /tmp/all_mp4.txt
  n=$(wc -l < /tmp/all_mp4.txt)
  head -n $((n - 40)) /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/train/$(basename "$f")"; done
  tail -n 40 /tmp/all_mp4.txt | while read -r f; do ln -s "$PWD/$f" "video_split/test/$(basename "$f")"; done
  log "STAGE package start: $((n - 40)) train / 40 test games"
  PACKAGE_WORKERS=24 uv run python package.py video_split/train npz shards/train --games-per-shard 20 > render/package_train.log 2>&1
  PACKAGE_WORKERS=24 uv run python package.py video_split/test npz shards/test --games-per-shard 20 > render/package_test.log 2>&1
  log "STAGE package done: $(tail -1 render/package_train.log) | $(tail -1 render/package_test.log) | $(du -sh shards | cut -f1)"
fi

# 3. codec training (batch 2 fits a 24 GB 4090; batch 4 OOMs)
cd /workspace/mira
export RS_DINO_WEIGHTS_DIR=$ROOT/weights
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "STAGE codec train start"
pixi run python scripts/train_codec.py model=raev2_codec_melee dataset=melee \
  wandb.mode=disabled run.output_dir=$ROOT/codec_run run.compile=false \
  run.steps=100_001 run.batch_size=2 optim.scheduler.decay_steps=99_000 \
  run.checkpoint_every=5% run.log_every=1% validation.val_every=5% validation.val_n_samples=256 \
  dataloader.num_workers=8 > $ROOT/render/codec_train.log 2>&1
log "STAGE codec train exit=$? : $(grep -E 'Step [0-9]+:' $ROOT/render/codec_train.log | tail -1)"
