#!/usr/bin/env bash
# Independent gate runner. The gate() inside the running wm_run.sh watcher predates the
# RS_DINO_WEIGHTS_DIR fix and cannot be repaired in place -- its function is already defined in a
# live shell. This runs the CORRECT eval on each milestone checkpoint and pushes clips to HF, so
# footage does not depend on anyone's laptop staying awake.
# Weight pushes are still handled by the original watcher; this only adds the evals.
set -u
LOG=/workspace/bench.log; log(){ echo "$(date -u +%FT%TZ) gates2: $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=/workspace/pyextra RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
REPO=justincg/melee-wm-weights; OUT=/workspace/wm_run; STEPS=${STEPS:-100001}
cd /workspace/mira || exit 1

hf_up(){ pixi run python -c "
import os,sys
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
    repo_id='$REPO', repo_type='model')" "$1" "$2" >> "$LOG" 2>&1 \
  && log "up $2" || log "UP FAILED $2"; }

while true; do
  [ -f /workspace/.wm_done ] && { log "wm done, exiting"; exit 0; }
  CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
  if [ -n "$CK" ] && [ -s "$CK/checkpoint.pth" ]; then
    STEP=${CK##*-}
    for pct in 5 10 25 50 75; do
      T=$((STEPS * pct / 100))
      [ "$STEP" -ge "$T" ] && [ ! -f /workspace/.g2_$pct ] && {
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "${FREE:-0}" -lt 9000 ]; then
          log "gate $pct% deferred: only ${FREE}MiB VRAM free"
        else
          D=/workspace/g2_$STEP; mkdir -p $D
          log "gate $pct% ($STEP) eval start"
          pixi run python scripts/eval_world_model_offline.py $CK/checkpoint.pth \
            --viz 2 --num-samples 32 --skip-validation --no-compile --output-dir $D >> $D/eval.log 2>&1
          log "gate $pct% ($STEP): $(grep -iE 'psnr|lpips|ssim|fid|drift' $D/eval.log | tail -4 | tr '\n' ' ')"
          # The ORIGINAL watcher in wm_run.sh also uploads its (crashed) eval.log to
          # wm/gate_<step>/eval.log and clobbers ours. Land the good one under a name it
          # never writes. The mp4s are ours alone -- that watcher dies before viz.
          for f in $(find $D -type f | head -8); do
            B=$(basename $f); [ "$B" = "eval.log" ] && B=eval_full.log
            hf_up "$f" "wm/gate_$STEP/$B"
          done
          touch /workspace/.g2_$pct
        fi
      }
    done
  fi
  sleep 180
done
