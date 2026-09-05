#!/usr/bin/env bash
# Single-player-architecture world model run (= the 2-player Melee WM: one shared camera,
# 32-key both-players action vocab). Env: HF_TOKEN. Optional: STEPS (default 100001).
# Logs: /workspace/bench.log (events), /workspace/wm_run.log (trainer)
# Resume: if /workspace/wm_run has a checkpoint, MIRA auto-resumes from it.
#
# This script NEVER deletes the pod. The previous version ended every exit path in
# `runpodctl remove pod`, which destroyed a 31k-step checkpoint that existed nowhere else.
# Stopping the pod is a human decision; the worst this script can do is idle.
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Without RS_DINO_WEIGHTS_DIR the eval's resolve_dino_weights() returns None and dinov3's hub code
# dies on Path(None). Training does not need it; every gate eval does.
export RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
DATA_REPO=justincg/melee-fox-falcon-bf          # private: shards, DINO weights, codec (read)
WEIGHTS_REPO=justincg/melee-wm-weights          # public model repo: no storage cap (write)
OUT=/workspace/wm_run; STEPS=${STEPS:-100001}
CODEC=/workspace/codec_frozen/checkpoint.pth
PYEXTRA=/workspace/pyextra
DONE=/workspace/.wm_done; NAN=/workspace/.wm_nan
MAX_ATTEMPTS=6

# hf_up REPORTS FAILURE. The old version always returned 0, so "pushed weights" was logged for
# uploads that had thrown a quota error -- which is why the 25k weights were believed safe and
# were not. A push that fails must say so.
hf_up() {
  pixi run python -c "
import os,sys
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(
    path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
    repo_id='$WEIGHTS_REPO', repo_type='model')
" "$1" "$2" >> $LOG 2>&1 \
    && { log "HF UP ok $2"; return 0; } \
    || { log "HF UP *** FAILED *** $2 (weights may exist only on this pod)"; return 1; }
}

# --- preflight -------------------------------------------------------------------------------
# Bounded wait: an unbounded `until grep` would idle the pod at full price forever if setup died.
for i in $(seq 1 240); do grep -q "pixi setup exit" $LOG && break; sleep 15; done
grep -q "pixi setup exit" $LOG || { log "ABORT: setup never finished (60 min)"; exit 1; }

cd /workspace/mira
# CUDA init blips are transient on these hosts: a check failed, then succeeded 30 s later. One
# failed probe must not end the run.
CUDA_OK=0
for i in 1 2 3; do
  pixi run python -c "import torch; assert torch.cuda.is_available(); print(torch.zeros(1).cuda().device, torch.cuda.get_device_name(0))" >> $LOG 2>&1 \
    && { CUDA_OK=1; break; }
  log "CUDA probe $i failed, retrying in 30s"; sleep 30
done
[ "$CUDA_OK" = 1 ] || { log "ABORT: CUDA unavailable after 3 probes"; exit 1; }

for i in $(seq 1 240); do grep -q "data download done" $LOG && break; sleep 15; done
grep -q "data download done" $LOG || { log "ABORT: data never downloaded (60 min)"; exit 1; }

# Frozen codec: pull the finished codec weights from the private dataset repo once.
if [ ! -f $CODEC ]; then
  mkdir -p /workspace/codec_frozen
  pixi run python -c "
import os,shutil
from huggingface_hub import hf_hub_download
[shutil.copy(hf_hub_download('$DATA_REPO', f'bench/codec/{f}', repo_type='dataset', token=os.environ['HF_TOKEN']),
             f'/workspace/codec_frozen/{f}') for f in ['checkpoint.pth','codec_config.yaml']]
" >> $LOG 2>&1 || { log "ABORT: codec checkpoint download failed"; exit 1; }
  log "frozen codec pulled from HF"
fi
[ -s $CODEC ] || { log "ABORT: codec checkpoint missing or empty"; exit 1; }

# Disk check by WRITE CANARY, not by df. df on this MooseFS mount reports the whole cluster
# (951TB total / 346TB avail) and is blind to the pod's 150GB quota -- on the volume that was
# actually full it still reported 780TB free. A full quota does not raise: dd returns success and
# the file lands at 0 bytes. Writing 8MB and reading the size back is the only honest test.
used_gb() { du -sBG /workspace 2>/dev/null | awk '{gsub("G","",$1); print $1+0}'; }
disk_ok() {
  dd if=/dev/zero of=/workspace/.canary bs=1M count=8 2>/dev/null
  local n; n=$(stat -c %s /workspace/.canary 2>/dev/null || echo 0)
  rm -f /workspace/.canary
  [ "${n:-0}" -ge 8388608 ]
}
disk_ok || { log "ABORT: /workspace drops writes at startup (quota full)"; exit 1; }
log "PREFLIGHT ok used=$(used_gb)G of 150G steps=$STEPS codec=$(du -h $CODEC | cut -f1)"

# --- gate ------------------------------------------------------------------------------------
# Runs the offline eval on the newest checkpoint and pushes its rollout clips + numbers to HF, so
# quality is reviewable without W&B. PYTHONPATH is what makes the metrics work: without pytorch_fid
# the eval dies on import and every gate so far has produced an empty log and no clips.
# Guarded on free VRAM: the eval shares the GPU with training and must never OOM it.
gate() {
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${FREE:-0}" -lt 9000 ] && { log "gate $1 deferred: only ${FREE}MiB VRAM free"; return 1; }
  D=/workspace/gate_$1; mkdir -p $D
  PYTHONPATH=$PYEXTRA pixi run python scripts/eval_world_model_offline.py $2/checkpoint.pth \
    --viz 2 --num-samples 32 --skip-validation --no-compile --output-dir $D >> $D/eval.log 2>&1
  log "gate $1: $(grep -iE 'psnr|lpips|ssim|fid|drift' $D/eval.log | tail -4 | tr '\n' ' ')"
  for f in $(find $D -type f | head -8); do hf_up $f wm/gate_$1/$(basename $f); done
  return 0
}

# --- watcher ---------------------------------------------------------------------------------
# NaN guard, disk guard, stall guard, milestone gates. Never hardcode step numbers: checkpoints
# land every 5% of STEPS, so fixed names silently never match (that is what broke the codec watcher).
# Exits on the done-marker, not on pgrep, so it survives the gap between training attempts.
( last=""; stall=0; prev=x
  while true; do
    sleep 120
    [ -f $DONE ] && exit

    if grep -qE "total loss nan" /workspace/wm_run.log 2>/dev/null; then
      log "NAN detected - stopping, this does not fix itself"; touch $NAN; pkill -f train_world_model.py; exit
    fi

    # Disk guard. A full /workspace does not raise: writes return success and store zero bytes,
    # which is how the 31k run died mid-checkpoint with no error in any log.
    if ! disk_ok; then
      log "DISK FULL (write canary truncated) at used=$(used_gb)G - pruning all but the newest checkpoint"
      ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | head -n -1 | xargs -r rm -rf
      if ! disk_ok; then
        log "DISK STILL FULL after prune - stopping cleanly before a checkpoint is silently truncated"
        touch $DONE; pkill -f train_world_model.py; exit
      fi
      log "DISK recovered after prune, used=$(used_gb)G"
    fi

    # Stall guard: a hung dataloader leaves the process alive at 0% GPU and would burn the whole
    # night at full price. Progress lines land every 1% (~6 min), so 30 min of no movement is dead.
    # Kill it and let the retry loop resume from the last checkpoint.
    NOW=$(grep -cE "Step [0-9]+:" /workspace/wm_run.log 2>/dev/null); NOW=${NOW:-0}
    if [ "$NOW" = "${prev:-x}" ]; then stall=$((stall+1)); else stall=0; prev=$NOW; fi
    if [ "$stall" -ge 15 ]; then
      log "STALL: no progress in 30 min at $NOW logged steps - killing, retry loop will resume"
      pkill -f train_world_model.py; stall=0
    fi

    CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CK" ] && [ -s "$CK/checkpoint.pth" ]; then
      STEP=${CK##*-}
      # 5% first: it lands ~35 min in and is the earliest point a broken setup becomes visible
      # as footage rather than as a scalar with no absolute scale.
      for pct in 5 10 25 50 75; do
        T=$((STEPS * pct / 100))
        [ "$STEP" -ge "$T" ] && [ ! -f /workspace/.gate_$pct ] && {
          gate $STEP "$CK" && { touch /workspace/.gate_$pct
            [ "$last" != "$CK" ] && { last=$CK; hf_up $CK/checkpoint.pth wm/checkpoint.pth && log "weights safe off-pod at step $STEP"; }; }
        }
      done
    fi
  done ) &
WATCHER=$!

# --- train -----------------------------------------------------------------------------------
# Retry loop: a transient CUDA blip killed the last run silently at step 31k and nothing restarted
# it. MIRA auto-resumes from the newest checkpoint, so a retry costs at most 5% of STEPS.
# width=384: model=latent_world_model hardcodes 512 (Rocket League); Melee is 288x384.
stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
last_step() { grep -oE 'Step [0-9]+:' /workspace/wm_run.log 2>/dev/null | tail -1 | tr -dc 0-9; }

RC=1
for attempt in $(seq 1 $MAX_ATTEMPTS); do
  [ -f $NAN ] && break
  log "WM RUN attempt $attempt/$MAX_ATTEMPTS from step $(last_step) steps=$STEPS bs=2 used=$(used_gb)G"
  pixi run python scripts/train_world_model.py dataset=melee \
    model/latent_world_model@model.architecture.config=200m \
    model.architecture.config.codec_checkpoint=$CODEC \
    model.architecture.config.video.width=384 \
    wandb.mode=disabled run.compile=false \
    run.steps=$STEPS run.batch_size=2 run.checkpoint_every=5% run.checkpoint_keep_recent=2 run.log_every=1% \
    validation.val_first=false validation.val_every=5% validation.val_n_samples=64 \
    optim.scheduler.decay_steps=$((STEPS-1001)) \
    dataloader.num_workers=8 run.output_dir=$OUT 2>&1 | stamp >> /workspace/wm_run.log
  RC=${PIPESTATUS[0]}
  S=$(last_step); S=${S:-0}
  log "WM RUN attempt $attempt exit=$RC at step $S"
  [ "$RC" = 0 ] && break
  [ -f $NAN ] && break
  [ "$S" -ge $((STEPS - 1500)) ] && { log "close enough to target at $S, stopping"; break; }
  sleep 60
done

# --- finish ----------------------------------------------------------------------------------
# Push weights on EVERY exit path, not only a clean one. A crashed run's checkpoint is exactly the
# one worth keeping, and the old script skipped the push unless RC was 0.
touch $DONE
sleep 3; kill $WATCHER 2>/dev/null
PUSHED=0
CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$CK" ] && [ -s "$CK/checkpoint.pth" ]; then
  hf_up $CK/checkpoint.pth wm/checkpoint.pth && PUSHED=1
  hf_up $CK/../world_model_config.yaml wm/world_model_config.yaml 2>/dev/null
  if [ "$RC" = 0 ]; then
    PYTHONPATH=$PYEXTRA pixi run python scripts/eval_world_model_offline.py $CK/checkpoint.pth \
      --viz 4 --num-samples 64 --no-compile --output-dir /workspace/gate_final >> /workspace/gate_final.log 2>&1
    for f in $(find /workspace/gate_final -type f 2>/dev/null | head -10); do hf_up $f wm/final/$(basename $f); done
  fi
else
  log "*** NO CHECKPOINT ON DISK AT EXIT ***"
fi
hf_up /workspace/bench.log wm/bench.log
hf_up /workspace/wm_run.log wm/wm_run.log
log "FINISH exit=$RC step=$(last_step) ckpt=$CK pushed=$PUSHED"

# STOP -- never REMOVE. `stop` halts GPU billing and keeps the volume and every checkpoint on it;
# `remove` is what destroyed the 31k run and appears nowhere in this script. Gated on a CONFIRMED
# weights upload, so the pod only ever stops once the result exists somewhere other than this disk.
if [ "$PUSHED" = 1 ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${POD_ID:-}" ]; then
  log "weights confirmed on HF - stopping (NOT deleting) the pod so it cannot idle at \$0.99/h"
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
  runpodctl stop pod "$POD_ID" >> $LOG 2>&1
else
  log "NOT stopping the pod: weights push unconfirmed (pushed=$PUSHED). Pod idles until stopped by hand."
fi
