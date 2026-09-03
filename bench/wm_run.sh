#!/usr/bin/env bash
# Single-player-architecture world model run (= the 2-player Melee WM: one shared camera,
# 32-key both-players action vocab). Env: HF_TOKEN, RUNPOD_API_KEY, POD_ID.
# Logs: /workspace/bench.log, /workspace/wm_run.log
# Resume: if /workspace/wm_run has a checkpoint, MIRA auto-resumes.
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=justincg/melee-fox-falcon-bf; OUT=/workspace/wm_run; STEPS=${STEPS:-100001}
# W&B is the mid-run quality gate: the trainer logs 8 rollout videos + drift/PSNR at every val.
WANDB_MODE=$([ -n "${WANDB_API_KEY:-}" ] && echo online || echo disabled); export WANDB_MODE
CODEC=/workspace/codec_frozen/checkpoint.pth
hf_up() { pixi run python -c "import os,sys; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], repo_id='$REPO', repo_type='dataset'); print('up', sys.argv[2])" "$1" "$2" >> $LOG 2>&1; }
finish() { log "FINISH $1"; hf_up /workspace/bench.log bench/wm/bench.log; hf_up /workspace/wm_run.log bench/wm/wm_run.log; runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1; runpodctl remove pod "$POD_ID" >> $LOG 2>&1; }

until grep -q "pixi setup exit" $LOG; do sleep 15; done
cd /workspace/mira
pixi run python -c "import torch; assert torch.cuda.is_available(); print(torch.zeros(1).cuda().device, torch.cuda.get_device_name(0))" >> $LOG 2>&1 || { log "BAD HOST: CUDA init failed"; finish badhost; exit 1; }
until grep -q "data download done" $LOG; do sleep 15; done

# Frozen codec: pull the finished codec weights from HF once.
if [ ! -f $CODEC ]; then
  mkdir -p /workspace/codec_frozen
  pixi run python -c "import os,shutil; from huggingface_hub import hf_hub_download; [shutil.copy(hf_hub_download('$REPO', f'bench/codec/{f}', repo_type='dataset', token=os.environ['HF_TOKEN']), f'/workspace/codec_frozen/{f}') for f in ['checkpoint.pth','codec_config.yaml']]" >> $LOG 2>&1 \
    || { log "BAD: codec checkpoint download failed"; finish nocodec; exit 1; }
  log "frozen codec pulled from HF"
fi

# watcher: NaN guard, weights push at 25/50/75%
# Pushes the NEWEST checkpoint every 30 min, whatever STEPS is. Do not hardcode step numbers:
# checkpoints land every 5% of STEPS, so fixed names silently never match (that broke the codec watcher).
( last=""; n=0; while true; do sleep 120; n=$((n+1))
    if grep -qE "total loss nan" /workspace/wm_run.log 2>/dev/null; then log "NAN detected"; pkill -f train_world_model.py; sleep 5; finish nan; exit; fi
    if [ $((n % 15)) = 0 ]; then
      CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
      [ -n "$CK" ] && [ -f "$CK/checkpoint.pth" ] && [ "$last" != "$CK" ] && { last=$CK; hf_up $CK/checkpoint.pth bench/wm/checkpoint.pth; log "pushed weights from $CK"; }
    fi
    pgrep -f train_world_model.py >/dev/null || exit
  done ) &

stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
log "WM RUN start steps=$STEPS bs=2 codec=$CODEC"
# width=384: model=latent_world_model hardcodes 512 (Rocket League); Melee is 288x384.
pixi run python scripts/train_world_model.py dataset=melee \
  model/latent_world_model@model.architecture.config=200m \
  model.architecture.config.codec_checkpoint=$CODEC \
  model.architecture.config.video.width=384 \
  wandb.mode=$WANDB_MODE wandb.project=melee-wm run.compile=false \
  run.steps=$STEPS run.batch_size=2 run.checkpoint_every=5% run.checkpoint_keep_recent=2 run.log_every=1% \
  validation.val_first=false validation.val_every=5% validation.val_n_samples=64 \
  optim.scheduler.decay_steps=$((STEPS-1001)) \
  dataloader.num_workers=8 run.output_dir=$OUT 2>&1 | stamp >> /workspace/wm_run.log
RC=${PIPESTATUS[0]}; log "WM RUN exit=$RC last: $(grep -E 'Step [0-9]+:' /workspace/wm_run.log | tail -1 | cut -d' ' -f2-)"
CK=$(ls -d $OUT/checkpoint-* | sort -V | tail -1)
[ "$RC" = 0 ] && { hf_up $CK/checkpoint.pth bench/wm/checkpoint.pth; pixi run python scripts/eval_world_model_offline.py $CK/checkpoint.pth >> $LOG 2>&1; hf_up $LOG bench/wm/eval.log; }
grep -E "total loss nan" /workspace/wm_run.log -q && exit 0   # watcher handles it
finish "exit=$RC ckpt=$CK"
