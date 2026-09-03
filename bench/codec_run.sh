#!/usr/bin/env bash
# Full codec run. Env: HF_TOKEN, RUNPOD_API_KEY, POD_ID. Logs: /workspace/bench.log, /workspace/codec_run.log
# Resume: if /workspace/codec_run has a checkpoint, MIRA auto-resumes. On a fresh pod, set RESUME_FROM_HF=1 to warm-start from the HF weights.
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
REPO=justincg/melee-fox-falcon-bf; OUT=/workspace/codec_run; STEPS=${STEPS:-100001}
hf_up() { pixi run python -c "import os,sys; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2], repo_id='$REPO', repo_type='dataset'); print('up', sys.argv[2])" "$1" "$2" >> $LOG 2>&1; }
finish() { log "FINISH $1"; hf_up /workspace/bench.log bench/codec/bench.log; hf_up /workspace/codec_run.log bench/codec/codec_run.log; runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1; runpodctl remove pod "$POD_ID" >> $LOG 2>&1; }

until grep -q "pixi setup exit" $LOG; do sleep 15; done
cd /workspace/mira
pixi run python -c "import torch; assert torch.cuda.is_available(); print(torch.zeros(1).cuda().device, torch.cuda.get_device_name(0))" >> $LOG 2>&1 || { log "BAD HOST: CUDA init failed"; finish badhost; exit 1; }
until grep -q "data download done" $LOG; do sleep 15; done
if [ "${RESUME_FROM_HF:-0}" = 1 ] && [ ! -d $OUT ]; then
  mkdir -p $OUT/warm && pixi run python -c "import os; from huggingface_hub import hf_hub_download; [hf_hub_download('$REPO', f'bench/codec/{f}', repo_type='dataset', token=os.environ['HF_TOKEN'], local_dir='$OUT/warm') for f in ['checkpoint.pth','codec_config.yaml']]" >> $LOG 2>&1
  WARM="run.finetune_from=$OUT/warm/bench/codec/checkpoint.pth"; log "warm-starting from HF weights"
else WARM=""; fi

# watcher: NaN guard, pictures at 10k/50k, weights push at 25/50/75%
( last=""; while true; do sleep 120
    if grep -qE "total loss nan" /workspace/codec_run.log 2>/dev/null; then log "NAN detected"; pkill -f train_codec.py; sleep 5; finish nan; exit; fi
    for n in 10000 50000; do [ -f $OUT/checkpoint-$n/checkpoint.pth ] && [ ! -f /workspace/recon_$n.png ] && { pixi run python /workspace/recon.py $OUT/checkpoint-$n /workspace/recon_$n.png >> $LOG 2>&1; hf_up /workspace/recon_$n.png bench/codec/recon_$n.png; }; done
    for n in 25000 50000 75000; do [ -f $OUT/checkpoint-$n/checkpoint.pth ] && [ "$last" != "$n" ] && { last=$n; hf_up $OUT/checkpoint-$n/checkpoint.pth bench/codec/checkpoint.pth; hf_up $OUT/codec_config.yaml bench/codec/codec_config.yaml; log "pushed weights at $n"; }; done
    pgrep -f train_codec.py >/dev/null || exit
  done ) &

stamp() { while IFS= read -r l; do echo "$(date +%s.%N) $l"; done; }
log "CODEC RUN start steps=$STEPS $WARM"
pixi run python scripts/train_codec.py model=raev2_codec_melee dataset=melee wandb.mode=disabled run.compile=false \
  run.steps=$STEPS run.batch_size=2 run.checkpoint_every=5% run.checkpoint_keep_recent=2 run.log_every=1% \
  validation.val_first=false validation.val_every=10% validation.val_n_samples=64 optim.scheduler.decay_steps=$((STEPS-1001)) \
  dataloader.num_workers=8 run.output_dir=$OUT $WARM 2>&1 | stamp >> /workspace/codec_run.log
RC=${PIPESTATUS[0]}; log "CODEC RUN exit=$RC last: $(grep -E 'Step [0-9]+:' /workspace/codec_run.log | tail -1 | cut -d' ' -f2-)"
CK=$(ls -d $OUT/checkpoint-* | sort -V | tail -1)
[ "$RC" = 0 ] && { hf_up $CK/checkpoint.pth bench/codec/checkpoint.pth; hf_up $OUT/codec_config.yaml bench/codec/codec_config.yaml; pixi run python /workspace/recon.py $CK /workspace/recon_final.png >> $LOG 2>&1; hf_up /workspace/recon_final.png bench/codec/recon_final.png; }
grep -E "total loss nan" /workspace/codec_run.log -q && exit 0   # watcher handles it
finish "exit=$RC ckpt=$CK"
