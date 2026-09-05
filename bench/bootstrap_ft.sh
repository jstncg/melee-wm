#!/usr/bin/env bash
# Rebuild a fresh pod from HF, then run the decoder-only fine-tune end to end.
# The previous pod's host filled up while it was stopped, so nothing is reused from it -- every
# artefact (shards, DINO weights, codec, world model) is pulled from HF.
set -u
LOG=/workspace/bench.log
log(){ echo "$(date -u +%FT%TZ) boot: $*" | tee -a $LOG; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH
DATA_REPO=justincg/melee-fox-falcon-bf

# Host network is the single biggest variance between pods: one host gave 1.4 MB/s from HF, which
# turns the 21 GB pull into 4.2 hours. Fail in 30 s on a slow host instead of an hour into setup.
log "STAGE speed gate"
SPD=$(curl -o /dev/null -sL -w "%{speed_download}" --max-time 20 \
      https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin)
SPD_MB=$(( ${SPD%.*} / 1000000 ))
log "HF sustained ${SPD_MB} MB/s"
if [ "$SPD_MB" -lt 8 ]; then
  log "ABORT slow host (${SPD_MB} MB/s < 8). Delete this pod and create another; do not wait."
  exit 1
fi

log "STAGE setup_wm.sh"
bash /workspace/setup_wm.sh >> $LOG 2>&1 || { log "ABORT setup failed"; exit 1; }

if [ ! -f /workspace/codec_frozen/checkpoint.pth ]; then
  log "STAGE codec download"
  mkdir -p /workspace/codec_frozen
  /root/mira/.pixi/envs/default/bin/python -c "
import os,shutil
from huggingface_hub import hf_hub_download
[shutil.copy(hf_hub_download('$DATA_REPO', f'bench/codec/{f}', repo_type='dataset', token=os.environ['HF_TOKEN']),
             f'/workspace/codec_frozen/{f}') for f in ['checkpoint.pth','codec_config.yaml']]
print('codec ok')" >> $LOG 2>&1 || { log "ABORT codec download failed"; exit 1; }
fi
[ -s /workspace/codec_frozen/checkpoint.pth ] || { log "ABORT codec missing"; exit 1; }
log "codec ready: $(du -h /workspace/codec_frozen/checkpoint.pth | cut -f1)"

log "STAGE fine-tune"
bash /workspace/ft_run.sh
log "BOOTSTRAP DONE"
