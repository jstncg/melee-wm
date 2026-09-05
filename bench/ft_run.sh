#!/usr/bin/env bash
# Decoder-only, motion-weighted codec fine-tune.
#
# Encoder (frozen DINOv3 + strided bottleneck) is frozen, so latents are unchanged and the existing
# world model stays valid. Only the 85M ViT decoder trains, with the reconstruction loss weighted
# toward whatever moves -- the audit measured characters reconstructing 9.66 dB worse than the
# static stage, which is what an unweighted mean over 97% background produces.
#
# Stages: selfcheck (free) -> pilot 2k -> gate on character PSNR -> full 20k -> verify + audit.
# pixi lives in /root and does not survive a pod restart, so the env python is called directly.
set -u
PY=/workspace/mira/.pixi/envs/default/bin/python
LOG=/workspace/ft.log
OUT=/workspace/codec_ft
BASE=/workspace/codec_frozen/checkpoint.pth
PILOT=${PILOT:-2000}
FULL=${FULL:-20000}
GATE_DB=${GATE_DB:-0.3}          # character-PSNR gain over baseline needed to justify the full run
export PYTHONPATH=/workspace/pyextra:/workspace/mira/src
export RS_DINO_WEIGHTS_DIR=/workspace/melee/weights
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
cd /workspace/mira || exit 1

log "STAGE patch"
$PY /workspace/apply_ft_patch.py /workspace/mira 2>&1 | tee -a $LOG
grep -q "motion_weight_strength: float" src/mira/codec/loss.py || { log "ABORT patch not applied"; exit 1; }

log "STAGE selfcheck (motion weight map)"
$PY /workspace/verify_ft.py --selfcheck 2>&1 | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || { log "ABORT selfcheck failed - weight map is wrong, not spending GPU"; exit 1; }

log "STAGE baseline audit"
$PY /workspace/codec_audit.py --ckpt $BASE --clips 40 2>&1 | grep -E "PSNR|LPIPS|gap" | tee -a $LOG
BASE_CHAR=$($PY -c "import json;print(json.load(open('/workspace/codec_audit.json'))['character_psnr'][0])")
cp /workspace/codec_audit.json /workspace/audit_baseline.json
log "baseline character PSNR = $BASE_CHAR"

train(){   # $1 = steps, $2 = output dir
  $PY scripts/train_codec.py model=raev2_codec_melee dataset=melee wandb.mode=disabled \
    run.compile=false run.steps=$1 run.batch_size=2 run.log_every=5% \
    run.checkpoint_every=50% run.checkpoint_keep_recent=1 run.output_dir=$2 \
    +run.freeze_encoder=true run.finetune_from=$BASE \
    +model.loss.weights.motion_weight_strength=9.0 \
    optim.optimizer.lr=5e-5 optim.scheduler.decay_steps=$(( $1 > 1001 ? $1 - 1001 : 1 )) \
    validation.val_first=false validation.val_every=50% validation.val_n_samples=32 \
    dataloader.num_workers=8 >> $LOG 2>&1
}
last_ckpt(){ ls -d $1/checkpoint-* 2>/dev/null | sort -V | tail -1; }

log "STAGE pilot ($PILOT steps)"
train $PILOT ${OUT}_pilot || { log "ABORT pilot training failed - see $LOG"; exit 1; }
PC=$(last_ckpt ${OUT}_pilot)/checkpoint.pth
[ -s "$PC" ] || { log "ABORT no pilot checkpoint"; exit 1; }
$PY /workspace/codec_audit.py --ckpt $PC --clips 40 2>&1 | grep -E "PSNR|LPIPS|gap" | tee -a $LOG
PILOT_CHAR=$($PY -c "import json;print(json.load(open('/workspace/codec_audit.json'))['character_psnr'][0])")
cp /workspace/codec_audit.json /workspace/audit_pilot.json
GAIN=$($PY -c "print(f'{$PILOT_CHAR-$BASE_CHAR:+.2f}')")
log "PILOT character PSNR $PILOT_CHAR vs baseline $BASE_CHAR -> $GAIN dB (gate ${GATE_DB})"

if $PY -c "import sys; sys.exit(0 if ($PILOT_CHAR-$BASE_CHAR) >= $GATE_DB else 1)"; then
  log "GATE PASS - the latent does carry the characters; running full $FULL steps"
else
  log "GATE FAIL - $GAIN dB in $PILOT steps. The encoder likely discarded the characters, so a"
  log "           better decoder cannot recover them. STOPPING before spending the full run."
  log "RESULT stopped_at_gate"; exit 0
fi

log "STAGE full ($FULL steps)"
train $FULL $OUT || { log "ABORT full training failed"; exit 1; }
FC=$(last_ckpt $OUT)/checkpoint.pth
[ -s "$FC" ] || { log "ABORT no final checkpoint"; exit 1; }

log "STAGE verify latents unchanged (world-model compatibility gate)"
$PY /workspace/verify_ft.py --latents $BASE $FC 2>&1 | tee -a $LOG
LAT=${PIPESTATUS[0]}

log "STAGE final audit"
$PY /workspace/codec_audit.py --ckpt $FC --clips 40 2>&1 | grep -E "PSNR|LPIPS|gap" | tee -a $LOG
cp /workspace/codec_audit.json /workspace/audit_final.json
FIN_CHAR=$($PY -c "import json;print(json.load(open('/workspace/audit_final.json'))['character_psnr'][0])")
FIN_BG=$($PY -c "import json;print(json.load(open('/workspace/audit_final.json'))['background_psnr'][0])")
BASE_BG=$($PY -c "import json;print(json.load(open('/workspace/audit_baseline.json'))['background_psnr'][0])")
log "RESULT character $BASE_CHAR -> $FIN_CHAR | background $BASE_BG -> $FIN_BG | latents_identical=$([ $LAT = 0 ] && echo yes || echo NO)"
log "STAGE push results to HF"
for f in /workspace/ft.log /workspace/audit_baseline.json /workspace/audit_pilot.json /workspace/audit_final.json; do
  [ -f "$f" ] || continue
  $PY - "$f" <<'PYPUSH' >> $LOG 2>&1 && log "pushed $(basename $f)" || log "push FAILED $(basename $f)"
import os, sys
from huggingface_hub import HfApi
f = sys.argv[1]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=f, path_in_repo="bench/codec_ft/" + os.path.basename(f),
    repo_id="justincg/melee-fox-falcon-bf", repo_type="dataset")
PYPUSH
done
log "DONE"
touch /workspace/.ft_done
