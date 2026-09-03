#!/usr/bin/env bash
# Chain the WM run onto the SAME secure pod after the codec finishes, instead of deleting the pod
# and building a new one: the env, the 20 GB dataset, the DINO weights and the codec checkpoint are
# all already here. Saves the setup hour and the bad-host lottery.
# codec_run.sh's finish() deletes the pod, so runpodctl is parked until that wrapper has exited.
# Env: HF_TOKEN, RUNPOD_API_KEY, POD_ID.
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) chain: $*" >> "$LOG"; }
RPCTL=$(command -v runpodctl)
mv "$RPCTL" "$RPCTL.off" && log "parked runpodctl so the codec finish() cannot delete this pod"
# Wait for the codec WRAPPER (not just the trainer) so finish() has fully run before restoring.
while pgrep -f "bash /workspace/codec_run.sh" >/dev/null; do sleep 30; done
mv "$RPCTL.off" "$RPCTL"; log "runpodctl restored"
if ! grep -q "CODEC RUN exit=0" "$LOG"; then log "codec did NOT exit 0 - not chaining, deleting pod"; runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1; runpodctl remove pod "$POD_ID" >> "$LOG" 2>&1; exit 1; fi
CK=$(ls -d /workspace/codec_run/checkpoint-* | sort -V | tail -1)
mkdir -p /workspace/codec_frozen
cp "$CK/checkpoint.pth" /workspace/codec_run/codec_config.yaml /workspace/codec_frozen/ || { log "codec copy failed"; exit 1; }
log "starting WM with frozen codec from $CK"
exec bash /workspace/wm_run.sh
