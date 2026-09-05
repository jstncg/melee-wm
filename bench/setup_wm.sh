#!/usr/bin/env bash
# One-time pod setup for the world-model run: pixi env, mira repo, Melee configs, shards,
# DINO weights, and the eval's pytorch_fid dependency.
# Writes the two markers wm_run.sh waits on: "pixi setup exit" and "data download done".
# Env: HF_TOKEN. Log: /workspace/bench.log
# Stages 1-2 are lifted verbatim from bench.sh, which is proven; the benchmark stages 3-4 are
# dropped (they burn ~40 min and disk re-measuring numbers we already have).
set -u
LOG=/workspace/bench.log; log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export PATH=$HOME/.pixi/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_REPO=justincg/melee-fox-falcon-bf

log "SETUP start GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) vcpu=$(nproc) free=$(df -h /workspace | awk 'NR==2{print $4}')"

# A full disk fails SILENTLY on this filesystem: writes return success and store zero bytes.
# df is useless here -- it reports the whole MooseFS cluster (951TB), not the pod's 150GB quota,
# and reported 780TB free on the volume that was actually full. Test by writing and reading back.
dd if=/dev/zero of=/workspace/.canary bs=1M count=8 2>/dev/null
CN=$(stat -c %s /workspace/.canary 2>/dev/null || echo 0); rm -f /workspace/.canary
[ "${CN:-0}" -ge 8388608 ] || { log "SETUP ABORT: /workspace drops writes (canary landed at ${CN} bytes)"; exit 1; }
log "SETUP disk canary ok, used=$(du -sBG /workspace 2>/dev/null | cut -f1) of 150G"

# 1. env
cd /workspace
[ -x $HOME/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash >/dev/null 2>&1
[ -d mira ] || git clone -q https://github.com/mira-wm/mira.git
cd mira
[ -d .pixi/envs/default ] || { log "STAGE pixi setup start"; pixi run setup > /workspace/pixi_setup.log 2>&1; log "STAGE pixi setup exit=$?"; }
grep -q "pixi setup exit" $LOG || log "STAGE pixi setup exit=0 (env already present)"

# 2. Melee configs. These live on HF under bench/melee_cfg/, not in the mira repo.
if [ ! -d /workspace/melee_cfg ]; then
  pixi run python - <<PY >> $LOG 2>&1
import os, shutil, pathlib
from huggingface_hub import snapshot_download
d = snapshot_download("$DATA_REPO", repo_type="dataset", token=os.environ["HF_TOKEN"],
                      local_dir="/workspace/cfgdl", allow_patterns=["bench/melee_cfg/**"])
shutil.copytree(pathlib.Path(d) / "bench" / "melee_cfg", "/workspace/melee_cfg", dirs_exist_ok=True)
print("cfg ok")
PY
fi
[ -f /workspace/melee_cfg/dataset/melee.yaml ] || { log "SETUP ABORT: melee_cfg missing after download"; exit 1; }
cp -r /workspace/melee_cfg/actions/. configs/actions/
cp -r /workspace/melee_cfg/dataset/. configs/dataset/
cp /workspace/melee_cfg/model/raev2_codec_melee.yaml configs/model/
mkdir -p configs/model/latent_world_model && cp /workspace/melee_cfg/model/latent_world_model/200m.yaml configs/model/latent_world_model/
log "STAGE configs copied target_fps=$(grep -h target_fps /workspace/melee_cfg/dataset/melee.yaml)"

# 3. data: shards + DINO weights from HF (~21 GB)
if [ ! -d /workspace/melee/shards/train ]; then
  log "STAGE data download start"
  pixi run python - <<PY >> $LOG 2>&1
import os
from huggingface_hub import snapshot_download
snapshot_download("$DATA_REPO", repo_type="dataset", token=os.environ["HF_TOKEN"],
                  local_dir="/workspace/melee", allow_patterns=["shards/**", "weights/**"])
print("data ok")
PY
fi
NTRAIN=$(ls /workspace/melee/shards/train 2>/dev/null | wc -l)
[ "$NTRAIN" -lt 30 ] && { log "SETUP ABORT: only $NTRAIN train shards, expected 36+"; exit 1; }

# 4. pytorch_fid for the offline eval. The pixi env has no pip, and `pixi add` would re-solve the
# environment the training run depends on, so install to a side directory the eval gets on PYTHONPATH.
# --no-deps is REQUIRED: without it uv pulls a second torch + the CUDA toolkit into pyextra (5.1 GB),
# and PYTHONPATH puts that shadow torch ahead of the pixi env's. scipy is already in the pixi env.
PYBIN=/workspace/mira/.pixi/envs/default/bin/python
if ! PYTHONPATH=/workspace/pyextra $PYBIN -c "import pytorch_fid" 2>/dev/null; then
  uv pip install --python $PYBIN --target /workspace/pyextra --no-deps pytorch-fid >> $LOG 2>&1 \
    || $PYBIN -m pip install --target /workspace/pyextra --no-deps pytorch-fid >> $LOG 2>&1
fi
PYTHONPATH=/workspace/pyextra $PYBIN -c "
import pytorch_fid, torch, scipy
assert '/pyextra/' not in torch.__file__, f'shadow torch on PYTHONPATH: {torch.__file__}'
" 2>/dev/null && log "STAGE pytorch_fid ok" \
  || log "WARN pytorch_fid broken - gate evals will produce no metrics"

# 5. The eval's Frechet metric loads a SECOND DINOv3, dinov3_vitb16, separate from the world
# model's vitl16. We only have vitl16, and Meta's default URL for vitb16 returns HTTP 403, so
# resolve_dino_weights() returns None, mira passes weights=None, and dinov3's hub code raises
# TypeError on Path(None). Point the metric at the backbone we actually have.
# Cost: the Frechet-DINO numbers are computed with vitl16, so they are internally consistent
# across gates but NOT comparable to MIRA's published vitb16 figures. PSNR/LPIPS/SSIM/FID unaffected.
WMM=/workspace/mira/src/mira/training/metrics/world_model_metrics.py
if grep -q 'DinoForMetrics(model_size="base")' $WMM; then
  sed -i 's/DinoForMetrics(model_size="base")/DinoForMetrics(model_size="large")/' $WMM
  log "STAGE patched metrics DINO base -> large (no vitb16 weights available)"
fi

log "STAGE data download done: $NTRAIN train files, weights=$(ls /workspace/melee/weights 2>/dev/null | tr '\n' ' ')"
log "SETUP done free=$(df -h /workspace | awk 'NR==2{print $4}')"
