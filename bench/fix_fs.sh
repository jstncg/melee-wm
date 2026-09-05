#!/usr/bin/env bash
# /workspace is MooseFS over FUSE. Installing the pixi env there died with
# "Stale file handle (os error 116)" mid-wheel-install -- the same network-FS class of failure that
# silently corrupted the WM run on 2026-09-03.
#
# Put the environment on the container's local overlay disk and symlink it back, so every existing
# /workspace/mira/... path still resolves. Keep the uv cache on the network volume: a stale handle
# there only costs a re-download, whereas one inside the env corrupts the install.
set -eu
log(){ echo "$(date -u +%FT%TZ) fs-fix: $*" | tee -a /workspace/bench.log; }

log "clearing partial install"
rm -rf /workspace/mira /root/mira
mkdir -p /root/mira /workspace/.uvcache
ln -sfn /root/mira /workspace/mira
log "env -> $(readlink -f /workspace/mira) on $(findmnt -no FSTYPE --target /root)"
log "overlay free: $(df -h / | awk 'NR==2{print $4}')"

cat > /root/.env_fix <<'ENV'
export UV_CACHE_DIR=/workspace/.uvcache
export UV_LINK_MODE=copy
ENV
log "ready"
