#!/usr/bin/env bash
# One-time setup on a RunPod Ubuntu pod: Slippi playback Dolphin (Linux AppImage), xvfb, ffmpeg, slp2mp4.
# Usage: bash setup_pod.sh   (run as root on the pod; ISO must be at /workspace/melee/iso/melee.iso)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
W=/workspace/melee
mkdir -p $W/tools $W/iso $W/slp $W/video

apt-get update -qq
apt-get install -y -qq xvfb ffmpeg curl unzip libfuse2 libgl1 libglu1-mesa libgtk2.0-0 libsdl2-2.0-0 \
  libasound2t64 libpulse0 libxi6 libxrandr2 libxxf86vm1 libegl1 mesa-utils libevdev2 libudev1 libpng16-16 \
  libsfml-system2.6 libsfml-network2.6 libminiupnpc17 libmbedtls14t64 libenet7 libhidapi-hidraw0 libgdk-pixbuf2.0-0 2>&1 | tail -1 || true

# Slippi playback build (frame dumping is compiled in on Linux, not on macOS)
if [ ! -x $W/tools/squashfs-root/AppRun ]; then
  cd $W/tools
  curl -sSL -o playback.zip https://github.com/project-slippi/Ishiiruka-Playback/releases/download/v3.5.2/playback-3.5.2-Linux.zip
  unzip -q -o playback.zip
  chmod +x Slippi_Playback-x86_64.AppImage
  ./Slippi_Playback-x86_64.AppImage --appimage-extract >/dev/null   # no FUSE needed
  cp -r Sys squashfs-root/usr/bin/ 2>/dev/null || true
fi

# slp2mp4 (Python), pinned to the fork that supports config + parallel
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null && export PATH=$HOME/.local/bin:$PATH)
export PATH=$HOME/.local/bin:$PATH
cd $W && [ -f pyproject.toml ] || uv init -q --python 3.12 --name melee-render
uv add -q "slp2mp4 @ git+https://github.com/davisdude/slp2mp4.git" numpy

cat > ~/.slp2mp4.toml <<EOF
[paths]
ffmpeg = "$(command -v ffmpeg)"
slippi_playback = "$W/tools/squashfs-root/AppRun"
ssbm_iso = "$W/iso/melee.iso"

[dolphin]
backend = "OGL"
resolution = "480p"
bitrate = 4000

[dolphin.gecko_codes]
"\$Optional: Show Player Names" = false
"\$Optional: Game Music OFF" = true
"\$Optional: Widescreen 16:9" = false
"\$Optional: Disable Screen Shake" = false
"\$Optional: Hide HUD" = false
"\$Optional: Hide Waiting For Game" = true

[runtime]
parallel = ${RENDER_PARALLEL:-1}
youtubify_names = false
EOF
echo "setup done. render with: cd $W && xvfb-run -a uv run slp2mp4 -o video directory slp"
