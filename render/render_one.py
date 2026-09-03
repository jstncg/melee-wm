"""Render one .slp to .mp4 with Slippi playback Dolphin under xvfb (Linux). Writes <out>.json sidecar
with frame bookkeeping so package.py can align video frames to .slp frames.

Usage: python render_one.py IN.slp OUT.mp4   (env: MELEE_ROOT=/workspace/melee)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ.get("MELEE_ROOT", "/workspace/melee"))
DOLPHIN = ROOT / "tools/squashfs-root/AppRun"
ISO = ROOT / "iso/melee.iso"

DOLPHIN_INI = "[Movie]\nDumpFrames = True\nDumpFramesSilent = True\n[DSP]\nDumpAudio = False\nBackend = No Audio Output\n[Display]\nRenderToMain = True\n[Core]\nAdapterRumble0 = False\n"
GFX_INI = "[Settings]\nAspectRatio = 0\nInternalResolutionFrameDumps = True\nEFBScale = 2\nBitrateKbps = 6000\n"
GECKO_INI = "[Gecko_Enabled]\n$Optional: Game Music OFF\n$Optional: Hide Waiting For Game\n[Gecko_Disabled]\n$Optional: Show Player Names\n$Optional: Widescreen 16:9\n"


def render(slp: Path, out: Path) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="dol_") as tmp:
        tmp = Path(tmp)
        (tmp / "user/Config").mkdir(parents=True)
        (tmp / "user/GameSettings").mkdir(parents=True)
        (tmp / "user/Config/Dolphin.ini").write_text(DOLPHIN_INI)
        (tmp / "user/Config/GFX.ini").write_text(GFX_INI)
        (tmp / "user/GameSettings/GALE01.ini").write_text(GECKO_INI)
        (tmp / "out").mkdir()
        comm = tmp / "comm.json"
        comm.write_text(json.dumps({"mode": "normal", "replay": str(slp.resolve()), "isRealTimeMode": False, "commandId": "r"}))
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", str(DOLPHIN), "--exec", str(ISO), "--batch",
               "--video_backend", "OGL", "--slippi-input", str(comm), "--hide-seekbar",
               "--output-directory", str(tmp / "out"), "--user", str(tmp / "user"), "--cout"]
        env = dict(os.environ, LP_NUM_THREADS=os.environ.get("LP_NUM_THREADS", "4"))  # llvmpipe threads per Dolphin
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                env=env, start_new_session=True)
        info = {"slp": str(slp), "start_frame": None, "end_frame": None, "last_frame": None}
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("[PLAYBACK_START_FRAME] "):
                info["start_frame"] = int(line.split()[-1])
            elif line.startswith("[GAME_END_FRAME] "):
                info["end_frame"] = int(line.split()[-1])
            elif line.startswith("[CURRENT_FRAME] "):
                info["last_frame"] = int(line.split()[-1])
                if info["end_frame"] is not None and info["last_frame"] >= info["end_frame"]:
                    break
        time.sleep(1.0)  # let the last frames flush to the avi
        os.killpg(proc.pid, 9)  # kill xvfb-run AND Dolphin (else Dolphin is orphaned and runs forever)
        proc.wait()
        avi = tmp / "out/framedump0.avi"
        if not avi.exists():
            raise RuntimeError("no framedump0.avi")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(avi), "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    nb = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                  "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)]).decode().strip()
    info.update(mp4_frames=int(nb), seconds=round(time.time() - t0, 1))
    out.with_suffix(".json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":
    print(json.dumps(render(Path(sys.argv[1]), Path(sys.argv[2]))))
