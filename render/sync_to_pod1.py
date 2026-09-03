"""Run on pod 2. Every 5 min: push finished (mp4+json) to pod 1, and pull pod 1's json list as
skip markers so the two pods never render the same game twice. Usage: python sync_to_pod1.py HOST PORT
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HOST, PORT = sys.argv[1], sys.argv[2]
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", "/root/.ssh/pod2pod", "-p", PORT, f"root@{HOST}"]
SCP = ["scp", "-q", "-o", "StrictHostKeyChecking=no", "-i", "/root/.ssh/pod2pod", "-P", PORT]
V = Path("/workspace/melee/video")
pushed: set[str] = set()

while True:
    # push: local games that have both mp4 and a real json (rendered here, not a marker)
    for j in V.glob("*.json"):
        b = j.stem
        mp4 = V / f"{b}.mp4"
        if b in pushed or not mp4.exists() or j.stat().st_size == 0:
            continue
        r = subprocess.run(SCP + [str(mp4), str(j), f"root@{HOST}:/workspace/melee/video/"])
        if r.returncode == 0:
            pushed.add(b)
    # pull: names finished on pod 1 -> empty-marker json is not enough (render checks -s), write a stub
    r = subprocess.run(SSH + ["ls /workspace/melee/video/*.json"], capture_output=True, text=True)
    n_mark = 0
    for line in r.stdout.splitlines():
        b = Path(line.strip()).stem
        if b and not (V / f"{b}.json").exists() and not (V / f"{b}.mp4").exists():
            (V / f"{b}.json").write_text(json.dumps({"marker": "rendered on pod1"}))
            n_mark += 1
    print(time.strftime("%H:%M:%S"), f"pushed_total={len(pushed)} new_markers={n_mark}", flush=True)
    time.sleep(300)
