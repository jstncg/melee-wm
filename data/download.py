"""Download the replays listed in data/fox_falcon_bf.txt into data/raw/ and parse each to .npz."""
import subprocess, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "erickfm/slippi-public-dataset-v3.7"
paths = Path("data/fox_falcon_bf.txt").read_text().split("\n")
for i, p in enumerate(paths, 1):
    dst = Path("data/raw") / p
    if not dst.with_suffix(".npz").exists():
        hf_hub_download(REPO, p, repo_type="dataset", local_dir="data/raw")
        subprocess.run([sys.executable, "data/parse.py", str(dst)], check=True, capture_output=True)
    if i % 50 == 0:
        print(f"{i}/{len(paths)}", flush=True)
print("done")
