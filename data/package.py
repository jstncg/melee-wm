"""Package rendered games into MIRA's WebDataset layout so `mira.data.RocketScienceDataset` loads them unchanged.

Input per game:  <name>.mp4 (60 fps, one shared view)  +  <name>.npz (from parse.py)
Output:          <out>/shard-XXXXX.tar  +  <out>/index.json

Layout per chunk (4 s = 240 frames):  {match}_c{i:05d}.p0.mp4  .p0.jsonl  .p0.physics.jsonl  .meta.json
One "perspective" (Melee has one camera). Both players' inputs live in one 32-key vocab: P1_A .. P2_trigger.

Usage: uv run python data/package.py <video_dir> <npz_dir> <out_dir> [--chunk 240] [--offset 0] [--games-per-shard 20]
"""
import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from parse import ACTION_KEYS, STATE_COLS

FPS = 60
WORKERS = int(__import__('os').environ.get('PACKAGE_WORKERS', '16'))
OUT_W, OUT_H = 384, 288  # training resolution, 4:3; the raw 642x528 dump is 4x larger
VOCAB = [f"P{p + 1}_{k}" for p in range(2) for k in ACTION_KEYS]  # 32 keys, MIRA-style multi-hot


def probe_frames(mp4: Path) -> int:
    side = mp4.with_suffix(".json")  # render_one.py already counted frames; decoding the whole file again is the slow path
    if side.exists():
        n = json.loads(side.read_text()).get("mp4_frames")
        if n: return int(n)
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)]
    )
    return int(out.decode().strip())


def cut_chunk(mp4: Path, start: int, n: int) -> bytes:
    """Re-encode frames [start, start+n) with a keyframe at 0 (each chunk must decode standalone)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        pass
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start / FPS:.6f}", "-i", str(mp4), "-frames:v", str(n),
           "-an", "-vf", f"scale={OUT_W}:{OUT_H}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-g", str(n), "-keyint_min", str(n), "-movflags", "+faststart", tmp.name]
    subprocess.run(cmd, check=True)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink()
    return data


def action_lines(actions: np.ndarray) -> bytes:
    """actions (2, T, K) 0/1 -> one jsonl line per frame with the pressed key names."""
    P, T, K = actions.shape
    lines = []
    for t in range(T):
        keys = [VOCAB[p * K + k] for p in range(P) for k in range(K) if actions[p, t, k]]
        lines.append(json.dumps({"keys": keys}))
    return ("\n".join(lines) + "\n").encode()


def physics_lines(state: np.ndarray) -> bytes:
    """state (2, T, S) -> one jsonl line per frame: {"p1": {...}, "p2": {...}}"""
    P, T, S = state.shape
    out = []
    for t in range(T):
        out.append(json.dumps({f"p{p + 1}": dict(zip(STATE_COLS, state[p, t].round(3).tolist())) for p in range(P)}))
    return ("\n".join(out) + "\n").encode()


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def package_game(tar: tarfile.TarFile, match_id: str, mp4: Path, npz: Path, chunk: int, offset: int) -> dict:
    d = np.load(npz)
    actions, state = d["actions"], d["state"]
    meta = json.loads(str(d["meta"]))
    T_npz = actions.shape[1]
    T_mp4 = probe_frames(mp4)
    T = min(T_npz - offset, T_mp4)
    n_chunks = T // chunk
    if n_chunks == 0:
        raise ValueError(f"{match_id}: too short ({T} frames)")
    chunk_frames = []
    with ThreadPoolExecutor(WORKERS) as ex:  # ffmpeg is a subprocess, so threads give real parallelism
        videos = list(ex.map(lambda c: cut_chunk(mp4, c * chunk, chunk), range(n_chunks)))
    for c in range(n_chunks):
        key = f"{match_id}_c{c:05d}"
        s = c * chunk
        add_bytes(tar, f"{key}.p0.mp4", videos[c])
        add_bytes(tar, f"{key}.p0.jsonl", action_lines(actions[:, offset + s: offset + s + chunk]))
        add_bytes(tar, f"{key}.p0.physics.jsonl", physics_lines(state[:, offset + s: offset + s + chunk]))
        add_bytes(tar, f"{key}.meta.json", json.dumps({"match_id": match_id, "chunk": c, **meta}).encode())
        chunk_frames.append(chunk)
    frames = n_chunks * chunk
    return {
        "match_id": match_id,
        "n_players": 1,
        "chunk_frames": chunk_frames,
        "arena": f"stage{meta['stage']}",
        "npz_frames": T_npz, "mp4_frames": T_mp4, "offset": offset,
        "perspectives": [{"player_id": 0, "team": 0, "frames": frames, "duration": frames / FPS,
                          "recording_offset_sec": 0.0, "anchors": []}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir"); ap.add_argument("npz_dir"); ap.add_argument("out_dir")
    ap.add_argument("--chunk", type=int, default=240)
    ap.add_argument("--offset", type=int, default=0, help="npz frame index that matches mp4 frame 0")
    ap.add_argument("--games-per-shard", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    npz_by_stem = {p.stem: p for p in Path(a.npz_dir).rglob("*.npz")}
    mp4s = sorted(Path(a.video_dir).rglob("*.mp4"))
    pairs = [(m, npz_by_stem[m.stem]) for m in mp4s if m.stem in npz_by_stem]
    print(f"{len(mp4s)} videos, {len(pairs)} with matching npz")
    entries, tar, shard_i = [], None, -1
    for i, (mp4, npz) in enumerate(pairs):
        if i % a.games_per_shard == 0:
            if tar: tar.close()
            shard_i += 1
            tar = tarfile.open(out / f"shard-{shard_i:05d}.tar", "w")
        match_id = f"g{i:05d}"  # no dots allowed in match_id
        try:
            e = package_game(tar, match_id, mp4, npz, a.chunk, a.offset)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {mp4.name}: {ex}"); continue
        e["shard"] = f"shard-{shard_i:05d}.tar"; e["source"] = mp4.stem
        entries.append(e)
        print(f"{match_id} {mp4.stem[:40]:40s} chunks={len(e['chunk_frames'])} npz={e['npz_frames']} mp4={e['mp4_frames']}")
    if tar: tar.close()
    (out / "index.json").write_text(json.dumps({"total_samples": len(entries), "vocab": VOCAB, "entries": entries}, indent=1))
    print(f"wrote {len(entries)} matches, {shard_i + 1} shards -> {out}")


if __name__ == "__main__":
    main()
