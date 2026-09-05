"""Unroll the WM past its 2s training window; record where (if) it breaks.

Seeds from a real clip, then keeps calling denoise_streaming with the sliding window + KV cache.
Actions are the real clip tiled (a real action distribution, not zeros).
Writes an mp4 + per-second latent stats so collapse (dz->0) or blow-up (absmax) is visible.
"""
import sys, json, time, argparse, os, torch
sys.path.insert(0, "/workspace/pyextra"); sys.path.insert(0, "/workspace/mira/scripts")
from pathlib import Path
import eval_world_model_offline as E
from mira.inference.loading import load_world_model
from mira.inference.rollout import _encode_window_actions
from mira.data.batch import VideoActionBatch

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30.0)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--out", type=str, default="/workspace/long")
a = p.parse_args()

CK = Path("/workspace/wm_run/checkpoint-100000/checkpoint.pth"); dev = torch.device("cuda")
cfg = E.load_run_config(CK)
model, _ = load_world_model(CK, device=dev); model = model.eval()
td, atd = model.temporal_downsampling, model.action_temporal_downsampling
nctx, W = model.n_context_latents, model.n_context_latents + 1
N = int(round(a.seconds * 20 / td))
print(f"nctx={nctx} window={W} td={td} atd={atd} target={N} latents = {N*td/20:.1f}s "
      f"steps={a.steps} noise={a.noise}", flush=True)

loader = E._build_loader(cfg, model, clip_len=80, batch_size=1, seed=7)
batch, _ = next(iter(loader)); batch = batch.to(dev); model.codec.preprocess_batch(batch)

acts = batch.actions
while acts.n_steps < N * atd + atd:
    acts = acts.cat_time(batch.actions)
long_batch = VideoActionBatch(video=batch.video, actions=acts)

stats = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    z = model.encode_video(batch).clone()
    z_t = torch.randn(z.shape[0], N, *z.shape[2:], device=dev, dtype=z.dtype)
    z_t[:, :nctx] = z[:, :nctx]
    kv, t0 = None, time.perf_counter()
    for start in range(N - W + 1):
        cur_a = _encode_window_actions(model, model, long_batch, start, W)
        z_t[:, start:start+W], kv = model.denoise_streaming(
            z_t[:, start:start+W], cur_a, n_diffusion_steps=a.steps,
            noise_level=a.noise, streaming_kv_caches=kv, schedule_type="linear")
        i = start + W - 1
        if i % (20 // td) == 0:
            f = z_t[:, i]
            s = {"t": round(i*td/20, 1), "std": float(f.std()),
                 "absmax": float(f.abs().max()),
                 "dz": float((z_t[:, i] - z_t[:, i-1]).abs().mean())}
            stats.append(s)
            print("  t={t:5.1f}s  std={std:.3f}  absmax={absmax:8.2f}  dz={dz:.4f}".format(**s), flush=True)
    wall = time.perf_counter() - t0
    print(f"rollout {N} latents in {wall:.1f}s = {wall/N*1000:.1f} ms/latent", flush=True)
    vid = torch.cat([model.decode_to_video(z_t[:, i:i+20]).float().cpu()
                     for i in range(0, N, 20)], dim=1)

import subprocess
os.makedirs(a.out, exist_ok=True)
tag = f"{int(a.seconds)}s_st{a.steps}_n{a.noise}"
frames = (vid[0].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()  # (T,H,W,C)
T, H, W, _ = frames.shape
mp4 = f"{a.out}/dream_{tag}.mp4"
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ff.stdin.write(frames.tobytes()); ff.stdin.close(); ff.wait()
json.dump(stats, open(f"{a.out}/stats_{tag}.json", "w"), indent=2)
print(f"WROTE {mp4} ({T} frames, {T/20:.1f}s)", flush=True)
