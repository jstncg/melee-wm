"""30s rollout on a REAL continuous action stream (consecutive chunks of one match).

Replaces the tiled-4s-loop confound in the first long-rollout test: an 8x repeated clip is an
out-of-distribution action sequence, so degradation there was not attributable to the model.
Chunks are 240 lines at 60 fps; the loader's 20 fps grid takes every 3rd line.
"""
import sys, os, io, json, time, tarfile, argparse, subprocess, torch
sys.path.insert(0, "/workspace/pyextra"); sys.path.insert(0, "/workspace/mira/scripts")
from pathlib import Path
import eval_world_model_offline as E
from mira.inference.loading import load_world_model
from mira.inference.rollout import _encode_window_actions
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionTensors

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30.0)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--match", type=str, default="g00000")
p.add_argument("--out", type=str, default="/workspace/long")
a = p.parse_args()

CK = Path("/workspace/wm_run/checkpoint-100000/checkpoint.pth"); dev = torch.device("cuda")
cfg = E.load_run_config(CK)
model, _ = load_world_model(CK, device=dev); model = model.eval()
td, atd = model.temporal_downsampling, model.action_temporal_downsampling
nctx, W = model.n_context_latents, model.n_context_latents + 1
N = int(round(a.seconds * 20 / td))
need = N * atd + atd

keys = list(model.config.actions.valid_keys); kidx = {k: i for i, k in enumerate(keys)}
SH = "/workspace/melee/shards/train/shard-00000.tar"
rows = []
with tarfile.open(SH) as tf:
    c = 0
    while len(rows) < need:
        name = f"{a.match}_c{c:05d}.p0.jsonl"
        try: f = tf.extractfile(name)
        except KeyError: break
        if f is None: break
        lines = f.read().decode().splitlines()[::3]          # 60 fps -> 20 fps
        for ln in lines:
            v = [0]*len(keys)
            for k in json.loads(ln)["keys"]:
                if k in kidx: v[kidx[k]] = 1
            rows.append(v)
        c += 1
print(f"real actions: {len(rows)} steps from {c} chunks of {a.match} ({len(rows)/20:.1f}s) need={need}", flush=True)
assert len(rows) >= need, "not enough consecutive chunks"

loader = E._build_loader(cfg, model, clip_len=80, batch_size=1, seed=7)
batch, _ = next(iter(loader)); batch = batch.to(dev); model.codec.preprocess_batch(batch)

acts = ActionTensors(config=model.config.actions, batch_size=1)
acts.key_presses = torch.tensor(rows[:need], dtype=torch.int32, device=dev)[None]
acts.mouse_movements = torch.zeros((1, need, 2), dtype=torch.float32, device=dev)
acts.game_mouse_sensitivity = batch.actions.game_mouse_sensitivity
long_batch = VideoActionBatch(video=batch.video, actions=acts)

stats = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    z = model.encode_video(batch).clone()
    torch.manual_seed(1234)
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
            s = {"t": round(i*td/20,1), "std": float(f.std()), "absmax": float(f.abs().max()),
                 "dz": float((z_t[:,i]-z_t[:,i-1]).abs().mean())}
            stats.append(s)
            print("  t={t:5.1f}s  std={std:.3f}  absmax={absmax:7.2f}  dz={dz:.4f}".format(**s), flush=True)
    wall = time.perf_counter()-t0
    print(f"rollout {N} latents in {wall:.1f}s = {wall/N*1000:.1f} ms/latent", flush=True)
    vid = torch.cat([model.decode_to_video(z_t[:, i:i+20]).float().cpu() for i in range(0,N,20)],dim=1)

os.makedirs(a.out, exist_ok=True)
tag = f"real{int(a.seconds)}s_st{a.steps}_n{a.noise}"
fr = (vid[0].clamp(0,1)*255).byte().permute(0,2,3,1).numpy(); T,H,Wd,_ = fr.shape
ff = subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{Wd}x{H}","-r","20",
                       "-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","18",f"{a.out}/dream_{tag}.mp4"],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ff.stdin.write(fr.tobytes()); ff.stdin.close(); ff.wait()
json.dump(stats, open(f"{a.out}/stats_{tag}.json","w"), indent=2)
print(f"WROTE {a.out}/dream_{tag}.mp4 ({T} frames, {T/20:.1f}s)", flush=True)
