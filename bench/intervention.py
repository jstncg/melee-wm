"""Tier 1 intervention gate: does the dream OBEY the controller?

Three rollouts from an identical seed and identical initial noise:
  A  baseline      - real actions, tiled
  A' control       - byte-identical to A. Must produce diff == 0, else the rig is non-deterministic
                     and any difference in B is unattributable.
  B  intervened    - identical to A except P1_X (jump) is forced ON from latent frame K onward.
Reports mean abs pixel diff vs A for both, per second. Writes a stacked A-vs-B mp4.
"""
import sys, os, json, argparse, subprocess, torch
sys.path.insert(0, "/workspace/pyextra"); sys.path.insert(0, "/workspace/mira/scripts")
from pathlib import Path
import eval_world_model_offline as E
from mira.inference.loading import load_world_model
from mira.inference.rollout import _encode_window_actions
from mira.data.batch import VideoActionBatch

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--key", type=str, default="P1_X")
p.add_argument("--out", type=str, default="/workspace/interv")
a = p.parse_args()

CK = Path("/workspace/wm_run/checkpoint-100000/checkpoint.pth"); dev = torch.device("cuda")
cfg = E.load_run_config(CK)
model, _ = load_world_model(CK, device=dev); model = model.eval()
td, atd = model.temporal_downsampling, model.action_temporal_downsampling
nctx, W = model.n_context_latents, model.n_context_latents + 1
N = int(round(a.seconds * 20 / td))
K = W  # intervene the first latent past the seeded context

loader = E._build_loader(cfg, model, clip_len=80, batch_size=1, seed=7)
batch, _ = next(iter(loader)); batch = batch.to(dev); model.codec.preprocess_batch(batch)

keys = list(model.config.actions.valid_keys)
ki = keys.index(a.key)
p1 = [i for i, k in enumerate(keys) if k.startswith("P1_")]
print(f"N={N} latents ({N*td/20:.1f}s)  intervene {a.key} (idx {ki}) from latent {K} (t={K*td/20:.1f}s)", flush=True)

def make_actions(intervene: bool):
    acts = batch.actions
    while acts.n_steps < N * atd + atd:
        acts = acts.cat_time(batch.actions)
    kp = acts.key_presses.clone()
    if intervene:
        kp[:, K*atd:, p1] = 0        # clear P1 inputs so the jump is unambiguous
        kp[:, K*atd:, ki] = 1        # hold jump
    out = acts.slice_time(None, None); out.key_presses = kp
    return VideoActionBatch(video=batch.video, actions=out)

def run(long_batch, seed=1234):
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode_video(batch).clone()
        z_t = torch.randn(z.shape[0], N, *z.shape[2:], device=dev, dtype=z.dtype)
        z_t[:, :nctx] = z[:, :nctx]
        kv = None
        for start in range(N - W + 1):
            cur_a = _encode_window_actions(model, model, long_batch, start, W)
            z_t[:, start:start+W], kv = model.denoise_streaming(
                z_t[:, start:start+W], cur_a, n_diffusion_steps=a.steps,
                noise_level=0.0, streaming_kv_caches=kv, schedule_type="linear")
        return torch.cat([model.decode_to_video(z_t[:, i:i+20]).float().cpu()
                          for i in range(0, N, 20)], dim=1)[0].clamp(0, 1)

base_b = make_actions(False)
A  = run(base_b)
Ac = run(base_b)          # control: same actions, same seed
B  = run(make_actions(True))

ctrl = (A - Ac).abs().mean().item()
print(f"\nCONTROL (identical actions): mean|diff| = {ctrl:.6f}   <- must be ~0", flush=True)
if ctrl > 1e-5:
    print("!! rig is NON-DETERMINISTIC; B's diff is not attributable to the action", flush=True)

rows = []
for s in range(int(N*td/20)):
    f0, f1 = s*20, (s+1)*20
    d_b = (A[f0:f1] - B[f0:f1]).abs().mean().item()
    d_c = (A[f0:f1] - Ac[f0:f1]).abs().mean().item()
    rows.append({"t": s, "diff_intervened": d_b, "diff_control": d_c})
    print(f"  t={s:2d}s  intervened={d_b:.5f}   control={d_c:.6f}   ratio={d_b/max(d_c,1e-9):8.1f}x", flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump({"control_mean_abs_diff": ctrl, "per_second": rows}, open(f"{a.out}/intervention.json","w"), indent=2)
stack = torch.cat([A, B], dim=2)                      # stack vertically: baseline over intervened
fr = (stack*255).byte().permute(0,2,3,1).numpy()
T,H,Wd,_ = fr.shape
ff = subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{Wd}x{H}","-r","20",
                       "-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
                       f"{a.out}/intervention.mp4"],
                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ff.stdin.write(fr.tobytes()); ff.stdin.close(); ff.wait()
print(f"WROTE {a.out}/intervention.mp4  (top=baseline, bottom={a.key} held)", flush=True)
