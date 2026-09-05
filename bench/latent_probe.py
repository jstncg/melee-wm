"""Do the LATENTS encode the characters, or does the encoder already throw them away?

Decides the cheapest viable fix:
  correlation high -> characters ARE in the latent, the decoder just fails to render them
                      -> decoder-only fine-tune, latents unchanged, WORLD MODEL SURVIVES
  correlation low  -> the encoder/bottleneck discarded them
                      -> encoder must be retrained, which forces a world-model retrain too
Compares, per latent cell, the latent's activation energy against how much that cell's pixels move.
"""
import sys, json, torch, numpy as np
sys.path.insert(0, "/workspace/pyextra")
import torch.nn.functional as F
from mira.codec.codec_model import VideoCodec
from mira.data.training_loader import create_loader

m = VideoCodec.load_from_checkpoint("/workspace/codec_frozen/checkpoint.pth", device="cuda").eval()
vc = m.config.encoder.video
keys = [f"P{p}_{k}" for p in (1, 2) for k in
        ["A","B","X","Y","Z","L","R","joy_x_neg","joy_x_pos","joy_y_neg","joy_y_pos",
         "c_x_neg","c_x_pos","c_y_neg","c_y_pos","trigger"]]
loader = create_loader("/workspace/melee/shards/test", clip_len=vc.timesteps, target_fps=vc.fps,
                       n_players=1, batch_size=1, num_workers=2, shuffle=False, infinite=False,
                       frame_size=(vc.height, vc.width), valid_keys=keys, exclude_replays=True)

def to01(v): return ((v + 1) / 2 if float(v.min()) < -0.05 else v).clamp(0, 1)
cors, n = [], 0
for batch, _ in loader:
    if n >= 20: break
    batch = batch.to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, enc = m.encode(batch.video, trim_video=False)
        z = enc.z.float()                                  # (b,t,c,h,w)
        x = to01(batch.video.float())
    while x.dim() > 4: x = x[0]
    while z.dim() > 4: z = z[0]
    T, C, H, W = z.shape
    # per-latent-cell energy, and per-cell pixel motion pooled to the same grid
    energy = z.pow(2).mean(1)                              # (t,h,w)
    mot = (x[1:] - x[:-1]).abs().mean(1, keepdim=True)     # (T-1,1,H,W)
    mot = F.adaptive_avg_pool2d(mot, (H, W))[:, 0]         # (T-1,h,w)
    k = min(energy.shape[0], mot.shape[0])
    e, mo = energy[:k].flatten(), mot[:k].flatten()
    e = (e - e.mean()) / e.std().clamp(min=1e-8)
    mo = (mo - mo.mean()) / mo.std().clamp(min=1e-8)
    cors.append(float((e * mo).mean()))
    n += 1

c = float(np.mean(cors))
print(f"\nlatent grid {H}x{W}, {n} clips")
print(f"correlation(latent energy, pixel motion) = {c:+.3f}")
print("interpretation:", "characters ARE represented -> decoder-only fix, WM survives" if c > 0.15
      else ("weak/none -> encoder discarded them, WM retrain needed" if c < 0.05 else "ambiguous"))
json.dump({"correlation": c, "per_clip": cors, "grid": [H, W]},
          open("/workspace/latent_probe.json", "w"), indent=2)
