"""Reconstruct one held-out clip with a codec checkpoint; save input/output grid PNG and print PSNR.
Usage: pixi run python recon.py <checkpoint_path> <out.png>"""
import sys, yaml, torch, numpy as np
from PIL import Image
from mira.codec.codec_model import VideoCodec
from mira.data.training_loader import create_loader

ckpt, out = sys.argv[1], sys.argv[2]
import os
if os.path.isdir(ckpt): ckpt = os.path.join(ckpt, "checkpoint.pth")
m = VideoCodec.load_from_checkpoint(ckpt, device="cuda").eval()
vc = m.config.encoder.video
keys = yaml.safe_load(open("/workspace/melee_cfg/actions/melee.yaml"))["valid_keys"]
loader = create_loader("/workspace/melee/shards/test", clip_len=vc.timesteps, target_fps=vc.fps, n_players=1, batch_size=1,
                       num_workers=2, shuffle=False, infinite=False, frame_size=(vc.height, vc.width), valid_keys=keys, exclude_replays=True)
batch, _ = next(iter(loader)); batch = batch.to("cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = m(batch)
x, y = o.input_video.float(), o.output_video.float()
print("input shape", tuple(x.shape), "range", float(x.min()), float(x.max()), "| output range", float(y.min()), float(y.max()))
def to01(v): return ((v + 1) / 2 if v.min() < -0.05 else v).clamp(0, 1)
x, y = to01(x), to01(y)
if x.dim() == 6: x, y = x[0], y[0]   # (P,T,C,H,W) -> drop batch
if x.dim() == 5: x, y = x[0], y[0]   # (T,C,H,W)
T = x.shape[0]; idx = [0, T // 2, T - 1]
mse = ((x - y) ** 2).mean().item(); print("PSNR %.2f dB over %d frames" % (10 * np.log10(1 / max(mse, 1e-10)), T))
row = lambda v: np.concatenate([(v[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for i in idx], axis=1)
Image.fromarray(np.concatenate([row(x), row(y)], axis=0)).save(out); print("saved", out)
