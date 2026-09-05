"""Codec audit: is the Base decoder saturated, or is the codec still undertrained?

Three measurements on held-out clips, encoder+decoder only (no world model):
  1. Whole-frame PSNR / LPIPS  -> compare to MIRA Table 7 Base row (PSNR 27.6, LPIPS 0.082)
  2. Character-region PSNR     -> the number nobody measured; characters are found by motion
                                  (the stage is static, the fighters are not), so no camera
                                  transform or coordinate mapping is needed
  3. PSNR vs character size    -> tests whether small characters specifically break down,
                                  which is what a /32 capacity limit would look like
"""
import sys, json, argparse, torch, numpy as np
sys.path.insert(0, "/workspace/pyextra")
import torch.nn.functional as F
from mira.codec.codec_model import VideoCodec
from mira.data.training_loader import create_loader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/workspace/codec_frozen/checkpoint.pth")
ap.add_argument("--clips", type=int, default=40)
a = ap.parse_args()

m = VideoCodec.load_from_checkpoint(a.ckpt, device="cuda").eval()
vc = m.config.encoder.video
keys = ["P1_A","P1_B","P1_X","P1_Y","P1_Z","P1_L","P1_R","P1_joy_x_neg","P1_joy_x_pos","P1_joy_y_neg",
        "P1_joy_y_pos","P1_c_x_neg","P1_c_x_pos","P1_c_y_neg","P1_c_y_pos","P1_trigger",
        "P2_A","P2_B","P2_X","P2_Y","P2_Z","P2_L","P2_R","P2_joy_x_neg","P2_joy_x_pos","P2_joy_y_neg",
        "P2_joy_y_pos","P2_c_x_neg","P2_c_x_pos","P2_c_y_neg","P2_c_y_pos","P2_trigger"]
loader = create_loader("/workspace/melee/shards/test", clip_len=vc.timesteps, target_fps=vc.fps,
                       n_players=1, batch_size=1, num_workers=2, shuffle=False, infinite=False,
                       frame_size=(vc.height, vc.width), valid_keys=keys, exclude_replays=True)

try:
    import lpips; LP = lpips.LPIPS(net="alex").cuda(); print("lpips ready", flush=True)
except Exception as e:
    LP = None; print("lpips unavailable:", e, flush=True)

def to01(v): return ((v + 1) / 2 if float(v.min()) < -0.05 else v).clamp(0, 1)
def psnr(x, y, mask=None):
    se = (x - y) ** 2
    mse = (se * mask).sum() / mask.sum().clamp(min=1) if mask is not None else se.mean()
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))

whole, char, bg, lp, buckets = [], [], [], [], []
n = 0
for batch, _ in loader:
    if n >= a.clips: break
    batch = batch.to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(batch)
    x, y = to01(o.input_video.float()), to01(o.output_video.float())
    while x.dim() > 4: x, y = x[0], y[0]              # -> (T,C,H,W)

    # Characters = what moves. Stage/HUD are static, so a frame difference isolates the fighters
    # without needing Melee's per-frame camera transform.
    d = (x[1:] - x[:-1]).abs().mean(1, keepdim=True)   # (T-1,1,H,W)
    d = F.avg_pool2d(d, 9, stride=1, padding=4)        # blur so the mask covers the whole body
    xs, ys = x[1:], y[1:]
    thr = torch.quantile(d.flatten(1), 0.97, dim=1).view(-1, 1, 1, 1)
    mask = (d > thr).float()
    area = float(mask.mean())                          # proxy for on-screen character size

    whole.append(psnr(xs, ys))
    char.append(psnr(xs, ys, mask.expand_as(xs)))
    bg.append(psnr(xs, ys, (1 - mask).expand_as(xs)))
    buckets.append((area, char[-1]))
    if LP is not None:
        with torch.no_grad():
            lp.append(float(LP(xs * 2 - 1, ys * 2 - 1).mean()))
    n += 1
    if n % 10 == 0: print(f"  {n}/{a.clips} clips", flush=True)

f = lambda v: (float(np.mean(v)), float(np.std(v)))
res = {"clips": n, "whole_frame_psnr": f(whole), "character_psnr": f(char), "background_psnr": f(bg),
       "lpips": f(lp) if lp else None,
       "MIRA_Table7_Base": {"psnr": 27.6, "lpips": 0.082},
       "MIRA_Table7_Large": {"psnr": 29.3, "lpips": 0.055}}
print("\n=================== CODEC AUDIT ===================")
print(f"  whole-frame PSNR   {res['whole_frame_psnr'][0]:6.2f} dB  (MIRA Base 27.6 / Large 29.3)")
print(f"  CHARACTER PSNR     {res['character_psnr'][0]:6.2f} dB   <- the number never measured")
print(f"  background PSNR    {res['background_psnr'][0]:6.2f} dB")
if lp: print(f"  LPIPS              {res['lpips'][0]:6.4f}      (MIRA Base 0.082 / Large 0.055)")
print(f"  gap character vs background: {res['background_psnr'][0]-res['character_psnr'][0]:.2f} dB")

# size dependence: if only SMALL characters break, that is a capacity limit, not undertraining
buckets.sort()
half = len(buckets) // 2
sm = float(np.mean([c for _, c in buckets[:half]]))     # least on-screen motion = smallest chars
lg = float(np.mean([c for _, c in buckets[half:]]))
res["char_psnr_small_area"], res["char_psnr_large_area"] = sm, lg
print(f"\n  character PSNR, smallest-motion half: {sm:.2f} dB")
print(f"  character PSNR, largest-motion half : {lg:.2f} dB")
print(f"  -> difference {lg-sm:+.2f} dB")
json.dump(res, open("/workspace/codec_audit.json", "w"), indent=2)
print("\nwrote /workspace/codec_audit.json")
