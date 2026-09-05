"""Is character LOCATION recoverable from the codec latent, or did the encoder drop it?

Fits both a linear (ridge) and a nonlinear (MLP) probe, because a linear probe failing alone
would not prove the information is absent -- only that it is not linearly readable.

latent_probe.py asked whether latent ENERGY sits where pixels move (corr 0.123 = ambiguous).
That is a weak instrument: it never tests whether the information is READABLE.

This one does. Per latent cell, fit a linear map from the 32 latent channels to how much of that
cell the characters cover (characters = what moves, the same mask codec_audit.py uses).

The control matters. A probe can score well just by learning "fighters are usually center-stage",
so two models are fitted and compared on held-out clips:

  POS   : cell position only            -> the static prior, knows nothing about this frame
  POS+Z : cell position + the 32 latents -> the prior plus whatever the latent adds

  delta R2 large  -> characters ARE in the latent; the DECODER fails to render them
                     -> decoder-only fix, latents unchanged, the world model survives
  delta R2 ~ 0    -> the encoder discarded them; no decoder can recover them
                     -> encoder retrain, which forces a world-model retrain
"""
import sys, json, argparse, torch, numpy as np
sys.path.insert(0, "/workspace/pyextra")
import torch.nn.functional as F
from mira.codec.codec_model import VideoCodec
from mira.data.training_loader import create_loader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/workspace/codec_frozen/checkpoint.pth")
ap.add_argument("--clips", type=int, default=40)
ap.add_argument("--out", default="/workspace/char_probe2.json")
a = ap.parse_args()

m = VideoCodec.load_from_checkpoint(a.ckpt, device="cuda").eval()
vc = m.config.encoder.video
keys = [f"P{p}_{k}" for p in (1, 2) for k in
        ["A", "B", "X", "Y", "Z", "L", "R", "joy_x_neg", "joy_x_pos", "joy_y_neg", "joy_y_pos",
         "c_x_neg", "c_x_pos", "c_y_neg", "c_y_pos", "trigger"]]
loader = create_loader("/workspace/melee/shards/test", clip_len=vc.timesteps, target_fps=vc.fps,
                       n_players=1, batch_size=1, num_workers=2, shuffle=False, infinite=False,
                       frame_size=(vc.height, vc.width), valid_keys=keys, exclude_replays=True)

def to01(v): return ((v + 1) / 2 if float(v.min()) < -0.05 else v).clamp(0, 1)

Z, Y, CLIP = [], [], []
n = 0
for batch, _ in loader:
    if n >= a.clips: break
    batch = batch.to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, enc = m.encode(batch.video, trim_video=False)
        z = enc.z.float()
        x = to01(batch.video.float())
    while x.dim() > 4: x = x[0]
    while z.dim() > 4: z = z[0]
    T, C, H, W = z.shape

    # Character mask, identical recipe to codec_audit.py: characters are what moves.
    d = (x[1:] - x[:-1]).abs().mean(1, keepdim=True)
    d = F.avg_pool2d(d, 9, stride=1, padding=4)
    thr = torch.quantile(d.flatten(1), 0.97, dim=1).view(-1, 1, 1, 1)
    mask = (d > thr).float()                                  # (T-1,1,Hpix,Wpix)
    cov = F.adaptive_avg_pool2d(mask, (H, W))[:, 0]           # (T-1,h,w) coverage per latent cell

    # One latent frame spans temporal_patch video frames; pool the mask onto the latent clock.
    k = min(T, cov.shape[0])
    if T < cov.shape[0]:
        stride = cov.shape[0] // T
        cov = cov[:T * stride].reshape(T, stride, H, W).mean(1)
        k = T
    Z.append(z[:k].permute(0, 2, 3, 1).reshape(-1, C).cpu())   # (k*h*w, C)
    Y.append(cov[:k].reshape(-1).cpu())
    CLIP.append(np.full(k * H * W, n))
    n += 1
    if n % 10 == 0: print(f"  {n}/{a.clips} clips", flush=True)

Z = torch.cat(Z).double()
Y = torch.cat(Y).double()
CLIP = np.concatenate(CLIP)
N, C = Z.shape
print(f"\n{n} clips, latent grid {H}x{W}, {N} cell samples, {C} channels")

# Position one-hot: the static prior "fighters are usually here".
cell = torch.arange(N) % (H * W)
POS = torch.zeros(N, H * W, dtype=torch.float64)
POS[torch.arange(N), cell] = 1.0

# Split by CLIP so a held-out clip is never seen during the fit.
test_clips = set(range(0, n, 3))                                # every 3rd clip held out
te = torch.tensor([c in test_clips for c in CLIP])
tr = ~te
print(f"train cells {int(tr.sum())}, test cells {int(te.sum())}")

def ridge_r2(feats, lam=1.0):
    A, b = feats[tr], Y[tr]
    G = A.T @ A + lam * torch.eye(A.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(G, A.T @ b)
    pred = feats[te] @ w
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot), pred

# Standardize latents so the ridge penalty is comparable across channels.
Zs = (Z - Z[tr].mean(0)) / Z[tr].std(0).clamp(min=1e-8)

r2_pos, _ = ridge_r2(POS)
r2_posz, pred = ridge_r2(torch.cat([POS, Zs], 1))
delta = r2_posz - r2_pos

# AUC of the POS+Z probe against a binarized mask, as a second readable number.
yt = (Y[te] > 0.5).numpy().astype(int)
p = pred.numpy()
if yt.sum() and (1 - yt).sum():
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = int(yt.sum()), int((1 - yt).sum())
    auc = float((ranks[yt == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
else:
    auc = float("nan")

verdict = ("characters ARE in the latent -> decoder is at fault, WM survives" if delta > 0.10 else
           "encoder discarded the characters -> no decoder fix, WM retrain needed" if delta < 0.02 else
           "partial: some character information survives, but weakly")

print("\n================ CHARACTER PROBE: LINEAR ================")
print(f"  R2  position only        {r2_pos:+.4f}   <- static prior, no frame knowledge")
print(f"  R2  position + latents   {r2_posz:+.4f}")
print(f"  delta R2 from latents    {delta:+.4f}")
print(f"  AUC (POS+Z, mask>0.5)    {auc:.4f}")
print(f"  verdict: {verdict}")


# ---- nonlinear control: a linear probe failing does not prove the information is absent.
# POS is one-hot, so the linear POS fit already IS the optimal per-cell constant; only POS+Z
# can improve. Fit an MLP on the same split and compare against the same r2_pos baseline.
dev = "cuda"
Xtr = torch.cat([POS[tr], Zs[tr]], 1).float().to(dev); ytr = Y[tr].float().to(dev)
Xte = torch.cat([POS[te], Zs[te]], 1).float().to(dev); yte = Y[te].float().to(dev)
net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 128), torch.nn.GELU(),
                          torch.nn.Linear(128, 128), torch.nn.GELU(),
                          torch.nn.Linear(128, 1)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
best = -9.0
for step in range(3000):
    idx = torch.randint(0, Xtr.shape[0], (4096,), device=dev)
    loss = F.mse_loss(net(Xtr[idx])[:, 0], ytr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 250 == 0:
        with torch.no_grad():
            pr = net(Xte)[:, 0]
            r2 = float(1 - ((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
        best = max(best, r2)
        print(f"  mlp step {step+1}: test R2 {r2:+.4f}", flush=True)
r2_mlp = best
delta_mlp = r2_mlp - r2_pos
print(f"\n  R2  MLP(position + latents)  {r2_mlp:+.4f}")
print(f"  delta R2 nonlinear           {delta_mlp:+.4f}")
print("  nonlinear verdict:", "information IS present, just not linear" if delta_mlp > 0.10
      else ("encoder discarded it (linear AND nonlinear agree)" if delta_mlp < 0.02 else "partial"))

json.dump({"clips": n, "grid": [H, W], "channels": C, "cells_train": int(tr.sum()),
           "cells_test": int(te.sum()), "r2_pos": r2_pos, "r2_pos_z": r2_posz,
           "delta_r2": delta, "auc": auc, "verdict": verdict, "r2_mlp": r2_mlp, "delta_r2_mlp": delta_mlp}, open(a.out, "w"), indent=2)
print(f"  wrote {a.out}")

