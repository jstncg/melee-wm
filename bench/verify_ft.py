"""Pre- and post-flight checks for the decoder-only fine-tune.

  --selfcheck        synthetic test that the motion weight map actually lands on what moves.
                     Runs BEFORE any training, so a broken weight map costs nothing.
  --latents A B      assert two codec checkpoints produce identical latents. This is the
                     non-negotiable gate: if it fails, the fine-tuned codec is NOT a drop-in
                     replacement and the world model would silently break.
"""
import sys, argparse, torch
sys.path.insert(0, "/workspace/pyextra")
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--selfcheck", action="store_true")
ap.add_argument("--latents", nargs=2, metavar=("OLD", "NEW"))
a = ap.parse_args()

if a.selfcheck:
    from mira.codec.loss import CodecLoss
    # static background, one bright square moving left->right: the only thing that moves
    b, t, c, h, w = 1, 8, 3, 64, 64
    x = torch.full((b, t, c, h, w), -0.5)
    x[..., 10:30, :] = 0.2                                   # a static "stage" band
    for i in range(t):
        x[0, i, :, 40:52, 5 + 5 * i: 17 + 5 * i] = 1.0       # the "character"
    wmap = CodecLoss._motion_weight(x, strength=9.0, q=0.95, blur=9)
    moving = wmap[0, 3, 0, 40:52, 20:32].mean()
    static_band = wmap[0, 3, 0, 10:30, :].mean()
    print(f"  weight on moving object : {moving:.2f}   (want >> 1)")
    print(f"  weight on static stage  : {static_band:.2f}   (want ~1)")
    print(f"  global min/max          : {wmap.min():.2f} / {wmap.max():.2f}")
    ok = moving > 2.0 and static_band < 2.0
    print("  SELFCHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if a.latents:
    from mira.codec.codec_model import VideoCodec
    from mira.data.training_loader import create_loader
    old, new = a.latents
    ms = [VideoCodec.load_from_checkpoint(p, device="cuda").eval() for p in (old, new)]
    vc = ms[0].config.encoder.video
    keys = [f"P{p}_{k}" for p in (1, 2) for k in
            ["A","B","X","Y","Z","L","R","joy_x_neg","joy_x_pos","joy_y_neg","joy_y_pos",
             "c_x_neg","c_x_pos","c_y_neg","c_y_pos","trigger"]]
    loader = create_loader("/workspace/melee/shards/test", clip_len=vc.timesteps, target_fps=vc.fps,
                           n_players=1, batch_size=1, num_workers=2, shuffle=False, infinite=False,
                           frame_size=(vc.height, vc.width), valid_keys=keys, exclude_replays=True)
    worst = 0.0
    for i, (batch, _) in enumerate(loader):
        if i >= 5: break
        batch = batch.to("cuda")
        zs = []
        for m in ms:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, enc = m.encode(batch.video, trim_video=False)
            zs.append(enc.z.float())
        worst = max(worst, float((zs[0] - zs[1]).abs().max()))
    print(f"  max |z_old - z_new| over 5 clips = {worst:.3e}")
    ok = worst < 1e-5
    print("  LATENTS", "IDENTICAL - world model stays valid" if ok
          else "DIFFER - world model would need retraining, do NOT ship as drop-in")
    sys.exit(0 if ok else 1)

ap.error("pass --selfcheck or --latents OLD NEW")
