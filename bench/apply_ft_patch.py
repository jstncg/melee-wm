"""Apply the decoder-only + motion-weighted-loss patch to MIRA, in place, idempotently.

Why: the codec audit measured characters reconstructing 9.66 dB worse than the static stage
(16.95 vs 26.61). The fighters are ~3% of pixels, so an unweighted mean reconstruction loss is
dominated by the background and the codec correctly learned to ignore them.

Two changes:
  loss.py         optional per-pixel motion weighting on BOTH the L1 anchor and LPIPS. Weighting
                  only L1 would be undone by auto_weight, which rescales the perceptual terms to
                  match the L1 gradient norm.
  train_codec.py  optional encoder freeze, so the latent space is untouched and an existing world
                  model trained on those latents stays valid.
"""
import sys, pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/mira")
loss_p = ROOT / "src/mira/codec/loss.py"
train_p = ROOT / "scripts/train_codec.py"

def edit(path, old, new, tag, sentinel):
    """`sentinel` must be a string unique to the patched result -- using the first line of `new`
    silently no-ops when that line already exists in the original."""
    s = path.read_text()
    if sentinel in s:
        print(f"  [skip] {tag} already applied"); return
    if old not in s:
        raise SystemExit(f"  [FAIL] {tag}: anchor not found in {path}")
    path.write_text(s.replace(old, new, 1)); print(f"  [ok]   {tag}")

# ---------------------------------------------------------------- loss.py: config fields
edit(loss_p,
"""    auto_weight: bool = False
    max_auto_weight: float = Field(default=1e4, gt=0)""",
"""    auto_weight: bool = False
    max_auto_weight: float = Field(default=1e4, gt=0)

    # Per-pixel reweighting toward whatever moves. The static stage/HUD dominate an unweighted
    # mean, so the codec spends its capacity there and renders the fighters as smudges. 0 disables.
    motion_weight_strength: float = Field(default=0.0, ge=0)
    motion_weight_quantile: float = Field(default=0.95, gt=0, lt=1)
    motion_weight_blur: int = Field(default=9, ge=1)""",
"loss weights fields", "motion_weight_strength: float")

# ---------------------------------------------------------------- loss.py: spatial LPIPS
edit(loss_p,
"""            self.lpips_perceptual_loss = lpips.LPIPS(net="vgg", verbose=False).eval()""",
"""            # spatial=True returns a per-position map instead of a scalar, so the same motion
            # weighting can be applied to LPIPS as to the L1 anchor.
            self.lpips_perceptual_loss = lpips.LPIPS(
                net="vgg", verbose=False, spatial=self.weights.motion_weight_strength > 0
            ).eval()""",
"spatial LPIPS", "spatial=self.weights.motion_weight_strength > 0")

# ---------------------------------------------------------------- loss.py: weight map helper
edit(loss_p,
"""    def _hook_clone(self, tensor: Tensor, loss_name: str) -> Tensor:""",
"""    @staticmethod
    def _motion_weight(target: Tensor, strength: float, q: float, blur: int) -> Tensor:
        \"\"\"Per-pixel weights, ~1 on static regions and ~1+strength on moving ones.

        The characters are the only thing that moves, so a frame difference localises them without
        labels or Melee's per-frame camera transform. Normalising by a per-clip quantile keeps the
        map scale-free, so a quiet clip is not weighted differently from a busy one.
        \"\"\"
        d = (target[:, 1:] - target[:, :-1]).abs().mean(2, keepdim=True)
        d = torch.cat([d[:, :1], d], dim=1)                       # (b,t,1,h,w); frame 0 reuses 1
        b, t = d.shape[:2]
        d = F.avg_pool2d(d.flatten(0, 1), blur, stride=1, padding=blur // 2).view(
            b, t, 1, *d.shape[-2:]
        )
        thr = torch.quantile(d.flatten(1).float(), q, dim=1).view(b, 1, 1, 1, 1)
        return 1.0 + strength * (d / thr.clamp(min=1e-6)).clamp(0, 1)

    def _hook_clone(self, tensor: Tensor, loss_name: str) -> Tensor:""",
"motion weight helper", "def _motion_weight(")

# ---------------------------------------------------------------- loss.py: weighted L1
edit(loss_p,
"""        if self.weights.loss_mae > 0:
            loss["loss_mae"] = F.l1_loss(self._hook_clone(predicted, "loss_mae"), target)""",
"""        w = None
        if self.weights.motion_weight_strength > 0:
            with torch.no_grad():
                w = self._motion_weight(
                    target,
                    self.weights.motion_weight_strength,
                    self.weights.motion_weight_quantile,
                    self.weights.motion_weight_blur,
                )

        if self.weights.loss_mae > 0:
            _p = self._hook_clone(predicted, "loss_mae")
            if w is None:
                loss["loss_mae"] = F.l1_loss(_p, target)
            else:
                loss["loss_mae"] = ((_p - target).abs() * w).sum() / (w.sum() * predicted.shape[2])""",
"weighted L1", "(w.sum() * predicted.shape[2])")

# ---------------------------------------------------------------- loss.py: weighted LPIPS
edit(loss_p,
"""            loss["loss_lpips_perceptual"] = self.lpips_perceptual_loss(pred_2d, tgt_2d).mean()""",
"""            _lp = self.lpips_perceptual_loss(pred_2d, tgt_2d)
            if w is None:
                loss["loss_lpips_perceptual"] = _lp.mean()
            else:
                _wl = rearrange(w[:, t_lpips], "b t c h w -> (b t) c h w")
                if _lp.shape[-2:] != _wl.shape[-2:]:
                    _wl = F.interpolate(_wl, size=_lp.shape[-2:], mode="area")
                loss["loss_lpips_perceptual"] = (_lp * _wl).sum() / _wl.sum().clamp(min=1e-6)""",
"weighted LPIPS", "_wl.sum().clamp(min=1e-6)")

# ---------------------------------------------------------------- train_codec.py: freeze
edit(train_p,
"""    optimizer = torch.optim.AdamW(
        model.parameters(),""",
"""    # Decoder-only fine-tune. Freezing the encoder (frozen DINO + strided bottleneck) leaves the
    # latent space bit-identical, so a world model already trained on these latents stays valid and
    # does not need retraining. eval() as well as requires_grad, so nothing stateful drifts.
    if cfg.run.get("freeze_encoder"):
        raw_model.encoder.eval()
        for _p in raw_model.encoder.parameters():
            _p.requires_grad = False
        if is_main_process:
            _n = sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
            logger.info(f"freeze_encoder: training {_n:.1f}M of {n_params:.1f}M parameters")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],""",
"encoder freeze", "freeze_encoder: training")

print("\npatch complete")
