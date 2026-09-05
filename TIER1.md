# Tier 1: interactive Melee world model — results

Checkpoint: `justincg/melee-wm-weights` `wm/checkpoint.pth` (step 100,000).
All numbers below were measured on the training pod (RTX 5090, EU-CZ-1) on 2026-09-04.
Scripts that produced them are in `bench/`.

## Verdict

**Tier 1 gates pass. The dream is controllable, runs faster than real time, and holds 30 s.
It is not a faithful Melee simulation.** Ship it as a controllable research demo, not as a
playable game.

| PLAN.md gate | Result |
|---|---|
| Full WM: 10 s rollout stable | **PASS** — held 30 s, 3x the gate, no collapse |
| Full WM: intervention (jump input → jump) | **PASS** — see below |
| Demo: person plays with keyboard, video recorded | **PASS** — `bench/play_server.py` + `bench/play.html` |

## Training

| | |
|---|---|
| Steps | 100,000 / 100,001, exit 0, 0 retries |
| Train / val loss | 0.3564 / 0.3573 |
| Cost | ~$10.50 |

## Offline eval at step 100k

`bench/` — `eval_world_model_offline.py --viz 4 --num-samples 64`

| Metric | WM | Codec floor |
|---|---|---|
| Fréchet DINO | 0.517 | **0.335** |
| `fdd_at_10` | 0.595 | 0.429 |
| `fdd_at_20` | 0.787 | **0.640** |
| PSNR / LPIPS / SSIM | 15.88 / 0.408 / 0.573 | — |
| FID | 57.87 | — |

**~80% of the error at the 2 s horizon is the codec, not the world model.** More world-model
training buys almost nothing here; a better codec is the only lever that matters for fidelity.

## Why it is blurry: four measurements, three hypotheses killed

The world model is not the problem. These are all encoder+decoder only, on the same 40 held-out
clips, with the frozen codec the world model actually uses. Characters are located by motion (the
stage and HUD are static, the fighters are not), so no camera transform is needed.
Scripts: `bench/codec_audit.py`, `bench/ft_run.sh`, `bench/latent_probe.py`, `bench/char_probe.py`.
Raw numbers: `bench/results/*.json`.

### 1. The gap. Characters reconstruct 9.66 dB worse than the stage.

| | PSNR | vs MIRA Table 7 |
|---|---|---|
| Whole frame | 25.61 ± 1.10 dB | Base 27.6, Large 29.3 |
| **Background** | **26.61 ± 1.41 dB** | within 1 dB of MIRA's whole-frame Base |
| **Characters** | **16.95 ± 0.56 dB** | — |
| LPIPS | 0.109 ± 0.019 | Base 0.082, Large 0.055 |

The static scene is near paper quality. The codec is not generically undertrained — it fails
specifically on the two things a Melee agent needs to see.

### 2. Killed: "characters are bad because they are small."

Character PSNR, smallest-motion half **16.91 dB** vs largest-motion half **16.99 dB**. A 0.09 dB
difference is noise. On-screen size does not predict the failure, so this is not a /32 spatial
capacity limit.

### 3. Killed: "the loss is dominated by 97% static background."

Decoder-only fine-tune with the reconstruction loss reweighted toward motion, encoder frozen so
the latents and the trained world model stay valid. Pre-registered gate: **+0.3 dB** character
PSNR after a 2,000-step pilot, or stop before the 20,000-step run.

| | Baseline | Pilot (2k steps) | Δ |
|---|---|---|---|
| Character PSNR | 16.95 dB | 17.12 dB | **+0.17** |
| Background PSNR | 26.61 | 26.66 | +0.05 |
| LPIPS | 0.1089 | 0.1122 | worse |

**Gate failed at +0.17 dB.** Run stopped at 2k instead of 20k. Reweighting the loss does not
recover the characters.

### 4. Killed: "the decoder is at fault." The information is not in the latent.

If character detail survived the encoder, a decoder fix could still work. Two probes on the
9×12×32 latent grid, split by clip so a held-out clip is never seen during the fit:

| Probe | Result | Reading |
|---|---|---|
| Latent energy vs pixel motion (`latent_probe.py`) | corr **0.123** | ambiguous — script's own bands are >0.15 present, <0.05 discarded |
| Linear, position vs position+latents (`char_probe.py`) | R² 0.1378 → 0.1434, **Δ +0.0056** | latents add ~nothing over "fighters are usually here" |
| Nonlinear control, same split, 2-hidden-layer MLP | best held-out R² 0.1344, **Δ −0.0033** | not a linearity artifact; MLP never beats the static prior |

A linear probe failing alone would prove nothing — the information could be nonlinearly coded.
The MLP is the control, and it agrees. **Character location is not recoverable from the latent
beyond a static positional prior.** The DINOv3-RAE encoder discards the players.

Caveats: n=1 model (200M, 36 h of video, 0.4% of MIRA's data); the character mask is motion-derived
and pooled to a coarse 9×12 grid; MIRA at their scale may not show this.

### 5. Corroboration: the dream sits exactly on the codec floor.

High-frequency energy (`bench/blur.py`), an instrument independent of PSNR:

| | high-freq energy |
|---|---|
| Real frames | 0.1016 |
| Codec round-trip | 0.0747 |
| Dream, 2 diffusion steps | **0.0752** |
| Dream, 4 steps | 0.0830 |
| Dream, 10 steps | 0.1270 (above real — sampling noise, not detail) |

At the 2 steps the live demo uses, the world model adds essentially nothing to the codec's loss of
detail. This matches the Fréchet DINO split above (WM 0.517 vs codec floor 0.335).

**Conclusion.** The ceiling is the encoder, not the decoder, not the loss weighting, and not the
world model. No amount of further world-model or decoder training moves it. Lifting it requires
retraining the encoder — which invalidates every latent and forces a full world-model retrain.

## Latency

`bench/speed_sweep.py`. One latent frame = 2 video frames = **100 ms** of game time at 20 fps,
so real time means ≤100 ms per call.

| Diffusion steps | ms / latent | Real time? |
|---|---|---|
| 1 | 46.3 | yes, 2.16x |
| 2 | 62.4 | yes, 1.60x |
| 4 | 94.8 | marginal, 1.05x |
| 8 | 159.5 | no |
| 10 (eval default) | 191.4 | no |

Live server measured end to end (denoise + decode + JPEG): **66 ms median, 70 ms p95 at 2 steps**,
sustained over 300 steps = **1.39x real time**. 2 steps is also mira-mini's shipped default.

## Long-horizon stability

`bench/long_real.py`, 30 s (300 latents) on a **real continuous action stream** (8 consecutive
chunks of match g00000; an earlier run that tiled one 4 s clip 8x was an out-of-distribution
input and is not a valid test).

Latent std stayed 0.92–1.00, `absmax` 3.5–4.5, and `dz` never approached 0 — no collapse, no
blow-up, scene still moving at t=29.75 s.

**Latent statistics are a lying instrument here.** They were flat across the full 30 s while the
characters visibly dissolved. Every stability claim in this file was confirmed by looking at
decoded frames.

MIRA reports flat quality to 5 minutes and rollouts "continuing for hours". The mechanism is
**diffusion forcing**, which this run does use (`sample_training_tau` draws an independent flow
time per frame). That is why 30 s was never the risk it looked like.

## Intervention gate

`bench/intervention.py`. Three rollouts from an identical seed and identical initial noise:
baseline, a byte-identical control, and one with `P1_X` (jump) forced on from t=2.0 s.

| | mean abs pixel diff vs baseline |
|---|---|
| Control (identical actions) | **0.000000** — rig is deterministic, so any diff is attributable |
| t=0–1 s (before intervention) | 0.00000 |
| t=2 s | 0.03563 |
| t=3 s | 0.11600 |
| t=4 s / t=5 s | 0.05740 / 0.06357 |

Direction is correct: with jump held the character rises above the platform and keeps climbing,
while the baseline stays at platform level. **The dream obeys the controller.**

## What is actually wrong with it

Read off decoded frames, worst first:

1. **Characters dissolve.** Clear at t=0 (the real seed), wispy by t=15 s. The codec's encoder
   discards them — measured, with the size and loss-weighting explanations both ruled out. See
   "Why it is blurry" above. This is the binding constraint on everything else in this file.
2. **Stage geometry drifts** — platforms move, multiply, sit at wrong heights.
3. **Game state is hallucinated** — percents jump (93→164→0→198→30), timer runs backward
   (07:12 → 07:32). Melee timers count down.
4. **Data.** 717 games ≈ 36 hours vs MIRA's 10,000 hours — about **0.4%**.
5. **Model size** 200M vs MIRA's 1B (ablations) / 5B (live demo).

## Deviations from the MIRA recipe

Matched: diffusion forcing, 16 layers, 16/4 GQA, temporal attention every 4, patch size 1,
AdaLN, attention gating, AdamW lr 1e-4 betas (0.9, 0.99) wd 0.1, 1000 warmup, pretrained-DINOv3
RAE codec, 20 fps, temporal patch 2, 32 latent channels, `use_clean_past: true`.

| Deviation | Ours | MIRA |
|---|---|---|
| Training clip length | 40 frames (2 s) | 80 frames (4 s) |
| Few-step distillation (PSD) | off (`psd_loss_prob: 0.0`) | distilled to 1–2 steps |
| LR schedule | decay over 99k | constant after warmup |
| `latent_mean_std` | null | 0.457 / 10.688 |

PSD was not needed: 2 undistilled steps already clear the real-time budget.

## How to play it

On the pod:

    setsid /workspace/start_play.sh < /dev/null &     # STEPS=2 by default

From the laptop:

    ssh -N -L 8765:localhost:8765 root@<pod-ip> -p <pod-port>

then open `bench/play.html`. Arrows are the P1 stick, `Z`/`X`/`C` are A/B/jump, `WASD` +
`1`/`2`/`3` drive P2. Every frame is recorded server-side to `/workspace/play/session_*.mp4`
at a true 20 fps, so the recording is clean no matter how laggy the link is.

**The pod is in Czechia, ~110 ms from Toronto.** Compute is 66 ms but the round trip adds
~110 ms, so play from Toronto feels like ~180 ms of input lag — real, not crisp. The recording
is unaffected. For a crisp session, redeploy on a US-East pod; the weights are on HF so it costs
~20 min and ~$1.

## Known ceilings

- KV cache grows with session length; `play_server.py` re-seeds at `--max-latents` (default 1200
  = 120 s). A ring buffer would be needed for unbounded sessions.
- P2 keys left at zero mean "opponent stands still", not "opponent is unconditioned". A passive
  opponent is the default unless a second player or a policy drives those 16 bits.
- Only Fox vs Falcon on Battlefield exists in the training data. Character identity is not an
  input; it is baked in.
