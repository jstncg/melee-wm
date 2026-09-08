# Tier 1: interactive Melee world model, results

Checkpoint: `justincg/melee-wm-weights`, `wm/checkpoint.pth`, step 100,000.
Measured on the training pod (RTX 5090, EU-CZ-1) on 2026-09-04, except the character probes,
which were run on 2026-09-05. The measurement scripts are in `bench/` and their raw output is in
`bench/results/`.

## Verdict

The Tier 1 gates pass. The dream is controllable, runs faster than real time, and holds 30 s.
It is not a faithful Melee simulation, and the reason is diagnosed below.

| Gate, from PLAN.md | Result |
|---|---|
| Full WM: 10 s rollout stable | Pass. Held 30 s, three times the gate, no collapse. |
| Full WM: intervention, jump input produces a jump | Pass. See below. |
| Demo: a person plays with a keyboard, video recorded | Pass. `bench/play_server.py` and `bench/play.html`. |

## Training

| | |
|---|---|
| Steps | 100,000 of 100,001, exit 0, no retries |
| Train / val loss | 0.3564 / 0.3573 |
| Cost | about $10.50 |

## Offline eval at step 100k

`eval_world_model_offline.py --viz 4 --num-samples 64`

| Metric | WM | Codec floor |
|---|---|---|
| Fréchet DINO | 0.517 | 0.335 |
| `fdd_at_10` | 0.595 | 0.429 |
| `fdd_at_20` | 0.787 | 0.640 |
| PSNR / LPIPS / SSIM | 15.88 / 0.408 / 0.573 | |
| FID | 57.87 | |

About 80% of the error at the 2 s horizon is the codec rather than the world model. More
world-model training buys very little; the codec is the lever that matters for fidelity.

## Why it is blurry

Four measurements, three hypotheses killed. All of these are encoder and decoder only, on the
same 40 held-out clips, using the frozen codec the world model was trained against. Characters
are located by motion, since the stage and HUD are static and the fighters are not, so no camera
transform is needed.

Scripts: `codec_audit.py`, `ft_run.sh`, `latent_probe.py`, `char_probe.py`.

### 1. The gap

![Source above, codec reconstruction below](docs/character_gap.png)

Held-out clip, codec round-trip only, no world model. Everything static survives and both
fighters do not. The full frame is in `bench/recon_40000.png`.

| | PSNR | MIRA Table 7 |
|---|---|---|
| Whole frame | 25.61 ± 1.10 dB | Base 27.6, Large 29.3 |
| Background | 26.61 ± 1.41 dB | |
| Characters | 16.95 ± 0.56 dB | |
| LPIPS | 0.109 ± 0.019 | Base 0.082, Large 0.055 |

The static scene lands within 1 dB of MIRA's whole-frame Base number, so the codec is not
generically undertrained. It fails on the two objects a Melee agent needs to see.

### 2. Killed: characters are bad because they are small

Character PSNR on the smallest-motion half is 16.91 dB, against 16.99 dB on the largest-motion
half. A 0.09 dB difference is noise. On-screen size does not predict the failure, so this is not
a /32 spatial capacity limit.

### 3. Killed: the loss is dominated by 97% static background

Decoder-only fine-tune with the reconstruction loss reweighted toward motion. The encoder stays
frozen so the latents, and the trained world model above them, stay valid. The gate was
registered before the run: +0.3 dB character PSNR after a 2,000-step pilot, or stop without
spending the 20,000-step run.

| | Baseline | Pilot, 2k steps | Change |
|---|---|---|---|
| Character PSNR | 16.95 dB | 17.12 dB | +0.17 |
| Background PSNR | 26.61 dB | 26.66 dB | +0.05 |
| LPIPS | 0.1089 | 0.1122 | worse |

The gate failed at +0.17 dB and the run stopped at 2k. Reweighting the loss does not recover the
characters.

### 4. Killed: the decoder is at fault

If character detail survived the encoder, a decoder-side fix could still work. Two probes on the
9x12x32 latent grid, split by clip so a held-out clip is never seen during the fit:

| Probe | Result | Reading |
|---|---|---|
| Latent energy against pixel motion (`latent_probe.py`) | corr 0.123 | Ambiguous. The script's own bands are above 0.15 present, below 0.05 discarded. |
| Linear, position against position plus latents (`char_probe.py`) | R2 0.1378 to 0.1434, change +0.0056 | Latents add almost nothing over "fighters are usually here". |
| Nonlinear control, same split, 2-hidden-layer MLP | best held-out R2 0.1344, change -0.0033 | Not a linearity artifact. The MLP never beats the static prior. |

A linear probe failing on its own would prove nothing, since the information could be coded
nonlinearly. The MLP is the control and it agrees. Character location is not recoverable from the
latent beyond a static positional prior, which means the DINOv3-RAE encoder discards the players.

Caveats: one model, 200M parameters, 36 hours of video, roughly 0.4% of MIRA's data. The
character mask is motion-derived and pooled to a coarse 9x12 grid. MIRA at their scale may not
show this.

### 5. Corroboration

High-frequency energy, an instrument independent of PSNR. Raw output is in
`bench/results/blur.json`. The script itself ran on the pod and was not pulled off before the
pod was deleted.

| | High-frequency energy |
|---|---|
| Real frames | 0.1016 |
| Codec round-trip | 0.0747 |
| Dream, 2 diffusion steps | 0.0752 |
| Dream, 4 steps | 0.0830 |
| Dream, 10 steps | 0.1270, above real, which is sampling noise rather than detail |

At the 2 steps the live demo uses, the world model adds almost nothing to the codec's loss of
detail. That matches the Fréchet DINO split above, 0.517 against a 0.335 codec floor.

The ceiling is the encoder. Not the decoder, not the loss weighting, and not the world model. No
amount of further world-model or decoder training moves it. Lifting it means retraining the
encoder, which invalidates every latent and forces a full world-model retrain.

## Latency

`bench/speed_sweep.py`. One latent frame is 2 video frames, so 100 ms of game time at 20 fps.
Real time means 100 ms or less per call.

| Diffusion steps | ms per latent | Real time |
|---|---|---|
| 1 | 46.3 | yes, 2.16x |
| 2 | 62.4 | yes, 1.60x |
| 4 | 94.8 | marginal, 1.05x |
| 8 | 159.5 | no |
| 10, the eval default | 191.4 | no |

The live server, measured end to end including denoise, decode and JPEG encoding: 66 ms median
and 70 ms p95 at 2 steps, sustained over 300 steps, so 1.39x real time. 2 steps is also
mira-mini's shipped default.

## Long-horizon stability

`bench/long_real.py`, 30 s or 300 latents, on a real continuous action stream: 8 consecutive
chunks of match g00000. An earlier run tiled one 4 s clip eight times, which is an
out-of-distribution input and not a valid test.

Latent std stayed between 0.92 and 1.00, `absmax` between 3.5 and 4.5, and `dz` never approached
0. No collapse, no blow-up, and the scene is still moving at t=29.75 s.

Those latent statistics are a lying instrument. They were flat across the full 30 s while the
characters visibly dissolved, so every stability claim in this file was confirmed by looking at
decoded frames.

MIRA reports flat quality out to 5 minutes and rollouts continuing for hours. The mechanism is
diffusion forcing, which this run does use, since `sample_training_tau` draws an independent flow
time per frame. 30 s was a smaller risk than it looked.

## Intervention gate

`bench/intervention.py`. Three rollouts from an identical seed and identical initial noise: a
baseline, a byte-identical control, and one with `P1_X` (jump) forced on from t=2.0 s.

The base action stream here is one 4 s clip tiled out to 6 s, which is the same
out-of-distribution input criticised in the stability section above. It is acceptable for this
test and not for that one: all three arms share the identical base stream, so the only difference
between baseline and intervened is the forced key, and the control arm proves the rig adds
nothing of its own. The stability test measures absolute quality over time, which tiling would
corrupt, so that one uses a real continuous stream.

| | Mean absolute pixel difference against baseline |
|---|---|
| Control, identical actions | 0.000000. The rig is deterministic, so any difference is attributable. |
| t=0 to 1 s, before the intervention | 0.00000 |
| t=2 s | 0.03563 |
| t=3 s | 0.11600 |
| t=4 s / t=5 s | 0.05740 / 0.06357 |

The direction is right. With jump held the character rises above the platform and keeps climbing
while the baseline stays at platform level.

## What is wrong with it

Read off decoded frames, worst first.

1. Characters dissolve. Clear at t=0, which is the real seed, and wispy by t=15 s. The encoder
   discards them, as measured above, with the size and loss-weighting explanations both ruled
   out. This is the binding constraint on everything else in this file.
2. Stage geometry drifts. Platforms move, multiply, and sit at wrong heights.
3. Game state is hallucinated. Percentages jump (93, 164, 0, 198, 30) and the timer runs backward
   (07:12 to 07:32). Melee timers count down.
4. Data. 717 games, about 36 hours, against MIRA's 10,000 hours, so roughly 0.4%.
5. Model size. 200M against MIRA's 1B for ablations and 5B for the live demo.

## Deviations from the MIRA recipe

Matched: diffusion forcing, 16 layers, 16/4 GQA, temporal attention every 4, patch size 1, AdaLN,
attention gating, AdamW at lr 1e-4 with betas (0.9, 0.99) and wd 0.1, 1000 warmup steps,
pretrained DINOv3 RAE codec, 20 fps, temporal patch 2, 32 latent channels, `use_clean_past: true`.

| Deviation | Here | MIRA |
|---|---|---|
| Training clip length | 40 frames, 2 s | 80 frames, 4 s |
| Few-step distillation (PSD) | off, `psd_loss_prob: 0.0` | distilled to 1 or 2 steps |
| LR schedule | decay over 99k | constant after warmup |
| `latent_mean_std` | null | 0.457 / 10.688 |

PSD was not needed, since 2 undistilled steps already clear the real-time budget.

## How to play it

On the pod:

    setsid /workspace/start_play.sh < /dev/null &     # STEPS=2 by default

From the laptop:

    ssh -N -L 8765:localhost:8765 root@<pod-ip> -p <pod-port>

then open `bench/play.html`. To point the page at a pod directly instead of through a tunnel,
append `?ws=wss://host:port`. Arrows are the P1 stick, `Z`/`X`/`C` are A/B/jump, and `WASD` with
`1`/`2`/`3` drive P2. Every frame is recorded server-side to `/workspace/play/session_*.mp4` at a
true 20 fps, so the recording stays clean however laggy the link is.

The pod used here was in Czechia, about 110 ms from Toronto. Compute is 66 ms, so play from
Toronto felt like roughly 180 ms of input lag. The recording is unaffected. A US-East pod fixes
it for about 20 minutes of setup and $1, since the weights are on Hugging Face.

## Known ceilings

* The KV cache grows with session length. `play_server.py` re-seeds at `--max-latents`, default
  1200, which is 120 s. Unbounded sessions would need a ring buffer.
* P2 keys left at zero mean the opponent stands still, not that the opponent is unconditioned. A
  passive opponent is the default unless a second player or a policy drives those 16 bits.
* Only Fox against Falcon on Battlefield exists in the training data. Character identity is not
  an input, it is baked in.
