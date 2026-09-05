# melee-wm

A playable neural world model of Super Smash Bros. Melee, built by reproducing the
MIRA recipe (General Intuition / Kyutai) on a new domain.

Fox vs Captain Falcon on Battlefield. 717 games (~36 h) of Slippi replays, rendered to video and
paired with exact per-frame controller inputs for **both** players. A DINOv3-RAE codec, then a
200M-parameter latent world model with diffusion forcing. You can hold a controller against it.

Built solo in 4 days for **~$69** of rented GPU.

---

## The finding

> A DINOv3-RAE codec at MIRA's own settings reconstructs Melee's static stage at near-paper
> quality (26.6 dB, vs MIRA's 27.6 dB whole-frame Base) but the two fighters **9.66 dB worse**
> — and the information is gone at the encoder, not the decoder.

Three explanations were tested and rejected:

| Hypothesis | Test | Result |
|---|---|---|
| Characters are small | PSNR vs on-screen size | 16.91 vs 16.99 dB — **no size dependence** |
| Loss is swamped by 97% static background | Motion-weighted decoder fine-tune, pre-registered +0.3 dB gate | **+0.17 dB, gate failed** |
| The decoder is at fault | Linear + nonlinear probes, latent → character location | **Δ R² +0.006 / −0.003** — latents add nothing over a static positional prior |

This matters for anyone training agents inside a learned simulator: the representation is worst
precisely on the pixels an agent needs. Full working in [TIER1.md](TIER1.md).

Caveats stated up front: n=1 model, 200M params, 36 hours of video — about 0.4% of MIRA's data.
Characters are located by motion, pooled to a coarse 9×12 latent grid. A model at MIRA's scale
may not show this.

---

## What works

| Gate (set before the run) | Result |
|---|---|
| 10 s rollout stays coherent | **PASS** — held 30 s on a real action stream |
| Intervention: change an input, the future changes | **PASS** — byte-identical control arm at exactly 0.000 |
| Real-time playable | **PASS** — 66 ms median per latent, 1.39x real time |
| Person plays it with a keyboard | **PASS** — `bench/play_server.py` |

Videos in [`bench/tier1_video/`](bench/tier1_video/):

- `intervention.mp4` — same seed, same noise; jump input forced on at t=2 s. The character rises,
  the baseline does not.
- `dream_real30s_st4_n0.0.mp4` — 30 s continuous rollout.
- `session_191319.mp4` — a live human-played session, recorded server-side at a true 20 fps.

Numbers, including the offline eval and the latency sweep: [TIER1.md](TIER1.md).

---

## Layout

    PLAN.md              goals, gates, and the Tier 2 experiment design
    TIER1.md             all results and the full measurement chain
    bench/               training, eval, and probe scripts (run on the pod)
      codec_audit.py       character vs background PSNR
      ft_run.sh            motion-weighted decoder fine-tune + gate
      latent_probe.py      latent energy vs pixel motion
      char_probe.py        linear + MLP probe, latent -> character location
      intervention.py      controllability test with a null control
      play_server.py       websocket server: keys in, decoded frames out
      results/*.json       every number quoted in TIER1.md
    render/              Slippi -> video pipeline (Linux, xvfb + playback Dolphin)
    bots/                slippi-ai harness for real-Dolphin evaluation

Weights and data: `justincg/melee-wm-weights` and `justincg/melee-fox-falcon-bf` on Hugging Face
(private). Pods are disposable; nothing depends on one existing.

---

## Reproducing

Everything ran on a single rented RTX 5090.

| Stage | Steps | Wall clock | Cost (approx) |
|---|---|---|---|
| Render + package 757 games | — | ~1 day | ~$24 |
| Codec | 80,000 | ~18 h | ~$18 |
| World model | 100,000 | ~17 h | ~$10.50 |
| Eval, probes, demo | — | — | ~$16 |

Per-stage costs are approximate; the RunPod billing total across the project is **~$69**.

Scripts assume `/workspace` on the pod, data pulled from HF, checkpoints pushed back at fixed
milestones. `bench/codec_run.sh` and `bench/wm_run.sh` are the entry points.

---

## What I would do next, and why I stopped

The natural next step is the transfer experiment in [PLAN.md](PLAN.md): put a slippi-ai agent
inside the dream, fine-tune with PPO, and measure what fraction of a real-Dolphin training gain
it recovers — `(dream-tuned − baseline) / (real-tuned − baseline)`, all evaluated in real Dolphin
against a frozen opponent.

Melee is a good place to ask that question: exact per-frame action labels from Slippi, a second
independently-controlled adversarial player, ground-truth state in the replay files, a free
headless real environment to measure against, and skill that reduces to a win rate.

I have not run it. Step 1 of that experiment needs character state read off decoded frames, and
the measurement above says that information is not in the latent. Fixing it means retraining the
encoder, which invalidates the world model and costs more than this project has. The honest
version of the result is the diagnosis, not a transfer number I did not measure.
