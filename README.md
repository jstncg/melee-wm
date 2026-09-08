# melee-wm

A playable neural world model of Super Smash Bros. Melee, built by running the
[MIRA](https://github.com/mira-wm/mira) recipe on a new domain.

Fox vs Captain Falcon on Battlefield. 757 games rendered to video and packaged, 717 for training
and 40 held out, about 36 hours in total, each paired with both players' controller inputs as a
single 32-key action stream. A
DINOv3-RAE codec, then a 200M-parameter latent world model with diffusion forcing. You can hold a
controller against it.

Solo, 4 days, about $69 of rented GPU.

## The finding

The codec reconstructs the stage, HUD and text almost perfectly, and the two fighters 9.66 dB
worse. The information is gone at the encoder, not the decoder.

![Source above, codec reconstruction below](docs/character_gap.png)

Held-out clip. Top is the source, bottom is the codec round-trip. The timer, "Ready", the damage
percentages, the platforms and the stage all survive. Both fighters are smears.

On 40 held-out clips: background 26.61 dB, characters 16.95 dB, whole frame 25.61 dB. MIRA Table 7
reports 27.6 dB whole-frame for their Base codec, so this model is roughly 2 dB behind overall and
10 dB behind on the fighters.

Three explanations, all tested and rejected:

| Hypothesis | Test | Result |
|---|---|---|
| Characters are small | PSNR against on-screen size | 16.91 vs 16.99 dB, no size dependence |
| Loss is swamped by 97% static background | Motion-weighted decoder fine-tune, pre-registered +0.3 dB gate | +0.17 dB, gate failed |
| The decoder is at fault | Linear and MLP probes, latent to character location | delta R2 +0.006 and -0.003 over a position-only prior |

Full working in [TIER1.md](TIER1.md). Every number quoted there has its raw output in
`bench/results/`.

Limits worth stating: one model, 200M parameters, 36 hours of video, roughly 0.4% of MIRA's
data.
Characters are located by motion and pooled to a coarse 9x12 latent grid. A model at MIRA's scale
may not show this at all.

## What works

Gates were written into [PLAN.md](PLAN.md) before the runs, not after.

| Gate | Result |
|---|---|
| 10 s rollout stays coherent | held 30 s on a real action stream |
| Change an input, the future changes | control arm at exactly 0.000, then divergence |
| Real-time playable | 66 ms median per latent, 1.39x real time |
| A person plays it with a keyboard | `bench/play_server.py` |

![Intervention: baseline above, jump input forced on below](docs/intervention.gif)

Same seed, same initial noise. Top is the baseline, bottom has the jump input held from t=2 s.
The two are identical until the intervention (mean absolute pixel difference 0.000), then diverge:
0.036 at t=2 s, 0.116 at t=3 s.

Videos in [`bench/tier1_video/`](bench/tier1_video/):

* `intervention.mp4`, the clip above
* `dream_real30s_st4_n0.0.mp4`, a 30 s continuous rollout
* `session_191319.mp4`, a live human-played session recorded server-side at a true 20 fps

## Layout

    PLAN.md              goals, gates, and the Tier 2 experiment design
    TIER1.md             results and the full measurement chain
    bench/               training, eval, and probe scripts
      codec_audit.py       character vs background PSNR
      ft_run.sh            motion-weighted decoder fine-tune and its gate
      latent_probe.py      latent energy against pixel motion
      char_probe.py        linear and MLP probes, latent to character location
      intervention.py      controllability test with a null control
      play_server.py       websocket server, keys in and decoded frames out
      results/             raw output behind every number in TIER1.md
    data/                Slippi replays to training shards
      select_games.py      regenerate the game list from the public dataset
      parse.py             .slp -> per-frame actions and state
      package.py           shards in MIRA's WebDataset layout
    render/              replays to video, on Linux under xvfb with playback Dolphin
    bots/                slippi-ai harness for real-Dolphin evaluation

## Running this yourself

This is a record of a set of experiments, not a package you can install. Five things are needed,
and two of them cannot be shipped.

| Need | Where |
|---|---|
| MIRA | [mira-wm/mira](https://github.com/mira-wm/mira), cloned by `bench/setup_wm.sh` |
| Slippi replays | [erickfm/slippi-public-dataset-v3.7](https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7), public |
| A Melee 1.02 ISO | Not distributable. Supply your own. |
| DINOv3-L/16 weights | Gated on Hugging Face, needs your own approved access |
| A GPU | 32 GB VRAM. Everything here ran on one rented RTX 5090. |

The rendered dataset is roughly 90 GB of Nintendo game footage, so it stays private. The replay
source above is public and `data/` holds the scripts that rebuild everything from it:
`select_games.py` regenerates the 889-game list (2-player Fox vs Captain Falcon on Battlefield),
`download.py` fetches and parses them, and `package.py` writes MIRA's shard layout. 889 games
match the filter, 757 of those were rendered before the render budget was called, and those became
717 training games and 40 held out.

Model weights live in `justincg/melee-wm-weights` on Hugging Face.

Scripts in `bench/` and `render/` are written for a RunPod pod with everything under `/workspace`,
and those paths are hardcoded. They are the scripts that produced the numbers rather than a
general-purpose CLI, and they are committed in that form on purpose.

`pyproject.toml` covers data preparation only. The training and eval environment is built on the
pod by `bench/setup_wm.sh`.

## Cost

| Stage | Steps | Wall clock | Cost |
|---|---|---|---|
| Render and package 757 games | | about 1 day | $24 |
| Codec | 80,000 | 18 h | $18 |
| World model | 100,000 | 17 h | $10.50 |
| Eval, probes, demo | | | $16 |

Per-stage figures are approximate. RunPod billing across the whole project came to about $69.

## What I would do next, and why I stopped

The next step is the transfer experiment in [PLAN.md](PLAN.md): put a slippi-ai agent inside the
dream, fine-tune it with PPO, and measure what fraction of a real-Dolphin training gain it
recovers, as `(dream-tuned - baseline) / (real-tuned - baseline)`, all evaluated in real Dolphin
against a frozen opponent.

Melee suits that question. Slippi gives exact per-frame action labels. There is a second
independently controlled adversarial player, ground-truth state in the replay files, a free
headless real environment to measure against, and skill reduces to a win rate.

I have not run it. Step 1 needs character state read off decoded frames, and the measurement above
says that information is not in the latent. Fixing that means retraining the encoder, which
invalidates the world model and costs more than this project has.
