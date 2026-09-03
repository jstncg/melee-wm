# melee-wm

Neural Melee: MIRA's recipe (github.com/mira-wm/mira) on Super Smash Bros. Melee.
Goal: 2-player world model, then self-play RL inside it, then transfer test in real Dolphin.

## Data pipeline

| Step | Tool | Status |
|---|---|---|
| 1. Replays | erickfm/slippi-public-dataset-v3.7 (95k .slp, CC0) | 5 downloaded to `data/raw/` |
| 2. Actions + state | `data/parse.py` -> .npz `(P,T,16)` actions, `(P,T,8)` state | done, self-check passes |
| 3. Video | `render/render_one.py` + `render_all.sh`: Slippi playback Dolphin (Linux AppImage) under xvfb on a RunPod pod, framedump -> h264 mp4 @60fps. macOS build cannot dump frames. | verified: 642x528@60, frame N == replay frame N. Full 889-game render running (16 parallel) |
| 4. Shards | `data/package.py` writes MIRA's WebDataset layout (4 s chunks, p0.mp4/p0.jsonl/physics/meta, index.json) so `RocketScienceDataset` loads unchanged | verified: MIRA `RocketScienceDataset` loads it, frames (1,40,3,288,384), actions (1,40,32), physics ok |
| 5. Configs | `mira_configs/`: actions/melee (32 keys), dataset/melee (288x384, 20 fps), codec raev2_codec_melee (small decoder), wm 200m | done, compose OK on pod |

## Action vector (K=16)
A B X Y Z L R, joy_x{neg,pos}, joy_y{neg,pos}, c_x{neg,pos}, c_y{neg,pos}, trigger>0.3

## State vector (S=8)
percent, stocks, x, y, direction, airborne, action_state, l_cancel

## Training (needs GPU, later)
codec -> WM (one perspective, 32-key both-players actions) -> eval (FID/horizon, action recoverability, counterfactual)

NO 2-player warm-start stage. Verified 2026-09-03: MIRA's `n_players` counts *camera
perspectives*, not players — `MultiWrapperWorldModel` tiles n clips into one vertically
stacked frame (`wm_config.video.height *= n_players`). That is Rocket League (4 players,
4 cameras). Melee is ONE shared camera, so `dataset/melee.yaml` correctly sets
`n_players: 1` and `actions/melee.yaml` puts both players' inputs in a 32-key vocab.
The single-perspective `LatentWorldModel` conditioned on those 32 keys IS the 2-player
Melee world model. Do NOT use `model=multi_wrapper_world_model` — it would tile two
copies of the same screen and double the compute for nothing. Deleting this stage saves
one full training run (~9 h, ~$6).

## Research extras, in order
1. Counterfactual replay: change one input at frame N, watch the alternate timeline
2. Bot in the dream: slippi-ai inside the model vs. in real Dolphin, measure the gap
3. Self-play RL inside the dream, test fine-tuned bot in real Dolphin

## Cloud (2026-09-02)
- RunPod pod `melee-gpu` (id 1u95fefmqmn48r): RTX 4090, 64 vCPU, 150 GB /workspace, EUR-IS-2, $0.74/h. ssh via `POD=gpu bash render/pod.sh '<cmd>'`.
- RunPod pod `melee-render` (id bq5rsj96nrjqvo): 2 vCPU test pod, $0.06/h. Delete once GPU pod render is confirmed.
- Network volume `melee-wm` (32zukc3iiz, CA-MTL-3, 100 GB): unused so far (MCP create-pod cannot attach it).
- On pod: /workspace/melee/{iso,slp,video,weights,render,tools}, /workspace/mira (pixi env ready, Melee configs installed).
- Design decision: 20 fps first (MIRA's exact recipe, loader strides 60->20). 60 fps is a later knob.
- Overnight 2026-09-02: `render/pipeline.sh` on pod = wait render -> package (train / 40 test games) -> codec 100k steps batch 2 (batch 4 OOMs on 24 GB). Logs: render/render_all.log, render/pipeline.log, render/codec_train.log.
- CPU test pod deleted. HF token in ~/melee-wm/.env (fine-grained, "melee").
- 2026-09-02 08:30 UTC: measured 34 games/h on pod 1 (24 Dolphins, CPU-bound). Added pod `melee-render2` (id 1o7d9ay7okk5xg, 4090, 96 vCPU, $0.74/h, `POD=gpu2`) rendering sorted indices [250,700) via `render/render_slice.sh`; `render/sync_to_pod1.py` pushes results to pod 1 every 5 min and pulls skip markers. Delete pod 2 when its slice is done.
- 2026-09-02 11:00 UTC: combined rate only ~55 games/h (shared hosts, CPU contention). Added pod `melee-render3` (id q45xwk3et8xy58, `POD=gpu3`) rendering [250,700) in REVERSE so it meets pod 2 in the middle; both sync to pod 1. Delete pods 2 and 3 once "render_slice done" appears in their logs.
- 2026-09-02 13:20 EDT: STOPPED on Justin's request (RunPod balance nearly out). Spent $17.59 (3 x 4090 secure pods at $0.74/h). Pod 1 `melee-gpu` is STOPPED (not deleted): 759/889 games rendered + ISO + replays + npz + mira env on its 150 GB disk (stopped-disk billing ~$0.20/GB/mo => ~$1/day). Pods 2 and 3 deleted. To resume: start pod 1, `RENDER_PARALLEL=24 bash render/render_all.sh` finishes the last ~130 games, then `bash render/pipeline.sh`.

## Goals, stops, and the transfer experiment (2026-09-02)

Data: 757 games (717 train / 40 test) packaged from 760 renders; 132 games never rendered (skipped, not worth $4+). Dataset lives at HF `justincg/melee-fox-falcon-bf` (shards + video + slp + npz). Pods are disposable; never rely on a stopped pod's disk.

### Tier 1 (must): interactive Melee world model
| Stage | Pass | Fail |
|---|---|---|
| Small codec run, 20 games, 1k steps | loss falls steadily; reconstructed frame shows two recognizable characters on Battlefield | flat/NaN loss, mush |
| Full codec | held-out (40 games) reconstruction error close to train | held-out much worse |
| Small WM run | 20-frame rollout stays a Melee scene | blur/drift within 10 frames |
| Full WM | 10 s rollout stable; intervention test: change one input at frame N, the future changes correctly (jump input -> jump) | collapse < 10 s, or inputs ignored |
| Demo | a person plays inside it with keyboard; video recorded; one-page writeup with the eval numbers | |
Hard stop: if the full WM fails the 10 s + intervention tests after two attempts, ship the counterfactual demo + writeup and stop.

### Tier 2 (stretch): the transfer number
Claim under test: a two-player world model learned from video can replace the real game for RL fine-tuning; report what fraction of the real gain it delivers.
1. Real-game harness: slippi-ai `run_evaluator.py`, headless Dolphin, N games vs a FROZEN opponent (the strong released checkpoint). Baseline win rate + stock diff with 95% CI (200 games ~ +/-7 pts).
2. Agent under test: the weakest released slippi-ai checkpoint (imitation-only if available), so there is headroom.
3. State from the dream. CORRECTED 2026-09-03: MIRA's WM does NOT emit state. Verified by grep — `physics` appears only in `src/mira/data/` (dataset, viz, state, physics helpers: consistency checks, frozen-clip detection, overlay badges). ZERO hits in `models/`, `training/`, `world_model/`, `codec/`, `inference/`. The shard format carrying physics is a DATA fact, not a model fact; adding state output means a new prediction head + loss + training wiring (real surgery). Cheaper path for the gate: train a small CNN state-reader on decoded frames, supervised by the 757 games of paired (frame, .npz state) already on disk — no MIRA changes. Use the reader to clear this gate; build the head only if RL actually proceeds (per-frame decoding is too slow for rollout-heavy RL). Gate unchanged: on the 40 held-out real games, state error must be small (positions within a few px, action_state accuracy high) or stop before any RL spend.
4. Bot-in-the-dream: run the agent inside the dream via the emitted state; compare damage/min, stock rate, off-stage rate vs real Dolphin. Large gap => dream not faithful, stop.
5. Fine-tune in the dream with slippi-ai's PPO, 5-20M frames, ~8 dreams in parallel (~160 fps). Reward = damage dealt - taken, stocks. Watch for reward hacking (dream damage/min >> real).
6. Control arm: same agent, same frame budget, fine-tuned in real fast-forward Dolphin.
7. Number = (dream-tuned - baseline) / (real-tuned - baseline), all measured in real Dolphin vs the frozen opponent.
Honest framing: Melee is the checkable stand-in (real env is cheap here, the dream is NOT faster than fast-forward Dolphin); the value is fidelity/transfer, not speed. Prior art: Dreamer 4 (single-player, real-game eval), MIRA (4-player WM, no agent), COMBAT (Tekken WM, no agent, no real-game eval, names policy-in-latent as future work), slippi-ai (state RL in real Dolphin). Nothing does agent-in-multiplayer-dream + real transfer.

### Efficiency rules (learned the expensive way)
- Measure before assuming: time a step, run `top`, test bandwidth. Every expensive mistake today was an unmeasured assumption.
- Small end-to-end run before any big run. Benchmark (200 steps on 4090 and H100, ~$1) before quoting training cost.
- Community cloud for training; secure pods only if a network volume is actually needed. Data on HF, pods disposable, checkpoints synced to HF every save, resume-from-latest in a restart loop. W&B on.
- Rendering is CPU-bound (llvmpipe); a 0-GPU "CPU start" pod has a 488 MB memory cap and cannot render.
- Turn on MIRA's built-in PSD few-step distillation (mira-mini recipe: zero-init delta pathway, lr 3e-5, ~10% of updates) for fps; KV caching already in MIRA. Codec quality caps the WM ("at 45k steps the WM is codec-limited", mira-mini).

### Laptop, $0
- `bots/play.sh`: slippi-ai medium-v2 bot vs bot in real Slippi Dolphin (verified 2026-09-02). Next: run_evaluator.py head-to-head with a fixed game count.
- slippi-ai released models (Dropbox "deployed_models", 2026-09-02 inventory): naming = `<char>_d<delay>_imitation_vN` (stage 1, imitation only) and `<char>_d<delay>_vs_<opp>` / `_ditto` (stage 2, RL-tuned vs that matchup). Rank-tier multi-char bots `silver/gold/plat/diamond/master/gm` (91 MB) = skill ladder rulers. For Fox vs Falcon on BF: agent under test = `fox_d18_imitation_v3`; frozen opponents = `falcon_delay_18_vs_fox` (RL Falcon) and `fox_delay_18_vs_falcon` (RL Fox). d18 = 18-frame input delay.
- `bots/evaluate.sh` runs run_evaluator.py rendered on the Mac at ~53 fps (real time). Headless/fast-forward needs vladfi1's custom Dolphin (Linux); do 200-game evals on a Linux CPU pod.
- 2026-09-02 22:00 EDT: DATASET DONE on HF `justincg/melee-fox-falcon-bf` (shards 36+2, video, slp, npz; 3,335 files). All pods deleted. Day's RunPod bill $24.15 total ($6.56 for the recovery path vs $2.50 quoted: 13.6-core packaging 3 h, upload 1.5 h, 1.4 h idle after a failed self-stop). Next: "benchmark" (~$2).
- 2026-09-02 22:25 EDT BENCHMARK (community RTX 5090 32 GB, $0.69/h, 10 vCPU quota, compile off, 61/41 steps): codec bs=2 0.805 s/step, bs=4 OOM. WM (200m, 288x384, 40 frames) bs=1 0.205, bs=2 0.329, bs=4 0.625 s/step, bs=8 OOM. => codec 100k steps = 22 h = $15; WM 100k steps @bs4 = 17 h = $12 (or bs2 9 h = $6). Env setup via `pixi run setup` takes 1 min; `bench/bench.sh` + configs live on HF under bench/. Community pods with no public IP need the account SSH key + `render/proxy_run.py` (proxy forces a PTY).
- 2026-09-02 22:55 EDT SMOKE (5090, 1,001 codec steps, bs 2, ~14 min, ~$0.25): loss 0.27 (step 150) -> 0.139 (step 1000), still falling. Held-out reconstruction PSNR 14.3 dB: dark, patchy, spatial layout roughly aligned (platforms, HUD blobs), characters not recognizable yet. Verdict: pipeline verified end to end (data, loss, checkpoint save/reload, reconstruct). Quality gate moves to 10k steps of the full run (~2 h, ~$1.50): abort if still mush. Checkpoint layout: `<out>/checkpoint-N/checkpoint.pth` (1.7 GB weights) + `training_state.pth` (3 GB). Image: bench/smoke_recon.png. HF quota ~84/100 GB: push only checkpoint.pth to HF, overwrite, every 25%.
- 2026-09-03 03:20 EDT GATE PASSED at step 8k of the 80k codec run (secure 5090 `u7d2dbveq3i5lx`, 0.83 s/step): held-out PSNR 17.2 dB (14.3 at 1k). "Ready"/"Go!" text legible, HUD/stocks/platforms right, characters as blobs, timer digits hallucinated (06:30 vs 08:00). Run continues. Note: checkpoints are every 5% = 4000 steps; watcher fixed (watch2.sh) for pictures at 12k/40k and weight pushes at 24k/40k/60k. Image: bench/recon_8000.png.
- 2026-09-03 11:50 EDT codec run at step 46,400/80,000, loss 0.0627, 0.84 s/step, GPU 99%. Held-out PSNR: 14.3 dB @1k -> 17.2 @8k -> 18.65 @12k -> 25.74 @40k (target was >25, hit at 40k). recon_40000.png: near-identical to source; "Ready"/"Go!"/percent/stocks/platforms/character silhouettes all correct, only the timer's small digits differ (06:00 vs 08:00). Weights pushed to HF at 24k and 40k. ETA ~18:30 EDT.

## WM run decisions (2026-09-03, verified against MIRA on the pod)

- **Batch size 2.** Bench: bs1 0.205, bs2 0.329, bs4 0.625 s/step, bs8 OOM. As samples/sec that is 4.88 / 6.08 / 6.40 — bs1->bs2 gains 25%, **bs2->bs4 gains only 5%** (GPU already saturated; bs4 pays 1.90x the time for 2x the work). Cost is set by TOTAL SAMPLES, not batch size: any batch reaches N samples in the same wall-clock +/-5%. bs2 chosen for OOM margin on an unattended run. MIRA's own default is `batch_size: 1, steps: 250_001`, so bs2 is already above their recipe. Do not re-litigate.
- **Mandatory overrides.** `model=latent_world_model` hardcodes `video.width: 512` (Rocket League). Melee is 384 -> must pass `model.architecture.config.video.width=384` or the run dies. Also `codec_checkpoint` defaults to null; size is picked with `model/latent_world_model@model.architecture.config=200m` (both `1b.yaml` and `200m.yaml` are installed on the pod).
- **Chunk math checks out.** `package.py` cuts 240 frames @ 60 fps = 4 s -> 80 frames @ 20 fps, matching MIRA's shipped chunk format. Training window is 40 frames (2 s); the eval defaults `n_context_frames: 38 + num_unrolled_frames: 20 * 2 = 78 <= 80` fit.
- **DINO note.** Codec needs the GATED DINOv3-L/16 (`RS_DINO_WEIGHTS_DIR`). WM training does not, but `world_model_metrics` loads a DINO/Inception backbone lazily via torch.hub for drift + Frechet curves. Keep network access on the pod.
- **Tier 1 gate risk.** The "10 s rollout" gate is 5x both the 2 s training window and MIRA's own ~2 s eval rollout. It is the most likely gate to fail; the hard-stop clause already points at it.
- **Interactive demo has no upstream tooling.** `examples/explore.py` is a marimo DATASET VIEWER (it overlays *recorded* keyboard actions), not play-in-the-model. `src/mira/inference/rollout.py` is the stepping engine and is written with interactive-server latency in mind, but no keyboard client ships. Budget ~100 lines: key -> 32-key multi-hot -> rollout step -> decode -> display.
- **Cost to finish Tier 1: ~$7.50 of new money** (WM 100k @ bs2 on a COMMUNITY pod $0.69/h = 9.1 h = $6.31; demo pod ~$1; offline eval marginal). ~$4 if the 45k "codec-limited" checkpoint is good enough to stop early. Use community, not secure: no network volume is needed since data comes from HF.
