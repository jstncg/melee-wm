# Design notes

How the Melee data is shaped for MIRA, and the decisions behind the training runs. Results live
in [RESULTS.md](RESULTS.md).

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

## Training stages
codec -> WM (one perspective, 32-key both-players actions) -> eval (FID/horizon, action recoverability, counterfactual)

NO 2-player warm-start stage. Verified 2026-09-03: MIRA's `n_players` counts *camera
perspectives*, not players. `MultiWrapperWorldModel` tiles n clips into one vertically
stacked frame (`wm_config.video.height *= n_players`). That is Rocket League (4 players,
4 cameras). Melee is ONE shared camera, so `dataset/melee.yaml` correctly sets
`n_players: 1` and `actions/melee.yaml` puts both players' inputs in a 32-key vocab.
The single-perspective `LatentWorldModel` conditioned on those 32 keys IS the 2-player
Melee world model. Do NOT use `model=multi_wrapper_world_model`: it would tile two
copies of the same screen and double the compute for nothing.

## World model run decisions

Verified against MIRA on the training machine.

- **Batch size 2.** Bench: bs1 0.205, bs2 0.329, bs4 0.625 s/step, bs8 OOM. As samples/sec that is 4.88 / 6.08 / 6.40. bs1->bs2 gains 25%, **bs2->bs4 gains only 5%** (GPU already saturated; bs4 pays 1.90x the time for 2x the work). Cost is set by TOTAL SAMPLES, not batch size: any batch reaches N samples in the same wall-clock +/-5%. bs2 chosen for OOM margin on an unattended run. MIRA's own default is `batch_size: 1, steps: 250_001`, so bs2 is already above their recipe. Do not re-litigate.
- **Mandatory overrides.** `model=latent_world_model` hardcodes `video.width: 512` (Rocket League). Melee is 384 -> must pass `model.architecture.config.video.width=384` or the run dies. Also `codec_checkpoint` defaults to null; size is picked with `model/latent_world_model@model.architecture.config=200m` (both `1b.yaml` and `200m.yaml` are installed on the pod).
- **Chunk math checks out.** `package.py` cuts 240 frames @ 60 fps = 4 s -> 80 frames @ 20 fps, matching MIRA's shipped chunk format. Training window is 40 frames (2 s); the eval defaults `n_context_frames: 38 + num_unrolled_frames: 20 * 2 = 78 <= 80` fit.
- **DINO note.** Codec needs the GATED DINOv3-L/16 (`RS_DINO_WEIGHTS_DIR`). WM training does not, but `world_model_metrics` loads a DINO/Inception backbone lazily via torch.hub for drift + Frechet curves. Keep network access on the pod.
- **Interactive demo has no upstream tooling.** `examples/explore.py` is a marimo DATASET VIEWER (it overlays *recorded* keyboard actions), not play-in-the-model. `src/mira/inference/rollout.py` is the stepping engine and is written with interactive-server latency in mind, but no keyboard client ships. Budget ~100 lines: key -> 32-key multi-hot -> rollout step -> decode -> display.
