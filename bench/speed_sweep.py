"""Latency sweep: ms per latent frame vs n_diffusion_steps. 1 latent = 2 video frames = 100ms @20fps."""
import sys, json, torch
sys.path.insert(0, "/workspace/pyextra")
from pathlib import Path
from mira.inference.loading import load_world_model
from mira.inference.rollout import measure_rollout_speed
from mira.world_model.config import WorldModelInferenceConfig
sys.argv = ["x", "/workspace/wm_run/checkpoint-100000/checkpoint.pth"]
sys.path.insert(0, "/workspace/mira/scripts")
import eval_world_model_offline as E

CK = Path("/workspace/wm_run/checkpoint-100000/checkpoint.pth")
dev = torch.device("cuda")
cfg = E.load_run_config(CK)
model, _ = load_world_model(CK, device=dev); model = model.eval()
# chunks are 80 frames; clip_len = n_context_frames + N*temporal_downsampling must fit
N = (80 - model.config.n_context_frames) // model.temporal_downsampling
print(f"n_context_frames={model.config.n_context_frames} td={model.temporal_downsampling} bench_frames={N}", flush=True)
loader = E._build_loader(cfg, model, clip_len=model.config.n_context_frames + N*model.temporal_downsampling, batch_size=1, seed=39)
batch, _ = next(iter(loader)); batch = batch.to(dev); model.codec.preprocess_batch(batch)

out = {}
for steps in [1, 2, 4, 8, 10, 16]:
    c = WorldModelInferenceConfig(n_diffusion_steps=steps, noise_level=0.0, schedule_type="linear")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        measure_rollout_speed(model, batch, c, n_frames=4)   # warmup
        r = measure_rollout_speed(model, batch, c, n_frames=N)
    ms = r["denoise_ms_per_latent_frame"]
    out[steps] = ms
    print(f"steps={steps:2d}  {ms:7.1f} ms/latent  ({100/ms:5.2f}x realtime budget)  wall_fps={1000/ms:5.2f}", flush=True)
json.dump(out, open("/workspace/speed_sweep.json","w"), indent=2)
