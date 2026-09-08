"""Play inside the Melee world model. WebSocket server: keypresses in, dreamed frames out.

One client message = one denoise_streaming call = 1 latent = 2 video frames = 100 ms of game time.
Measured 81 ms/latent at 4 diffusion steps on a 5090, so this runs real time on the pod itself.
Every frame is also written to an mp4 server-side, so the recording is clean at a true 20 fps
regardless of how laggy the network is between the browser and the pod.

  python play_server.py [--steps 4] [--port 8765]
  then tunnel:  ssh -N -L 8765:localhost:8765 root@<pod> -p <port>
  and open play.html
"""
import sys, os, io, json, time, asyncio, argparse, subprocess, tarfile, torch
sys.path.insert(0, "/workspace/pyextra"); sys.path.insert(0, "/workspace/mira/scripts")
from pathlib import Path
import websockets
from PIL import Image
import eval_world_model_offline as E
from mira.inference.loading import load_world_model
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionTensors

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--max-latents", type=int, default=1200, help="KV cache ceiling; re-seeds past it")
ap.add_argument("--out", type=str, default="/workspace/play")
A = ap.parse_args()

CK = Path("/workspace/wm_run/checkpoint-100000/checkpoint.pth"); DEV = torch.device("cuda")
cfg = E.load_run_config(CK)
model, _ = load_world_model(CK, device=DEV); model = model.eval()
TD, ATD = model.temporal_downsampling, model.action_temporal_downsampling
NCTX, WIN = model.n_context_latents, model.n_context_latents + 1
KEYS = list(model.config.actions.valid_keys)
print(f"loaded. window={WIN} latents, {TD} video frames/latent, {len(KEYS)} keys, {A.steps} steps", flush=True)
# The wire format below is header | jpeg0 | jpeg1, and play.html unpacks exactly two. Fail here
# with a readable message rather than inside the first handler with an unpack error.
assert TD == 2, f"wire format sends 2 frames per latent, but this checkpoint decodes {TD}"

_loader = E._build_loader(cfg, model, clip_len=80, batch_size=1, seed=7)
_seed_batch, _ = next(iter(_loader)); _seed_batch = _seed_batch.to(DEV)
model.codec.preprocess_batch(_seed_batch)

os.makedirs(A.out, exist_ok=True)
MP4 = f"{A.out}/session_{time.strftime('%H%M%S')}.mp4"
# rawvideo carries no geometry, so a wrong -s silently skews every frame. Take it from the clip
# the decoder was seeded with rather than hardcoding the training resolution.
_H, _W = _seed_batch.video.shape[-2:]
REC = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{_W}x{_H}", "-r", "20",
     "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", MP4],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Dream:
    """Holds the rolling latent buffer + streaming KV cache for one play session.

    Two known limits, both needing a live pod to change safely:
      * reset() zeroes the whole key_presses buffer, including the NCTX slots that back the
        seeded context latents. Those frames show real in-game motion, so for the first WIN
        steps of a session, and again after every re-seed, the model is told nothing was
        pressed while it looks at motion. _seed_batch.actions.key_presses holds the real ones.
      * REC is one module-level ffmpeg pipe while a Dream is created per connection, so two
        simultaneous clients interleave into one mp4. Single-player is the intended use.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode_video(_seed_batch).clone()
        self.z = torch.randn(1, A.max_latents, *z.shape[2:], device=DEV, dtype=z.dtype)
        self.z[:, :NCTX] = z[:, :NCTX]
        self.kv, self.i = None, 0          # i = index of the window start
        self.acts = ActionTensors(config=model.config.actions, batch_size=1)
        self.acts.key_presses = torch.zeros((1, A.max_latents * ATD + ATD), dtype=torch.int32, device=DEV)[..., None].repeat(1, 1, len(KEYS))
        self.acts.mouse_movements = torch.zeros((1, A.max_latents * ATD + ATD, 2), dtype=torch.float32, device=DEV)
        self.acts.game_mouse_sensitivity = _seed_batch.actions.game_mouse_sensitivity
        print(f"[reset] fresh dream seeded from a real clip", flush=True)

    def step(self, pressed: list[str]) -> torch.Tensor:
        """Advance one latent frame under ``pressed``; returns (TD, 3, H, W) decoded video."""
        if self.i + WIN >= A.max_latents:
            self.reset()   # ponytail: hard re-seed at the KV ceiling; a ring buffer if sessions must be unbounded
        # write this step's action into every action slot the upcoming window will read
        v = torch.zeros(len(KEYS), dtype=torch.int32, device=DEV)
        for k in pressed:
            if k in KEYS:
                v[KEYS.index(k)] = 1
        lo = (self.i + WIN - 1) * ATD
        self.acts.key_presses[:, lo:lo + ATD + 1] = v

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            off = ATD - 1
            a_enc = model.action_encoder(
                self.acts.slice_time(self.i * ATD + off, (self.i + WIN - 1) * ATD + off)).clone()
            self.z[:, self.i:self.i + WIN], self.kv = model.denoise_streaming(
                self.z[:, self.i:self.i + WIN], a_enc, n_diffusion_steps=A.steps,
                noise_level=A.noise, streaming_kv_caches=self.kv, schedule_type="linear")
            new = self.z[:, self.i + WIN - 1: self.i + WIN]
            vid = model.decode_to_video(new).float()[0].clamp(0, 1)
        self.i += 1
        return vid


def to_jpegs(vid: torch.Tensor) -> list[bytes]:
    out = []
    for f in (vid * 255).byte().permute(0, 2, 3, 1).cpu().numpy():
        REC.stdin.write(f.tobytes())
        b = io.BytesIO(); Image.fromarray(f).save(b, "JPEG", quality=80)
        out.append(b.getvalue())
    return out


async def handler(ws):
    dream, n, t0 = Dream(), 0, time.perf_counter()
    print("[client connected]", flush=True)
    try:
        async for msg in ws:
            m = json.loads(msg)
            if m.get("reset"):
                dream.reset(); continue
            t = time.perf_counter()
            vid = dream.step(m.get("keys", []))
            ms = (time.perf_counter() - t) * 1000
            n += 1
            # One frame per step, not three. Three separate sends cost extra round trips over a
            # 146 ms link, which dominated everything: header | jpeg0 | jpeg1 in a single message.
            a, b = to_jpegs(vid)
            hdr = json.dumps({"step": n, "ms": round(ms, 1),
                              "dream_s": round(dream.i * TD / 20, 1)}).encode()
            await ws.send(len(hdr).to_bytes(4, "big") + hdr
                          + len(a).to_bytes(4, "big") + a + b)
            if n % 50 == 0:
                print(f"  step {n}  {ms:.0f} ms/latent  dream t={dream.i*TD/20:.1f}s", flush=True)
    finally:
        print(f"[client gone] {n} steps in {time.perf_counter()-t0:.0f}s", flush=True)


async def main():
    async with websockets.serve(handler, "0.0.0.0", A.port, max_size=None):
        print(f"listening on :{A.port}  recording -> {MP4}", flush=True)
        await asyncio.Future()

try:
    asyncio.run(main())
finally:
    REC.stdin.close(); REC.wait()
    print(f"recording written: {MP4}", flush=True)
