"""Convert a Slippi .slp replay into MIRA-shaped arrays.

Output .npz:
  actions  (P, T, K) int32   multi-hot, K = len(ACTION_KEYS)
  state    (P, T, S) float32 columns = STATE_COLS
  frame_id (T,)      int32   Slippi frame index (game starts at 0, countdown is negative)
  meta     json str          characters, stage, ports

Usage: uv run python data/parse.py IN.slp [OUT.npz]
"""
import json
import sys
from pathlib import Path

import numpy as np
import peppi_py

# Physical GameCube buttons, Slippi bit layout.
BUTTON_BITS = {"A": 1 << 8, "B": 1 << 9, "X": 1 << 10, "Y": 1 << 11, "Z": 1 << 4, "L": 1 << 6, "R": 1 << 5}
STICK_DEADZONE = 0.2875  # Melee's own threshold
AXES = ["joy_x", "joy_y", "c_x", "c_y"]
ACTION_KEYS = list(BUTTON_BITS) + [f"{a}_{lvl}" for a in AXES for lvl in ("neg", "pos")] + ["trigger"]
STATE_COLS = ["percent", "stocks", "x", "y", "direction", "airborne", "action_state", "l_cancel"]


def _axis_bits(arr):
    """3-level stick: returns (neg, pos) bool columns. Neutral = both 0."""
    return (arr < -STICK_DEADZONE).astype(np.int32), (arr > STICK_DEADZONE).astype(np.int32)


def parse(path: str):
    g = peppi_py.read_slippi(path)
    ports = g.frames.ports
    frame_id = g.frames.id.to_numpy().astype(np.int32)
    T = len(frame_id)
    actions, state = [], []
    for p in ports:
        pre, post = p.leader.pre, p.leader.post
        buttons = pre.buttons_physical.to_numpy().astype(np.uint32)
        cols = [((buttons & bit) > 0).astype(np.int32) for bit in BUTTON_BITS.values()]
        for axis in (pre.joystick.x, pre.joystick.y, pre.cstick.x, pre.cstick.y):
            cols.extend(_axis_bits(axis.to_numpy()))
        cols.append((pre.triggers.to_numpy() > 0.3).astype(np.int32))  # analog shield press
        actions.append(np.stack(cols, axis=1))
        state.append(
            np.stack(
                [
                    post.percent.to_numpy(),
                    post.stocks.to_numpy(),
                    post.position.x.to_numpy(),
                    post.position.y.to_numpy(),
                    post.direction.to_numpy(),
                    post.airborne.to_numpy(),
                    post.state.to_numpy(),
                    post.l_cancel.to_numpy(),
                ],
                axis=1,
            ).astype(np.float32)
        )
    meta = {
        "stage": g.start.stage,
        "players": [{"port": int(pl.port), "character": pl.character} for pl in g.start.players],
        "n_frames": T,
    }
    return np.stack(actions), np.stack(state), frame_id, meta


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(src).with_suffix(".npz"))
    actions, state, frame_id, meta = parse(src)
    np.savez_compressed(out, actions=actions, state=state, frame_id=frame_id, meta=json.dumps(meta))
    P, T, K = actions.shape
    print(f"{Path(src).name}: P={P} T={T} K={K} S={state.shape[2]} -> {out}")


if __name__ == "__main__":
    main()
