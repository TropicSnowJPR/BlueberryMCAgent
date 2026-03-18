"""trainer/server.py – Flask HTTP server + online PPO training loop.

Start from the project root:
    python trainer/server.py

Endpoints
---------
POST /step   – receive observation JSON, return action JSON
GET  /status – human-readable training summary
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request

# ── Path setup so `trainer.*` imports work when run as a script ───────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import (
    CHECKPOINT_EVERY_N_UPDATES,
    HOST,
    PORT,
    ROLLOUT_STEPS,
    SEED,
    get_device,
    set_seed,
)
from trainer.ppo import ActorCritic, RolloutBuffer, ppo_update
from trainer.ppo.utils import decode_screenshot, obs_to_state_vector, save_screenshot
from trainer.reward import compute_reward

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("blueberry")

# ── Run directory ─────────────────────────────────────────────────────────────
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs") / RUN_ID
SCREENS_DIR = RUN_DIR / "screens"
STEPS_LOG = RUN_DIR / "steps.jsonl"
CKPT_DIR = RUN_DIR / "checkpoints"

for d in (SCREENS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Initialise model ──────────────────────────────────────────────────────────
set_seed(SEED)
DEVICE = get_device()
log.info("Using device: %s", DEVICE)

model = ActorCritic().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

IMG_SHAPE = (3, 90, 160)   # C, H, W  (matches config.IMG_H / IMG_W)

buffer = RolloutBuffer(
    capacity=ROLLOUT_STEPS,
    img_shape=IMG_SHAPE,
    state_dim=ActorCritic.STATE_DIM,
    device=DEVICE,
)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()

_state = {
    "prev_obs": None,
    "prev_img": np.zeros(IMG_SHAPE, dtype=np.float32),
    "prev_state_vec": np.zeros(ActorCritic.STATE_DIM, dtype=np.float32),
    "prev_action": 0,
    "prev_log_prob": 0.0,
    "prev_value": 0.0,
    "milestones": {},
    "total_steps": 0,
    "total_updates": 0,
    "episode_reward": 0.0,
    "total_reward": 0.0,
    "last_stats": {},
}

# ── Background training thread ────────────────────────────────────────────────
_train_event = threading.Event()


def _training_worker():
    """Wait for the buffer to fill, then run a PPO update."""
    while True:
        _train_event.wait()
        _train_event.clear()
        with _lock:
            if not buffer.full:
                continue
            # Bootstrap value for the last state
            with torch.no_grad():
                img_t = torch.tensor(
                    _state["prev_img"][None], device=DEVICE
                )
                st_t = torch.tensor(
                    _state["prev_state_vec"][None], device=DEVICE
                )
                _, last_val = model(img_t, st_t)
            last_value = float(last_val.item())
            stats = ppo_update(model, optimizer, buffer, last_value)
            buffer.reset()
            _state["total_updates"] += 1
            _state["last_stats"] = stats

            n = _state["total_updates"]
            log.info(
                "PPO update #%d | pl=%.4f vl=%.4f ent=%.4f",
                n, stats.get("policy_loss", 0), stats.get("value_loss", 0),
                stats.get("entropy", 0),
            )

            if n % CHECKPOINT_EVERY_N_UPDATES == 0:
                ckpt_path = CKPT_DIR / f"ckpt_{n:06d}.pt"
                torch.save(
                    {
                        "update": n,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                log.info("Checkpoint saved → %s", ckpt_path)


_train_thread = threading.Thread(target=_training_worker, daemon=True)
_train_thread.start()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.post("/step")
def step():
    """Receive an observation, log transition, maybe trigger PPO, return action."""
    obs = request.get_json(force=True, silent=True)
    if obs is None:
        return jsonify({"action": 0, "error": "invalid JSON"}), 400

    with _lock:
        # ── Decode inputs ──────────────────────────────────────────────────────
        b64 = obs.pop("screenshot_b64", None)
        img = decode_screenshot(b64)
        state_vec = obs_to_state_vector(obs)

        # ── Save screenshot to disk ────────────────────────────────────────────
        if b64:
            step_n = _state["total_steps"]
            screen_path = SCREENS_DIR / f"step_{step_n:08d}.png"
            threading.Thread(
                target=save_screenshot, args=(b64, str(screen_path)), daemon=True
            ).start()

        # ── Select action ──────────────────────────────────────────────────────
        img_t = torch.tensor(img[None], device=DEVICE)
        st_t = torch.tensor(state_vec[None], device=DEVICE)
        action, log_prob, value = model.act(img_t, st_t)

        # ── Compute reward from previous → current obs transition ─────────────
        prev_obs = _state["prev_obs"]
        reward = 0.0
        if prev_obs is not None:
            reward, _state["milestones"] = compute_reward(
                prev_obs, obs, _state["milestones"]
            )

        _state["episode_reward"] += reward
        _state["total_reward"] += reward

        # ── Store transition to rollout buffer ─────────────────────────────────
        if prev_obs is not None:
            buffer.add(
                img=_state["prev_img"],
                state=_state["prev_state_vec"],
                action=_state["prev_action"],
                reward=reward,
                value=_state["prev_value"],
                log_prob=_state["prev_log_prob"],
                done=False,
            )
            if buffer.full:
                _train_event.set()

        # ── Log transition to JSONL ────────────────────────────────────────────
        record = {
            "step": _state["total_steps"],
            "t": time.time(),
            "action": action,
            "reward": reward,
            "obs": {
                "pos": obs.get("pos"),
                "yaw": obs.get("yaw"),
                "pitch": obs.get("pitch"),
                "health": obs.get("health"),
                "inventory": obs.get("inventory"),
                "targeted_block": obs.get("targeted_block"),
            },
        }
        try:
            with open(STEPS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

        # ── Advance state ──────────────────────────────────────────────────────
        _state["prev_obs"] = obs
        _state["prev_img"] = img
        _state["prev_state_vec"] = state_vec
        _state["prev_action"] = action
        _state["prev_log_prob"] = log_prob
        _state["prev_value"] = value
        _state["total_steps"] += 1

    return jsonify({"action": action})


@app.get("/status")
def status():
    """Return a human-readable JSON status summary."""
    with _lock:
        return jsonify(
            {
                "run_id": RUN_ID,
                "device": str(DEVICE),
                "total_steps": _state["total_steps"],
                "total_updates": _state["total_updates"],
                "episode_reward": round(_state["episode_reward"], 3),
                "total_reward": round(_state["total_reward"], 3),
                "buffer_size": min(buffer._ptr, ROLLOUT_STEPS),
                "milestones": _state["milestones"],
                "last_ppo_stats": _state["last_stats"],
            }
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("BlueberryMCAgent trainer starting on http://%s:%d", HOST, PORT)
    log.info("Run directory: %s", RUN_DIR.resolve())
    # threaded=True lets Flask handle each HTTP request in its own thread so
    # the training daemon thread is never blocked by a slow /step handler.
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
