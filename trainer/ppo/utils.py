"""trainer/ppo/utils.py – Utility helpers shared across the trainer."""

from __future__ import annotations

import base64
import io
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from trainer.config import IMG_H, IMG_W
from trainer.ppo.model import ActorCritic

# Ordered list of inventory items the state vector tracks.
# Add more items here to expand the state representation.
TRACKED_ITEMS = [
    "oak_log", "birch_log", "spruce_log",
    "oak_planks", "birch_planks", "spruce_planks",
    "crafting_table",
    "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "cobblestone",
    "apple", "bread", "carrot", "potato",
    "cooked_beef", "cooked_porkchop", "cooked_chicken",
]


def obs_to_state_vector(obs: dict) -> np.ndarray:
    """Convert an observation dict to a fixed-length float32 state vector.

    Layout (STATE_DIM = 32):
      [0]   pos_x  (normalised ÷1000)
      [1]   pos_y  (normalised ÷256)
      [2]   pos_z  (normalised ÷1000)
      [3]   yaw    (÷180)
      [4]   pitch  (÷90)
      [5]   health (÷20)
      [6]   targeted_block flag (1.0 if something targeted)
      [7..24]  inventory counts for TRACKED_ITEMS (÷64, capped at 1.0)
      [25..31] padding zeros
    """
    vec = np.zeros(ActorCritic.STATE_DIM, dtype=np.float32)

    pos = obs.get("pos", [0.0, 64.0, 0.0])
    vec[0] = float(pos[0]) / 1000.0
    vec[1] = float(pos[1]) / 256.0
    vec[2] = float(pos[2]) / 1000.0
    vec[3] = float(obs.get("yaw", 0.0)) / 180.0
    vec[4] = float(obs.get("pitch", 0.0)) / 90.0
    vec[5] = float(obs.get("health", 20.0)) / 20.0
    vec[6] = 1.0 if obs.get("targeted_block") is not None else 0.0

    inv = obs.get("inventory", {})
    for i, item in enumerate(TRACKED_ITEMS):
        if 7 + i < ActorCritic.STATE_DIM:
            vec[7 + i] = min(inv.get(item, 0) / 64.0, 1.0)

    return vec


def decode_screenshot(b64_png: str | None) -> np.ndarray:
    """Decode a base-64 PNG string to a float32 (C, H, W) array in [0, 1].

    Returns a black image tensor of shape (3, IMG_H, IMG_W) if decoding fails.
    """
    black = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
    if not b64_png or not PIL_AVAILABLE:
        return black
    try:
        raw = base64.b64decode(b64_png)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, C)
        return arr.transpose(2, 0, 1)                    # (C, H, W)
    except Exception:
        return black


def save_screenshot(b64_png: str, path: str) -> None:
    """Save a base-64 PNG string to *path* on disk."""
    if not PIL_AVAILABLE:
        return
    try:
        raw = base64.b64decode(b64_png)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(path, format="PNG")
    except Exception:
        pass
