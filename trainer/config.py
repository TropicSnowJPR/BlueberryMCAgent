"""trainer/config.py – central configuration for the BlueberryMCAgent trainer."""

import random
import torch
import numpy as np

# ── Server ─────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 25566

# ── PPO hyper-parameters ───────────────────────────────────────────────────────
ROLLOUT_STEPS = 64       # collect this many transitions before each PPO update
PPO_EPOCHS = 4           # number of passes over the rollout per update
MINIBATCH_SIZE = 32      # minibatch size during each PPO epoch
GAMMA = 0.99             # discount factor
GAE_LAMBDA = 0.95        # GAE λ
CLIP_EPS = 0.2           # PPO clipping ε
LR = 3e-4                # learning rate
VALUE_COEF = 0.5         # value-loss coefficient
ENTROPY_COEF = 0.01      # entropy bonus coefficient
MAX_GRAD_NORM = 0.5      # gradient clipping

# ── Action space ───────────────────────────────────────────────────────────────
ACTION_NAMES = [
    "noop",        # 0
    "forward",     # 1
    "turn_left",   # 2
    "turn_right",  # 3
    "look_up",     # 4
    "look_down",   # 5
    "attack",      # 6
    "use",         # 7
    "jump",        # 8
]
NUM_ACTIONS = len(ACTION_NAMES)   # must match blueberry_bridge.py

# ── Vision ─────────────────────────────────────────────────────────────────────
IMG_W = 160              # width fed to the CNN (downscaled from full res)
IMG_H = 90               # height fed to the CNN

# ── Checkpointing ──────────────────────────────────────────────────────────────
CHECKPOINT_EVERY_N_UPDATES = 50   # save checkpoint every N PPO updates

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
