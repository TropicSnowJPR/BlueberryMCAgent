"""trainer/bc_train.py – Offline behaviour-cloning trainer for BlueberryMCAgent.

Loads demonstration data recorded by ``minescript/blueberry_record.py`` and
trains the ActorCritic policy head with supervised cross-entropy loss
(behaviour cloning).

Usage
-----
    # Train on a single demo run:
    python trainer/bc_train.py --demo demos/20240415_100000

    # Train on multiple demo runs, more epochs, GPU:
    python trainer/bc_train.py \\
        --demo demos/run1 demos/run2 \\
        --epochs 20 \\
        --batch_size 64 \\
        --lr 3e-4 \\
        --device cuda

    # Then warm-start online PPO from the resulting checkpoint:
    python trainer/server.py --init_from checkpoints/bc_latest.pt

Output
------
Checkpoints are saved to:
    checkpoints/bc_<timestamp>.pt   – final checkpoint for this run
    checkpoints/bc_latest.pt        – symlink / copy of the most recent run

Checkpoint format
-----------------
    {
        "epoch": <int>,
        "model_state": <state_dict>,
        "optimizer_state": <state_dict>,
        "train_loss": <float>,
        "num_samples": <int>,
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.actions import NUM_ACTIONS
from trainer.config import IMG_H, IMG_W, get_device, set_seed, SEED
from trainer.ppo import ActorCritic
from trainer.ppo.utils import decode_screenshot, obs_to_state_vector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bc_train")


# ── Dataset ───────────────────────────────────────────────────────────────────

class DemoDataset(Dataset):
    """Dataset built from one or more demo JSONL files.

    Each sample is a ``(img, state_vec, action_id)`` tuple where:
    * ``img``       – float32 tensor of shape (3, IMG_H, IMG_W), or zeros if no screenshot.
    * ``state_vec`` – float32 tensor of shape (STATE_DIM,).
    * ``action_id`` – int64 scalar in [0, NUM_ACTIONS).
    """

    def __init__(self, demo_paths: list[Path]):
        self._samples: list[dict] = []

        for p in demo_paths:
            if p.is_dir():
                jsonl_files = list(p.glob("steps*.jsonl"))
                if not jsonl_files:
                    log.warning("No steps*.jsonl found in %s – skipping.", p)
                    continue
                for jf in sorted(jsonl_files):
                    self._load_jsonl(jf)
            else:
                self._load_jsonl(p)

        log.info(
            "DemoDataset: loaded %d labelled samples from %d path(s).",
            len(self._samples),
            len(demo_paths),
        )

    def _load_jsonl(self, path: Path) -> None:
        skipped = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                action = rec.get("action", -1)
                if not isinstance(action, int) or not (0 <= action < NUM_ACTIONS):
                    skipped += 1
                    continue
                self._samples.append(rec)
        if skipped:
            log.debug("Skipped %d unlabelled/invalid lines in %s.", skipped, path)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        rec = self._samples[idx]
        obs = rec.get("obs", {})

        # Screenshot: load from path if available, else fall back to b64 in obs
        img = None
        screen_path = rec.get("screenshot_path")
        if screen_path:
            try:
                with open(screen_path, "rb") as f:
                    import base64
                    b64 = base64.b64encode(f.read()).decode("ascii")
                img = decode_screenshot(b64)
            except Exception:
                img = None
        if img is None:
            b64 = obs.get("screenshot_b64")
            img = decode_screenshot(b64)  # returns zeros on None

        state_vec = obs_to_state_vector(obs)
        action = int(rec["action"])

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(state_vec, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
        )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model: ActorCritic,
    dataset: DemoDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    ckpt_dir: Path,
) -> None:
    if len(dataset) == 0:
        log.error("Dataset is empty – nothing to train on.  Check demo paths.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    log.info(
        "Starting BC training | samples=%d epochs=%d batch=%d lr=%g device=%s",
        len(dataset), epochs, batch_size, lr, device,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        for imgs, states, actions in loader:
            imgs    = imgs.to(device)
            states  = states.to(device)
            actions = actions.to(device)

            logits, _ = model(imgs, states)
            loss = criterion(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(actions)
            correct    += (logits.argmax(dim=-1) == actions).sum().item()
            n          += len(actions)

        avg_loss = total_loss / max(n, 1)
        accuracy = correct / max(n, 1)
        log.info(
            "Epoch %d/%d | loss=%.4f  acc=%.3f  (%d samples)",
            epoch, epochs, avg_loss, accuracy, n,
        )

    # ── Save final checkpoint ─────────────────────────────────────────────────
    ckpt = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": avg_loss,
        "num_samples": len(dataset),
    }

    named_path = ckpt_dir / f"bc_{ts}.pt"
    torch.save(ckpt, named_path)
    log.info("Checkpoint saved → %s", named_path.resolve())

    # Also save / overwrite bc_latest.pt for easy use with --init_from
    latest_path = ckpt_dir / "bc_latest.pt"
    torch.save(ckpt, latest_path)
    log.info("Latest checkpoint → %s", latest_path.resolve())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behaviour-cloning offline trainer for BlueberryMCAgent."
    )
    parser.add_argument(
        "--demo",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more demo directories (containing steps.jsonl) or direct "
            "paths to steps*.jsonl files.  Multiple paths can be given to train "
            "on several demo runs at once."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "PyTorch device string, e.g. 'cuda', 'cpu'.  "
            "Defaults to CUDA if available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints",
        metavar="DIR",
        help="Directory to save checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        metavar="SEED",
        help=f"Random seed (default: {SEED}).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    log.info("Using device: %s", device)

    demo_paths = [Path(p) for p in args.demo]
    for p in demo_paths:
        if not p.exists():
            parser.error(f"Demo path not found: {p}")

    dataset = DemoDataset(demo_paths)
    model = ActorCritic().to(device)

    train(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        ckpt_dir=Path(args.ckpt_dir),
    )


if __name__ == "__main__":
    main()
