"""trainer/ppo/model.py – Actor-Critic neural network for BlueberryMCAgent.

Architecture
------------
* **Vision encoder** – small CNN that processes a downscaled screenshot
  (C=3, H=IMG_H, W=IMG_W) into a 256-d feature vector.
* **State encoder** – MLP that encodes the compact state vector (pos, yaw,
  pitch, health, inventory counts, targeted-block flag) into a 64-d vector.
* **Fusion** – concatenate vision + state features → 320-d.
* **Actor head** – outputs logits over NUM_ACTIONS discrete actions.
* **Critic head** – outputs a scalar value estimate V(s).

When no screenshot is available the vision encoder receives a zero tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from trainer.config import IMG_H, IMG_W, NUM_ACTIONS


class _VisionEncoder(nn.Module):
    """Lightweight CNN for 3×H×W input → 256-d feature."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),  # H/4 × W/4
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),           # H/8 × W/8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size dynamically
        dummy = torch.zeros(1, in_channels, IMG_H, IMG_W)
        flat_size = self.net(dummy).shape[1]
        self.fc = nn.Linear(flat_size, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.net(x)))


class _StateEncoder(nn.Module):
    """Small MLP for compact state vector → 64-d feature."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Combined actor-critic model used by the PPO trainer."""

    # Dimension of the compact state vector (see utils.obs_to_state_vector)
    STATE_DIM = 32

    def __init__(self):
        super().__init__()
        self.vision_enc = _VisionEncoder()
        self.state_enc = _StateEncoder(self.STATE_DIM)

        fusion_dim = 256 + 64  # 320
        self.actor_head = nn.Linear(fusion_dim, NUM_ACTIONS)
        self.critic_head = nn.Linear(fusion_dim, 1)

        # Orthogonal init (common practice for PPO)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head gets smaller gain
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def _fuse(
        self,
        img: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        vis_feat = self.vision_enc(img)
        st_feat = self.state_enc(state)
        return torch.cat([vis_feat, st_feat], dim=-1)

    def forward(
        self,
        img: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, value)."""
        fused = self._fuse(img, state)
        logits = self.actor_head(fused)
        value = self.critic_head(fused).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        img: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[int, float, float]:
        """Sample an action; return (action, log_prob, value)."""
        logits, value = self.forward(img, state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
        )
