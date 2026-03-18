"""trainer/ppo/update.py – PPO clipped-objective policy update."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from trainer.ppo.model import ActorCritic
from trainer.config import (
    PPO_EPOCHS,
    MINIBATCH_SIZE,
    CLIP_EPS,
    VALUE_COEF,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
)


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer,
    last_value: float,
) -> dict[str, float]:
    """Run one round of PPO updates over the filled rollout buffer.

    Returns a dict with mean training statistics for logging.
    """
    advantages, returns = buffer.compute_gae(last_value)

    # Normalise advantages across the rollout
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    stats: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
    }

    for _ in range(PPO_EPOCHS):
        for imgs, states, actions, old_log_probs, advs, rets in buffer.minibatches(
            advantages, returns, MINIBATCH_SIZE
        ):
            logits, values = model(imgs, states)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, rets)

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["entropy"].append(float(entropy.item()))
            stats["total_loss"].append(float(loss.item()))

    return {k: float(sum(v) / len(v)) for k, v in stats.items() if v}
