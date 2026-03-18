"""trainer/ppo/buffer.py – Rollout buffer with Generalised Advantage Estimation."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Stores transitions collected during a single rollout.

    Parameters
    ----------
    capacity:   Maximum number of transitions (= ROLLOUT_STEPS).
    img_shape:  Shape of the image observation (C, H, W).
    state_dim:  Dimension of the compact state vector.
    gamma:      Discount factor γ.
    gae_lambda: GAE λ.
    device:     PyTorch device.
    """

    def __init__(
        self,
        capacity: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        C, H, W = img_shape
        self.imgs    = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values  = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0
        self._full = False

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        img: np.ndarray,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        i = self._ptr % self.capacity
        self.imgs[i]      = img
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = float(done)
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    @property
    def full(self) -> bool:
        return self._full

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    def clone(self) -> "RolloutBuffer":
        """Return a deep copy of the filled buffer for lock-free PPO updates."""
        buf = RolloutBuffer.__new__(RolloutBuffer)
        buf.capacity = self.capacity
        buf.gamma = self.gamma
        buf.gae_lambda = self.gae_lambda
        buf.device = self.device
        buf.imgs      = self.imgs.copy()
        buf.states    = self.states.copy()
        buf.actions   = self.actions.copy()
        buf.rewards   = self.rewards.copy()
        buf.values    = self.values.copy()
        buf.log_probs = self.log_probs.copy()
        buf.dones     = self.dones.copy()
        buf._ptr  = self._ptr
        buf._full = self._full
        return buf

    # ── Compute returns / advantages ───────────────────────────────────────────

    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value: V(s_{T}) bootstrapped from the model for the step after
                    the last stored transition.

        Returns
        -------
        (advantages, returns)  – both shape (capacity,)
        """
        advantages = np.zeros(self.capacity, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.capacity)):
            # Use current step's done flag: if the episode ended at step t
            # the next state has no value (non-terminal = 0).
            next_non_terminal = 1.0 - self.dones[t]
            # Bootstrap from last_value for the final step, otherwise from
            # the stored value of the very next step.
            next_val = last_value if t == self.capacity - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    # ── Iterate minibatches ────────────────────────────────────────────────────

    def minibatches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        minibatch_size: int,
    ):
        """Yield shuffled minibatches as (imgs, states, actions, log_probs, advantages, returns)."""
        idx = np.random.permutation(self.capacity)
        for start in range(0, self.capacity, minibatch_size):
            mb = idx[start: start + minibatch_size]
            yield (
                torch.tensor(self.imgs[mb],    device=self.device),
                torch.tensor(self.states[mb],  device=self.device),
                torch.tensor(self.actions[mb], device=self.device),
                torch.tensor(self.log_probs[mb], device=self.device),
                torch.tensor(advantages[mb],   device=self.device),
                torch.tensor(returns[mb],      device=self.device),
            )
