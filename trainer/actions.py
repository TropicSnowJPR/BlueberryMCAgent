"""trainer/actions.py – Shared action-space definitions for BlueberryMCAgent.

This module is the **single source of truth** for the discrete action space.
The MineScript scripts (blueberry_bridge.py, blueberry_record.py) mirror the
constants defined here.  Any change here must be reflected in those files.

Discrete action IDs
-------------------
0   noop
1   forward_250ms
2   forward_1000ms
3   forward_2000ms
4   turn_left_small
5   turn_right_small
6   look_up_small
7   look_down_small
8   jump_250ms
9   use_250ms
10  attack_250ms
11  attack_hold_1000ms
12  attack_hold_2000ms
13  attack_hold_4000ms
"""

from __future__ import annotations

# ── Action ID constants ────────────────────────────────────────────────────────
ACTION_NOOP         = 0
ACTION_FORWARD_250  = 1
ACTION_FORWARD_1000 = 2
ACTION_FORWARD_2000 = 3
ACTION_TURN_LEFT    = 4
ACTION_TURN_RIGHT   = 5
ACTION_LOOK_UP      = 6
ACTION_LOOK_DOWN    = 7
ACTION_JUMP_250     = 8
ACTION_USE_250      = 9
ACTION_ATTACK_250   = 10
ACTION_ATTACK_1000  = 11
ACTION_ATTACK_2000  = 12
ACTION_ATTACK_4000  = 13

NUM_ACTIONS: int = 14

# Human-readable names (index = action ID)
ACTION_NAMES: list[str] = [
    "noop",               # 0
    "forward_250ms",      # 1
    "forward_1000ms",     # 2
    "forward_2000ms",     # 3
    "turn_left_small",    # 4
    "turn_right_small",   # 5
    "look_up_small",      # 6
    "look_down_small",    # 7
    "jump_250ms",         # 8
    "use_250ms",          # 9
    "attack_250ms",       # 10
    "attack_hold_1000ms", # 11
    "attack_hold_2000ms", # 12
    "attack_hold_4000ms", # 13
]

# How long (seconds) to hold the relevant key for each action.
# 0.0 = instant / no hold (orientation changes, noop).
ACTION_HOLD_SECONDS: list[float] = [
    0.0,   # noop
    0.25,  # forward_250ms
    1.0,   # forward_1000ms
    2.0,   # forward_2000ms
    0.0,   # turn_left_small  (instant orientation change)
    0.0,   # turn_right_small
    0.0,   # look_up_small
    0.0,   # look_down_small
    0.25,  # jump_250ms
    0.25,  # use_250ms
    0.25,  # attack_250ms
    1.0,   # attack_hold_1000ms
    2.0,   # attack_hold_2000ms
    4.0,   # attack_hold_4000ms
]

# Orientation deltas (degrees) used by the bridge / recorder
YAW_DELTA: float = 5.0
PITCH_DELTA: float = 4.0

# ── Hold bucketing ─────────────────────────────────────────────────────────────
# Ordered LARGEST → SMALLEST so we "round up" (assign to the smallest bucket
# whose threshold the observed hold duration meets or exceeds).
_FORWARD_HOLD_BUCKETS: tuple[float, ...] = (2.0, 1.0)
_ATTACK_HOLD_BUCKETS: tuple[float, ...] = (4.0, 2.0, 1.0)


def bucket_hold(duration_s: float, buckets: tuple[float, ...]) -> int:
    """Return the first bucket index where ``duration_s >= buckets[index]``.

    Buckets must be ordered **largest first**.  Examples with
    ``buckets = (4.0, 2.0, 1.0)``:

    * duration_s = 5.0  → 0   (≥ 4.0)
    * duration_s = 2.0  → 1   (≥ 2.0, not ≥ 4.0)
    * duration_s = 0.8  → 3   (below all buckets → shortest)

    This implements "round up": a hold of 0.9 s maps to the 1.0 s bucket.

    Parameters
    ----------
    duration_s:
        Observed continuous hold duration in seconds.
    buckets:
        Tuple of thresholds in descending order.

    Returns
    -------
    Integer index in ``[0, len(buckets)]``.
    """
    for i, threshold in enumerate(buckets):
        if duration_s >= threshold:
            return i
    return len(buckets)


def forward_action(hold_duration_s: float) -> int:
    """Return the forward action ID for a given hold duration (seconds).

    * ``hold_duration_s >= 2.0`` → :data:`ACTION_FORWARD_2000`
    * ``hold_duration_s >= 1.0`` → :data:`ACTION_FORWARD_1000`
    * otherwise                  → :data:`ACTION_FORWARD_250`
    """
    idx = bucket_hold(hold_duration_s, _FORWARD_HOLD_BUCKETS)
    # idx=0 → ACTION_FORWARD_2000 (3), idx=1 → 2, idx=2 → 1
    return ACTION_FORWARD_2000 - idx


def attack_action(hold_duration_s: float) -> int:
    """Return the attack action ID for a given hold duration (seconds).

    * ``hold_duration_s >= 4.0`` → :data:`ACTION_ATTACK_4000`
    * ``hold_duration_s >= 2.0`` → :data:`ACTION_ATTACK_2000`
    * ``hold_duration_s >= 1.0`` → :data:`ACTION_ATTACK_1000`
    * otherwise                  → :data:`ACTION_ATTACK_250`
    """
    idx = bucket_hold(hold_duration_s, _ATTACK_HOLD_BUCKETS)
    # idx=0 → ACTION_ATTACK_4000 (13), idx=1 → 12, idx=2 → 11, idx=3 → 10
    return ACTION_ATTACK_4000 - idx


# ── Key → action helper ────────────────────────────────────────────────────────
# Minimum mouse delta (degrees) before a turn/look action is triggered.
_TURN_THRESHOLD: float = 2.5
_LOOK_THRESHOLD: float = 2.0


def keys_to_action(
    *,
    key_forward: bool = False,
    key_attack: bool = False,
    key_use: bool = False,
    key_jump: bool = False,
    yaw_delta: float = 0.0,
    pitch_delta: float = 0.0,
    forward_hold_s: float = 0.0,
    attack_hold_s: float = 0.0,
) -> int:
    """Convert instantaneous key/mouse state to a single discrete action ID.

    Priority (highest first):
    1. Attack
    2. Use
    3. Jump
    4. Forward movement
    5. Orientation change (turn/look)
    6. Noop

    Parameters
    ----------
    key_forward:
        W key (or equivalent) is currently pressed.
    key_attack:
        Left mouse button is currently pressed.
    key_use:
        Right mouse button is currently pressed.
    key_jump:
        Space bar is currently pressed.
    yaw_delta:
        Mouse horizontal delta since last step (degrees; positive = right).
    pitch_delta:
        Mouse vertical delta since last step (degrees; positive = down).
    forward_hold_s:
        Cumulative seconds the forward key has been held continuously.
    attack_hold_s:
        Cumulative seconds the attack (left-click) key has been held.

    Returns
    -------
    Discrete action ID in ``[0, NUM_ACTIONS)``.
    """
    if key_attack:
        return attack_action(attack_hold_s)
    if key_use:
        return ACTION_USE_250
    if key_jump:
        return ACTION_JUMP_250
    if key_forward:
        return forward_action(forward_hold_s)
    if abs(yaw_delta) >= _TURN_THRESHOLD and abs(yaw_delta) >= abs(pitch_delta):
        return ACTION_TURN_LEFT if yaw_delta < 0 else ACTION_TURN_RIGHT
    if abs(pitch_delta) >= _LOOK_THRESHOLD:
        return ACTION_LOOK_UP if pitch_delta < 0 else ACTION_LOOK_DOWN
    return ACTION_NOOP
