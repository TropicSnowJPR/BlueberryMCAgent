"""
blueberry_bridge.py – MineScript 5.0b11 (Fabric 1.21.11) bridge for BlueberryMCAgent.

Run in-game:  \\blueberry_bridge
Stop:         \\killjob

Every DT seconds the script:
  1. Reads player state (pos, yaw, pitch, health, inventory, targeted block).
  2. Optionally captures a screenshot (every SCREENSHOT_EVERY_N steps).
  3. POSTs an observation JSON (with optional base-64 PNG) to the trainer server.
  4. Receives a discrete action index and executes it.
  5. Holds keys for the action-specific duration, then releases them.

If the trainer is unreachable the agent defaults to action 0 (noop) and keeps running.

Action space (mirrors trainer/actions.py)
-----------------------------------------
0  noop            4  turn_left_small    8  jump_250ms
1  forward_250ms   5  turn_right_small   9  use_250ms
2  forward_1000ms  6  look_up_small     10  attack_250ms
3  forward_2000ms  7  look_down_small   11  attack_hold_1000ms
                                        12  attack_hold_2000ms
                                        13  attack_hold_4000ms
"""

# ── Configuration ──────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 25566
DT = 0.25                 # step duration in seconds (~4 steps/s)
SCREENSHOT_EVERY_N = 3    # take a screenshot once every N steps (0 = never)
YAW_DELTA = 5.0           # degrees per turn-left / turn-right action
PITCH_DELTA = 4.0         # degrees per look-up / look-down action
# ───────────────────────────────────────────────────────────────────────────────

import time
import json
import base64
import io
import urllib.request
import urllib.error

import minescript  # type: ignore – provided by the MineScript mod

# ── Action space (mirrored from trainer/actions.py) ────────────────────────────
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
NUM_ACTIONS         = 14

ACTION_NAMES = [
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

# Hold duration (seconds) for each action.  0.0 = instant (no hold).
_ACTION_HOLD_S = [
    0.0,   # noop
    0.25,  # forward_250ms
    1.0,   # forward_1000ms
    2.0,   # forward_2000ms
    0.0,   # turn_left_small
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

# ── Helpers ────────────────────────────────────────────────────────────────────

def _release_all():
    """Release every control key the agent may have pressed."""
    for fn in (
        minescript.player_press_forward,
        minescript.player_press_back,
        minescript.player_press_left,
        minescript.player_press_right,
        minescript.player_press_jump,
        minescript.player_press_sprint,
        minescript.player_press_sneak,
        minescript.player_press_attack,
        minescript.player_press_use,
        minescript.player_press_pick_item,
    ):
        try:
            fn(False)
        except Exception:
            pass


def _get_inventory() -> dict:
    """Return a dict of {item_id: count} for the player's inventory."""
    inv: dict = {}
    try:
        items = minescript.player_inventory()
        for slot in items:
            if slot is None:
                continue
            item_id = getattr(slot, "item_id", None) or getattr(slot, "type", None) or str(slot)
            count = int(getattr(slot, "count", 1))
            if item_id:
                inv[item_id] = inv.get(item_id, 0) + count
    except Exception:
        pass
    return inv


def _get_targeted_block() -> dict | None:
    """Return info about the block the player is looking at, or None."""
    try:
        tb = minescript.targeted_block(6.0)
        if tb is None:
            return None
        return {
            "type": str(getattr(tb, "block_type", "unknown")),
            "pos": list(getattr(tb, "pos", [0, 0, 0])),
            "distance": float(getattr(tb, "distance", 0.0)),
        }
    except Exception:
        return None


def _capture_screenshot_b64() -> str | None:
    """Take a screenshot and return it as a base-64-encoded PNG string."""
    try:
        # minescript.screenshot() returns a PIL Image or a file path depending on version.
        result = minescript.screenshot()
        if result is None:
            return None
        # If it's a PIL Image:
        if hasattr(result, "tobytes"):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        # If it's a file path string:
        with open(str(result), "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _build_observation(step: int) -> dict:
    """Collect all observable state and package it into a dict."""
    obs: dict = {"step": step, "t": time.time()}

    # Player position & orientation
    try:
        pos = minescript.player_position()
        obs["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        obs["pos"] = [0.0, 64.0, 0.0]

    try:
        yaw, pitch = minescript.player_orientation()
        obs["yaw"] = float(yaw)
        obs["pitch"] = float(pitch)
    except Exception:
        obs["yaw"] = 0.0
        obs["pitch"] = 0.0

    # Health
    try:
        obs["health"] = float(minescript.player_health())
    except Exception:
        obs["health"] = 20.0

    obs["inventory"] = _get_inventory()
    obs["targeted_block"] = _get_targeted_block()

    # Screenshot (base-64 PNG), only every N steps; omit key if capture failed
    if SCREENSHOT_EVERY_N > 0 and step % SCREENSHOT_EVERY_N == 0:
        b64 = _capture_screenshot_b64()
        if b64:
            obs["screenshot_b64"] = b64

    return obs


def _post_step(obs: dict) -> int:
    """POST obs to the trainer server; return action index (default 0 = noop)."""
    url = f"http://{HOST}:{PORT}/step"
    payload = json.dumps(obs).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DT * 0.8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            action = int(body.get("action", 0))
            if 0 <= action < NUM_ACTIONS:
                return action
            return 0
    except (urllib.error.URLError, json.JSONDecodeError, Exception):
        return 0  # noop if unreachable


def _execute_action(action: int, current_yaw: float, current_pitch: float) -> float:
    """Execute a discrete action via MineScript controls.

    Returns the desired hold duration in seconds for this action.  The main
    loop will sleep for ``max(DT, hold_s)`` so that long holds block the
    observation loop for their full duration before the next step.
    """
    _release_all()

    hold_s = _ACTION_HOLD_S[action] if 0 <= action < NUM_ACTIONS else 0.0

    if action == ACTION_NOOP:
        pass
    elif action in (ACTION_FORWARD_250, ACTION_FORWARD_1000, ACTION_FORWARD_2000):
        minescript.player_press_forward(True)
    elif action == ACTION_TURN_LEFT:
        minescript.player_set_orientation(current_yaw - YAW_DELTA, current_pitch)
    elif action == ACTION_TURN_RIGHT:
        minescript.player_set_orientation(current_yaw + YAW_DELTA, current_pitch)
    elif action == ACTION_LOOK_UP:
        new_pitch = max(-89.0, current_pitch - PITCH_DELTA)
        minescript.player_set_orientation(current_yaw, new_pitch)
    elif action == ACTION_LOOK_DOWN:
        new_pitch = min(89.0, current_pitch + PITCH_DELTA)
        minescript.player_set_orientation(current_yaw, new_pitch)
    elif action == ACTION_JUMP_250:
        minescript.player_press_jump(True)
    elif action == ACTION_USE_250:
        minescript.player_press_use(True)
    elif action in (ACTION_ATTACK_250, ACTION_ATTACK_1000,
                    ACTION_ATTACK_2000, ACTION_ATTACK_4000):
        minescript.player_press_attack(True)

    return hold_s


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    minescript.echo("[BlueberryBridge] Starting – connecting to trainer on "
                    f"{HOST}:{PORT} …")
    step = 0
    while True:
        t_start = time.time()

        obs = _build_observation(step)
        action = _post_step(obs)
        yaw = obs.get("yaw", 0.0)
        pitch = obs.get("pitch", 0.0)
        hold_s = _execute_action(action, yaw, pitch)

        # Hold keys for max(DT, hold_s) minus elapsed time, then release.
        # Long-hold actions (e.g. attack_hold_4000ms) intentionally block the
        # loop so the key stays pressed for the full hold duration.
        elapsed = time.time() - t_start
        target = max(DT, hold_s)
        sleep_t = max(0.0, target - elapsed - 0.02)
        if sleep_t > 0:
            time.sleep(sleep_t)
        _release_all()

        step += 1


main()
