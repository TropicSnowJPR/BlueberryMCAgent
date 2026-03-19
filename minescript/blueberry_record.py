"""
blueberry_record.py – MineScript demo recorder for BlueberryMCAgent.

Run in-game:  \\blueberry_record
Stop:         \\killjob

While you play normally the script records an observation + a discrete action
label every DT seconds (default 0.25 s) into:

    demos/<run_id>/steps.jsonl   – one JSON object per line
    demos/<run_id>/screens/      – periodic PNG screenshots

Action labelling strategy
-------------------------
1. Primary: MineScript key-state queries (player_get_key_*).  Most MineScript
   builds do NOT expose these, so this is a best-effort call.
2. Fallback (state-change inference): compare consecutive observations to
   estimate what the player did (movement → forward, yaw change → turn, etc.).
   Attack / use cannot be reliably inferred; those steps get action = -1.
3. External logger: run ``tools/input_logger.py`` in a separate terminal while
   you record.  After recording, run ``trainer/demo_align.py`` to merge the
   input log with this file and replace the -1 labels with real action IDs.

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
TRAINER_HOST = "127.0.0.1"
TRAINER_PORT = 25566          # set to 0 to disable posting to trainer
DT = 0.25                     # step duration in seconds
SCREENSHOT_EVERY_N = 3        # 0 = never
DEMOS_DIR = "demos"           # relative to where Minecraft looks for scripts
# ───────────────────────────────────────────────────────────────────────────────

import time
import json
import math
import base64
import io
import os

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

# Hold thresholds for forward/attack bucketing (seconds, largest first).
_FORWARD_BUCKETS = (2.0, 1.0)
_ATTACK_BUCKETS  = (4.0, 2.0, 1.0)

# Minimum mouse delta (degrees) to register a turn / look action.
_TURN_THRESHOLD = 2.5
_LOOK_THRESHOLD = 2.0


def _bucket_hold(duration_s: float, buckets: tuple) -> int:
    for i, thr in enumerate(buckets):
        if duration_s >= thr:
            return i
    return len(buckets)


def _forward_action(hold_s: float) -> int:
    return ACTION_FORWARD_2000 - _bucket_hold(hold_s, _FORWARD_BUCKETS)


def _attack_action(hold_s: float) -> int:
    return ACTION_ATTACK_4000 - _bucket_hold(hold_s, _ATTACK_BUCKETS)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_inventory() -> dict:
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


def _get_targeted_block():
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


def _capture_screenshot_b64():
    try:
        result = minescript.screenshot()
        if result is None:
            return None
        if hasattr(result, "tobytes"):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        with open(str(result), "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _build_observation(step: int) -> dict:
    obs: dict = {"step": step, "t": time.time()}
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
    try:
        obs["health"] = float(minescript.player_health())
    except Exception:
        obs["health"] = 20.0
    obs["inventory"] = _get_inventory()
    obs["targeted_block"] = _get_targeted_block()
    if SCREENSHOT_EVERY_N > 0 and step % SCREENSHOT_EVERY_N == 0:
        b64 = _capture_screenshot_b64()
        if b64:
            obs["screenshot_b64"] = b64
    return obs


def _try_get_key_states() -> dict | None:
    """Attempt to read raw key states from MineScript.

    Returns a dict with boolean values if supported by this MineScript build,
    or None if the API is unavailable.  Keys: forward, back, left, right,
    jump, attack, use.
    """
    try:
        # These functions are NOT in the standard MineScript API; they are
        # hypothetical.  The try/except handles the AttributeError gracefully.
        return {
            "forward": bool(minescript.player_get_key_forward()),
            "back":    bool(minescript.player_get_key_back()),
            "left":    bool(minescript.player_get_key_left()),
            "right":   bool(minescript.player_get_key_right()),
            "jump":    bool(minescript.player_get_key_jump()),
            "attack":  bool(minescript.player_get_key_attack()),
            "use":     bool(minescript.player_get_key_use()),
        }
    except Exception:
        return None


def _infer_action_from_state(obs_prev: dict, obs_curr: dict) -> int:
    """Infer the most likely action by comparing consecutive observations.

    Attack and use cannot be inferred reliably from state changes alone; those
    steps are assigned :data:`ACTION_NOOP` here and should be relabelled using
    ``trainer/demo_align.py`` with external input logs.
    """
    pos_p = obs_prev.get("pos", [0.0, 64.0, 0.0])
    pos_c = obs_curr.get("pos", [0.0, 64.0, 0.0])
    yaw_p = float(obs_prev.get("yaw", 0.0))
    yaw_c = float(obs_curr.get("yaw", 0.0))
    pitch_p = float(obs_prev.get("pitch", 0.0))
    pitch_c = float(obs_curr.get("pitch", 0.0))

    dx = pos_c[0] - pos_p[0]
    dz = pos_c[2] - pos_p[2]
    horiz_dist = math.sqrt(dx * dx + dz * dz)

    # Normalise yaw delta to [-180, 180]
    dyaw = yaw_c - yaw_p
    if dyaw > 180:
        dyaw -= 360
    elif dyaw < -180:
        dyaw += 360

    dpitch = pitch_c - pitch_p

    # Priority: orientation > movement > noop
    if abs(dyaw) >= _TURN_THRESHOLD and abs(dyaw) >= abs(dpitch):
        return ACTION_TURN_LEFT if dyaw < 0 else ACTION_TURN_RIGHT
    if abs(dpitch) >= _LOOK_THRESHOLD:
        return ACTION_LOOK_UP if dpitch < 0 else ACTION_LOOK_DOWN
    if horiz_dist > 0.05:
        return ACTION_FORWARD_250
    if pos_c[1] - pos_p[1] > 0.3:
        return ACTION_JUMP_250
    return ACTION_NOOP


def _action_from_keys(keys: dict, yaw_delta: float, pitch_delta: float,
                      fwd_hold_s: float, atk_hold_s: float) -> int:
    """Convert key states dict to discrete action (same priority as keys_to_action)."""
    if keys.get("attack"):
        return _attack_action(atk_hold_s)
    if keys.get("use"):
        return ACTION_USE_250
    if keys.get("jump"):
        return ACTION_JUMP_250
    if keys.get("forward"):
        return _forward_action(fwd_hold_s)
    if abs(yaw_delta) >= _TURN_THRESHOLD and abs(yaw_delta) >= abs(pitch_delta):
        return ACTION_TURN_LEFT if yaw_delta < 0 else ACTION_TURN_RIGHT
    if abs(pitch_delta) >= _LOOK_THRESHOLD:
        return ACTION_LOOK_UP if pitch_delta < 0 else ACTION_LOOK_DOWN
    return ACTION_NOOP


# ── Main recording loop ────────────────────────────────────────────────────────

def main():
    from datetime import datetime
    import pathlib
    import urllib.request
    import urllib.error

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve demos directory relative to the Minecraft minescript folder.
    # MineScript scripts run with cwd = .minecraft/minescript; adjust as needed.
    demos_root = pathlib.Path(DEMOS_DIR) / run_id
    screens_dir = demos_root / "screens"
    screens_dir.mkdir(parents=True, exist_ok=True)
    steps_path = demos_root / "steps.jsonl"

    minescript.echo(f"[BlueberryRecord] Run {run_id} – saving to {demos_root}")
    minescript.echo("[BlueberryRecord] Play normally.  Stop with \\killjob")

    step = 0
    obs_prev: dict | None = None
    prev_yaw = 0.0
    prev_pitch = 0.0

    # Hold-duration tracking (seconds)
    fwd_hold_start: float | None = None
    atk_hold_start: float | None = None

    while True:
        t_start = time.time()
        obs = _build_observation(step)

        # ── Determine action label ────────────────────────────────────────────
        action: int = -1  # -1 = unknown (to be filled by demo_align)

        yaw_now = float(obs.get("yaw", 0.0))
        pitch_now = float(obs.get("pitch", 0.0))
        yaw_delta = yaw_now - prev_yaw
        if yaw_delta > 180:
            yaw_delta -= 360
        elif yaw_delta < -180:
            yaw_delta += 360
        pitch_delta = pitch_now - prev_pitch

        keys = _try_get_key_states()
        if keys is not None:
            # MineScript key-state API is available — track hold durations.
            now = time.time()
            if keys.get("forward"):
                if fwd_hold_start is None:
                    fwd_hold_start = now
                fwd_hold_s = now - fwd_hold_start
            else:
                fwd_hold_start = None
                fwd_hold_s = 0.0
            if keys.get("attack"):
                if atk_hold_start is None:
                    atk_hold_start = now
                atk_hold_s = now - atk_hold_start
            else:
                atk_hold_start = None
                atk_hold_s = 0.0
            action = _action_from_keys(keys, yaw_delta, pitch_delta,
                                       fwd_hold_s, atk_hold_s)
        elif obs_prev is not None:
            # Fallback: infer from state changes (attack/use labelled as noop).
            action = _infer_action_from_state(obs_prev, obs)
            keys = {}
        else:
            keys = {}

        # ── Persist screenshot if present ─────────────────────────────────────
        screenshot_path: str | None = None
        b64 = obs.pop("screenshot_b64", None)
        if b64:
            screenshot_path = str(screens_dir / f"step_{step:08d}.png")
            try:
                import struct
                raw = base64.b64decode(b64)
                with open(screenshot_path, "wb") as f:
                    f.write(raw)
            except Exception:
                screenshot_path = None

        # ── Write step record ─────────────────────────────────────────────────
        record = {
            "step": step,
            "t": obs["t"],
            "obs": {
                "pos": obs.get("pos"),
                "yaw": obs.get("yaw"),
                "pitch": obs.get("pitch"),
                "health": obs.get("health"),
                "inventory": obs.get("inventory"),
                "targeted_block": obs.get("targeted_block"),
            },
            "action": action,
            "keys": keys if keys else None,
            "screenshot_path": screenshot_path,
        }
        try:
            with open(str(steps_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            minescript.echo(f"[BlueberryRecord] WARNING: cannot write steps.jsonl: {exc}")

        obs_prev = obs
        prev_yaw = yaw_now
        prev_pitch = pitch_now
        step += 1

        # ── Sleep for remainder of DT ─────────────────────────────────────────
        elapsed = time.time() - t_start
        sleep_t = max(0.0, DT - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)


main()
