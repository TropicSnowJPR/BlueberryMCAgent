"""trainer/demo_align.py – Align recorder frames with external input logs.

When ``minescript/blueberry_record.py`` cannot query raw key states from
MineScript it writes ``"action": -1`` for frames it cannot label (mainly
attack and use).  This utility merges a recorder JSONL with an input logger
JSONL (from ``tools/input_logger.py``) and replaces those -1 labels with
properly bucketed action IDs.

Usage
-----
    python trainer/demo_align.py \\
        --demo  demos/20240415_100000/steps.jsonl \\
        --input input_log_20240415_100000.jsonl \\
        --out   demos/20240415_100000/steps_aligned.jsonl

    # Align all demos in a directory (looks for steps.jsonl in each subdir):
    python trainer/demo_align.py \\
        --demo  demos/20240415_100000 \\
        --input input_log_20240415_100000.jsonl

Output
------
A new JSONL with the same schema as steps.jsonl but with ``"action"`` filled
in for every frame.  Frames where the input log has no coverage are assigned
``ACTION_NOOP`` (0).
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from trainer.actions import (
    NUM_ACTIONS,
    ACTION_NOOP,
    ACTION_USE_250,
    ACTION_JUMP_250,
    keys_to_action,
)


# ── Key-name normalisation ─────────────────────────────────────────────────────
# pynput uses different names depending on platform and key.
_FORWARD_KEYS  = {"w", "'w'"}
_BACK_KEYS     = {"s", "'s'"}
_LEFT_KEYS     = {"a", "'a'"}
_RIGHT_KEYS    = {"d", "'d'"}
_JUMP_KEYS     = {"space", "Key.space"}
_ATTACK_BTNS   = {"left", "Button.left"}
_USE_BTNS      = {"right", "Button.right"}


def _norm(name: str) -> str:
    """Strip extra quotes that pynput sometimes adds to single char keys."""
    return name.strip("'\"")


# ── Input-log loading ──────────────────────────────────────────────────────────

def load_input_log(path: Path) -> list[dict]:
    """Return a list of input events sorted by timestamp."""
    events: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda e: e.get("t", 0.0))
    return events


def load_demo(path: Path) -> list[dict]:
    """Return a list of step records sorted by step index."""
    steps: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    steps.sort(key=lambda s: s.get("step", 0))
    return steps


# ── State reconstruction ───────────────────────────────────────────────────────

class _InputState:
    """Replay the input event log and answer 'what was pressed at time t?'."""

    def __init__(self, events: list[dict]):
        self._events = events
        self._ts = [e["t"] for e in events]

    def state_at(self, t_start: float, t_end: float) -> dict:
        """Return the key / mouse state *active* during (t_start, t_end).

        A key is considered 'active' if it was held at the start of the window
        (from prior events) **or** received a key-down event within the window.
        Hold duration is measured from the key-down timestamp to t_end (capped
        at zero if the key was released before t_end).
        """
        end_idx = bisect.bisect_right(self._ts, t_end)
        start_idx = bisect.bisect_left(self._ts, t_start)

        # ── Phase 1: replay all events before t_start to build baseline state ──
        key_down: set[str] = set()
        btn_down: set[str] = set()
        _held_since: dict[str, float] = {}

        for ev in self._events[:start_idx]:
            t = ev["t"]
            etype = ev.get("type", "")
            key = _norm(ev.get("key", ""))
            btn = _norm(ev.get("button", ""))
            if etype == "key_down":
                key_down.add(key)
                _held_since.setdefault(key, t)
            elif etype == "key_up":
                key_down.discard(key)
                _held_since.pop(key, None)
            elif etype == "mouse_button_down":
                btn_down.add(btn)
                _held_since.setdefault(btn, t)
            elif etype == "mouse_button_up":
                btn_down.discard(btn)
                _held_since.pop(btn, None)

        # ── Phase 2: replay events within [t_start, t_end] ────────────────────
        # Track keys that were active at ANY point in the window, accumulate
        # mouse movement deltas.
        keys_active_in_window: set[str] = set()
        btns_active_in_window: set[str] = set()
        dx_total = 0.0
        dy_total = 0.0

        # Keys already held going into the window count as active
        keys_active_in_window.update(key_down)
        btns_active_in_window.update(btn_down)

        # Keep a separate dict so that releasing within the window does not
        # erase the hold-start time (needed for duration calculation).
        _held_since_window: dict[str, float] = dict(_held_since)

        for ev in self._events[start_idx:end_idx]:
            t = ev["t"]
            etype = ev.get("type", "")
            key = _norm(ev.get("key", ""))
            btn = _norm(ev.get("button", ""))
            if etype == "key_down":
                key_down.add(key)
                keys_active_in_window.add(key)
                _held_since.setdefault(key, t)
                _held_since_window.setdefault(key, t)
            elif etype == "key_up":
                key_down.discard(key)
                _held_since.pop(key, None)
                # Do NOT remove from _held_since_window – keep start time
            elif etype == "mouse_button_down":
                btn_down.add(btn)
                btns_active_in_window.add(btn)
                _held_since.setdefault(btn, t)
                _held_since_window.setdefault(btn, t)
            elif etype == "mouse_button_up":
                btn_down.discard(btn)
                _held_since.pop(btn, None)
                # Do NOT remove from _held_since_window
            elif etype == "mouse_move":
                dx_total += float(ev.get("dx", 0))
                dy_total += float(ev.get("dy", 0))

        # ── Compute hold durations ─────────────────────────────────────────────
        fwd_hold_s = 0.0
        atk_hold_s = 0.0
        fwd_key_set = {_norm(x) for x in _FORWARD_KEYS}
        atk_btn_set = {_norm(x) for x in _ATTACK_BTNS}

        for k, since in _held_since_window.items():
            if k in fwd_key_set and k in keys_active_in_window:
                fwd_hold_s = max(fwd_hold_s, t_end - since)
        for b, since in _held_since_window.items():
            if b in atk_btn_set and b in btns_active_in_window:
                atk_hold_s = max(atk_hold_s, t_end - since)

        return {
            "key_forward": any(_norm(k) in fwd_key_set for k in keys_active_in_window),
            "key_attack":  any(_norm(b) in atk_btn_set for b in btns_active_in_window),
            "key_use":     any(_norm(b) in {_norm(x) for x in _USE_BTNS} for b in btns_active_in_window),
            "key_jump":    any(_norm(k) in {_norm(x) for x in _JUMP_KEYS} for k in keys_active_in_window),
            "yaw_delta":   dx_total * 0.3,
            "pitch_delta": dy_total * 0.3,
            "forward_hold_s": fwd_hold_s,
            "attack_hold_s":  atk_hold_s,
        }


# ── Alignment ──────────────────────────────────────────────────────────────────

def align(steps: list[dict], input_state: _InputState, dt: float = 0.25) -> list[dict]:
    """Fill in action labels for every step using the input log.

    Steps that already have a valid action (0 ≤ action < NUM_ACTIONS) are
    preserved as-is unless ``overwrite=True`` is set (not exposed in the CLI
    for safety).

    Parameters
    ----------
    steps:
        List of step records from the demo recorder.
    input_state:
        Replay object built from the input log.
    dt:
        Step duration in seconds (default 0.25).

    Returns
    -------
    New list of step dicts with ``"action"`` filled in.
    """
    out: list[dict] = []
    for rec in steps:
        rec = dict(rec)  # shallow copy
        t = float(rec.get("t", 0.0))
        existing = rec.get("action", -1)
        if isinstance(existing, int) and 0 <= existing < NUM_ACTIONS:
            out.append(rec)
            continue

        state = input_state.state_at(t - dt, t)
        action = keys_to_action(
            key_forward=state["key_forward"],
            key_attack=state["key_attack"],
            key_use=state["key_use"],
            key_jump=state["key_jump"],
            yaw_delta=state["yaw_delta"],
            pitch_delta=state["pitch_delta"],
            forward_hold_s=state["forward_hold_s"],
            attack_hold_s=state["attack_hold_s"],
        )
        rec["action"] = action
        out.append(rec)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align BlueberryMCAgent demo frames with input logger events."
    )
    parser.add_argument(
        "--demo",
        required=True,
        metavar="PATH",
        help="Path to steps.jsonl or a demo directory containing steps.jsonl.",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Input log JSONL produced by tools/input_logger.py.",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Output path for aligned JSONL.  Defaults to steps_aligned.jsonl "
             "in the same directory as the demo.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.25,
        help="Step duration in seconds (default: 0.25).",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo)
    if demo_path.is_dir():
        demo_file = demo_path / "steps.jsonl"
    else:
        demo_file = demo_path

    if not demo_file.exists():
        parser.error(f"Demo file not found: {demo_file}")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input log not found: {input_path}")

    out_path = Path(args.out) if args.out else demo_file.parent / "steps_aligned.jsonl"

    print(f"[demo_align] Loading demo:       {demo_file}")
    print(f"[demo_align] Loading input log:  {input_path}")

    steps = load_demo(demo_file)
    events = load_input_log(input_path)
    inp_state = _InputState(events)

    aligned = align(steps, inp_state, dt=args.dt)

    labelled = sum(1 for s in aligned if s.get("action", -1) != -1)
    print(f"[demo_align] Steps: {len(aligned)}, labelled: {labelled}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in aligned:
            fh.write(json.dumps(rec) + "\n")

    print(f"[demo_align] Saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
