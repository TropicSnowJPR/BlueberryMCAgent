"""tools/input_logger.py – Lightweight Windows keyboard/mouse input logger.

Records every key press/release and mouse button/move event with a high-
resolution timestamp to a JSONL file.  Run this in a separate terminal
*while* you also run ``\\blueberry_record`` in-game so that the timestamps
can be aligned later with ``trainer/demo_align.py``.

Usage
-----
    python tools/input_logger.py                          # → input_log_<timestamp>.jsonl
    python tools/input_logger.py --out logs/my_session.jsonl
    python tools/input_logger.py --no-mouse-move         # suppress move spam

Requirements
------------
    pip install pynput   (already in requirements.txt)

Output format (one JSON object per line)
----------------------------------------
    {"t": 1712345678.123456, "type": "key_down",          "key": "w"}
    {"t": 1712345678.234567, "type": "key_up",            "key": "w"}
    {"t": 1712345678.345678, "type": "mouse_button_down", "button": "left"}
    {"t": 1712345678.456789, "type": "mouse_button_up",   "button": "left"}
    {"t": 1712345678.567890, "type": "mouse_move",        "dx": 12, "dy": -3}

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _require_pynput():
    try:
        import pynput  # noqa: F401
    except ImportError:
        print(
            "ERROR: pynput is not installed.\n"
            "Run:  pip install pynput\n"
            "or:   pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


def run(out_path: Path, no_mouse_move: bool) -> None:
    """Start listening and write events to *out_path* until Ctrl+C."""
    from pynput import keyboard as kb
    from pynput import mouse as ms

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a", encoding="utf-8", buffering=1)

    def _write(obj: dict) -> None:
        fh.write(json.dumps(obj) + "\n")

    # ── Keyboard listeners ────────────────────────────────────────────────────

    def on_press(key):
        try:
            name = key.char or key.name
        except AttributeError:
            name = str(key)
        _write({"t": time.time(), "type": "key_down", "key": name})

    def on_release(key):
        try:
            name = key.char or key.name
        except AttributeError:
            name = str(key)
        _write({"t": time.time(), "type": "key_up", "key": name})
        if key == kb.Key.esc:
            return False  # stop listener on Escape (optional; Ctrl+C also works)

    # ── Mouse listeners ───────────────────────────────────────────────────────

    _last_pos: list[int | None] = [None, None]

    def on_move(x, y):
        if no_mouse_move:
            return
        lx, ly = _last_pos
        dx = (x - lx) if lx is not None else 0
        dy = (y - ly) if ly is not None else 0
        _last_pos[0] = x
        _last_pos[1] = y
        if dx != 0 or dy != 0:
            _write({"t": time.time(), "type": "mouse_move", "dx": dx, "dy": dy})

    def on_click(x, y, button, pressed):
        event_type = "mouse_button_down" if pressed else "mouse_button_up"
        _write({"t": time.time(), "type": event_type, "button": button.name})

    def on_scroll(x, y, dx, dy):
        _write({"t": time.time(), "type": "mouse_scroll", "dx": dx, "dy": dy})

    print(f"[InputLogger] Writing to: {out_path.resolve()}")
    print("[InputLogger] Press Ctrl+C to stop.\n")

    kb_listener = kb.Listener(on_press=on_press, on_release=on_release)
    ms_listener = ms.Listener(on_move=on_move, on_click=on_click,
                               on_scroll=on_scroll)

    kb_listener.start()
    ms_listener.start()

    try:
        kb_listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        kb_listener.stop()
        ms_listener.stop()
        fh.close()
        print(f"\n[InputLogger] Stopped.  Log saved to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record keyboard/mouse events with timestamps for demo alignment."
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Output JSONL file path.  Defaults to input_log_<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--no-mouse-move",
        action="store_true",
        help="Suppress mouse-move events (reduces file size; turn/look inferred "
             "from position diffs in the recorder instead).",
    )
    args = parser.parse_args()

    _require_pynput()

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"input_log_{ts}.jsonl")

    run(out_path, no_mouse_move=args.no_mouse_move)


if __name__ == "__main__":
    main()
