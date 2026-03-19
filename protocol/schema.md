# BlueberryMCAgent – JSON Protocol

This document describes every JSON message exchanged between the MineScript
bridge (`minescript/blueberry_bridge.py`) running inside Minecraft and the
Python trainer server (`trainer/server.py`) running on the same Windows PC.

---

## Transport

| Property   | Value                   |
|------------|-------------------------|
| Protocol   | HTTP/1.1                |
| Host       | `127.0.0.1` (localhost) |
| Port       | `25566`                 |
| Encoding   | UTF-8 JSON              |
| CORS       | not required (localhost)|

---

## POST /step

The bridge calls this endpoint every `DT` seconds (default 0.25 s).

### Request body (observation)

```json
{
  "step": 42,
  "t": 1712345678.12,
  "pos": [12.5, 64.0, -30.1],
  "yaw": 45.0,
  "pitch": -10.0,
  "health": 20.0,
  "inventory": {
    "oak_log": 3,
    "crafting_table": 1
  },
  "targeted_block": {
    "type": "oak_log",
    "pos": [13, 65, -30],
    "distance": 2.8
  },
  "screenshot_b64": "<base-64-encoded PNG string or absent>"
}
```

| Field             | Type            | Notes                                                  |
|-------------------|-----------------|--------------------------------------------------------|
| `step`            | integer         | Monotonically increasing step counter (from bridge)    |
| `t`               | float           | Unix timestamp (seconds) of the observation            |
| `pos`             | [x, y, z] float | Player world position                                  |
| `yaw`             | float           | Player yaw (−180 … 180°, 0 = south)                   |
| `pitch`           | float           | Player pitch (−90 = straight up, +90 = straight down) |
| `health`          | float           | Player health (0.0 – 20.0)                            |
| `inventory`       | object          | Item ID → count for non-zero stacks                   |
| `targeted_block`  | object or null  | Block the crosshair aims at; `null` if none           |
| `screenshot_b64`  | string or absent| Base-64 PNG; only present every `SCREENSHOT_EVERY_N` steps |

### Response body (action)

```json
{
  "action": 1
}
```

| Field    | Type    | Notes                                        |
|----------|---------|----------------------------------------------|
| `action` | integer | Discrete action index (see action table below)|

### On error (trainer unreachable)

The bridge catches all network and JSON errors and defaults to **action 0 (noop)**.

---

## GET /status

Returns a human-readable JSON summary of the current training state.

### Response

```json
{
  "run_id": "20240415_143012",
  "device": "cuda",
  "total_steps": 1280,
  "total_updates": 20,
  "episode_reward": 7.42,
  "total_reward": 143.1,
  "buffer_size": 0,
  "milestones": {
    "planks_unlocked": true,
    "crafting_table_unlocked": false
  },
  "last_ppo_stats": {
    "policy_loss": 0.0312,
    "value_loss": 0.1184,
    "entropy": 2.1905,
    "total_loss": 0.0749
  }
}
```

---

## Discrete Action Space

| Index | Name                  | MineScript effect                                          | Hold     |
|-------|-----------------------|------------------------------------------------------------|----------|
| 0     | `noop`                | No key presses; agent stands still                         | —        |
| 1     | `forward_250ms`       | `player_press_forward(True)` for 0.25 s                   | 0.25 s   |
| 2     | `forward_1000ms`      | `player_press_forward(True)` for 1.0 s                    | 1.0 s    |
| 3     | `forward_2000ms`      | `player_press_forward(True)` for 2.0 s                    | 2.0 s    |
| 4     | `turn_left_small`     | Decrease yaw by `YAW_DELTA` degrees (default 5°)          | instant  |
| 5     | `turn_right_small`    | Increase yaw by `YAW_DELTA` degrees (default 5°)          | instant  |
| 6     | `look_up_small`       | Decrease pitch by `PITCH_DELTA` degrees (default 4°)      | instant  |
| 7     | `look_down_small`     | Increase pitch by `PITCH_DELTA` degrees (default 4°)      | instant  |
| 8     | `jump_250ms`          | `player_press_jump(True)` for 0.25 s                      | 0.25 s   |
| 9     | `use_250ms`           | `player_press_use(True)` for 0.25 s                       | 0.25 s   |
| 10    | `attack_250ms`        | `player_press_attack(True)` for 0.25 s                    | 0.25 s   |
| 11    | `attack_hold_1000ms`  | `player_press_attack(True)` held for 1.0 s                | 1.0 s    |
| 12    | `attack_hold_2000ms`  | `player_press_attack(True)` held for 2.0 s                | 2.0 s    |
| 13    | `attack_hold_4000ms`  | `player_press_attack(True)` held for 4.0 s                | 4.0 s    |

Keys are **released** after the specified hold duration.  For actions with a
hold duration greater than `DT` (0.25 s) the bridge pauses observation
collection for the full hold duration before taking the next step.

### Hold rounding

Continuous user input is discretised into hold buckets by **rounding up** to
the nearest threshold:

**Forward:** 2.0 s, 1.0 s, else 0.25 s  
**Attack:** 4.0 s, 2.0 s, 1.0 s, else 0.25 s

---

## Observation State Vector (internal, 32-d)

The trainer converts the JSON observation to a 32-element float32 vector for
the neural network:

| Index  | Feature                                   | Normalisation   |
|--------|-------------------------------------------|-----------------|
| 0      | pos_x                                     | ÷ 1000          |
| 1      | pos_y                                     | ÷ 256           |
| 2      | pos_z                                     | ÷ 1000          |
| 3      | yaw                                       | ÷ 180           |
| 4      | pitch                                     | ÷ 90            |
| 5      | health                                    | ÷ 20            |
| 6      | targeted_block flag (0 or 1)              | —               |
| 7–24   | inventory counts (18 tracked item types)  | ÷ 64, capped 1  |
| 25–31  | padding zeros                             | —               |

---

## Screenshot Format

* Captured in-game by MineScript's `screenshot()` function.
* Encoded as base-64 PNG before inclusion in the JSON payload.
* The trainer downscales every received screenshot to **160 × 90** pixels
  (RGB) before feeding it to the CNN vision encoder.
* Downscaled copies are **not** saved; originals (or the downscaled version)
  are written to `runs/<run_id>/screens/step_<N>.png`.
