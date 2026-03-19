# BlueberryMCAgent

A **real reinforcement-learning agent** that learns to play Minecraft Survival
from scratch, running entirely on your own Windows PC.  
No LLM, no scripted rules — it genuinely improves through trial-and-error.

Two modes are supported:

| Mode | Description |
|------|-------------|
| **A – Offline demo recording + behaviour cloning** | Play normally; record demonstrations; train a policy offline (supervised) |
| **B – Online PPO** | Agent plays and learns continuously in real-time (can warm-start from Mode A) |

## How it works

```
 Minecraft (Fabric 1.21.11)
  └─ MineScript 5.0b11
      └─ blueberry_bridge.py     ←──────────────────────────────┐
           │  POST /step (obs + screenshot)                      │
           ▼                                                      │
     trainer/server.py           (Flask on 127.0.0.1:25566)      │
           │  returns {action}  ──────────────────────────────── ┘
           │
           ├─ computes shaped reward from consecutive observations
           ├─ stores transition in rollout buffer
           ├─ when buffer is full → PPO update (background thread)
           └─ saves screenshots to runs/<id>/screens/
                       step log to runs/<id>/steps.jsonl
                       checkpoints to runs/<id>/checkpoints/
```

**Every ~250 ms the loop:**
1. Reads player state (pos, yaw, pitch, health, inventory, targeted block).
2. Optionally captures a screenshot (every 3 steps by default).
3. POSTs an observation to the trainer server.
4. Receives a discrete action index (0–13) and executes it in-game.
5. Holds keys for the action-specific duration, then releases them.

The trainer runs **online PPO** — weights are updated *while the agent plays*,
exactly like a newborn learning from experience, not a chatbot pretending.

---

## Mode A – Offline demo recording + behaviour cloning

### Step A1 – Record a demonstration while you play

1. Copy `minescript/blueberry_record.py` to `%APPDATA%\.minecraft\minescript\`.
2. Open the MineScript console in-game and run:
   ```
   \blueberry_record
   ```
3. Play Minecraft normally.  The script records observations + inferred action
   labels to `demos/<run_id>/steps.jsonl`.
4. Stop recording with `\killjob`.

> **Tip – External input logger (for better attack/use labels)**
>
> MineScript cannot query raw key states in most builds, so attack and use
> actions may be labelled as `noop` in the raw recorder output.  To fix this:
>
> 1. Before recording, open a second terminal and run:
>    ```bat
>    python tools/input_logger.py --out logs/session.jsonl
>    ```
> 2. Record as normal with `\blueberry_record`.
> 3. After recording, merge the two logs:
>    ```bat
>    python trainer/demo_align.py \
>        --demo demos/<run_id> \
>        --input logs/session.jsonl
>    ```
>    This writes `demos/<run_id>/steps_aligned.jsonl` with proper action labels.

### Step A2 – Train offline with behaviour cloning

```bat
python trainer/bc_train.py --demo demos/<run_id>
```

Or with custom options:
```bat
python trainer/bc_train.py \
    --demo demos/run1 demos/run2 \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-4 \
    --device cuda
```

Checkpoints are saved to `checkpoints/bc_<timestamp>.pt` and
`checkpoints/bc_latest.pt`.

### Step A3 – Run online PPO warm-started from the BC checkpoint

```bat
python trainer/server.py --init_from checkpoints/bc_latest.pt
```

---

## Mode B – Online PPO (agent plays and learns in real time)

### Online training vs offline training

| | Online (Mode B) | Offline (Mode A) |
|---|---|---|
| When weights update | Continuously, every N steps | After a separate training run |
| Data required upfront | None | Demo recordings |
| Latency to improvement | Minutes | Hours / days |
| Risk of catastrophic forgetting | Higher (mitigated by PPO clip) | Lower |

---

## Project layout

```
BlueberryMCAgent/
├── minescript/
│   ├── blueberry_bridge.py   ← online agent (runs inside Minecraft via \blueberry_bridge)
│   └── blueberry_record.py   ← demo recorder  (runs via \blueberry_record)
├── trainer/
│   ├── server.py             ← Flask HTTP server + online PPO training loop
│   ├── config.py             ← all hyper-parameters and seeds
│   ├── actions.py            ← shared action-space definitions (14 actions)
│   ├── reward.py             ← shaped reward function
│   ├── bc_train.py           ← offline behaviour-cloning trainer
│   ├── demo_align.py         ← merge recorder JSONL with input logger JSONL
│   └── ppo/
│       ├── __init__.py
│       ├── model.py          ← CNN + MLP actor-critic network
│       ├── buffer.py         ← rollout buffer with GAE
│       ├── update.py         ← PPO clipped objective
│       └── utils.py          ← obs → state vector, screenshot decode
├── tools/
│   └── input_logger.py       ← optional Windows key/mouse logger (pynput)
├── protocol/
│   └── schema.md             ← full JSON protocol documentation
├── requirements.txt
└── README.md
```

---

## Windows setup (step by step)

### Prerequisites

| Requirement | Where to get it |
|---|---|
| Python 3.11 or 3.12 | https://www.python.org/downloads/ |
| Minecraft Java Edition | Minecraft Launcher |
| Fabric Loader 1.21.11 | https://fabricmc.net/use/installer/ |
| MineScript 5.0b11 mod | MineScript GitHub / Modrinth |
| NVIDIA GPU driver (latest) | https://www.nvidia.com/drivers |

### 1 — Clone the repo

```bat
git clone https://github.com/TropicSnowJPR/BlueberryMCAgent.git
cd BlueberryMCAgent
```

### 2 — Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3 — Install PyTorch with CUDA 12.1 (RTX 30xx)

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `CUDA: True NVIDIA GeForce RTX 3050 Ti Laptop GPU`

### 4 — Install remaining dependencies

```bat
pip install -r requirements.txt
```

### 5 — Copy MineScript scripts into Minecraft

```bat
copy minescript\blueberry_bridge.py "%APPDATA%\.minecraft\minescript\blueberry_bridge.py"
copy minescript\blueberry_record.py "%APPDATA%\.minecraft\minescript\blueberry_record.py"
```

MineScript looks for scripts in `%APPDATA%\.minecraft\minescript\`.

---

## Running the agent (Mode B — online PPO)

### Step B1 — Start the trainer server

Open a terminal **in the project root** (with the virtual environment active):

```bat
.venv\Scripts\activate
python trainer/server.py
```

Or warm-start from a BC checkpoint:
```bat
python trainer/server.py --init_from checkpoints/bc_latest.pt
```

You should see:

```
10:00:00 [INFO] Using device: cuda
10:00:00 [INFO] BlueberryMCAgent trainer starting on http://127.0.0.1:25566
10:00:00 [INFO] Run directory: C:\...\BlueberryMCAgent\runs\20240415_100000
```

Leave this window open.

### Step B2 — Launch Minecraft

1. Open the Minecraft Launcher.
2. Select the **Fabric 1.21.11** profile and click **Play**.
3. Load or create a **Survival** world.  
   > **Tip:** Start in **Peaceful** difficulty — the agent learns faster without
   > hostile mobs.

### Step B3 — Start the bridge in-game

Open the MineScript console (default key: `\`) and run:

```
\blueberry_bridge
```

The agent will start acting immediately.  
Watch the trainer terminal for PPO update messages every 64 steps.

### Step B4 — Check training status

In a browser (or `curl`) visit:

```
http://127.0.0.1:25566/status
```

### Step B5 — Stop the agent

In-game MineScript console:

```
\killjob
```

The trainer server keeps running; you can restart the bridge any time.  
Press **Ctrl+C** in the trainer terminal to stop the server.

---

## Action space reference

Hold durations are bucketed ("rounded up") so continuous holds are mapped to the
nearest discrete hold action.

| Index | Name | Effect | Hold |
|---|---|---|---|
| 0 | noop | Stand still | — |
| 1 | forward_250ms | Walk forward | 0.25 s |
| 2 | forward_1000ms | Walk forward | 1.0 s |
| 3 | forward_2000ms | Walk forward | 2.0 s |
| 4 | turn_left_small | Rotate yaw −5° | instant |
| 5 | turn_right_small | Rotate yaw +5° | instant |
| 6 | look_up_small | Decrease pitch by 4° | instant |
| 7 | look_down_small | Increase pitch by 4° | instant |
| 8 | jump_250ms | Jump | 0.25 s |
| 9 | use_250ms | Right-click (place/open) | 0.25 s |
| 10 | attack_250ms | Left-click (break/hit) | 0.25 s |
| 11 | attack_hold_1000ms | Hold left-click | 1.0 s |
| 12 | attack_hold_2000ms | Hold left-click | 2.0 s |
| 13 | attack_hold_4000ms | Hold left-click | 4.0 s |

### Hold rounding

When the player holds a key for a continuous duration *d* seconds, the
recorder and demo_align assign the action according to these rules:

**Forward:**
- *d* ≥ 2.0 s → `forward_2000ms`
- *d* ≥ 1.0 s → `forward_1000ms`
- *d* < 1.0 s → `forward_250ms`

**Attack:**
- *d* ≥ 4.0 s → `attack_hold_4000ms`
- *d* ≥ 2.0 s → `attack_hold_2000ms`
- *d* ≥ 1.0 s → `attack_hold_1000ms`
- *d* < 1.0 s → `attack_250ms`

---

## Reward milestones (Peaceful survival progression)

| Milestone | Reward |
|---|---|
| Each new log collected | +0.5 |
| First plank crafted | +1.0 (one time) |
| First crafting table | +2.0 (one time) |
| First pickaxe | +3.0 (one time) |
| Each new cobblestone | +0.3 |
| Each new food item | +0.4 |
| Surviving each step | +0.01 |
| Taking damage (per HP lost) | −1.0 |
| Death | −10.0 |

---

## Troubleshooting

### Minecraft stutters / low FPS
- Increase `SCREENSHOT_EVERY_N` in `blueberry_bridge.py` (e.g., 5 or 10).
- Lower Minecraft render distance.
- Set `DT = 0.35` for a slower step rate.

### Trainer connection errors in bridge
The bridge catches all network errors and defaults to action 0 (noop), so the
game keeps running. Check that `trainer/server.py` is still running in its
terminal window.

### PyTorch says CUDA is not available
- Make sure you installed the CUDA wheel (`cu121`), not the CPU-only wheel.
- Make sure your NVIDIA driver is up to date (≥ 527 for CUDA 12).
- Run `nvidia-smi` in a terminal to confirm the driver is installed.

### Agent does nothing / always noop
During the very first rollout (first 64 steps) the model uses a random
initialisation, so behaviour will look random. After the first PPO update
(step 64) it begins improving. Give it at least a few hundred steps.

### Screenshots not saving
MineScript's `screenshot()` API may differ between minor versions. If
screenshots are absent from `runs/<id>/screens/`, set `SCREENSHOT_EVERY_N = 0`
in `blueberry_bridge.py` to disable them; the agent still trains on state
observations only.

### MineScript cannot capture inputs (attack/use labelled as noop)
Run `tools/input_logger.py` in a separate terminal while recording, then use
`trainer/demo_align.py` to merge the logs.  See **Mode A, Step A1** above.

### Want to resume online PPO training from a checkpoint?
Checkpoints are saved to `runs/<id>/checkpoints/ckpt_<N>.pt` every 50 updates.
Load one with:

```python
import torch
from trainer.ppo import ActorCritic
model = ActorCritic()
ckpt = torch.load("runs/20240415_100000/checkpoints/ckpt_000050.pt")
model.load_state_dict(ckpt["model_state"])
```

Or pass it as `--init_from` to the server to warm-start online PPO.

