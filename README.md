# BlueberryMCAgent

A **real reinforcement-learning agent** that learns to play Minecraft Survival
from scratch, running entirely on your own Windows PC.  
No LLM, no scripted rules — it genuinely improves through trial-and-error.

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
4. Receives a discrete action index (0–8) and executes it in-game.
5. Releases all key presses.

The trainer runs **online PPO** — weights are updated *while the agent plays*,
exactly like a newborn learning from experience, not a chatbot pretending.

---

## Online training vs offline training

| | Online (what we do) | Offline |
|---|---|---|
| When weights update | Continuously, every N steps | After a separate training run |
| Data required upfront | None | Large replay dataset |
| Latency to improvement | Minutes | Hours / days |
| Risk of catastrophic forgetting | Higher (mitigated by PPO clip) | Lower |

---

## Project layout

```
BlueberryMCAgent/
├── minescript/
│   └── blueberry_bridge.py   ← runs inside Minecraft via \blueberry_bridge
├── trainer/
│   ├── server.py             ← Flask HTTP server + online training loop
│   ├── config.py             ← all hyper-parameters and seeds
│   ├── reward.py             ← shaped reward function
│   └── ppo/
│       ├── __init__.py
│       ├── model.py          ← CNN + MLP actor-critic network
│       ├── buffer.py         ← rollout buffer with GAE
│       ├── update.py         ← PPO clipped objective
│       └── utils.py          ← obs → state vector, screenshot decode
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

### 5 — Copy the MineScript bridge into Minecraft

```bat
copy minescript\blueberry_bridge.py "%APPDATA%\.minecraft\minescript\blueberry_bridge.py"
```

MineScript looks for scripts in `%APPDATA%\.minecraft\minescript\`.

---

## Running the agent

### Step A — Start the trainer server

Open a terminal **in the project root** (with the virtual environment active):

```bat
.venv\Scripts\activate
python trainer/server.py
```

You should see:

```
10:00:00 [INFO] Using device: cuda
10:00:00 [INFO] BlueberryMCAgent trainer starting on http://127.0.0.1:25566
10:00:00 [INFO] Run directory: C:\...\BlueberryMCAgent\runs\20240415_100000
```

Leave this window open.

### Step B — Launch Minecraft

1. Open the Minecraft Launcher.
2. Select the **Fabric 1.21.11** profile and click **Play**.
3. Load or create a **Survival** world.  
   > **Tip:** Start in **Peaceful** difficulty — the agent learns faster without
   > hostile mobs.

### Step C — Start the bridge in-game

Open the MineScript console (default key: `\`) and run:

```
\blueberry_bridge
```

The agent will start acting immediately.  
Watch the trainer terminal for PPO update messages every 64 steps.

### Step D — Check training status

In a browser (or `curl`) visit:

```
http://127.0.0.1:25566/status
```

### Step E — Stop the agent

In-game MineScript console:

```
\killjob
```

The trainer server keeps running; you can restart the bridge any time.  
Press **Ctrl+C** in the trainer terminal to stop the server.

---

## Action space reference

| Index | Name | Effect |
|---|---|---|
| 0 | noop | Stand still |
| 1 | forward | Walk forward for one step (0.25 s) |
| 2 | turn_left | Rotate yaw −5° |
| 3 | turn_right | Rotate yaw +5° |
| 4 | look_up | Decrease pitch by 4° |
| 5 | look_down | Increase pitch by 4° |
| 6 | attack | Left-click (break block / hit entity) |
| 7 | use | Right-click (place block / open menu) |
| 8 | jump | Jump |

All key presses are released at the end of every step.

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

### Want to resume training from a checkpoint?
Checkpoints are saved to `runs/<id>/checkpoints/ckpt_<N>.pt` every 50 updates.
Load one with:

```python
import torch
from trainer.ppo import ActorCritic
model = ActorCritic()
ckpt = torch.load("runs/20240415_100000/checkpoints/ckpt_000050.pt")
model.load_state_dict(ckpt["model_state"])
```
