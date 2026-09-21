# Minecraft Survival Gym

> A Gymnasium environment for training reinforcement learning agents to survive in Minecraft Java, with visual observations, full player controls, and keyboard/mouse demonstration recording.

Minecraft Survival Gym connects Python policies to Minecraft Java 1.21 through a Fabric client mod and exposes the game as `minecraft_gym/MinecraftSurvival-v0`. It is designed as an experimental foundation for reinforcement learning, imitation learning, DAgger, and future VLM/LLM-driven objectives and rewards.

> [!IMPORTANT]
> This is an independent experimental project. It is not an official Minecraft product and is not affiliated with Mojang or Microsoft.

## Features

- Gymnasium-compatible `reset()`, `step()`, `render()`, and `close()` API;
- RGB observations combined with structured player state;
- movement, camera, combat, item use, hotbar, and GUI actions;
- controlled simulation time with an exact number of ticks per action;
- agent, human, and DAgger control modes;
- atomic trajectory recording for behavior cloning and imitation learning;
- deterministic mock backend for development without Minecraft;
- pluggable reward functions decoupled from the game loop.

## Architecture

```text
RL policy / Human operator
            │
            ▼
 MinecraftSurvivalEnv
      (Gymnasium)
            │  Local TCP / JSON
            ▼
     Fabric bridge mod
            │
            ▼
 Minecraft Java — Survival
            │
            └── RGB + vitals + pose + inventory + events
```

The bridge listens only on `127.0.0.1:25570`. Requests are processed sequentially to preserve the relationship between each action, the executed game ticks, and the returned observation.

## Project status

The environment has been validated end to end against a real Minecraft client:

- Python-to-Fabric handshake;
- `128 × 128 × 3` RGB capture;
- exact four-tick advancement per action;
- player movement and camera rotation;
- DAgger execution;
- human episode recording;
- Gymnasium environment checker compliance;
- automated Python test suite.

## Requirements

- Python 3.10 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- JDK 21 — not only a JRE, and not Java 8 or 17;
- Minecraft Java Edition 1.21;
- Fabric Loader;
- Fabric API compatible with Minecraft 1.21.

## Installation

### 1. Clone the repository and install Python dependencies

```bash
git clone https://github.com/rlsgarcia-code/Minecraft-Survival-Gym.git
cd Minecraft-Survival-Gym
uv sync --extra dev
```

Verify the Gymnasium API without launching Minecraft:

```bash
uv run python scripts/smoke_env.py --backend mock --steps 20
```

### 2. Select JDK 21

Check the active Java version:

```bash
java -version
```

The first line must report Java 21. On macOS with Homebrew:

```bash
brew install openjdk@21
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

To keep this configuration across terminal sessions, add the two `export` lines to `~/.zshrc`.

### 3. Build the Fabric mod

```bash
cd fabric
./gradlew build
cd ..
```

The mod artifact is written to:

```text
fabric/build/libs/minecraft-gym-bridge-0.1.0.jar
```

## Running the real environment

The real backend requires **two terminals plus the Minecraft window**:

| Component | Purpose | Must remain open? |
|---|---|---|
| Terminal 1 | launches Minecraft with the Fabric bridge | yes |
| Minecraft window | hosts the loaded single-player world | yes |
| Terminal 2 | runs the Python Gymnasium environment | while the agent is running |

Starting Minecraft is not enough: you must enter a single-player world and wait until the player HUD and terrain are visible before calling `env.reset()`.

### One-command development launcher

The helper script starts both Terminal 1 services in the background: the
Minecraft Fabric development client and JupyterLab with the test notebook.

```bash
./scripts/run_dev.sh
```

The script detects a Homebrew JDK 21 installation automatically, stores PID
files under `.minecraft-gym/`, opens JupyterLab, and writes service logs to:

```text
.minecraft-gym/logs/minecraft.log
.minecraft-gym/logs/jupyter.log
```

After it starts, enter a single-player Survival world in the Minecraft window.
The script cannot select a world on your behalf unless a development save name
is explicitly provided:

```bash
MINECRAFT_GYM_QUICK_PLAY_WORLD=MyWorld ./scripts/run_dev.sh
```

`MyWorld` must already exist under `fabric/run/saves/`. To inspect what would
be launched without starting processes:

```bash
./scripts/run_dev.sh --dry-run
```

Stop both Minecraft and JupyterLab with:

```bash
./scripts/stop_dev.sh
```

The stop script terminates only the process trees recorded by the launcher and
preserves the logs for debugging.

### Option A — development client

This is the quickest way to run the project during development.

#### Terminal 1: start Minecraft

From the repository root:

```bash
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

cd fabric
./gradlew runClient
```

Leave Terminal 1 running. A separate Minecraft window will open.

#### Minecraft window: open a world

1. Click **Singleplayer**.
2. Create or open a world.
3. Make sure the player is in **Survival** mode.
4. Wait until the terrain and player HUD are fully visible.
5. Leave Minecraft open inside the world.

The title screen does not satisfy this requirement. If Python connects while Minecraft is still at the title screen, the bridge returns:

```text
Open a single-player world before using the bridge
```

#### Terminal 2: run the Gym environment

Open a new terminal and return to the repository root:

```bash
cd /path/to/Minecraft-Survival-Gym
uv run python scripts/smoke_env.py --steps 20
```

Do not close Terminal 1 or the Minecraft window while this command is running.

A successful run starts with output similar to:

```text
reset tick=... seed=... rgb=(128, 128, 3)
step=1 tick=... reward=... terminated=False truncated=False
```

### Option B — regular Minecraft installation

1. Install Fabric Loader for Minecraft 1.21.
2. Place Fabric API in that installation's `mods` directory.
3. Copy `fabric/build/libs/minecraft-gym-bridge-0.1.0.jar` into the same `mods` directory.
4. Launch Minecraft with the Fabric profile.
5. Enter a single-player Survival world and wait for it to finish loading.
6. Run the Python command from Terminal 2 as shown above.

## Test notebook

The repository includes a guided notebook that validates observation shapes,
renders an RGB frame, executes a controlled action, checks tick advancement,
and optionally connects to the real Fabric bridge:

[`output/jupyter-notebook/minecraft-survival-gym-quickstart.ipynb`](output/jupyter-notebook/minecraft-survival-gym-quickstart.ipynb)

Launch it from the repository root:

```bash
uv run --extra notebook jupyter lab \
  output/jupyter-notebook/minecraft-survival-gym-quickstart.ipynb
```

The notebook uses the mock backend by default, so **Run All** is safe without
Minecraft. For the real test, keep Terminal 1 running `./gradlew runClient`,
enter a loaded single-player world, and then set `REAL_BACKEND = True` in the
notebook.

## Basic usage

```python
import gymnasium as gym
import minecraft_gym  # registers the environment

env = gym.make(
    "minecraft_gym/MinecraftSurvival-v0",
    backend="socket",
    width=128,
    height=128,
    frame_skip=4,
    max_episode_steps=9_000,
    control_mode="agent",
    render_mode="rgb_array",
)

observation, info = env.reset(seed=42)

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Use `backend="mock"` for fast and deterministic tests that do not require Minecraft.

`MinecraftSurvival-v0` remains available as a compatibility alias. New code
should use the namespaced ID to avoid collisions with other third-party
environment packages.

## Action space

The action space is a 15-component `gymnasium.spaces.MultiDiscrete`:

```text
MultiDiscrete([3, 3, 2, 2, 2, 2, 2, 2, 2, 5, 5, 10, 5, 5, 3])
```

The value `0` is `NOOP` for every component.

| Index | Control | Values |
|---:|---|---|
| 0 | Strafe | idle, left, right |
| 1 | Movement | idle, forward, backward |
| 2 | Jump | no, yes |
| 3 | Sprint | no, yes |
| 4 | Sneak | no, yes |
| 5 | Attack | no, yes |
| 6 | Use | no, yes |
| 7 | Inventory | keep, toggle |
| 8 | Drop item | no, yes |
| 9 | Horizontal camera | neutral, ±6°, ±18° |
| 10 | Vertical camera | neutral, ±6°, ±18° |
| 11 | Hotbar | keep or select slots 1–9 |
| 12 | Horizontal GUI cursor | neutral, slow, or fast in either direction |
| 13 | Vertical GUI cursor | neutral, slow, or fast in either direction |
| 14 | GUI click | none, primary, secondary |

The canonical `ControlState` retains continuous mouse deltas. Human demonstrations therefore preserve more information than the discrete projection used by the initial policy space.

## Observation space

The environment returns a `gymnasium.spaces.Dict`:

| Field | Type and shape | Contents |
|---|---|---|
| `rgb` | `uint8[H, W, 3]` | rendered RGB frame |
| `vitals` | `float32[8]` | health, hunger, armor, air, XP, and body flags |
| `pose` | `float32[10]` | position, velocity, camera, ground contact, and fall state |
| `inventory_ids` | `int32[36]` | item registry identifiers |
| `inventory_counts` | `int32[36]` | item count for each slot |
| `equipped_slot` | `Discrete(9)` | selected hotbar slot |
| `ui_mode` | `Discrete(8)` | gameplay, inventory, container, chat, death, or other screen |

The `info` dictionary includes:

- current tick and actual world seed;
- events such as damage, death, and inventory increases;
- individual reward terms;
- `policy_action`, `human_action`, and `executed_action`;
- `control_source`, indicating whether the agent or a human controlled the step.

## Control modes

| Mode | Behavior |
|---|---|
| `agent` | executes only the action supplied by the policy |
| `human` | ignores the policy action and records keyboard/mouse input |
| `dagger` | uses the policy by default and accepts human intervention |

### Recording demonstrations

Keep Minecraft focused and inside a loaded world. In Terminal 2, run:

```bash
uv run python scripts/record_human.py \
  --mode human \
  --steps 9000 \
  --output datasets/demonstrations
```

To collect human corrections during policy execution:

```bash
uv run python scripts/record_human.py \
  --mode dagger \
  --steps 9000 \
  --output datasets/dagger
```

Each episode produces a compressed `.npz` archive and a `.json` metadata file. Episodes are committed atomically to reduce the chance of partially written datasets.

Main arrays:

| Field | Purpose |
|---|---|
| `actions` | action originally submitted to the environment |
| `policy_actions` | policy action projected onto the discrete action space |
| `human_actions` | human action projected onto the discrete action space |
| `executed_actions` | action actually executed by the game |
| `policy_controls` | continuous canonical policy controls |
| `human_controls` | keyboard state and continuous mouse deltas |
| `executed_controls` | canonical controls actually applied |
| `human_action_present` | mask indicating human action availability/intervention |

> [!NOTE]
> Locking the screen does not prevent Python-only tests or training, but Minecraft may stop rendering. RGB capture and physical keyboard/mouse demonstrations require an unlocked graphical session.

## Rewards

The default reward is deliberately conservative:

- a small reward for remaining alive;
- a penalty proportional to health lost;
- a larger penalty for death.

A different strategy can be injected without modifying the environment:

```python
from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.rewards import RewardResult


class MyReward:
    def __call__(self, previous, transition):
        collected = sum(
            event.get("count", 0)
            for event in transition.events
            if event.get("type") == "inventory_increased"
        )
        return RewardResult(
            total=float(collected),
            terms={"collected_items": float(collected)},
        )


env = MinecraftSurvivalEnv(reward_function=MyReward())
```

This separation makes it possible to experiment with VLM/LLM-generated semantic rewards, intermediate objectives, and knowledge retrieved from survival guides without coupling those experiments to the Minecraft bridge.

## Time and episode semantics

During a Gym session, the bridge freezes the integrated server and advances exactly `frame_skip` ticks for each `step()` call.

- `terminated=True`: the player died;
- `truncated=True`: an operational limit such as `max_episode_steps` was reached.

### Reset behavior

The current version implements a **soft reset**. It:

- forces Survival mode;
- returns the player to the session's initial position;
- clears inventory and status effects;
- restores health, hunger, and saturation;
- restores time and weather.

The reset does not yet rebuild modified blocks or change the seed of the currently loaded world. `requested_seed_matches_world` reports whether the requested seed matches the actual world seed. For experiments requiring identical terrain, start each batch from a clean copy of the save.

## Environment configuration

| Parameter | Default | Description |
|---|---:|---|
| `backend` | `"socket"` | real backend or `"mock"` |
| `host` | `127.0.0.1` | bridge address |
| `port` | `25570` | local TCP port |
| `bridge_timeout` | `30.0` | connection timeout in seconds |
| `width` | `128` | RGB observation width |
| `height` | `128` | RGB observation height |
| `frame_skip` | `4` | ticks executed per action |
| `max_episode_steps` | `9000` | internal episode limit |
| `control_mode` | `"agent"` | `agent`, `human`, or `dagger` |
| `render_mode` | `None` | use `"rgb_array"` for `render()` |
| `reward_function` | sparse survival | pluggable reward strategy |

## Troubleshooting

### `Gradle requires JVM 17 or later` / `configured to use JVM 8`

Gradle found an older Java installation. Select JDK 21 before running the wrapper:

```bash
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
java -version

cd fabric
./gradlew build
```

### `ConnectionRefusedError: [Errno 61] Connection refused`

The Python package could not find the bridge at `127.0.0.1:25570`. Confirm that:

1. Terminal 1 is still running `./gradlew runClient`, or Minecraft was launched from a Fabric installation containing the bridge mod;
2. the Minecraft process has finished starting;
3. no other process is using port `25570`.

### `Open a single-player world before using the bridge`

Python successfully connected to the mod, but Minecraft is still at the title screen or not fully inside a world. In the Minecraft window, click **Singleplayer**, load or create a world, wait for the player HUD and terrain to appear, and then rerun the Python command from Terminal 2.

The `--backend mock` smoke test neither launches Minecraft nor tests the Fabric connection.

## Development

Run the Python test suite:

```bash
uv run pytest
```

Build and validate the Fabric mod:

```bash
cd fabric
./gradlew build
```

Repository structure:

```text
src/minecraft_gym/       environment, actions, rewards, recorder, and transport
fabric/                  Fabric mod and local bridge
scripts/smoke_env.py     real or simulated smoke test
scripts/record_human.py  human demonstration collection
scripts/run_dev.sh       start Minecraft and JupyterLab
scripts/stop_dev.sh      stop the development session
tests/                   Gym contract and protocol tests
docs/protocol.md         TCP protocol specification
```

## Roadmap

- hard resets and automatic world restoration;
- vectorized environments and multiple Minecraft instances;
- hierarchical and temporal action wrappers;
- behavior cloning and reinforcement learning baselines;
- VLM-based reward evaluation and generation;
- language-model subgoal planning with survival-guide retrieval;
- demonstration inspection, replay, and curation tools.

## Protocol

The bridge specification is available in [`docs/protocol.md`](docs/protocol.md).

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the development workflow. When reporting a problem, include the Minecraft,
Fabric Loader, Fabric API, Java, and operating system versions, along with
minimal reproduction steps.
