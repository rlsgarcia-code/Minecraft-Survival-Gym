# Minecraft Survival Gym

[![PyPI](https://img.shields.io/pypi/v/minecraft-gym.svg)](https://pypi.org/project/minecraft-gym/)
[![CI](https://github.com/rlsgarcia-code/Minecraft-Survival-Gym/actions/workflows/ci.yml/badge.svg)](https://github.com/rlsgarcia-code/Minecraft-Survival-Gym/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/minecraft-gym.svg)](https://pypi.org/project/minecraft-gym/)
[![License](https://img.shields.io/github/license/rlsgarcia-code/Minecraft-Survival-Gym)](LICENSE)

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
- Minecraft Java Edition 1.21;
- Fabric Loader;
- Fabric API compatible with Minecraft 1.21.

[`uv`](https://docs.astral.sh/uv/) is required only for development from a
source checkout. It is not required when installing the published Python
package with `pip`. The published mod runs with the Java 21 runtime selected by
the Minecraft Launcher; a full JDK 21 is required only to build or run the
Fabric development client from source.

## Installation

Choose **one** route. The published package opens your regular Minecraft
Launcher; a source checkout runs a separate Fabric development client. Do not
run `minecraft-gym start` and `./gradlew runClient` as consecutive setup steps.

| Route | Use it when | Minecraft command | Where worlds are saved |
|---|---|---|---|
| [Published release](#published-release-regular-launcher) | You want to use the installed Python package | `minecraft-gym setup`, then `minecraft-gym start` (version 0.3.2+) | Your Launcher installation's game directory |
| [Source checkout](#development-installation-from-source) | You are developing this repository | `cd fabric && ./gradlew runClient` | `fabric/run/saves/` |

### Published release (regular Launcher)

#### One-time setup command (version 0.3.2 and later)

The normal Launcher flow is:

```bash
python -m pip install --upgrade minecraft-gym==0.3.2
minecraft-gym setup
minecraft-gym doctor
```

`minecraft-gym setup` installs Fabric Loader for Minecraft 1.21, the matching
Fabric API, and the bridge mod into the official Minecraft Launcher's game
directory. The bridge 0.1.2 is included and verified inside the Python wheel;
users do not need Git, Gradle, a source checkout, or a separate bridge download.
The setup command leaves existing worlds untouched and refuses conflicting mod
versions rather than deleting them. Install and open the official Launcher once
before running it. Java 21 must be available for the one-time Fabric Loader
installation. You can preview the target with `minecraft-gym setup --dry-run`
or choose a nondefault Launcher directory with
`minecraft-gym setup --game-dir /path/to/minecraft`.

Running `setup` again upgrades the project's own bridge mod when a supported
older release is present. The previous `.jar` is preserved outside `mods` in
the `.minecraft-gym-backup` directory.

The command cannot create a Minecraft account or click through the Launcher:
after setup, run `minecraft-gym start`, select the **Minecraft 1.21 Fabric**
profile, click **Play**, create or enter a single-player Survival world, wait
for the terrain and HUD, then press Enter in the terminal and choose recording
or an agent. No repository checkout or Gradle command is needed.

#### Public command-line interface

The installed package exposes reusable operations rather than project-specific
scripts:

```text
minecraft-gym setup                 install/upgrade Fabric and the packaged bridge
minecraft-gym doctor                inspect Launcher, mods, saves, focus pause and bridge
minecraft-gym world snapshot        create a hash-verified template from a closed save
minecraft-gym world clone           create a fresh verified save from a template
minecraft-gym start record          record a keyboard/mouse demonstration
minecraft-gym start agent           run an external Python policy
```

For example, create a reusable template and a disposable run without copying
world directories by hand:

```bash
minecraft-gym world snapshot \
  "$HOME/Library/Application Support/minecraft/saves/My Survival World" \
  --output "$HOME/minecraft-gym-templates/survival-001"

minecraft-gym world clone \
  "$HOME/minecraft-gym-templates/survival-001"
```

Both operations reject unsafe overwrites. Snapshot requires a closed source
world; clone requires that no Launcher world is currently open and prints the
new save name and SHA-256 as JSON. Use `--saves-dir` for a nonstandard Launcher.

#### Earlier PyPI release (version 0.2.0)

1. Install Minecraft **Java Edition 1.21** and [Fabric Loader for 1.21](https://docs.fabricmc.net/players/installing-fabric). The installer must create a Fabric profile in the Minecraft Launcher.
2. Download [Fabric API for Minecraft 1.21](https://docs.fabricmc.net/players/installing-mods) and the [`minecraft-gym-bridge-0.1.0.jar`](https://github.com/rlsgarcia-code/Minecraft-Survival-Gym/releases/download/v0.1.0/minecraft-gym-bridge-0.1.0.jar) release asset. Put **both `.jar` files** in that installation's `mods` directory. The default macOS location is `~/Library/Application Support/minecraft/mods/`; a custom Launcher game directory has its own `mods` folder.
3. Install the Python package in the environment where you will run your agent:

   ```bash
   pip install minecraft-gym==0.2.0
   ```

4. Start the interactive helper in a terminal:

   ```bash
   minecraft-gym start
   ```

   On macOS it opens the Minecraft Launcher. Select the **1.21 Fabric** profile,
   click **Play**, create or open a **single-player Survival** world, and wait
   until terrain and the player HUD are visible. The title screen is not enough.
   Return to the terminal and press Enter when prompted. Choose **1** to record
   keyboard/mouse actions or **2** to run an agent script. Keep Minecraft
   focused while recording.

To select a mode without the final menu, use:

```bash
minecraft-gym start record --output datasets/demonstrations --steps 9000
minecraft-gym start record --output datasets/visual-16x9 --steps 9000 \
  --width 320 --height 180 --frame-skip 4 \
  --metadata collector=human-01 --metadata split=train
minecraft-gym start agent path/to/agent.py
```

Run **one** of these commands per session, not both. The agent script must
create and close its own Gymnasium environment; `start agent` does not supply
a built-in policy. On Linux, `start` uses `minecraft-launcher` if it is on
`PATH`; otherwise pass `--launcher`. If Minecraft is already open, add
`--no-launch` after `start`. The Python package does not install Minecraft
itself; `minecraft-gym setup` installs Fabric Loader, Fabric API, and the
bridge into an existing official Launcher directory.

For agent and external human-recording loops, press `F3+P` until Minecraft
shows `Pause on lost focus: disabled`. Otherwise the integrated server can
pause while the Python process is active and `step()` will wait indefinitely.

Recording uses `reset()` before its first step. This soft reset clears the
player inventory, restores vitals, time, and weather, and moves the player to
the session anchor. It does not rebuild modified blocks.

To check the connection without creating a file:

```bash
python -c 'import gymnasium as gym, minecraft_gym; env = gym.make("minecraft_gym/MinecraftSurvival-v0"); observation, info = env.reset(seed=42); print(observation["rgb"].shape, info["tick"]); env.close()'
```

This command only checks a running world with the bridge mod loaded; it does
not launch Minecraft.
To stop a `start` session, press `Ctrl+C` and close Minecraft normally.
The repository's `stop_dev.sh` command manages only processes started by the
source-development launcher.

### Development installation from source

#### 1. Clone the repository and install Python dependencies

```bash
git clone https://github.com/rlsgarcia-code/Minecraft-Survival-Gym.git
cd Minecraft-Survival-Gym
uv sync --extra dev
```

Verify the Gymnasium API without launching Minecraft:

```bash
uv run python scripts/smoke_env.py --backend mock --steps 20
```

#### 2. Select JDK 21

Minecraft 1.21 development requires JDK 21. On macOS, `fabric/gradlew` now selects an installed JDK 21 for that command even if the shell's `JAVA_HOME` points to an older JDK. It checks the macOS Java registry and Homebrew's `openjdk@21`. Verify the JVM used by Gradle with `cd fabric && ./gradlew --version`; the `JVM` line should report 21.

If JDK 21 is not installed, install it with Homebrew:

```bash
brew install openjdk@21
export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

The `export` lines are optional for direct Gradle commands on macOS, but are useful when other Java commands also need JDK 21. The wrapper does not modify `~/.zshrc`.

#### 3. Run the development client

From the repository root, start the development client in Terminal 1:

```bash
cd fabric
./gradlew runClient
```

`runClient` builds and loads the bridge mod automatically. You do **not** need
to install the published bridge `.jar` into this client's `mods` folder, and
you do not need to run `minecraft-gym start` for this route. When Minecraft
opens, create or select a single-player **Survival** world and wait until the
terrain and player HUD are visible. Keep this terminal and Minecraft open.

In Terminal 2, from the repository root, verify the real connection:

```bash
uv run python scripts/smoke_env.py --steps 20
```

The development world's save directory is `fabric/run/saves/`. If you also
want a standalone mod `.jar`, build it separately:

```bash
cd fabric
./gradlew build
```

Install that exact verified local build into the official Launcher instead of
downloading a release artifact:

```bash
uv run minecraft-gym setup \
  --bridge-jar fabric/build/libs/minecraft-gym-bridge-0.1.2.jar
```

The mod artifact is written to:

```text
fabric/build/libs/minecraft-gym-bridge-0.1.2.jar
```

## Running the real environment

The real backend always requires a running Minecraft window with a loaded
single-player world. The number of terminals depends on the installation:

| Installation | Start Minecraft | Start the Python environment |
|---|---|---|
| Published release (`pip`) | `minecraft-gym start` opens the launcher; select Fabric and click Play | `start record` captures input; `start agent path/to/agent.py` runs your script |
| Source checkout | Terminal 1: `cd fabric && ./gradlew runClient` | Terminal 2: agent, smoke test, or notebook |

Starting Minecraft is not enough: you must enter a single-player world and wait until the player HUD and terrain are visible before calling `env.reset()`.

### One-command development launcher

The helper script starts the Minecraft Fabric development client and JupyterLab
with the test notebook in the background.
It is available only in a cloned source checkout and is not installed by
`pip`.

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

From the repository root (after installing JDK 21 as described above):

```bash
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
- structured events: aggregate `inventory_delta`, `recipe_unlocked`,
  `vital_delta`, `damage`, `death`, and `ui_changed`;
- canonical item/recipe identifiers such as `minecraft:oak_log` in events,
  while the numeric inventory arrays remain available for compatibility;
- `privileged_context` for dataset labelling: day time, rain, light, sky
  exposure, submersion, and nearby hostile summary;
- individual reward terms;
- `policy_action`, `human_action`, and `executed_action`;
- `control_source`, indicating whether the agent or a human controlled the step.

`privileged_context` and the structured state are instrumentation. A visual-only
agent should filter them before policy inference. For learning from UI and HUD
at the aspect ratio used by the game, the research collection profile uses
`width=320`, `height=180`; the environment defaults remain small for backwards
compatibility.

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
