# Navigation Academy v3

The Navigation Academy is an isolated benchmark for learning embodied travel
before a policy is allowed to interact with an open Minecraft world. It is
owned by `Minecraft-Survival-Gym`: BrincaBot may start runs and display them,
but it must not carry a private copy of the task, reward, reset, or dataset
logic.

## Safety boundary

- `brincadeiras` is never a training target or reset target.
- A datapack can only be installed when both the candidate training save and
  the family save are supplied, exist, and have disjoint paths.
- Bundle creation never opens or edits a Minecraft save.
- The datapack has no load hook. Arena creation begins only after the explicit
  `/function minecraft_gym:academy/setup` command inside an isolated world.
- Training episodes declare `environment_mode=isolated_training` and a world
  snapshot identifier in the canonical dataset v2 manifest.

## Primary benchmark: random target navigation

The default comparison command now trains `target_navigation` through five
ordered curriculum phases. Episodes remain interleaved inside each phase:

1. orientation and movement: 80% easy, 20% moderate;
2. small obstacles: 20% easy, 60% moderate, 20% hard;
3. complex navigation: 10% easy, 20% moderate, 50% hard, 20% expert;
4. robustness: 10% easy, 15% moderate, 25% hard, 25% expert, 25% mixed;
5. consolidation: 20% in each difficulty.

Every deterministic seed varies target distance and lateral offset, obstacle
density, terrain and biome. Training and held-out evaluation use disjoint
schedules and arena hashes.

Available biome dynamics are:

- plains for speed and orientation;
- jungle with breakable tree obstacles and optional longer detours;
- snowy terrain with powder-snow delay, sinking exposure and cold damage;
- rivers with slower swimming traversal;
- volcanic terrain where lava is highly costly and damaging;
- mixed arenas combining these dynamics.

The target is known initially and persists in the map pyramid when it leaves
the current gnomonic view. Breaking and construction are mandatory affordance
strata: full-width walls and gaps make selected episodes unsolvable without
the declared interaction. A combined episode requires separate break and
place decisions; simultaneous mouse buttons cannot satisfy both. Interaction
receives credit only when it reduces the
estimated remaining travel time.

The previous five-stage academy remains available as
`--benchmark legacy_curriculum`. It is a regression suite, not the score of
the deployable target-navigation policy:

| Stage | Capability | Required interaction |
|---|---|---|
| `flat` | A to B on level ground | walk, orient, sprint |
| `climb` | one-block elevation changes | jump while moving |
| `break` | blocked route | break only the obstructing blocks |
| `build` | gap and two-block rise | place a bridge block and a stair block |
| `combined` | composed route | move, break, bridge, climb, recover |

The same `ArenaBlueprint` generates both the deterministic reference
environment and the Minecraft datapack commands. Every blueprint has a
logical SHA-256 recorded by the task and bundle manifests, so a world arena
cannot silently drift away from the task used by training and evaluation.

## Observations and privileged evaluation

The current public state is `minecraft-academy-world-map-window-time/v4`. It
combines two spatial observations without confusing persistent memory with
the currently exposed crop:

- a north-up `65 × 65` map window with exactly one Minecraft block per cell;
- an `18 × 32` gnomonic lattice for the current field of view.

The complete accumulated X/Z evidence remains in the actor-scoped sparse world
map. Moving the window never deletes cells outside its 32-block radius; they
reappear at the exact metric location when the actor returns. The map window
has evidence, visited, relative elevation, material, solid-stack height, head
clearance, traversability and target/direction channels. Unknown cells stay
zero and the actor occupies the centre. The window is world-aligned instead of
being resampled after every camera rotation, preserving the one-cell/one-block
metric. Orientation remains body state.

When the objective lies outside the window, its world bearing is intersected
with the window border and the target channel marks that border cell. The
policy therefore knows which way the objective lies without receiving an
absolute coordinate or pretending that distant geometry is locally known.

The encoder preserves the full map resolution, resizes the gnomonic lattice,
and concatenates them into one joint tensor. The
linear SARSA encoder flattens that tensor; Expected SARSA applies its learned
convolution to the joint spatial tensor. Inventory remains a small body-state
input. The former `8 × 14 × 9` global grid stays
in the environment only for loading and evaluating historical checkpoints; no
new modern policy is trained from it.

Absolute position, absolute goal, shortest-route cost, and world deltas appear
only under `privileged_evaluator` and `privileged_context`. Neither policy
encoder accepts those dictionaries. The dataset recorder stores them as
privileged evaluation data, never as policy inputs.

The current `minecraft-control-camera-keyboard/v3` action is
`(delta_longitude, delta_latitude, strafe, move, jump, sprint, sneak, attack, use)`.
Longitude spans the continuous relative range `[-180°, 180°]`; latitude spans
`[-90°, 90°]`. They are genuine mouse displacements, not labels such as
left/right/up/down and not an automatic bearing lock on the waypoint. A/D,
S/W and the five buttons are separate actor heads, so the policy may look in
one direction while moving forward, backward or sideways, including diagonal
and modifier combinations, without enumerating a Cartesian action catalogue.
Movement remains Minecraft-relative to camera orientation, but camera motion
does not imply movement. The waypoint remains state: the actor may look
elsewhere or route around an obstacle while the target persists in the map
window or at its directional border.

The actor produces a circular von Mises distribution over longitude, a
bounded squashed Gaussian over latitude, categorical distributions for W/S
and A/D, and Bernoulli distributions for the five buttons. `attack` mines the
centered breakable block and `use` places the selected construction block. A separate critic
estimates `V(s)` in the physically possible
episodic interval `[-40, +20]`. Training is episodic and strictly on-policy: the
current actor generates one complete trajectory, its observed undiscounted
returns train the critic, and the bounded return advantage trains the actor
with entropy regularization. There is no replay and no value bootstrap from a
second policy. Evaluation freezes learning and uses the camera means and most
likely independent W/S, A/D and button values. The older 25-chord, 12-action
and factorized checkpoints
remain readable for comparison; the obsolete unbounded actor-critic v1
checkpoint is deliberately incompatible with v2.

The fast observation is intentionally structured. It validates learnability,
reward accounting, resets, and checkpointing; it is not a substitute for the
Fabric RGB domain. A structured Q checkpoint is marked
`deployable_on_rgb=false`. Visual deployment requires training or adaptation
with the real Fabric-backed environment.

## Reward

The reward ranks successful trajectories by Minecraft time until the target. For
normalized elapsed time `u`, cumulative cost is the bounded asymptotic curve
`F(u) = 20 × (1 - exp(-3u²))`. A transition receives the negative increment
`-(F(u_after) - F(u_before))`; when no cell displacement occurs the increment
is multiplied by `1.5`. The square makes delay increasingly expensive before
the nominal horizon, while the negative exponential guarantees `F(u) < 20`
for every finite duration and `F(u) → 20` even when an action consumes far
more than four ticks. Reaching the target ends the episode and adds a terminal
bonus of `+20`; a timeout adds `-5`. This makes even a late arrival explicitly
better than failing while the time cost still orders successful routes from
fastest to slowest. A death is charged the remaining base cost through the episode
limit so that dying early cannot avoid time cost. Terrain, blocked motion,
rotation, damage and world changes remain events and evaluator diagnostics,
but contribute no separate reward term.

The joint policy observes `elapsed_fraction` and uses `gamma=1.0`, keeping the
finite-horizon shortest-time objective undiscounted. Actor-critic v2 fits the
realized remaining return at every visited state, rather than recursively
bootstrapping a value estimate. Critic predictions are bounded to `[-40, +20]`
and the actor advantage to `[-10, 10]`; this prevents an estimator from
creating magnitudes that the reward contract cannot produce. The canonical task
contracts returned by `target_navigation_task_suite()` expose the exponential
parameters and stationary multiplier. Success remains a termination predicate
and evaluation metric.

## Create and install the real arena

Create a bundle without touching any world:

```bash
PYTHONPATH=src python -m minecraft_gym.cli academy create \
  --output training-worlds/navigation-academy-v1 \
  --seed 42
```

Install it into a closed, disposable save. The family path is mandatory only
to prove that the destination is different:

```bash
PYTHONPATH=src python -m minecraft_gym.cli academy install \
  training-worlds/navigation-academy-v1 \
  --world /path/to/isolated-training-save \
  --live-world /path/to/brincadeiras
```

Open only the isolated save, then run:

```text
/reload
/function minecraft_gym:academy/setup
/function minecraft_gym:academy/attach
```

Reset a stage before each real episode with one of:

```text
/function minecraft_gym:academy/reset_flat
/function minecraft_gym:academy/reset_climb
/function minecraft_gym:academy/reset_break
/function minecraft_gym:academy/reset_build
/function minecraft_gym:academy/reset_combined
```

The reset function reconstructs the arena, clears effects and inventory,
restores the task items, teleports the tagged trainee, and resets its task
score. For automated real-world training, the episode runner must invoke the
same function through the dedicated training-server control channel before
calling Gym `reset`; it must never send that command to the family server.

## Train and compare movement policies

The original tabular Q baseline remains reproducible:

```bash
PYTHONPATH=src python -m minecraft_gym.cli academy train \
  --output checkpoints/navigation-academy/q-v1.json \
  --episodes-per-stage 1000 \
  --progress-file runs/navigation-academy/q-v1-progress.json \
  --evaluation-episodes 20 \
  --seed 42

PYTHONPATH=src python -m minecraft_gym.cli academy evaluate \
  checkpoints/navigation-academy/q-v1.json \
  --episodes-per-stage 50
```

The trainer adds a decaying count-based novelty bonus to learning updates so
untried body actions are explored. That intrinsic bonus is not written into the
environment reward or dataset reward terms. Later stages reuse the same local
state keys, allowing locomotion learned on flat ground to transfer without
revealing the curriculum label or privileged coordinates.

`--progress-file` is an atomic, machine-readable
`minecraft-academy-training-progress/v1` document. It identifies the current
stage and episode, completed and total episodes, training success, actor and
critic diagnostics, elapsed ticks, evaluation results, recovery checkpoint,
and terminal status. During actor-critic training an atomic checkpoint is
written every 100 completed episodes. It contains actor and critic weights,
both optimizer states, Torch and NumPy RNG state, the deterministic schedule
cursor, reward/success histories and all live diagnostic curves. Restarting
the same output directory rolls back at most 99 episodes and resumes the exact
online policy rather than silently starting over.

BrincaBot's Observer starts this canonical CLI through `/api/academy/runs` and
reads only that public progress contract; it does not duplicate the task or
training implementation. Closing the Observer does not terminate the child
run. If the child disappears while progress still says `running`, the
orchestrator may restart it from the latest atomic checkpoint at most twice;
otherwise it marks both run state and progress as failed with a diagnostic.

This checkpoint is a foundation diagnostic. The production sequence is:

1. validate every task and reward in the structured environment;
2. run the same hashed arenas through the Fabric client;
3. record exact structured bot rollouts with `DatasetRecorderV2`;
4. record aligned RGB rollouts through the visual collector;
5. train the visual motor policy and evaluate on held-out arena seeds;
6. expose status and playback in Observer without moving ownership out of Gym.

Promotion should require at least 80% success over a 20-episode window. Test
and validation seeds must remain disjoint from training seeds.

The comparison runner trains any subset of the available strategies on exactly the
same arena seeds, then evaluates every result on a second, held-out seed set:

```bash
PYTHONPATH=src python -m minecraft_gym.cli academy compare \
  --output-dir checkpoints/navigation-academy/comparison-42 \
  --strategies tabular_q linear_sarsa_lambda expected_sarsa_cnn \
  --episodes-per-stage 25 \
  --evaluation-episodes 20 \
  --epsilon-start 0.40 \
  --epsilon-end 0.05 \
  --progress-file runs/navigation-academy/comparison-42.json \
  --seed 42
```

This command uses `--benchmark target_navigation` by default. To reproduce the
old diagnostic stages, add `--benchmark legacy_curriculum`.

- `tabular_q` is the unchanged hash/table reference.
- `linear_sarsa_lambda` is on-policy SARSA with replacing eligibility traces
  over the flattened metric-window state.
- `expected_sarsa_cnn` is Expected SARSA with a small trainable convolutional
  Q estimator implemented in NumPy.
- `joint_control_expected_sarsa_cnn` is the historical Expected SARSA policy
  over 25 complete keyboard/mouse chords.
- `hybrid_actor_critic` is the continuous-camera, categorical-body on-policy
  actor-critic. It requires the optional `learning` dependencies.

The SARSA variants learn online. Their behaviour policy is epsilon-greedy and
the update is performed after every environment transition. For ordinary
SARSA, the bootstrap action is the actual next action sampled by that same
behaviour policy. Expected SARSA uses the expectation under that epsilon-greedy
policy. Exploration decays linearly inside each curriculum stage and restarts
at `--epsilon-start` for the next stage, so a later task family is not silently
trained with the nearly-greedy tail of an earlier family.

The hybrid actor-critic does not use epsilon. Exploration comes from its
learned camera variance, categorical sampling and entropy term. Consequently
its artifacts record `stochastic_hybrid_actor`; epsilon fields are null rather
than presenting a fictitious epsilon schedule.

The v3 comparison keeps three measurements separate:

- `online_training`: success while weights are changing and exploration is
  active; the recent window is useful for monitoring but is not a deployment
  score;
- `training_seed_evaluation`: frozen, epsilon-zero evaluation on seeds the
  learner actually practiced; this measures consolidation;
- `evaluation`: frozen, epsilon-zero evaluation on disjoint seeds; this
  measures generalization.

Each difficulty stores `reward_curve`, `reward_moving_average`,
`success_moving_average`, and `moving_average_window`. In addition,
`minecraft-academy-chronology/v1` stores every episode in actual execution
order with phase, difficulty, requirement, seed, return and success. It also
publishes the chronological global moving average, an equal-weight rolling
mean over all five difficulties, per-difficulty progress and a recent global
trend. Observer renders those canonical values and does not reconstruct order
by concatenating difficulty buckets.

Observer exposes two distinct training operations. **Train new policy** starts
new parameters. **Continue training** supplies the selected immutable
`movement-policy.json` through `--resume-policy-artifact`, loads its checkpoint,
adds a disjoint deterministic training-seed block, and appends new episode
curves to the inherited history. The child artifact records `parent_policy_id`
and `training_seed_history`; the parent remains unchanged.

With `--stop-when-stable`, each phase uses its episode count as a maximum. It
may stop only during consolidation, after every difficulty has at least 40
episodes, when the absolute slope of the recent moving reward is at most 0.05
and recent success is at least 80% for consecutive checks. A flat,
unsuccessful curve is therefore not classified as convergence.
Removing a policy in Observer removes only the runtime building block and adds
an audit tombstone. Research checkpoints and run manifests remain immutable.

Obstacle positions vary deterministically with the seed. The comparison report
stores every training/evaluation seed and logical arena hash, so the claim of a
fair comparison is auditable.

Each completed strategy also emits `movement-policy.json` using
`minecraft-academy-movement-policy/v2`. This is the deployable building-block
manifest: policy ID, strategy, checkpoint SHA-256, benchmark version,
observation/action contracts, input channels, online curves, training-seed and
held-out metrics, and runtime adapter. BrincaBot
catalogs these artifacts without copying the learner implementation. Selecting
one in the Observer creates a persistent `movement` runtime binding; subsequent
movement and exploration executions record the exact policy version used, and
the previous artifacts remain available for rollback.

The Gym-owned BrincaBot runtime adapter projects its persistent sparse world
memory through the same single-scale metric-window contract. Observer does not
own or duplicate the encoder. Stable, high-confidence cells are cached; a
small displacement rolls the previous crop and fills only its newly exposed
strip plus topology changes. Break/place remain masked unless authoritative
affordance evidence is available. If spatial projection is unavailable, the
runtime reports that condition and uses the existing target controller as a
non-learned fallback rather than fabricating policy inputs.
