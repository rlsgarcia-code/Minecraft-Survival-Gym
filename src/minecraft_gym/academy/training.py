"""Dependency-light Q-learning baseline for validating the academy curriculum.

The checkpoint is deliberately tied to the structured academy observation
domain.  It is a research baseline and curriculum sanity check, not a claim
that the resulting table can be deployed directly on RGB Minecraft frames.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_gym.actions import noop_action

from .arena import (
    AcademyStage,
    NavigationDifficulty,
    NavigationRequirement,
    build_target_arena,
)
from .env import NavigationAcademyEnv
from .learning import (
    BranchingExpectedSarsaCnnAgent,
    ExpectedSarsaCnnAgent,
    JointControlExpectedSarsaCnnAgent,
    LinearSarsaLambdaAgent,
)
from .policy import (
    abstract_action,
    factorized_control_action,
    hybrid_control_action,
    joint_control_action,
)


def _action_library() -> tuple[np.ndarray, ...]:
    actions: list[np.ndarray] = []

    def make(**values: int) -> np.ndarray:
        action = noop_action()
        fields = {
            "strafe": 0,
            "move": 1,
            "jump": 2,
            "sprint": 3,
            "attack": 5,
            "use": 6,
            "yaw": 9,
        }
        for key, value in values.items():
            action[fields[key]] = value
        return action

    actions.extend(
        [
            make(),
            make(move=1),
            make(move=1, sprint=1),
            make(move=1, jump=1),
            make(strafe=1),
            make(strafe=2),
            make(yaw=3),
            make(yaw=4),
            make(attack=1),
            make(use=1),
        ]
    )
    return tuple(actions)


ACTION_LIBRARY = _action_library()


CURRICULUM_PHASES: tuple[dict[str, Any], ...] = (
    {
        "slug": "orientation_movement",
        "label": "Orientacao e movimento",
        "weights": {"easy": 0.80, "moderate": 0.20},
    },
    {
        "slug": "small_obstacles",
        "label": "Pequenos obstaculos",
        "weights": {"easy": 0.20, "moderate": 0.60, "hard": 0.20},
    },
    {
        "slug": "complex_navigation",
        "label": "Navegacao complexa",
        "weights": {
            "easy": 0.10,
            "moderate": 0.20,
            "hard": 0.50,
            "expert": 0.20,
        },
    },
    {
        "slug": "robustness",
        "label": "Robustez",
        "weights": {
            "easy": 0.10,
            "moderate": 0.15,
            "hard": 0.25,
            "expert": 0.25,
            "mixed": 0.25,
        },
    },
    {
        "slug": "consolidation",
        "label": "Consolidacao",
        "weights": {
            "easy": 0.20,
            "moderate": 0.20,
            "hard": 0.20,
            "expert": 0.20,
            "mixed": 0.20,
        },
    },
)


def _write_progress(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "interface": "minecraft-academy-training-progress/v1",
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


class TabularAcademyAgent:
    """Observable-state Q table with no access to evaluator coordinates."""

    def __init__(self, *, learning_rate: float = 0.15, discount: float = 0.98) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.q: dict[str, np.ndarray] = {}
        self.visits: dict[tuple[str, int], int] = {}

    @staticmethod
    def state_key(observation: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(observation["local_voxels"], dtype=np.uint8).tobytes())
        goal = np.rint(np.asarray(observation["goal_hint"]) * 32).astype(np.int8)
        digest.update(goal.tobytes())
        digest.update(np.asarray(observation["inventory"], dtype=np.int16).tobytes())
        # Do not encode the curriculum label: identical local situations must
        # transfer from flat terrain into later stages.  Obstacles and task
        # affordances are already present in local voxels and inventory.
        return digest.hexdigest()[:32]

    def values(self, observation: dict[str, Any]) -> np.ndarray:
        key = self.state_key(observation)
        if key not in self.q:
            self.q[key] = np.zeros(len(ACTION_LIBRARY), dtype=np.float32)
        return self.q[key]

    def choose(
        self,
        observation: dict[str, Any],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        values = self.values(observation)
        if rng.random() < epsilon:
            return int(rng.integers(len(ACTION_LIBRARY)))
        maximum = float(values.max())
        candidates = np.flatnonzero(np.isclose(values, maximum))
        return int(rng.choice(candidates))

    def update(
        self,
        observation: dict[str, Any],
        action_index: int,
        reward: float,
        next_observation: dict[str, Any],
        *,
        done: bool,
    ) -> None:
        current = self.values(observation)
        target = reward
        if not done:
            target += self.discount * float(self.values(next_observation).max())
        current[action_index] += self.learning_rate * (target - current[action_index])

    def exploration_bonus(
        self, observation: dict[str, Any], action_index: int, *, scale: float
    ) -> float:
        key = (self.state_key(observation), action_index)
        visits = self.visits.get(key, 0) + 1
        self.visits[key] = visits
        return scale / float(np.sqrt(visits))

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "interface": "minecraft-academy-tabular-q/v1",
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "action_contract": "minecraft-native-control/v2",
            "observation_contract": "minecraft-academy-structured/v1",
            "metadata": metadata,
            "q": {key: value.tolist() for key, value in sorted(self.q.items())},
        }
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["TabularAcademyAgent", dict[str, Any]]:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if payload.get("interface") != "minecraft-academy-tabular-q/v1":
            raise ValueError("not a Navigation Academy Q checkpoint")
        agent = cls(
            learning_rate=float(payload["learning_rate"]),
            discount=float(payload["discount"]),
        )
        agent.q = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in payload["q"].items()
        }
        if any(value.shape != (len(ACTION_LIBRARY),) for value in agent.q.values()):
            raise ValueError("checkpoint action space is incompatible")
        return agent, dict(payload.get("metadata", {}))


def train_curriculum(
    output: str | Path,
    *,
    episodes_per_stage: int = 1000,
    seed: int = 0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    intrinsic_bonus_scale: float = 0.20,
    progress_path: str | Path | None = None,
    evaluation_episodes: int = 20,
) -> dict[str, Any]:
    if episodes_per_stage < 1:
        raise ValueError("episodes_per_stage must be positive")
    rng = np.random.default_rng(seed)
    agent = TabularAcademyAgent()
    stages: dict[str, dict[str, float | int]] = {}
    total_episodes = episodes_per_stage * len(AcademyStage)
    episode_number = 0
    _write_progress(
        progress_path,
        {
            "status": "running",
            "phase": "training",
            "progress": 0.0,
            "completed_episodes": 0,
            "total_episodes": total_episodes,
            "episodes_per_stage": episodes_per_stage,
            "seed": seed,
            "stages": {},
        },
    )
    for stage_index, stage in enumerate(AcademyStage, start=1):
        successes = 0
        returns: list[float] = []
        env = NavigationAcademyEnv(stage=stage)
        for local_episode in range(episodes_per_stage):
            fraction = episode_number / max(1, total_episodes - 1)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            observation, _ = env.reset(seed=seed + local_episode)
            episode_return = 0.0
            while True:
                action_index = agent.choose(observation, epsilon=epsilon, rng=rng)
                next_observation, reward, terminated, truncated, _ = env.step(
                    ACTION_LIBRARY[action_index]
                )
                done = terminated or truncated
                learning_reward = reward + agent.exploration_bonus(
                    observation,
                    action_index,
                    scale=intrinsic_bonus_scale,
                )
                agent.update(
                    observation,
                    action_index,
                    learning_reward,
                    next_observation,
                    done=done,
                )
                observation = next_observation
                episode_return += reward
                if done:
                    successes += int(terminated)
                    returns.append(episode_return)
                    break
            episode_number += 1
            _write_progress(
                progress_path,
                {
                    "status": "running",
                    "phase": "training",
                    "stage": stage.slug,
                    "stage_index": stage_index,
                    "stage_count": len(AcademyStage),
                    "episode": local_episode + 1,
                    "episodes_per_stage": episodes_per_stage,
                    "completed_episodes": episode_number,
                    "total_episodes": total_episodes,
                    "progress": 0.9 * episode_number / total_episodes,
                    "success_rate": successes / (local_episode + 1),
                    "mean_return": float(np.mean(returns)),
                    "seed": seed,
                    "stages": stages,
                },
            )
        env.close()
        stages[stage.slug] = {
            "episodes": episodes_per_stage,
            "success_rate": successes / episodes_per_stage,
            "mean_return": float(np.mean(returns)),
        }
    checkpoint = agent.save(
        output,
        metadata={
            "seed": seed,
            "episodes_per_stage": episodes_per_stage,
            "stages": stages,
            "policy_input": "structured_academy_only",
            "deployable_on_rgb": False,
            "intrinsic_bonus_scale": intrinsic_bonus_scale,
        },
    )
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "states": len(agent.q),
        "stages": stages,
    }
    evaluation: dict[str, Any] | None = None
    if evaluation_episodes > 0:
        _write_progress(
            progress_path,
            {
                "status": "running",
                "phase": "evaluation",
                "progress": 0.94,
                "completed_episodes": total_episodes,
                "total_episodes": total_episodes,
                "episodes_per_stage": episodes_per_stage,
                "seed": seed,
                "stages": stages,
                "checkpoint": str(checkpoint),
            },
        )
        evaluation = evaluate_checkpoint(
            checkpoint,
            episodes_per_stage=evaluation_episodes,
            seed=seed + 10_000,
        )
        report["evaluation"] = evaluation
    _write_progress(
        progress_path,
        {
            "status": "succeeded",
            "phase": "complete",
            "progress": 1.0,
            "completed_episodes": total_episodes,
            "total_episodes": total_episodes,
            "episodes_per_stage": episodes_per_stage,
            "seed": seed,
            "states": len(agent.q),
            "stages": stages,
            "evaluation": evaluation,
            "checkpoint": str(checkpoint),
        },
    )
    return report


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    episodes_per_stage: int = 20,
    seed: int = 10_000,
) -> dict[str, Any]:
    if episodes_per_stage < 1:
        raise ValueError("episodes_per_stage must be positive")
    agent, metadata = TabularAcademyAgent.load(checkpoint)
    rng = np.random.default_rng(seed)
    stages: dict[str, dict[str, float | int]] = {}
    for stage in AcademyStage:
        successes = 0
        returns: list[float] = []
        env = NavigationAcademyEnv(stage=stage)
        for episode in range(episodes_per_stage):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action_index = agent.choose(observation, epsilon=0.0, rng=rng)
                observation, reward, terminated, truncated, _ = env.step(
                    ACTION_LIBRARY[action_index]
                )
                episode_return += reward
                if terminated or truncated:
                    successes += int(terminated)
                    returns.append(episode_return)
                    break
        env.close()
        stages[stage.slug] = {
            "episodes": episodes_per_stage,
            "success_rate": successes / episodes_per_stage,
            "mean_return": float(np.mean(returns)),
        }
    return {"checkpoint_metadata": metadata, "stages": stages}


STRATEGY_LABELS = {
    "tabular_q": "Tabela Q atual",
    "linear_sarsa_lambda": "SARSA(lambda) linear",
    "expected_sarsa_cnn": "Expected SARSA com CNN",
    "branching_expected_sarsa_cnn": "Expected SARSA · controles fatorados",
    "joint_control_expected_sarsa_cnn": "Expected SARSA · 25 controles conjuntos",
    "hybrid_actor_critic": "Actor–critic · câmera contínua",
}

TARGET_NAVIGATION_BENCHMARK = "target_navigation"
LEGACY_CURRICULUM_BENCHMARK = "legacy_curriculum"
BENCHMARK_CONTRACTS = {
    TARGET_NAVIGATION_BENCHMARK: "minecraft-target-navigation/random-biomes-interactions-v3",
    LEGACY_CURRICULUM_BENCHMARK: "minecraft-navigation-academy/variable-obstacles-v3",
}


def _is_hybrid_strategy(strategy: str) -> bool:
    return strategy == "hybrid_actor_critic"


def _behaviour_policy(strategy: str) -> str:
    return "stochastic_hybrid_actor" if _is_hybrid_strategy(strategy) else "epsilon_greedy"


def _benchmark_tasks(benchmark: str) -> tuple[AcademyStage | NavigationDifficulty, ...]:
    if benchmark == TARGET_NAVIGATION_BENCHMARK:
        return tuple(NavigationDifficulty)
    if benchmark == LEGACY_CURRICULUM_BENCHMARK:
        return tuple(AcademyStage)
    raise ValueError(f"unknown academy benchmark: {benchmark}")


def _target_schedule(
    *, seed: int, episodes_per_difficulty: int
) -> list[tuple[NavigationDifficulty, int, NavigationRequirement]]:
    """Balance physical requirements inside every difficulty band."""

    schedule: list[tuple[NavigationDifficulty, int, NavigationRequirement]] = []
    arena_seed = seed
    for difficulty in NavigationDifficulty:
        requirements = (
            (
                NavigationRequirement.TRAVEL,
                NavigationRequirement.BREAK,
                NavigationRequirement.PLACE,
            )
            if difficulty is NavigationDifficulty.EASY
            else tuple(NavigationRequirement)
        )
        for episode in range(episodes_per_difficulty):
            schedule.append(
                (
                    difficulty,
                    arena_seed,
                    requirements[(episode + int(difficulty)) % len(requirements)],
                )
            )
            arena_seed += 1
    np.random.default_rng(seed).shuffle(schedule)
    return schedule


def _weighted_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an exact integer budget using deterministic largest remainder."""

    if total < 1:
        raise ValueError("curriculum phase budget must be positive")
    weight_total = float(sum(weights.values()))
    if not np.isclose(weight_total, 1.0):
        raise ValueError("curriculum difficulty weights must sum to one")
    raw = {key: total * float(value) for key, value in weights.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    insertion_order = list(weights)
    order = sorted(
        weights,
        key=lambda key: (raw[key] - counts[key], -insertion_order.index(key)),
        reverse=True,
    )
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _curriculum_target_schedule(
    *, seed: int, episodes_per_phase: int
) -> list[tuple[int, str, NavigationDifficulty, int, NavigationRequirement]]:
    """Return the five-phase curriculum in real chronological order.

    Episodes are shuffled only inside their phase. This preserves the phase
    distribution contract while preventing long homogeneous difficulty runs.
    Physical affordances remain stratified inside each difficulty.
    """

    schedule: list[
        tuple[int, str, NavigationDifficulty, int, NavigationRequirement]
    ] = []
    arena_seed = int(seed)
    occurrences = {difficulty: 0 for difficulty in NavigationDifficulty}
    for phase_index, phase in enumerate(CURRICULUM_PHASES, start=1):
        phase_rows: list[
            tuple[int, str, NavigationDifficulty, int, NavigationRequirement]
        ] = []
        counts = _weighted_counts(episodes_per_phase, phase["weights"])
        for difficulty in NavigationDifficulty:
            count = counts.get(difficulty.slug, 0)
            requirements = (
                (NavigationRequirement.TRAVEL,)
                if phase_index == 1
                else (
                    NavigationRequirement.TRAVEL,
                    NavigationRequirement.BREAK,
                    NavigationRequirement.PLACE,
                )
                if phase_index == 2
                else
                (
                    NavigationRequirement.TRAVEL,
                    NavigationRequirement.BREAK,
                    NavigationRequirement.PLACE,
                )
                if difficulty is NavigationDifficulty.EASY
                else tuple(NavigationRequirement)
            )
            for _ in range(count):
                occurrence = occurrences[difficulty]
                requirement = requirements[
                    (occurrence + int(difficulty)) % len(requirements)
                ]
                phase_rows.append(
                    (
                        phase_index,
                        str(phase["slug"]),
                        difficulty,
                        arena_seed,
                        requirement,
                    )
                )
                occurrences[difficulty] += 1
                arena_seed += 1
        np.random.default_rng(seed + phase_index * 10_003).shuffle(phase_rows)
        schedule.extend(phase_rows)
    return schedule


def _timeline_metrics(
    chronology: list[dict[str, Any]],
    *,
    difficulty_targets: dict[str, int],
    phase_targets: dict[str, int],
    moving_window: int,
    balanced_window: int,
) -> dict[str, Any]:
    """Build the public chronology contract from immutable episode records."""

    difficulties = tuple(NavigationDifficulty)
    rewards = [float(item["reward"]) for item in chronology]
    moving_window = max(1, int(moving_window))
    balanced_window = max(1, int(balanced_window))
    reward_average = [
        float(np.mean(rewards[max(0, index - moving_window + 1) : index + 1]))
        for index in range(len(rewards))
    ]
    per_difficulty: dict[str, list[float]] = {
        difficulty.slug: [] for difficulty in difficulties
    }
    balanced_curve: list[float | None] = []
    for item in chronology:
        per_difficulty[str(item["difficulty"])].append(float(item["reward"]))
        if all(per_difficulty[difficulty.slug] for difficulty in difficulties):
            balanced_curve.append(
                float(
                    np.mean(
                        [
                            np.mean(per_difficulty[difficulty.slug][-balanced_window:])
                            for difficulty in difficulties
                        ]
                    )
                )
            )
        else:
            balanced_curve.append(None)
    recent = reward_average[-min(100, len(reward_average)) :]
    slope = (
        float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        if len(recent) >= 2
        else 0.0
    )
    threshold = 0.002
    direction = "up" if slope > threshold else "down" if slope < -threshold else "stable"
    phase_labels = {str(phase["slug"]): str(phase["label"]) for phase in CURRICULUM_PHASES}
    return {
        "interface": "minecraft-academy-chronology/v1",
        "episodes": chronology,
        "reward_curve": rewards,
        "reward_moving_average": reward_average,
        "moving_average_window": moving_window,
        "balanced_reward_curve": balanced_curve,
        "balanced_window_per_difficulty": balanced_window,
        "balanced_mean_return": next(
            (value for value in reversed(balanced_curve) if value is not None),
            None,
        ),
        "difficulty_progress": {
            difficulty.slug: {
                "completed": len(per_difficulty[difficulty.slug]),
                "target": int(difficulty_targets.get(difficulty.slug, 0)),
            }
            for difficulty in difficulties
        },
        "phase_progress": {
            phase_slug: {
                "completed": sum(
                    1 for item in chronology if item["phase"] == phase_slug
                ),
                "target": int(target),
                "label": phase_labels.get(phase_slug, phase_slug),
            }
            for phase_slug, target in phase_targets.items()
        },
        "recent_trend": {
            "direction": direction,
            "slope_per_episode": slope,
            "slope_per_100_episodes": slope * 100.0,
            "window": len(recent),
        },
    }


def _arena_manifest(
    *,
    seed: int,
    episodes_per_stage: int,
    benchmark: str,
    curriculum: bool = False,
) -> dict[str, Any]:
    from .arena import build_arena

    if benchmark == TARGET_NAVIGATION_BENCHMARK:
        grouped: dict[str, list[dict[str, Any]]] = {
            difficulty.slug: [] for difficulty in NavigationDifficulty
        }
        rows = (
            (
                (difficulty, arena_seed, requirement)
                for _, _, difficulty, arena_seed, requirement in _curriculum_target_schedule(
                    seed=seed,
                    episodes_per_phase=episodes_per_stage,
                )
            )
            if curriculum
            else iter(
                _target_schedule(
                    seed=seed,
                    episodes_per_difficulty=episodes_per_stage,
                )
            )
        )
        for difficulty, arena_seed, requirement in rows:
            arena = build_target_arena(
                difficulty, arena_seed, requirement=requirement
            )
            grouped[difficulty.slug].append(
                {
                    "seed": arena_seed,
                    "arena_sha256": arena.logical_sha256,
                    "biome": arena.biome.value,
                    "requirement": arena.requirement.value,
                    "initial_blocks": arena.initial_blocks,
                }
            )
        return grouped
    return {
        task.slug: [
            {
                "seed": seed + episode,
                "arena_sha256": build_arena(task, seed + episode).logical_sha256,
                "biome": "plains",
            }
            for episode in range(episodes_per_stage)
        ]
        for task in _benchmark_tasks(benchmark)
    }


def _modern_episode(
    agent: LinearSarsaLambdaAgent | ExpectedSarsaCnnAgent,
    env: NavigationAcademyEnv,
    observation: dict[str, Any],
    *,
    epsilon: float,
    rng: np.random.Generator,
    training: bool,
    return_diagnostics: bool = False,
) -> tuple[float, bool] | tuple[float, bool, dict[str, float]]:
    total = 0.0
    update_metrics: dict[str, list[float]] = {}
    event_counts = {
        "blocks_broken": 0,
        "blocks_placed": 0,
        "attack_misses": 0,
        "place_failures": 0,
    }
    hybrid_trajectory: list[tuple[dict[str, Any], Any, float]] = []
    hybrid = (
        getattr(agent, "action_contract", None)
        in {
            "minecraft-control-camera-keyboard/v2",
            "minecraft-control-camera-keyboard/v3",
        }
    )
    if isinstance(agent, (LinearSarsaLambdaAgent, ExpectedSarsaCnnAgent)):
        agent.begin_episode()
    elif hybrid:
        agent.begin_episode()
    action = (
        agent.choose(observation, stochastic=training, rng=rng)
        if hybrid
        else agent.choose(observation, epsilon=epsilon, rng=rng)
    )
    while True:
        if hybrid:
            next_observation, reward, terminated, truncated, info = env.step_control(
                hybrid_control_action(action)
            )
        else:
            next_observation, reward, terminated, truncated, info = env.step(
                joint_control_action(action)
                if isinstance(agent, JointControlExpectedSarsaCnnAgent)
                else factorized_control_action(action)
                if isinstance(agent, BranchingExpectedSarsaCnnAgent)
                else abstract_action(action)
            )
        done = terminated or truncated
        for event in info.get("events", []):
            event_type = event.get("type")
            if event_type == "block_broken":
                event_counts["blocks_broken"] += 1
            elif event_type == "block_placed":
                event_counts["blocks_placed"] += 1
            elif event_type == "attack_miss":
                event_counts["attack_misses"] += 1
            elif event_type == "place_failed":
                event_counts["place_failures"] += 1
        next_action = None
        if not done:
            next_action = (
                agent.choose(next_observation, stochastic=training, rng=rng)
                if hybrid
                else agent.choose(next_observation, epsilon=epsilon, rng=rng)
            )
        if training:
            if hybrid:
                hybrid_trajectory.append((observation, action, float(reward)))
            elif isinstance(agent, LinearSarsaLambdaAgent):
                agent.update(
                    observation,
                    action,
                    reward,
                    next_observation,
                    next_action,
                    done=done,
                )
            else:
                agent.update(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done=done,
                    epsilon=epsilon,
                )
        total += reward
        if done:
            if training and hybrid:
                metrics = agent.update_episode(hybrid_trajectory)
                for key, value in metrics.items():
                    update_metrics.setdefault(key, []).append(float(value))
            result = (total, bool(info.get("is_success", terminated)))
            if not return_diagnostics:
                return result
            diagnostics = {
                "elapsed_ticks": float(info.get("elapsed_ticks", 0.0)),
                "decision_steps": float(getattr(env, "_steps", 0)),
                "requirement_satisfied": float(
                    bool(info.get("requirement_satisfied", False))
                ),
                **{key: float(value) for key, value in event_counts.items()},
                **{
                    key: float(np.mean(values))
                    for key, values in update_metrics.items()
                    if values
                },
            }
            return (*result, diagnostics)
        observation = next_observation
        if hybrid:
            action = next_action
        else:
            action = (
                np.asarray(next_action, dtype=np.int64)
                if isinstance(agent, BranchingExpectedSarsaCnnAgent)
                else int(next_action)
            )


def _train_random_target_strategy(
    strategy: str,
    agent: Any,
    checkpoint: Path,
    *,
    episodes_per_difficulty: int,
    seed: int,
    epsilon_start: float,
    epsilon_end: float,
    stop_when_stable: bool,
    on_episode: Any,
    resume_state: dict[str, Any] | None = None,
    checkpoint_interval: int = 100,
    callback_interval: int = 1,
) -> tuple[Any, dict[str, Any], Path]:
    """Train on an interleaved stream of random biomes and target ranges."""

    rng = np.random.default_rng(seed)
    difficulties = tuple(NavigationDifficulty)
    schedule = _curriculum_target_schedule(
        seed=seed,
        episodes_per_phase=episodes_per_difficulty,
    )
    diagnostic_names = (
        "elapsed_ticks",
        "decision_steps",
        "requirement_satisfied",
        "blocks_broken",
        "blocks_placed",
        "attack_misses",
        "place_failures",
        "actor_loss",
        "critic_loss",
        "advantage",
        "raw_advantage",
        "state_value",
        "q_value",
        "entropy",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "td_target",
    )
    histories: dict[str, dict[str, list[Any]]] = {
        difficulty.slug: {
            "returns": [],
            "successes": [],
            "biomes": [],
            "requirements": [],
            **{name: [] for name in diagnostic_names},
        }
        for difficulty in difficulties
    }
    chronology: list[dict[str, Any]] = []
    difficulty_targets = {
        difficulty.slug: sum(
            1 for _, _, scheduled, _, _ in schedule if scheduled is difficulty
        )
        for difficulty in difficulties
    }
    phase_targets = {
        str(phase["slug"]): episodes_per_difficulty for phase in CURRICULUM_PHASES
    }
    total = len(schedule)
    stable_checks = 0
    stopped_when_stable = False
    completed_before = 0
    if resume_state and resume_state.get("status") == "running":
        saved_histories = resume_state.get("histories") or {}
        for difficulty in difficulties:
            saved = saved_histories.get(difficulty.slug) or {}
            history = histories[difficulty.slug]
            for key in history:
                values = saved.get(key) or []
                history[key] = list(values) if isinstance(values, list) else []
        completed_before = min(
            total,
            sum(len(item["returns"]) for item in histories.values()),
        )
        stable_checks = int(resume_state.get("stable_checks") or 0)
        stopped_when_stable = bool(resume_state.get("stopped_when_stable", False))
        saved_chronology = resume_state.get("chronology") or []
        if isinstance(saved_chronology, list):
            chronology = [dict(item) for item in saved_chronology if isinstance(item, dict)]
        if len(chronology) != completed_before:
            # v1 recovery checkpoints did not persist chronological order. They
            # cannot safely resume a curriculum whose phase is order-sensitive.
            chronology = []
            completed_before = 0
            for history in histories.values():
                for key in history:
                    history[key] = []
        rng_state = resume_state.get("numpy_rng_state")
        if isinstance(rng_state, dict):
            rng.bit_generator.state = rng_state

    def timeline_snapshot() -> dict[str, Any]:
        return _timeline_metrics(
            chronology,
            difficulty_targets=difficulty_targets,
            phase_targets=phase_targets,
            moving_window=min(100, max(10, episodes_per_difficulty // 10)),
            balanced_window=min(50, max(10, episodes_per_difficulty // 20)),
        )

    def snapshot() -> dict[str, Any]:
        result: dict[str, Any] = {}
        for difficulty in difficulties:
            history = histories[difficulty.slug]
            returns = [float(value) for value in history["returns"]]
            successes = [bool(value) for value in history["successes"]]
            window = min(20, max(3, episodes_per_difficulty // 10))
            reward_average = [
                float(np.mean(returns[max(0, index - window + 1) : index + 1]))
                for index in range(len(returns))
            ]
            success_average = [
                float(np.mean(successes[max(0, index - window + 1) : index + 1]))
                for index in range(len(successes))
            ]
            diagnostics = {
                f"{name}_curve": [float(value) for value in history[name]]
                for name in diagnostic_names
                if history[name]
            }
            elapsed = diagnostics.get("elapsed_ticks_curve", [])
            successful_elapsed = [
                value
                for value, success in zip(elapsed, successes, strict=False)
                if success
            ]
            failed_elapsed = [
                value
                for value, success in zip(elapsed, successes, strict=False)
                if not success
            ]
            result[difficulty.slug] = {
                "episodes": len(returns),
                "max_episodes": difficulty_targets[difficulty.slug],
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "recent_success_rate": (
                    float(np.mean(successes[-min(20, len(successes)) :]))
                    if successes
                    else 0.0
                ),
                "recent_window": min(20, len(successes)),
                "mean_return": float(np.mean(returns)) if returns else 0.0,
                "reward_curve": returns,
                "reward_moving_average": reward_average,
                "success_curve": [int(value) for value in successes],
                "success_moving_average": success_average,
                "moving_average_window": window,
                "stopped_when_stable": stopped_when_stable,
                "biome_counts": {
                    biome: history["biomes"].count(biome)
                    for biome in sorted(set(history["biomes"]))
                },
                "requirement_counts": {
                    requirement: history["requirements"].count(requirement)
                    for requirement in sorted(set(history["requirements"]))
                },
                "success_by_requirement": {
                    requirement: float(
                        np.mean(
                            [
                                success
                                for success, observed_requirement in zip(
                                    successes,
                                    history["requirements"],
                                    strict=False,
                                )
                                if observed_requirement == requirement
                            ]
                        )
                    )
                    for requirement in sorted(set(history["requirements"]))
                },
                "successful_elapsed_ticks_mean": (
                    float(np.mean(successful_elapsed)) if successful_elapsed else None
                ),
                "failed_elapsed_ticks_mean": (
                    float(np.mean(failed_elapsed)) if failed_elapsed else None
                ),
                **diagnostics,
            }
        return result

    def checkpoint_metadata(status: str) -> dict[str, Any]:
        stages = snapshot()
        return {
            "strategy": strategy,
            "seed": seed,
            "episodes_per_difficulty": episodes_per_difficulty,
            "benchmark": TARGET_NAVIGATION_BENCHMARK,
            "benchmark_contract": BENCHMARK_CONTRACTS[TARGET_NAVIGATION_BENCHMARK],
            "observation_contract": (
                "minecraft-academy-structured/v1"
                if strategy == "tabular_q"
                else "minecraft-academy-world-map-window-time/v4"
                if strategy == "hybrid_actor_critic"
                else "minecraft-academy-world-map-window-time/v1"
                if strategy == "joint_control_expected_sarsa_cnn"
                else "minecraft-academy-world-map-window/v1"
            ),
            "action_contract": (
                "minecraft-academy-legacy-10/v1"
                if strategy == "tabular_q"
                else "minecraft-control-joint-25/v1"
                if strategy == "joint_control_expected_sarsa_cnn"
                else "minecraft-control-camera-keyboard/v3"
                if strategy == "hybrid_actor_critic"
                else "minecraft-control-factorized/v3"
                if strategy == "branching_expected_sarsa_cnn"
                else "minecraft-academy-body-12/v2"
            ),
            "policy_input": "public_policy_channels_only",
            "reward_contract": "minecraft-bounded-asymptotic-time-terminal/v3",
            "reward_parameters": {
                "maximum_time_cost": 20.0,
                "shape_beta": 3.0,
                "stationary_multiplier": 1.5,
            },
            "privileged_channels_used": [],
            "learning_mode": "online_on_policy",
            "behaviour_policy": _behaviour_policy(strategy),
            "epsilon_start": None if _is_hybrid_strategy(strategy) else epsilon_start,
            "epsilon_end": None if _is_hybrid_strategy(strategy) else epsilon_end,
            "epsilon_schedule": (
                None
                if _is_hybrid_strategy(strategy)
                else "global_interleaved_linear_decay"
            ),
            "difficulty_schedule": "five_phase_curriculum_with_affordance_strata",
            "curriculum": list(CURRICULUM_PHASES),
            "timeline": timeline_snapshot(),
            "stop_when_stable": stop_when_stable,
            "stages": stages,
            "training_state": {
                "interface": "minecraft-academy-recovery/v1",
                "status": status,
                "completed_episodes": sum(
                    len(item["returns"]) for item in histories.values()
                ),
                "total_episodes": total,
                "histories": histories,
                "chronology": chronology,
                "stable_checks": stable_checks,
                "stopped_when_stable": stopped_when_stable,
                "numpy_rng_state": rng.bit_generator.state,
            },
        }

    for completed, (
        phase_index,
        phase_slug,
        difficulty,
        arena_seed,
        requirement,
    ) in enumerate(
        schedule[completed_before:], start=completed_before + 1
    ):
        fraction = (completed - 1) / max(1, total - 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        env = NavigationAcademyEnv(
            difficulty=difficulty,
            requirement=requirement,
            max_episode_steps=190,
            camera_fast_degrees=90.0 if strategy != "tabular_q" else 18.0,
        )
        observation, reset_info = env.reset(seed=arena_seed)
        if strategy == "tabular_q":
            episode_return = 0.0
            while True:
                action = agent.choose(observation, epsilon=epsilon, rng=rng)
                next_observation, reward, terminated, truncated, info = env.step(
                    ACTION_LIBRARY[action]
                )
                done = terminated or truncated
                agent.update(observation, action, reward, next_observation, done=done)
                observation = next_observation
                episode_return += reward
                if done:
                    success = bool(info.get("is_success", terminated))
                    break
        else:
            episode_return, success, episode_diagnostics = _modern_episode(
                agent,
                env,
                observation,
                epsilon=epsilon,
                rng=rng,
                training=True,
                return_diagnostics=True,
            )
        if strategy == "tabular_q":
            episode_diagnostics = {
                "elapsed_ticks": float(info.get("elapsed_ticks", 0.0)),
                "decision_steps": float(getattr(env, "_steps", 0)),
            }
        biome = str(reset_info.get("biome") or env.blueprint.biome.value)
        requirement = str(
            reset_info.get("requirement") or env.blueprint.requirement.value
        )
        env.close()
        history = histories[difficulty.slug]
        history["returns"].append(float(episode_return))
        history["successes"].append(bool(success))
        history["biomes"].append(biome)
        history["requirements"].append(requirement)
        for name in diagnostic_names:
            if name in episode_diagnostics:
                history[name].append(float(episode_diagnostics[name]))
        chronology.append(
            {
                "index": completed,
                "phase_index": phase_index,
                "phase": phase_slug,
                "difficulty": difficulty.slug,
                "arena_seed": arena_seed,
                "requirement": requirement,
                "reward": float(episode_return),
                "success": bool(success),
                "requirement_satisfied": bool(
                    episode_diagnostics.get("requirement_satisfied", False)
                ),
            }
        )
        should_notify = (
            completed == 1
            or completed == total
            or completed % max(1, callback_interval) == 0
        )
        if on_episode is not None and should_notify:
            stages = snapshot()
            current = stages[difficulty.slug]
            on_episode(
                difficulty,
                int(current["episodes"]),
                completed,
                float(current["success_rate"]),
                float(current["mean_return"]),
                current,
                {key: value for key, value in stages.items() if key != difficulty.slug},
                timeline_snapshot(),
            )
        if stop_when_stable and phase_index == len(CURRICULUM_PHASES) and all(
            len(histories[item.slug]["returns"]) >= 40 for item in difficulties
        ):
            stable_now = True
            for item in difficulties:
                item_history = histories[item.slug]
                recent_rewards = np.asarray(
                    item_history["returns"][-20:], dtype=np.float32
                )
                slope = float(
                    np.polyfit(np.arange(recent_rewards.size), recent_rewards, 1)[0]
                )
                recent_success = float(np.mean(item_history["successes"][-20:]))
                if abs(slope) > 0.05 or recent_success < 0.8:
                    stable_now = False
                    break
            stable_checks = stable_checks + 1 if stable_now else 0
            if stable_checks >= 10:
                stopped_when_stable = True
        should_checkpoint = completed % max(1, checkpoint_interval) == 0
        if should_checkpoint or stopped_when_stable:
            agent.save(checkpoint, metadata=checkpoint_metadata("running"))
        if stopped_when_stable:
            break

    stages = snapshot()
    metadata = checkpoint_metadata("complete")
    saved = agent.save(checkpoint, metadata=metadata)
    return agent, stages, saved


def _train_one_strategy(
    strategy: str,
    output_dir: Path,
    *,
    episodes_per_stage: int,
    seed: int,
    epsilon_start: float,
    epsilon_end: float,
    benchmark: str = TARGET_NAVIGATION_BENCHMARK,
    resume_checkpoint: Path | None = None,
    stop_when_stable: bool = False,
    stability_min_episodes: int = 40,
    stability_window: int = 20,
    stability_success_rate: float = 0.8,
    stability_slope_tolerance: float = 0.05,
    on_episode: Any = None,
    checkpoint_interval: int = 100,
    callback_interval: int = 1,
) -> tuple[Any, dict[str, Any], Path]:
    recovery_state: dict[str, Any] | None = None
    if strategy == "tabular_q":
        agent: Any = (
            TabularAcademyAgent.load(resume_checkpoint)[0]
            if resume_checkpoint is not None
            else TabularAcademyAgent()
        )
        checkpoint = output_dir / "tabular-q.json"
    elif strategy == "linear_sarsa_lambda":
        agent = (
            LinearSarsaLambdaAgent.load(resume_checkpoint)[0]
            if resume_checkpoint is not None
            else LinearSarsaLambdaAgent()
        )
        checkpoint = output_dir / "linear-sarsa-lambda.npz"
    elif strategy == "expected_sarsa_cnn":
        agent = (
            ExpectedSarsaCnnAgent.load(resume_checkpoint)[0]
            if resume_checkpoint is not None
            else ExpectedSarsaCnnAgent(seed=seed)
        )
        checkpoint = output_dir / "expected-sarsa-cnn.npz"
    elif strategy == "branching_expected_sarsa_cnn":
        agent = (
            BranchingExpectedSarsaCnnAgent.load(resume_checkpoint)[0]
            if resume_checkpoint is not None
            else BranchingExpectedSarsaCnnAgent(seed=seed)
        )
        checkpoint = output_dir / "branching-expected-sarsa-cnn.npz"
    elif strategy == "joint_control_expected_sarsa_cnn":
        agent = (
            JointControlExpectedSarsaCnnAgent.load(resume_checkpoint)[0]
            if resume_checkpoint is not None
            else JointControlExpectedSarsaCnnAgent(seed=seed)
        )
        checkpoint = output_dir / "joint-control-expected-sarsa-cnn.npz"
    elif strategy == "hybrid_actor_critic":
        from .actor_critic import HybridActorCriticAgent

        checkpoint = output_dir / "hybrid-actor-critic.pt"
        if resume_checkpoint is not None:
            agent = HybridActorCriticAgent.load(resume_checkpoint)[0]
        elif checkpoint.is_file():
            candidate, candidate_metadata = HybridActorCriticAgent.load(checkpoint)
            candidate_recovery = candidate_metadata.get("training_state") or {}
            compatible = (
                candidate_recovery.get("status") == "running"
                and int(candidate_metadata.get("seed", -1)) == seed
                and int(candidate_metadata.get("episodes_per_difficulty", -1))
                == episodes_per_stage
            )
            if compatible:
                agent = candidate
                recovery_state = candidate_recovery
            else:
                agent = HybridActorCriticAgent(seed=seed)
        else:
            agent = HybridActorCriticAgent(seed=seed)
    else:
        raise ValueError(f"unknown academy strategy: {strategy}")

    if benchmark == TARGET_NAVIGATION_BENCHMARK:
        return _train_random_target_strategy(
            strategy,
            agent,
            checkpoint,
            episodes_per_difficulty=episodes_per_stage,
            seed=seed,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            stop_when_stable=stop_when_stable,
            on_episode=on_episode,
            resume_state=recovery_state,
            checkpoint_interval=checkpoint_interval,
            callback_interval=callback_interval,
        )
    if benchmark != LEGACY_CURRICULUM_BENCHMARK:
        raise ValueError(f"unknown academy benchmark: {benchmark}")

    rng = np.random.default_rng(seed)
    completed = 0
    stages: dict[str, Any] = {}
    for stage in AcademyStage:
        successes = 0
        success_history: list[bool] = []
        returns: list[float] = []
        reward_moving_average: list[float] = []
        success_moving_average: list[float] = []
        curve_window = min(20, max(3, episodes_per_stage // 10))
        env = NavigationAcademyEnv(
            stage=stage,
            max_episode_steps=120,
            camera_fast_degrees=90.0 if strategy != "tabular_q" else 18.0,
        )
        stable_checks = 0
        stopped_when_stable = False
        for episode in range(episodes_per_stage):
            # Exploration belongs to the behaviour policy and is restarted for
            # every task family.  A single global decay made later stages see a
            # materially different policy and could leave a new task with
            # almost no exploration at all.
            fraction = episode / max(1, episodes_per_stage - 1)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            observation, _ = env.reset(seed=seed + episode)
            if strategy == "tabular_q":
                episode_return = 0.0
                while True:
                    action = agent.choose(observation, epsilon=epsilon, rng=rng)
                    next_observation, reward, terminated, truncated, _ = env.step(
                        ACTION_LIBRARY[action]
                    )
                    done = terminated or truncated
                    agent.update(observation, action, reward, next_observation, done=done)
                    observation = next_observation
                    episode_return += reward
                    if done:
                        success = terminated
                        break
            else:
                episode_return, success = _modern_episode(
                    agent,
                    env,
                    observation,
                    epsilon=epsilon,
                    rng=rng,
                    training=True,
                )
            successes += int(success)
            success_history.append(bool(success))
            returns.append(float(episode_return))
            reward_moving_average.append(float(np.mean(returns[-curve_window:])))
            success_moving_average.append(
                float(np.mean(success_history[-curve_window:]))
            )
            completed += 1
            if on_episode is not None:
                on_episode(
                    stage,
                    episode + 1,
                    completed,
                    successes / (episode + 1),
                    float(np.mean(returns)),
                    {
                        "episodes": episode + 1,
                        "reward_curve": list(returns),
                        "reward_moving_average": list(reward_moving_average),
                        "success_curve": [int(value) for value in success_history],
                        "success_moving_average": list(success_moving_average),
                        "moving_average_window": curve_window,
                    },
                    dict(stages),
                )
            if stop_when_stable and len(returns) >= max(
                stability_min_episodes, stability_window
            ):
                recent_rewards = np.asarray(
                    reward_moving_average[-stability_window:], dtype=np.float32
                )
                slope = float(
                    np.polyfit(np.arange(recent_rewards.size), recent_rewards, 1)[0]
                )
                recent_success = float(
                    np.mean(success_history[-stability_window:])
                )
                stable_now = (
                    abs(slope) <= stability_slope_tolerance
                    and recent_success >= stability_success_rate
                )
                stable_checks = stable_checks + 1 if stable_now else 0
                if stable_checks >= min(10, max(3, stability_window // 2)):
                    stopped_when_stable = True
                    break
        env.close()
        actual_episodes = len(returns)
        recent_window = min(20, max(1, episodes_per_stage // 5))
        stages[stage.slug] = {
            "episodes": actual_episodes,
            "max_episodes": episodes_per_stage,
            "success_rate": successes / actual_episodes,
            "recent_success_rate": float(
                np.mean(success_history[-recent_window:])
            ),
            "recent_window": recent_window,
            "mean_return": float(np.mean(returns)),
            "reward_curve": returns,
            "reward_moving_average": reward_moving_average,
            "success_curve": [int(value) for value in success_history],
            "success_moving_average": success_moving_average,
            "moving_average_window": curve_window,
            "stopped_when_stable": stopped_when_stable,
        }
    metadata = {
        "strategy": strategy,
        "seed": seed,
        "episodes_per_stage": episodes_per_stage,
        "observation_contract": (
            "minecraft-academy-structured/v1"
            if strategy == "tabular_q"
            else "minecraft-academy-world-map-window-time/v4"
            if strategy == "hybrid_actor_critic"
            else "minecraft-academy-world-map-window-time/v1"
            if strategy == "joint_control_expected_sarsa_cnn"
            else "minecraft-academy-world-map-window/v1"
        ),
        "action_contract": (
            "minecraft-academy-legacy-10/v1"
            if strategy == "tabular_q"
            else "minecraft-control-joint-25/v1"
            if strategy == "joint_control_expected_sarsa_cnn"
            else "minecraft-control-camera-keyboard/v3"
            if strategy == "hybrid_actor_critic"
            else "minecraft-control-factorized/v3"
            if strategy == "branching_expected_sarsa_cnn"
            else "minecraft-academy-body-12/v2"
        ),
        "policy_input": "public_policy_channels_only",
        "reward_contract": "minecraft-bounded-asymptotic-time-terminal/v3",
        "privileged_channels_used": [],
        "learning_mode": "online_on_policy",
        "behaviour_policy": _behaviour_policy(strategy),
        "epsilon_start": None if _is_hybrid_strategy(strategy) else epsilon_start,
        "epsilon_end": None if _is_hybrid_strategy(strategy) else epsilon_end,
        "epsilon_schedule": None if _is_hybrid_strategy(strategy) else "per_stage_linear_decay",
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "stop_when_stable": stop_when_stable,
        "stability": {
            "minimum_episodes": stability_min_episodes,
            "window": stability_window,
            "success_rate": stability_success_rate,
            "reward_slope_tolerance": stability_slope_tolerance,
        },
        "stages": stages,
    }
    saved = agent.save(checkpoint, metadata=metadata)
    return agent, stages, saved


def _evaluate_strategy(
    strategy: str,
    agent: Any,
    *,
    episodes_per_stage: int,
    seed: int,
    held_out: bool,
    benchmark: str = TARGET_NAVIGATION_BENCHMARK,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stages: dict[str, Any] = {}
    target_seeds: dict[str, list[int]] = {}
    if benchmark == TARGET_NAVIGATION_BENCHMARK:
        target_seeds = {difficulty.slug: [] for difficulty in NavigationDifficulty}
        target_requirements: dict[str, list[NavigationRequirement]] = {
            difficulty.slug: [] for difficulty in NavigationDifficulty
        }
        for difficulty, arena_seed, requirement in _target_schedule(
            seed=seed,
            episodes_per_difficulty=episodes_per_stage,
        ):
            target_seeds[difficulty.slug].append(arena_seed)
            target_requirements[difficulty.slug].append(requirement)
    for stage in _benchmark_tasks(benchmark):
        successes = 0
        returns: list[float] = []
        env = NavigationAcademyEnv(
            **(
                {"difficulty": stage}
                if benchmark == TARGET_NAVIGATION_BENCHMARK
                else {"stage": stage}
            ),
            max_episode_steps=(190 if benchmark == TARGET_NAVIGATION_BENCHMARK else 120),
            camera_fast_degrees=90.0 if strategy != "tabular_q" else 18.0,
        )
        for episode in range(episodes_per_stage):
            arena_seed = (
                target_seeds[stage.slug][episode]
                if benchmark == TARGET_NAVIGATION_BENCHMARK
                else seed + episode
            )
            observation, _ = env.reset(
                seed=arena_seed,
                options=(
                    {"requirement": target_requirements[stage.slug][episode]}
                    if benchmark == TARGET_NAVIGATION_BENCHMARK
                    else None
                ),
            )
            if strategy == "tabular_q":
                episode_return = 0.0
                while True:
                    action = agent.choose(observation, epsilon=0.0, rng=rng)
                    observation, reward, terminated, truncated, info = env.step(
                        ACTION_LIBRARY[action]
                    )
                    episode_return += reward
                    if terminated or truncated:
                        success = bool(info.get("is_success", terminated))
                        break
            else:
                episode_return, success = _modern_episode(
                    agent,
                    env,
                    observation,
                    epsilon=0.0,
                    rng=rng,
                    training=False,
                )
            successes += int(success)
            returns.append(float(episode_return))
        env.close()
        stages[stage.slug] = {
            "episodes": episodes_per_stage,
            "success_rate": successes / episodes_per_stage,
            "mean_return": float(np.mean(returns)),
        }
    return {
        "benchmark": benchmark,
        "benchmark_contract": BENCHMARK_CONTRACTS[benchmark],
        "seed": seed,
        "held_out": held_out,
        "learning_enabled": False,
        "epsilon": 0.0,
        "overall_success_rate": float(
            np.mean([value["success_rate"] for value in stages.values()])
        ),
        "stages": stages,
    }


def _merge_training_history(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Append a continuation run without rewriting its immutable parent."""

    if not previous:
        return current
    merged_stages: dict[str, Any] = {}
    previous_stages = previous.get("stages") or {}
    current_stages = current.get("stages") or {}
    stage_slugs = list(dict.fromkeys([*previous_stages, *current_stages]))
    for stage_slug in stage_slugs:
        old = previous_stages.get(stage_slug) or {}
        new = current_stages.get(stage_slug) or {}
        rewards = [
            *[float(value) for value in old.get("reward_curve") or []],
            *[float(value) for value in new.get("reward_curve") or []],
        ]
        successes = [
            *[int(value) for value in old.get("success_curve") or []],
            *[int(value) for value in new.get("success_curve") or []],
        ]
        # v2 artifacts created before success_curve existed still preserve an
        # aggregate success rate. Keep that history in the aggregate while new
        # continuations record every episode exactly.
        old_episodes = int(old.get("episodes") or len(old.get("reward_curve") or []))
        new_episodes = int(new.get("episodes") or len(new.get("reward_curve") or []))
        total_episodes = old_episodes + new_episodes
        weighted_success = (
            float(old.get("success_rate", 0.0)) * old_episodes
            + float(new.get("success_rate", 0.0)) * new_episodes
        ) / max(1, total_episodes)
        weighted_return = (
            float(old.get("mean_return", 0.0)) * old_episodes
            + float(new.get("mean_return", 0.0)) * new_episodes
        ) / max(1, total_episodes)
        window = int(new.get("moving_average_window") or old.get("moving_average_window") or 10)
        moving_rewards = [
            float(np.mean(rewards[max(0, index - window + 1) : index + 1]))
            for index in range(len(rewards))
        ]
        moving_success = [
            float(np.mean(successes[max(0, index - window + 1) : index + 1]))
            for index in range(len(successes))
        ]
        diagnostic_keys = (
            "elapsed_ticks_curve",
            "decision_steps_curve",
            "actor_loss_curve",
            "critic_loss_curve",
            "advantage_curve",
            "q_value_curve",
            "entropy_curve",
            "actor_gradient_norm_curve",
            "critic_gradient_norm_curve",
            "td_target_curve",
        )
        merged_diagnostics = {
            key: [
                *[float(value) for value in old.get(key) or []],
                *[float(value) for value in new.get(key) or []],
            ]
            for key in diagnostic_keys
            if old.get(key) or new.get(key)
        }
        elapsed = merged_diagnostics.get("elapsed_ticks_curve", [])
        successful_elapsed = [
            value
            for value, success in zip(elapsed, successes, strict=False)
            if success
        ]
        failed_elapsed = [
            value
            for value, success in zip(elapsed, successes, strict=False)
            if not success
        ]
        merged_stages[stage_slug] = {
            **old,
            **new,
            "episodes": total_episodes,
            "continuation_episodes": new_episodes,
            "success_rate": weighted_success,
            "mean_return": weighted_return,
            "reward_curve": rewards,
            "reward_moving_average": moving_rewards,
            "success_curve": successes,
            "success_moving_average": moving_success,
            "moving_average_window": window,
            "successful_elapsed_ticks_mean": (
                float(np.mean(successful_elapsed)) if successful_elapsed else None
            ),
            "failed_elapsed_ticks_mean": (
                float(np.mean(failed_elapsed)) if failed_elapsed else None
            ),
            **merged_diagnostics,
        }
    previous_timeline = previous.get("timeline") or {}
    current_timeline = current.get("timeline") or {}
    merged_timeline: dict[str, Any] | None = None
    if previous_timeline or current_timeline:
        old_episodes = [
            dict(item) for item in previous_timeline.get("episodes") or []
        ]
        new_episodes = [dict(item) for item in current_timeline.get("episodes") or []]
        merged_episodes = [*old_episodes, *new_episodes]
        for index, episode in enumerate(merged_episodes, start=1):
            episode["index"] = index
        difficulty_targets = {
            difficulty.slug: int(
                (previous_timeline.get("difficulty_progress") or {})
                .get(difficulty.slug, {})
                .get("target", 0)
            )
            + int(
                (current_timeline.get("difficulty_progress") or {})
                .get(difficulty.slug, {})
                .get("target", 0)
            )
            for difficulty in NavigationDifficulty
        }
        phase_slugs = {
            *(previous_timeline.get("phase_progress") or {}),
            *(current_timeline.get("phase_progress") or {}),
        }
        phase_targets = {
            slug: int(
                (previous_timeline.get("phase_progress") or {})
                .get(slug, {})
                .get("target", 0)
            )
            + int(
                (current_timeline.get("phase_progress") or {})
                .get(slug, {})
                .get("target", 0)
            )
            for slug in phase_slugs
        }
        merged_timeline = _timeline_metrics(
            merged_episodes,
            difficulty_targets=difficulty_targets,
            phase_targets=phase_targets,
            moving_window=int(
                current_timeline.get("moving_average_window")
                or previous_timeline.get("moving_average_window")
                or 20
            ),
            balanced_window=int(
                current_timeline.get("balanced_window_per_difficulty")
                or previous_timeline.get("balanced_window_per_difficulty")
                or 20
            ),
        )
    return {
        **previous,
        **current,
        "continuation_count": int(previous.get("continuation_count", 0)) + 1,
        "timeline": merged_timeline,
        "stages": merged_stages,
        "overall_success_rate": float(
            np.mean([value["success_rate"] for value in merged_stages.values()])
        ),
        "recent_success_rate": float(
            np.mean(
                [
                    (value.get("success_moving_average") or [0.0])[-1]
                    for value in merged_stages.values()
                ]
            )
        ),
    }


def _movement_policy_artifact(
    strategy: str,
    checkpoint: Path,
    evaluation: dict[str, Any],
    training_seed_evaluation: dict[str, Any],
    online_training: dict[str, Any],
    parent_policy_id: str | None = None,
    *,
    seed: int,
    episodes_per_stage: int,
    benchmark: str,
) -> dict[str, Any]:
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy_id = f"movement:{strategy}:{checkpoint_sha256[:12]}"
    modern = strategy != "tabular_q"
    return {
        "interface": "minecraft-academy-movement-policy/v2",
        "benchmark": benchmark,
        "benchmark_contract": BENCHMARK_CONTRACTS[benchmark],
        "policy_id": policy_id,
        "parent_policy_id": parent_policy_id,
        "building_block": "movement",
        "competence": "target_navigation",
        "title": "Navegacao rapida ate alvo",
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "observation_contract": (
            "minecraft-academy-world-map-window-time/v4"
            if strategy == "hybrid_actor_critic"
            else "minecraft-academy-world-map-window-time/v1"
            if strategy == "joint_control_expected_sarsa_cnn"
            else "minecraft-academy-world-map-window/v1"
            if modern
            else "minecraft-academy-structured/v1"
        ),
        "action_contract": (
            "minecraft-control-joint-25/v1"
            if strategy == "joint_control_expected_sarsa_cnn"
            else "minecraft-control-camera-keyboard/v3"
            if strategy == "hybrid_actor_critic"
            else "minecraft-control-factorized/v3"
            if strategy == "branching_expected_sarsa_cnn"
            else "minecraft-academy-body-12/v2"
            if modern
            else "minecraft-academy-legacy-10/v1"
        ),
        "reward_contract": "minecraft-bounded-asymptotic-time-terminal/v3",
        "input_channels": (
            [
                "map_window.1_block_per_cell",
                "gnomonic_lattice",
                "inventory",
                *(
                    ["elapsed_fraction"]
                    if strategy in ("joint_control_expected_sarsa_cnn", "hybrid_actor_critic")
                    else []
                ),
                "action_mask",
            ]
            if modern
            else ["local_voxels", "goal_hint", "inventory"]
        ),
        "privileged_channels": [],
        "runtime_adapter": "brincabot-movement-policy/v1",
        "deployment_state": "assignable",
        "training_seed": seed,
        "episodes_per_stage": episodes_per_stage,
        "online_training": online_training,
        "training_seed_evaluation": training_seed_evaluation,
        "evaluation": evaluation,
    }


def _compare_strategies(
    output_dir: str | Path,
    *,
    strategies: list[str] | tuple[str, ...],
    episodes_per_stage: int = 500,
    evaluation_episodes: int = 20,
    seed: int = 0,
    progress_path: str | Path | None = None,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    resume_policy_artifact: str | Path | None = None,
    stop_when_stable: bool = False,
    benchmark: str = TARGET_NAVIGATION_BENCHMARK,
) -> dict[str, Any]:
    """Train one or more strategies on identical arenas and held-out seeds."""

    selected = list(dict.fromkeys(strategies))
    if not selected:
        raise ValueError("select at least one academy strategy")
    unknown = set(selected) - set(STRATEGY_LABELS)
    if unknown:
        raise ValueError(f"unknown academy strategies: {sorted(unknown)}")
    if episodes_per_stage < 1 or evaluation_episodes < 1:
        raise ValueError("training and evaluation episode counts must be positive")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
    tasks = _benchmark_tasks(benchmark)
    resume_artifact: dict[str, Any] | None = None
    resume_checkpoint: Path | None = None
    if resume_policy_artifact is not None:
        artifact_path = Path(resume_policy_artifact).expanduser().resolve()
        resume_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if len(selected) != 1 or resume_artifact.get("strategy") != selected[0]:
            raise ValueError("a continuation must use exactly its parent strategy")
        resume_checkpoint = Path(str(resume_artifact.get("checkpoint", ""))).expanduser()
        if not resume_checkpoint.is_file():
            raise ValueError("the parent policy checkpoint is unavailable")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total = episodes_per_stage * len(tasks) * len(selected)
    completed_before = 0
    reports: dict[str, Any] = {}
    train_manifest = _arena_manifest(
        seed=seed,
        episodes_per_stage=episodes_per_stage,
        benchmark=benchmark,
        curriculum=benchmark == TARGET_NAVIGATION_BENCHMARK,
    )
    evaluation_seed = seed + 10_000
    evaluation_manifest = _arena_manifest(
        seed=evaluation_seed,
        episodes_per_stage=evaluation_episodes,
        benchmark=benchmark,
    )

    def publish(**updates: Any) -> None:
        _write_progress(
            progress_path,
            {
                "interface": "minecraft-academy-comparison-progress/v3",
                "status": "running",
                "phase": "training",
                "progress": 0.0,
                "strategies": selected,
                "strategy_count": len(selected),
                "completed_episodes": 0,
                "total_episodes": total,
                "episodes_per_stage": episodes_per_stage,
                "evaluation_episodes": evaluation_episodes,
                "seed": seed,
                "benchmark": benchmark,
                "benchmark_contract": BENCHMARK_CONTRACTS[benchmark],
                "learning_mode": "online_on_policy",
                "behaviour_policy": (
                    "stochastic_hybrid_actor"
                    if all(_is_hybrid_strategy(item) for item in selected)
                    else "strategy_specific"
                    if any(_is_hybrid_strategy(item) for item in selected)
                    else "epsilon_greedy"
                ),
                "epsilon_start": (
                    None if all(_is_hybrid_strategy(item) for item in selected) else epsilon_start
                ),
                "epsilon_end": (
                    None if all(_is_hybrid_strategy(item) for item in selected) else epsilon_end
                ),
                "epsilon_schedule": (
                    None
                    if all(_is_hybrid_strategy(item) for item in selected)
                    else "global_interleaved_linear_decay"
                    if benchmark == TARGET_NAVIGATION_BENCHMARK
                    else "per_stage_linear_decay"
                ),
                "stop_when_stable": stop_when_stable,
                "parent_policy_id": (
                    resume_artifact.get("policy_id") if resume_artifact else None
                ),
                "reports": reports,
                **updates,
            },
        )

    publish()
    for strategy_index, strategy in enumerate(selected, start=1):
        strategy_dir = destination / strategy
        latest_timeline: dict[str, Any] | None = None

        def on_episode(
            stage: AcademyStage | NavigationDifficulty,
            episode: int,
            completed_local: int,
            success_rate: float,
            mean_return: float,
            current_curve: dict[str, Any],
            completed_stages: dict[str, Any],
            timeline: dict[str, Any] | None = None,
        ) -> None:
            nonlocal latest_timeline
            latest_timeline = timeline
            completed = completed_before + completed_local
            latest_episode = (
                timeline["episodes"][-1]
                if timeline and timeline.get("episodes")
                else None
            )
            publish(
                current_strategy=strategy,
                current_strategy_label=STRATEGY_LABELS[strategy],
                strategy_index=strategy_index,
                stage=stage.slug,
                stage_index=int(stage) + 1,
                stage_count=len(tasks),
                curriculum_phase=(
                    latest_episode.get("phase") if latest_episode else None
                ),
                curriculum_phase_index=(
                    latest_episode.get("phase_index") if latest_episode else None
                ),
                curriculum_phase_count=len(CURRICULUM_PHASES),
                episode=episode,
                completed_episodes=completed,
                progress=0.90 * completed / total,
                success_rate=success_rate,
                mean_return=mean_return,
                live_training={
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "timeline": timeline,
                    "stages": {
                        **completed_stages,
                        stage.slug: current_curve,
                    },
                },
            )

        agent, stages, checkpoint = _train_one_strategy(
            strategy,
            strategy_dir,
            episodes_per_stage=episodes_per_stage,
            seed=seed,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            benchmark=benchmark,
            resume_checkpoint=resume_checkpoint,
            stop_when_stable=stop_when_stable,
            on_episode=on_episode,
            callback_interval=10,
        )
        publish(
            phase="evaluation",
            current_strategy=strategy,
            current_strategy_label=STRATEGY_LABELS[strategy],
            strategy_index=strategy_index,
            completed_episodes=completed_before + episodes_per_stage * len(tasks),
            progress=0.90 + 0.09 * strategy_index / len(selected),
        )
        training_seed_evaluation = _evaluate_strategy(
            strategy,
            agent,
            episodes_per_stage=min(evaluation_episodes, episodes_per_stage),
            seed=seed,
            held_out=False,
            benchmark=benchmark,
        )
        evaluation = _evaluate_strategy(
            strategy,
            agent,
            episodes_per_stage=evaluation_episodes,
            seed=evaluation_seed,
            held_out=True,
            benchmark=benchmark,
        )
        current_online_training = {
            "learning_enabled": True,
            "behaviour_policy": _behaviour_policy(strategy),
            "epsilon_start": None if _is_hybrid_strategy(strategy) else epsilon_start,
            "epsilon_end": None if _is_hybrid_strategy(strategy) else epsilon_end,
            "epsilon_schedule": (
                None
                if _is_hybrid_strategy(strategy)
                else "global_interleaved_linear_decay"
                if benchmark == TARGET_NAVIGATION_BENCHMARK
                else "per_stage_linear_decay"
            ),
            "overall_success_rate": float(
                np.mean([value["success_rate"] for value in stages.values()])
            ),
            "recent_success_rate": float(
                np.mean([value["recent_success_rate"] for value in stages.values()])
            ),
            "timeline": latest_timeline,
            "stages": stages,
        }
        online_training = _merge_training_history(
            (resume_artifact or {}).get("online_training"),
            current_online_training,
        )
        policy_artifact = _movement_policy_artifact(
            strategy,
            checkpoint,
            evaluation,
            training_seed_evaluation,
            online_training,
            parent_policy_id=(
                str(resume_artifact.get("policy_id")) if resume_artifact else None
            ),
            seed=seed,
            episodes_per_stage=episodes_per_stage,
            benchmark=benchmark,
        )
        policy_artifact["training_seed_history"] = [
            *(
                list(resume_artifact.get("training_seed_history") or [])
                if resume_artifact
                else []
            ),
            *(
                [int(resume_artifact.get("training_seed"))]
                if resume_artifact
                and not resume_artifact.get("training_seed_history")
                and resume_artifact.get("training_seed") is not None
                else []
            ),
            seed,
        ]
        policy_path = strategy_dir / "movement-policy.json"
        policy_temporary = policy_path.with_suffix(".json.partial")
        policy_temporary.write_text(
            json.dumps(policy_artifact, sort_keys=True), encoding="utf-8"
        )
        policy_temporary.replace(policy_path)
        reports[strategy] = {
            "label": STRATEGY_LABELS[strategy],
            "checkpoint": str(checkpoint),
            "training": online_training,
            "training_seed_evaluation": training_seed_evaluation,
            "evaluation": evaluation,
            "policy_artifact": policy_artifact,
            "policy_artifact_path": str(policy_path),
        }
        completed_before += episodes_per_stage * len(tasks)

    comparison = sorted(
        (
            {
                "strategy": strategy,
                "label": STRATEGY_LABELS[strategy],
                "online_success_rate": reports[strategy]["training"][
                    "overall_success_rate"
                ],
                "recent_online_success_rate": reports[strategy]["training"][
                    "recent_success_rate"
                ],
                "trained_seed_success_rate": reports[strategy][
                    "training_seed_evaluation"
                ]["overall_success_rate"],
                "held_out_success_rate": reports[strategy]["evaluation"][
                    "overall_success_rate"
                ],
                "generalization_gap": reports[strategy]["training_seed_evaluation"][
                    "overall_success_rate"
                ]
                - reports[strategy]["evaluation"]["overall_success_rate"],
            }
            for strategy in selected
        ),
        key=lambda row: row["held_out_success_rate"],
        reverse=True,
    )
    report = {
        "interface": "minecraft-academy-strategy-comparison/v3",
        "benchmark": benchmark,
        "benchmark_contract": BENCHMARK_CONTRACTS[benchmark],
        "learning_mode": "online_on_policy",
        "behaviour_policy": (
            "stochastic_hybrid_actor"
            if all(_is_hybrid_strategy(item) for item in selected)
            else "strategy_specific"
            if any(_is_hybrid_strategy(item) for item in selected)
            else "epsilon_greedy"
        ),
        "epsilon_start": (
            None if all(_is_hybrid_strategy(item) for item in selected) else epsilon_start
        ),
        "epsilon_end": (
            None if all(_is_hybrid_strategy(item) for item in selected) else epsilon_end
        ),
        "epsilon_schedule": (
            None
            if all(_is_hybrid_strategy(item) for item in selected)
            else "global_interleaved_linear_decay"
            if benchmark == TARGET_NAVIGATION_BENCHMARK
            else "per_stage_linear_decay"
        ),
        "stop_when_stable": stop_when_stable,
        "parent_policy_id": (
            resume_artifact.get("policy_id") if resume_artifact else None
        ),
        "strategies": selected,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "same_training_arenas": True,
        "same_evaluation_arenas": True,
        "train_arena_manifest": train_manifest,
        "evaluation_arena_manifest": evaluation_manifest,
        "reports": reports,
        "comparison": comparison,
    }
    report_path = destination / "comparison.json"
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)
    _write_progress(
        progress_path,
        {
            "interface": "minecraft-academy-comparison-progress/v3",
            "status": "succeeded",
            "phase": "complete",
            "progress": 1.0,
            "completed_episodes": total,
            "total_episodes": total,
            "episodes_per_stage": episodes_per_stage,
            "evaluation_episodes": evaluation_episodes,
            "seed": seed,
            "strategies": selected,
            "strategy_count": len(selected),
            "reports": reports,
            "comparison": comparison,
            "report_path": str(report_path),
        },
    )
    return report


def compare_strategies(
    output_dir: str | Path,
    *,
    strategies: list[str] | tuple[str, ...],
    episodes_per_stage: int = 500,
    evaluation_episodes: int = 20,
    seed: int = 0,
    progress_path: str | Path | None = None,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    resume_policy_artifact: str | Path | None = None,
    stop_when_stable: bool = False,
    benchmark: str = TARGET_NAVIGATION_BENCHMARK,
) -> dict[str, Any]:
    """Run a comparison while making failures explicit in the progress WAL."""

    try:
        return _compare_strategies(
            output_dir,
            strategies=strategies,
            episodes_per_stage=episodes_per_stage,
            evaluation_episodes=evaluation_episodes,
            seed=seed,
            progress_path=progress_path,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            resume_policy_artifact=resume_policy_artifact,
            stop_when_stable=stop_when_stable,
            benchmark=benchmark,
        )
    except Exception as error:
        previous: dict[str, Any] = {}
        if progress_path is not None:
            try:
                loaded = json.loads(
                    Path(progress_path).expanduser().read_text(encoding="utf-8")
                )
                if isinstance(loaded, dict):
                    previous = loaded
            except (OSError, ValueError):
                pass
        _write_progress(
            progress_path,
            {
                **previous,
                "status": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "recoverable": any(
                    Path(output_dir).expanduser().glob("*/hybrid-actor-critic.pt")
                ),
            },
        )
        raise
