"""Curriculum manifests and promotion rules for locomotion training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from minecraft_gym.dataset.contracts import TaskManifestV1

from .arena import (
    AcademyStage,
    NavigationDifficulty,
    build_arena,
    build_target_arena,
)


def target_navigation_task_suite(seed: int = 0) -> tuple[TaskManifestV1, ...]:
    """Public task contracts for random-biome, time-efficient navigation."""

    tasks: list[TaskManifestV1] = []
    titles = {
        NavigationDifficulty.EASY: "Alvo próximo em ambiente aberto",
        NavigationDifficulty.MODERATE: "Alvo com relevo e terreno lento",
        NavigationDifficulty.HARD: "Alvo distante entre obstáculos e riscos",
        NavigationDifficulty.EXPERT: "Alvo em ambiente severo",
        NavigationDifficulty.MIXED: "Alvo aleatório em bioma misto",
    }
    for difficulty in NavigationDifficulty:
        arena = build_target_arena(difficulty, seed)
        tasks.append(
            TaskManifestV1(
                task_id=f"target-navigation/{difficulty.slug}-v3",
                version=3,
                title=titles[difficulty],
                family="navigate",
                world_snapshot_id=f"target-navigation:{arena.logical_sha256}",
                initial_state={
                    "difficulty": difficulty.slug,
                    "biome": arena.biome.value,
                    "target_initially_known": True,
                    "required_affordance": arena.requirement.value,
                    "construction_allowed": True,
                    "inventory_blocks": arena.initial_blocks,
                },
                goal={"marker": "target", "radius": 0.75},
                success_predicates=(
                    "distance_to_goal <= 0.75",
                    "required_affordance_satisfied",
                ),
                failure_predicates=("player_dead", "left_arena_bounds"),
                timeout_steps=arena.max_steps,
                reward_terms={
                    "time_cost_max": 20.0,
                    "time_cost_beta": 3.0,
                    "stationary_time_multiplier": 1.5,
                    "terminal_success_bonus": 20.0,
                    "terminal_timeout_penalty": 5.0,
                },
                policy_channels=(
                    "map_window",
                    "gnomonic_lattice",
                    "orientation",
                    "inventory",
                    "elapsed_fraction",
                    "action_mask",
                ),
                privileged_reward_channels=("elapsed_ticks", "task_success"),
            )
        )
    return tuple(tasks)


def academy_task_suite(seed: int = 0) -> tuple[TaskManifestV1, ...]:
    tasks: list[TaskManifestV1] = []
    titles = {
        AcademyStage.FLAT: "Chegar ao alvo em terreno plano",
        AcademyStage.CLIMB: "Subir degraus e pequenos desníveis",
        AcademyStage.BREAK: "Abrir uma passagem quebrando o necessário",
        AcademyStage.BUILD: "Construir ponte e degrau para alcançar o alvo",
        AcademyStage.COMBINED: "Navegar combinando movimento, quebra e construção",
    }
    for stage in AcademyStage:
        arena = build_arena(stage, seed)
        family = "build" if stage in {AcademyStage.BUILD, AcademyStage.COMBINED} else "navigate"
        tasks.append(
            TaskManifestV1(
                task_id=f"navigation-academy/{stage.slug}-v2",
                version=2,
                title=titles[stage],
                family=family,
                world_snapshot_id=f"navigation-academy:{arena.logical_sha256}",
                initial_state={
                    "arena_stage": stage.slug,
                    "spawn": "fixed_by_snapshot",
                    "inventory_blocks": arena.initial_blocks,
                },
                goal={"marker": "lime_concrete", "radius": 0.75},
                success_predicates=("distance_to_goal <= 0.75",),
                failure_predicates=("player_dead", "left_arena_bounds"),
                timeout_steps=arena.max_steps,
                reward_terms={
                    "time_cost_max": 20.0,
                    "time_cost_beta": 3.0,
                    "stationary_time_multiplier": 1.5,
                    "terminal_success_bonus": 20.0,
                    "terminal_timeout_penalty": 5.0,
                },
                policy_channels=(
                    "rgb",
                    "inventory",
                    "vitals",
                    "previous_action",
                    "task_goal_hint",
                ),
                privileged_reward_channels=("elapsed_ticks", "task_success"),
            )
        )
    return tuple(tasks)


@dataclass(slots=True)
class CurriculumController:
    """Advance only after sustained success, never after one lucky episode."""

    window_size: int = 20
    promotion_success_rate: float = 0.8
    stage: AcademyStage = AcademyStage.FLAT
    _history: dict[AcademyStage, deque[bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 < self.promotion_success_rate <= 1:
            raise ValueError("promotion_success_rate must be in (0, 1]")
        self._history = {
            stage: deque(maxlen=self.window_size) for stage in AcademyStage
        }

    def record(self, success: bool) -> AcademyStage:
        history = self._history[self.stage]
        history.append(bool(success))
        if (
            len(history) == self.window_size
            and sum(history) / self.window_size >= self.promotion_success_rate
            and self.stage is not AcademyStage.COMBINED
        ):
            self.stage = AcademyStage(int(self.stage) + 1)
        return self.stage

    def report(self) -> dict[str, object]:
        history = self._history[self.stage]
        return {
            "stage": self.stage.slug,
            "episodes_in_window": len(history),
            "success_rate": sum(history) / len(history) if history else 0.0,
            "window_size": self.window_size,
            "promotion_success_rate": self.promotion_success_rate,
        }
