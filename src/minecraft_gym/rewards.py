"""Replaceable reward strategies for Minecraft trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from minecraft_gym.backend import BackendObservation, BackendTransition


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    terms: dict[str, float]


class RewardFunction(Protocol):
    def __call__(
        self,
        previous: BackendObservation,
        transition: BackendTransition,
    ) -> RewardResult: ...


@dataclass(slots=True)
class SparseSurvivalReward:
    """Conservative default reward; richer rewards belong in wrappers."""

    alive_per_second: float = 0.001
    damage_per_health_fraction: float = -5.0
    death_penalty: float = -20.0
    ticks_per_second: float = 20.0

    def __call__(
        self,
        previous: BackendObservation,
        transition: BackendTransition,
    ) -> RewardResult:
        current = transition.observation
        elapsed_seconds = max(0, current.tick - previous.tick) / self.ticks_per_second
        health_lost = max(0.0, float(previous.vitals[0] - current.vitals[0]))
        terms: dict[str, float] = {
            "alive": self.alive_per_second * elapsed_seconds,
            "damage": self.damage_per_health_fraction * health_lost,
            "death": self.death_penalty if transition.terminated else 0.0,
        }
        return RewardResult(total=float(sum(terms.values())), terms=terms)
