"""Versioned benchmark suite for the first play-and-build milestone."""

from __future__ import annotations

from .contracts import TaskManifestV1


def play_and_build_suite(world_snapshot_id: str) -> tuple[TaskManifestV1, ...]:
    common = {
        "version": 1,
        "world_snapshot_id": world_snapshot_id,
        "failure_predicates": ("player_dead", "left_task_bounds"),
        "policy_channels": ("rgb", "inventory", "vitals", "previous_action"),
        "privileged_reward_channels": (
            "absolute_position",
            "target_blocks",
            "world_delta",
        ),
    }
    return (
        TaskManifestV1(
            **common,
            task_id="play-build/navigate-visual-v1",
            title="Navegar até um alvo visual",
            family="navigate",
            initial_state={"spawn": "fixed", "inventory": []},
            goal={"visual_target": "colored_marker", "radius": 1.5},
            success_predicates=("distance_to_target <= 1.5",),
            timeout_steps=1200,
            reward_terms={"progress": 1.0, "success": 20.0, "collision": -0.05},
        ),
        TaskManifestV1(
            **common,
            task_id="play-build/follow-safe-v1",
            title="Acompanhar um personagem",
            family="follow",
            initial_state={"leader_route": "fixed", "safe_distance": [2.0, 5.0]},
            goal={"follow_ticks": 400},
            success_predicates=("safe_follow_ticks >= 400",),
            timeout_steps=900,
            reward_terms={"safe_distance": 1.0, "lost_target": -1.0, "success": 20.0},
        ),
        TaskManifestV1(
            **common,
            task_id="play-build/fetch-block-v1",
            title="Buscar um bloco indicado",
            family="collect",
            initial_state={"inventory": [], "target_visible": True},
            goal={"inventory_contains": "task_target_block", "count": 1},
            success_predicates=("target_block_count >= 1",),
            timeout_steps=1800,
            reward_terms={"target_acquired": 20.0, "wrong_block": -0.2},
        ),
        TaskManifestV1(
            **common,
            task_id="play-build/place-pattern-v1",
            title="Colocar blocos seguindo um padrão",
            family="place_pattern",
            initial_state={"inventory": {"task_block": 9}, "pattern": "checker_3x3"},
            goal={"match_ratio": 1.0},
            success_predicates=("pattern_match_ratio == 1.0",),
            timeout_steps=1200,
            reward_terms={"correct_delta": 1.0, "incorrect_delta": -1.0, "success": 20.0},
        ),
        TaskManifestV1(
            **common,
            task_id="play-build/wall-3x3-v1",
            title="Construir uma parede 3×3",
            family="build",
            initial_state={"inventory": {"task_block": 12}, "demonstration": "wall_3x3"},
            goal={"structure": "wall_3x3", "orientation": "task_marker"},
            success_predicates=("structure_match_ratio == 1.0",),
            timeout_steps=1800,
            reward_terms={"correct_delta": 1.0, "incorrect_delta": -1.0, "success": 30.0},
        ),
        TaskManifestV1(
            **common,
            task_id="play-build/short-demo-v1",
            title="Reproduzir uma estrutura curta demonstrada",
            family="build",
            initial_state={"inventory": {"task_blocks": 16}, "demonstration": "episode_reference"},
            goal={"structure": "demonstrated_short_structure"},
            success_predicates=("structure_match_ratio >= 0.95",),
            timeout_steps=2400,
            reward_terms={"correct_delta": 1.0, "incorrect_delta": -0.5, "success": 40.0},
        ),
    )
