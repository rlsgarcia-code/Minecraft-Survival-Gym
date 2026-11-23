"""Fast reference environment for the navigation academy.

This environment validates task mechanics, reward accounting, curricula, and
dataset contracts without pretending to be a visual Minecraft replacement.
Policies intended for deployment must still be trained or adapted against the
real Fabric-backed environment and its RGB observations.
"""

from __future__ import annotations

from heapq import heappop, heappush
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from minecraft_gym.actions import ControlState, decode_action, make_action_space

from .arena import (
    AcademyStage,
    ArenaBlueprint,
    NavigationBiome,
    NavigationDifficulty,
    NavigationRequirement,
    SurfaceMaterial,
    build_arena,
    build_target_arena,
)
from .policy import (
    ACTION_COUNT,
    GNOMONIC_SHAPE,
    GRID_CHANNELS,
    MAP_LATTICE_CHANNELS,
    WORLD_MAP_CHANNELS,
    WORLD_MAP_WINDOW_SIZE,
    VIEW_LATTICE_CHANNELS,
    PYRAMID_SCALES,
    PYRAMID_SIZE,
    AcademyAction,
)


TIME_COST_MAX = 20.0
TIME_COST_BETA = 3.0
STATIONARY_TIME_MULTIPLIER = 1.5
SUCCESS_TERMINAL_BONUS = 20.0
TIMEOUT_TERMINAL_PENALTY = 5.0


class NavigationAcademyEnv(gym.Env[dict[str, Any], np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        stage: AcademyStage | int | str = AcademyStage.FLAT,
        difficulty: NavigationDifficulty | int | str | None = None,
        biome: NavigationBiome | str | None = None,
        requirement: NavigationRequirement | str | None = None,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        camera_fast_degrees: float = 18.0,
    ) -> None:
        super().__init__()
        if isinstance(stage, str):
            stage = AcademyStage[stage.upper()]
        self.stage = AcademyStage(stage)
        if isinstance(difficulty, str):
            difficulty = NavigationDifficulty[difficulty.upper()]
        self.difficulty = (
            NavigationDifficulty(difficulty) if difficulty is not None else None
        )
        self.biome = NavigationBiome(biome) if biome is not None else None
        self.requirement = (
            NavigationRequirement(requirement) if requirement is not None else None
        )
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self._configured_max_steps = max_episode_steps
        self._camera_fast_degrees = float(camera_fast_degrees)
        self.action_space = make_action_space()
        self.observation_space = spaces.Dict(
            {
                "local_voxels": spaces.Box(
                    0, 5, shape=(5, 9, 9), dtype=np.uint8
                ),
                "proprioception": spaces.Box(
                    -1.0, 1.0, shape=(10,), dtype=np.float32
                ),
                "inventory": spaces.Box(0, 64, shape=(2,), dtype=np.int32),
                "goal_hint": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "stage": spaces.Discrete(len(AcademyStage)),
                "global_grid": spaces.Box(
                    0.0, 1.0, shape=(len(GRID_CHANNELS), 14, 9), dtype=np.float32
                ),
                "grid_pyramid": spaces.Box(
                    0.0,
                    1.0,
                    shape=(
                        len(PYRAMID_SCALES),
                        len(MAP_LATTICE_CHANNELS),
                        PYRAMID_SIZE,
                        PYRAMID_SIZE,
                    ),
                    dtype=np.float32,
                ),
                "map_window": spaces.Box(
                    0.0,
                    1.0,
                    shape=(
                        len(WORLD_MAP_CHANNELS),
                        WORLD_MAP_WINDOW_SIZE,
                        WORLD_MAP_WINDOW_SIZE,
                    ),
                    dtype=np.float32,
                ),
                "gnomonic_lattice": spaces.Box(
                    0.0,
                    1.0,
                    shape=(len(VIEW_LATTICE_CHANNELS), *GNOMONIC_SHAPE),
                    dtype=np.float32,
                ),
                "orientation": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                "elapsed_fraction": spaces.Box(
                    0.0, 1.0, shape=(), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(ACTION_COUNT),
            }
        )
        self.blueprint: ArenaBlueprint | None = None
        self._top: np.ndarray | None = None
        self._breakable: set[tuple[int, int, int]] = set()
        self._position = (0, 0)
        self._yaw = 0.0
        self._pitch = 0.0
        self._blocks = 0
        self._health = 1.0
        self._steps = 0
        self._max_steps = 1
        self._previous_rotated = False
        self._last_moved = 0
        self._last_route_cost = math.inf
        self._elapsed_ticks = 0
        self._step_ticks = 0
        self._step_damage = 0.0
        self._blocks_broken = 0
        self._blocks_placed = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        actual_seed = int(seed if seed is not None else self.np_random.integers(2**31))
        options = options or {}
        selected_difficulty = options.get("difficulty", self.difficulty)
        if selected_difficulty is not None:
            if isinstance(selected_difficulty, str):
                selected_difficulty = NavigationDifficulty[selected_difficulty.upper()]
            self.difficulty = NavigationDifficulty(selected_difficulty)
            selected_biome = options.get("biome", self.biome)
            selected_requirement = options.get("requirement", self.requirement)
            self.blueprint = build_target_arena(
                self.difficulty,
                actual_seed,
                biome=selected_biome,
                requirement=selected_requirement,
            )
            self.biome = self.blueprint.biome
            self.stage = AcademyStage.FLAT
        else:
            selected = options.get("stage", self.stage)
            if isinstance(selected, str):
                selected = AcademyStage[selected.upper()]
            self.stage = AcademyStage(selected)
            self.blueprint = build_arena(self.stage, actual_seed)
            self.biome = self.blueprint.biome
        self._top = self.blueprint.top_y.copy()
        self._breakable = set(self.blueprint.breakable)
        self._position = self.blueprint.spawn
        self._yaw = 0.0
        self._pitch = 0.0
        self._blocks = self.blueprint.initial_blocks
        self._health = 1.0
        self._steps = 0
        self._max_steps = self._configured_max_steps or self.blueprint.max_steps
        self._previous_rotated = False
        self._last_moved = 0
        self._last_route_cost = self._route_cost()
        self._elapsed_ticks = 0
        self._step_ticks = 0
        self._step_damage = 0.0
        self._blocks_broken = 0
        self._blocks_placed = 0
        observation = self._observation()
        return observation, self._info(events=[])

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action is outside action_space: {action!r}")
        return self.step_control(
            decode_action(action, camera_fast_degrees=self._camera_fast_degrees)
        )

    def step_control(
        self, control: ControlState
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply an unquantized canonical control.

        The ordinary Gym action remains backwards compatible.  Hybrid
        camera/body policies use this explicit adapter so latitude and
        longitude are never collapsed into the five legacy mouse bins.
        """

        if self.blueprint is None or self._top is None:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(control, ControlState):
            raise TypeError("step_control() expects a canonical ControlState")
        if not math.isfinite(control.yaw_delta) or not math.isfinite(control.pitch_delta):
            raise ValueError("camera displacement must be finite")
        self._steps += 1
        events: list[dict[str, Any]] = []
        # Time still ranks successful trajectories: reaching the target sooner
        # always yields a larger return.  Terminal terms make a late arrival
        # unambiguously better than timing out, which the bounded time cost
        # alone could not guarantee.
        terms = {
            "elapsed_time": 0.0,
            "failure_remaining_time": 0.0,
            "terminal_success": 0.0,
            "terminal_timeout": 0.0,
        }
        before_cost = self._route_cost()
        world_changed = False
        # Large mouse movements take longer than small corrections.  This
        # keeps the continuous action physically meaningful under the time
        # objective instead of making a 180-degree snap artificially free.
        camera_ticks = math.ceil(
            max(abs(control.yaw_delta) / 90.0, abs(control.pitch_delta) / 45.0) * 4.0
        )
        self._step_ticks = max(4, camera_ticks)
        self._step_damage = 0.0

        if control.yaw_delta:
            self._yaw = (self._yaw + control.yaw_delta) % 360.0
        if control.pitch_delta:
            self._pitch = float(np.clip(self._pitch + control.pitch_delta, -90.0, 90.0))

        # Minecraft exposes both mouse buttons, but one decision must not be
        # allowed to destroy and immediately replace the same block. Besides
        # being physically implausible, that would let a random chord satisfy
        # the combined affordance without learning two distinct interactions.
        # Primary action has deterministic precedence when both are sampled.
        if control.attack:
            world_changed = self._attack(events)
            self._step_ticks = max(self._step_ticks, 12)
            if control.use:
                events.append({"type": "use_suppressed", "reason": "attack_precedence"})
        elif control.use:
            world_changed = self._place(events) or world_changed
            self._step_ticks = max(self._step_ticks, 8)

        moved = 0
        if control.move_x or control.move_z:
            attempts = 2 if control.sprint and not control.sneak else 1
            for _ in range(attempts):
                if self._move(control.move_x, control.move_z, control.jump, events):
                    moved += 1
                    if self._position == self.blueprint.goal:
                        break
                else:
                    break
        self._last_moved = moved
        if moved and control.sneak:
            self._step_ticks += 4 * moved

        after_cost = self._route_cost() if (moved or world_changed) else before_cost
        elapsed_before = self._elapsed_ticks
        elapsed_after = elapsed_before + self._step_ticks
        base_time_delta = self._cumulative_time_cost(
            elapsed_after
        ) - self._cumulative_time_cost(elapsed_before)
        time_multiplier = 1.0 if moved else STATIONARY_TIME_MULTIPLIER
        terms["elapsed_time"] = -time_multiplier * base_time_delta
        self._elapsed_ticks = elapsed_after

        reached_goal = self._position == self.blueprint.goal
        requirement_satisfied = self._requirement_satisfied()
        success = reached_goal and requirement_satisfied
        failed = self._health <= 0.0
        terminated = success or failed
        if success:
            terms["terminal_success"] = SUCCESS_TERMINAL_BONUS
            events.append({"type": "task_success", "step": self._steps})
        elif failed:
            # Dying must not be a shortcut to stop paying time.  Charge the
            # minimum remaining four-tick decisions through the episode limit.
            minimum_budget_ticks = self._max_steps * 4
            terms["failure_remaining_time"] = -max(
                0.0,
                self._cumulative_time_cost(minimum_budget_ticks)
                - self._cumulative_time_cost(self._elapsed_ticks),
            )
            events.append({"type": "task_failure", "reason": "environmental_damage"})
        truncated = self._steps >= self._max_steps and not terminated
        if truncated:
            terms["terminal_timeout"] = -TIMEOUT_TERMINAL_PENALTY
        self._previous_rotated = bool(control.yaw_delta and not moved)
        self._last_route_cost = after_cost
        reward = float(sum(terms.values()))
        info = self._info(events=events)
        info["reward_terms"] = terms
        info["control_mode"] = "agent"
        info["control_source"] = "agent"
        info["executed_action"] = control.to_dict()
        info["is_success"] = success
        info["reached_goal"] = reached_goal
        info["requirement_satisfied"] = requirement_satisfied
        info["elapsed_ticks"] = self._elapsed_ticks
        info["time_multiplier"] = time_multiplier
        if truncated:
            info["truncation_reason"] = "task_timeout"
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None or self.blueprint is None or self._top is None:
            return None
        scale = 12
        image = np.full(
            (self.blueprint.length * scale, self.blueprint.width * scale, 3),
            18,
            dtype=np.uint8,
        )
        palette = {
            -1: np.asarray([22, 28, 36], dtype=np.uint8),
            0: np.asarray([175, 162, 132], dtype=np.uint8),
            1: np.asarray([153, 139, 111], dtype=np.uint8),
            2: np.asarray([126, 112, 88], dtype=np.uint8),
            3: np.asarray([102, 91, 74], dtype=np.uint8),
        }
        for x in range(self.blueprint.width):
            for z in range(self.blueprint.length):
                color = palette.get(int(self._top[x, z]), palette[3])
                image[z * scale : (z + 1) * scale, x * scale : (x + 1) * scale] = color
        gx, gz = self.blueprint.goal
        image[gz * scale : (gz + 1) * scale, gx * scale : (gx + 1) * scale] = (80, 210, 95)
        px, pz = self._position
        image[pz * scale : (pz + 1) * scale, px * scale : (px + 1) * scale] = (55, 125, 240)
        return image

    def close(self) -> None:
        self.blueprint = None
        self._top = None

    def _forward(self) -> tuple[int, int]:
        # Project the continuous camera longitude onto the nearest 8-neighbour
        # lattice direction.  Forward therefore follows the mouse heading;
        # it is not a hidden waypoint tracker and is not limited to four
        # cardinal camera bins.
        radians = math.radians(self._yaw)
        return int(round(-math.sin(radians))), int(round(math.cos(radians)))

    def _cumulative_time_cost(self, elapsed_ticks: int) -> float:
        """Return a strong but bounded cost for elapsed environment time.

        Squaring normalized time makes procrastination increasingly expensive
        near the nominal horizon.  The saturating exponential guarantees that
        unusually expensive actions can never make the return explode.
        """
        reference_ticks = max(1, self._max_steps * 4)
        fraction = max(0.0, float(elapsed_ticks) / reference_ticks)
        return TIME_COST_MAX * (1.0 - math.exp(-TIME_COST_BETA * fraction**2))

    def _target(self) -> tuple[int, int]:
        dx, dz = self._forward()
        return self._position[0] + dx, self._position[1] + dz

    def _inside(self, x: int, z: int) -> bool:
        assert self.blueprint is not None
        return 0 <= x < self.blueprint.width and 0 <= z < self.blueprint.length

    def _attack(self, events: list[dict[str, Any]]) -> bool:
        assert self._top is not None
        x, z = self._target()
        if not self._inside(x, z):
            return False
        y = int(self._top[x, z])
        if (x, y, z) not in self._breakable:
            events.append({"type": "attack_miss", "target": [x, y, z]})
            return False
        self._breakable.remove((x, y, z))
        self._top[x, z] -= 1
        self._blocks_broken += 1
        events.append({"type": "block_broken", "position": [x, y, z]})
        return True

    def _place(self, events: list[dict[str, Any]]) -> bool:
        assert self._top is not None
        if self._blocks <= 0:
            events.append({"type": "place_failed", "reason": "empty_inventory"})
            return False
        x, z = self._target()
        if not self._inside(x, z) or self._top[x, z] >= 3:
            return False
        current_y = int(self._top[self._position])
        next_y = int(self._top[x, z]) + 1
        if next_y > current_y + 1:
            return False
        self._top[x, z] = next_y
        self._blocks -= 1
        self._blocks_placed += 1
        events.append({"type": "block_placed", "position": [x, next_y, z]})
        return True

    def _requirement_satisfied(self) -> bool:
        assert self.blueprint is not None
        requirement = self.blueprint.requirement
        if requirement is NavigationRequirement.TRAVEL:
            return True
        if requirement is NavigationRequirement.BREAK:
            return self._blocks_broken > 0
        if requirement is NavigationRequirement.PLACE:
            return self._blocks_placed > 0
        return self._blocks_broken > 0 and self._blocks_placed > 0

    def _move(
        self,
        strafe: float,
        forward: float,
        jump: bool,
        events: list[dict[str, Any]],
    ) -> bool:
        assert self._top is not None
        fx, fz = self._forward()
        rx, rz = fz, -fx
        dx = int(np.sign(np.sign(forward) * fx + np.sign(strafe) * rx))
        dz = int(np.sign(np.sign(forward) * fz + np.sign(strafe) * rz))
        target = self._position[0] + dx, self._position[1] + dz
        if not self._inside(*target) or self._top[target] < 0:
            events.append({"type": "movement_blocked", "reason": "void_or_bounds"})
            return False
        current_y = int(self._top[self._position])
        target_y = int(self._top[target])
        ceiling_y = int(self.blueprint.ceiling_y[target])
        clearance = math.inf if ceiling_y < 0 else ceiling_y - target_y - 1
        if clearance < 2:
            events.append(
                {"type": "movement_blocked", "reason": "insufficient_head_clearance"}
            )
            return False
        if target_y > current_y and (target_y - current_y > 1 or not jump):
            events.append({"type": "movement_blocked", "reason": "step_requires_jump"})
            return False
        self._position = target
        surface = SurfaceMaterial(int(self.blueprint.surface[target]))
        if surface is SurfaceMaterial.POWDER_SNOW:
            self._step_ticks += 6
            damage = 0.015
            self._health = max(0.0, self._health - damage)
            self._step_damage += damage
            events.append({"type": "powder_snow_exposure", "position": list(target)})
        elif surface is SurfaceMaterial.WATER:
            self._step_ticks += 8
            events.append({"type": "swimming", "position": list(target)})
        elif surface is SurfaceMaterial.LAVA:
            self._step_ticks += 4
            damage = 0.55
            self._health = max(0.0, self._health - damage)
            self._step_damage += damage
            events.append({"type": "lava_damage", "position": list(target)})
        return True

    def _route_cost(self) -> float:
        assert self.blueprint is not None and self._top is not None
        start, goal = self._position, self.blueprint.goal
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        best = {start: 0.0}
        while queue:
            cost, cell = heappop(queue)
            if cell == goal:
                return cost
            if cost != best[cell]:
                continue
            current_y = int(self._top[cell])
            for dx, dz in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nxt = cell[0] + dx, cell[1] + dz
                if not self._inside(*nxt):
                    continue
                target_y = int(self._top[nxt])
                edge = math.inf
                if target_y < 0:
                    if self._blocks > 0:
                        edge = 5.0
                elif target_y <= current_y:
                    edge = 1.0
                elif target_y == current_y + 1:
                    edge = 1.5
                else:
                    breakable_layers = sum(
                        (nxt[0], y, nxt[1]) in self._breakable
                        for y in range(current_y + 1, target_y + 1)
                    )
                    if breakable_layers >= target_y - current_y - 1:
                        edge = 1.5 + 2.0 * (target_y - current_y - 1)
                    elif target_y == current_y + 2 and self._blocks > 0:
                        edge = 5.0
                if math.isfinite(edge):
                    surface = SurfaceMaterial(int(self.blueprint.surface[nxt]))
                    edge *= {
                        SurfaceMaterial.NORMAL: 1.0,
                        SurfaceMaterial.POWDER_SNOW: 2.5,
                        SurfaceMaterial.WATER: 3.0,
                        SurfaceMaterial.LAVA: 20.0,
                    }[surface]
                candidate = cost + edge
                if candidate < best.get(nxt, math.inf):
                    best[nxt] = candidate
                    heappush(queue, (candidate, nxt))
        return math.inf

    def _observation(self) -> dict[str, Any]:
        assert self.blueprint is not None and self._top is not None
        px, pz = self._position
        py = int(self._top[self._position]) + 1
        voxels = np.zeros((5, 9, 9), dtype=np.uint8)
        for ix, x in enumerate(range(px - 4, px + 5)):
            for iz, z in enumerate(range(pz - 4, pz + 5)):
                if not self._inside(x, z):
                    voxels[:, iz, ix] = 5
                    continue
                top = int(self._top[x, z])
                for iy, y in enumerate(range(py - 2, py + 3)):
                    if 0 <= y <= top:
                        voxels[iy, iz, ix] = 2 if (x, y, z) in self._breakable else 1
                    if (x, z) == self.blueprint.goal and y == top + 1:
                        voxels[iy, iz, ix] = 3
                    if y > int(self.blueprint.base_y[x, z]) and y <= top and voxels[iy, iz, ix] == 1:
                        voxels[iy, iz, ix] = 4
        dx = self.blueprint.goal[0] - px
        dz = self.blueprint.goal[1] - pz
        radians = math.radians(self._yaw)
        ego_x = math.cos(radians) * dx - math.sin(radians) * dz
        ego_z = math.sin(radians) * dx + math.cos(radians) * dz
        scale = max(self.blueprint.width, self.blueprint.length)
        goal_hint = np.asarray([ego_x / scale, ego_z / scale], dtype=np.float32)
        progress = min(1.0, self._elapsed_ticks / max(1, self._max_steps * 4))
        proprioception = np.asarray(
            [
                math.sin(radians),
                math.cos(radians),
                np.clip(py / 8.0, -1.0, 1.0),
                1.0,
                np.clip(self._last_moved / 2.0, 0.0, 1.0),
                np.clip(self._blocks / 8.0, 0.0, 1.0),
                int(self.stage) / max(1, len(AcademyStage) - 1),
                progress,
                self._health,
                0.0,
            ],
            dtype=np.float32,
        )
        grid = np.zeros(
            (len(GRID_CHANNELS), self.blueprint.length, self.blueprint.width),
            dtype=np.float32,
        )
        for x in range(self.blueprint.width):
            for z in range(self.blueprint.length):
                top = int(self._top[x, z])
                base = int(self.blueprint.base_y[x, z])
                grid[0, z, x] = np.clip((top + 1) / 4.0, 0.0, 1.0)
                grid[1, z, x] = float(top < 0)
                grid[2, z, x] = float(any((x, y, z) in self._breakable for y in range(4)))
                grid[3, z, x] = float(top > base)
                grid[4, z, x] = float(top >= 0)
                grid[7, z, x] = 1.0
        grid[5, pz, px] = 1.0
        gx, gz = self.blueprint.goal
        grid[6, gz, gx] = 1.0
        orientation = np.zeros(4, dtype=np.float32)
        orientation[int(round(self._yaw / 90.0)) % 4] = 1.0
        pyramid = self._grid_pyramid()
        map_window = self._map_window()
        gnomonic = self._gnomonic_lattice()
        action_mask = np.ones(ACTION_COUNT, dtype=np.int8)
        tx, tz = self._target()
        break_valid = False
        place_valid = False
        if self._inside(tx, tz):
            target_top = int(self._top[tx, tz])
            break_valid = (tx, target_top, tz) in self._breakable
            current_top = int(self._top[self._position])
            place_valid = self._blocks > 0 and target_top < 3 and target_top + 1 <= current_top + 1
        action_mask[AcademyAction.BREAK] = int(break_valid)
        action_mask[AcademyAction.PLACE] = int(place_valid)
        return {
            "local_voxels": voxels,
            "proprioception": proprioception,
            "inventory": np.asarray([self._blocks, 1], dtype=np.int32),
            "goal_hint": goal_hint,
            "stage": int(self.stage),
            "global_grid": grid,
            "grid_pyramid": pyramid,
            "map_window": map_window,
            "gnomonic_lattice": gnomonic,
            "orientation": orientation,
            "elapsed_fraction": np.asarray(progress, dtype=np.float32),
            "action_mask": action_mask,
        }

    def _info(self, *, events: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.blueprint is not None and self._top is not None
        privileged = {
            "absolute_position": list(self._position),
            "absolute_goal": list(self.blueprint.goal),
            "route_cost": self._last_route_cost,
            "minimum_expected_travel_time": self._last_route_cost,
            "world_delta_columns": int(np.count_nonzero(self._top != self.blueprint.base_y)),
        }
        return {
            "tick": self._elapsed_ticks,
            "task_id": (
                f"target-navigation/{self.blueprint.difficulty.slug}-v3"
                if self.blueprint.difficulty is not None
                else f"navigation-academy/{self.stage.slug}-v2"
            ),
            "arena_sha256": self.blueprint.logical_sha256,
            "biome": self.blueprint.biome.value,
            "difficulty": (
                self.blueprint.difficulty.slug
                if self.blueprint.difficulty is not None
                else None
            ),
            "requirement": self.blueprint.requirement.value,
            "events": events,
            "interaction_counts": {
                "blocks_broken": self._blocks_broken,
                "blocks_placed": self._blocks_placed,
            },
            "requirement_satisfied": self._requirement_satisfied(),
            "policy_channels": [
                "map_window",
                "gnomonic_lattice",
                "orientation",
                "inventory",
                "elapsed_fraction",
                "action_mask",
            ],
            "privileged_evaluator": privileged,
            "privileged_context": privileged,
        }

    def _map_window(self) -> np.ndarray:
        """Render the authoritative arena into one metric world-aligned crop."""

        assert self.blueprint is not None and self._top is not None
        result = np.zeros(
            (len(WORLD_MAP_CHANNELS), WORLD_MAP_WINDOW_SIZE, WORLD_MAP_WINDOW_SIZE),
            dtype=np.float32,
        )
        px, pz = self._position
        current_top = int(self._top[px, pz])
        half = WORLD_MAP_WINDOW_SIZE // 2
        for x in range(self.blueprint.width):
            for z in range(self.blueprint.length):
                row = half - (z - pz)
                column = half + (x - px)
                if not (
                    0 <= row < WORLD_MAP_WINDOW_SIZE
                    and 0 <= column < WORLD_MAP_WINDOW_SIZE
                ):
                    continue
                top = int(self._top[x, z])
                base = int(self.blueprint.base_y[x, z])
                ceiling = int(self.blueprint.ceiling_y[x, z])
                breakable = any((x, y, z) in self._breakable for y in range(4))
                surface = SurfaceMaterial(int(self.blueprint.surface[x, z]))
                material = (
                    0.0
                    if top < 0
                    else 0.75
                    if breakable
                    else 1.0
                    if top > base
                    else {
                        SurfaceMaterial.NORMAL: 0.4,
                        SurfaceMaterial.POWDER_SNOW: 0.25,
                        SurfaceMaterial.WATER: 0.1,
                        SurfaceMaterial.LAVA: 0.0,
                    }[surface]
                )
                clearance = 4.0 if ceiling < 0 else float(max(0, ceiling - top - 1))
                stack = float(max(0, top - base))
                result[0, row, column] = 1.0
                result[1, row, column] = float((x, z) == self._position)
                result[2, row, column] = np.clip(
                    (top - current_top + 4.0) / 8.0, 0.0, 1.0
                )
                result[3, row, column] = material
                result[4, row, column] = np.clip(stack / 4.0, 0.0, 1.0)
                result[5, row, column] = np.clip(clearance / 4.0, 0.0, 1.0)
                result[6, row, column] = float(
                    top >= 0 and clearance >= 2.0 and surface is not SurfaceMaterial.LAVA
                )

        gx, gz = self.blueprint.goal
        dx, dz = gx - px, gz - pz
        magnitude = max(abs(dx), abs(dz))
        scale = 1.0 if magnitude <= half else half / float(magnitude)
        goal_column = half + int(round(dx * scale))
        goal_row = half - int(round(dz * scale))
        result[7, goal_row, goal_column] = 1.0
        return result

    def _grid_pyramid(self) -> np.ndarray:
        """Project one authoritative arena into three egocentric map scales."""

        assert self.blueprint is not None and self._top is not None
        result = np.zeros(
            (
                len(PYRAMID_SCALES),
                len(MAP_LATTICE_CHANNELS),
                PYRAMID_SIZE,
                PYRAMID_SIZE,
            ),
            dtype=np.float32,
        )
        px, pz = self._position
        half = PYRAMID_SIZE // 2
        radians = math.radians(self._yaw)
        fx, fz = -math.sin(radians), math.cos(radians)
        rx, rz = fz, -fx
        for level, scale in enumerate(PYRAMID_SCALES):
            buckets: dict[
                tuple[int, int], list[tuple[int, float, bool, float, float]]
            ] = {}
            for x in range(self.blueprint.width):
                for z in range(self.blueprint.length):
                    dx, dz = x - px, z - pz
                    right = dx * rx + dz * rz
                    forward = dx * fx + dz * fz
                    column = half + math.floor((right + scale / 2.0) / scale)
                    row = half - math.floor((forward + scale / 2.0) / scale)
                    if not (0 <= row < PYRAMID_SIZE and 0 <= column < PYRAMID_SIZE):
                        continue
                    top = int(self._top[x, z])
                    if top < 0:
                        material = 0.0
                    elif any((x, y, z) in self._breakable for y in range(4)):
                        material = 0.75
                    elif top > int(self.blueprint.base_y[x, z]):
                        material = 1.0
                    else:
                        material = {
                            SurfaceMaterial.NORMAL: 0.4,
                            SurfaceMaterial.POWDER_SNOW: 0.25,
                            SurfaceMaterial.WATER: 0.1,
                            SurfaceMaterial.LAVA: 0.0,
                        }[SurfaceMaterial(int(self.blueprint.surface[x, z]))]
                    buckets.setdefault((row, column), []).append(
                        (
                            top,
                            material,
                            (x, z) == self.blueprint.goal,
                            float(max(0, top - int(self.blueprint.base_y[x, z]))),
                            (
                                4.0
                                if int(self.blueprint.ceiling_y[x, z]) < 0
                                else float(
                                    max(
                                        0,
                                        int(self.blueprint.ceiling_y[x, z]) - top - 1,
                                    )
                                )
                            ),
                        )
                    )
            for (row, column), values in buckets.items():
                heights = [value[0] for value in values]
                result[level, 0, row, column] = min(
                    1.0, len(values) / float(scale * scale)
                )
                result[level, 1, row, column] = np.clip(
                    (float(np.mean(heights)) + 1.0) / 4.0, 0.0, 1.0
                )
                result[level, 2, row, column] = np.clip(
                    (max(heights) - min(heights)) / 4.0, 0.0, 1.0
                )
                result[level, 3, row, column] = sum(value[2] for value in values) / len(values)
                result[level, 4, row, column] = float(
                    np.mean([value[1] for value in values])
                )
                # Barrier height uses the worst (tallest) column in a coarse
                # bin; clearance uses the worst (lowest) overhead gap.
                result[level, 5, row, column] = np.clip(
                    max(value[3] for value in values) / 4.0, 0.0, 1.0
                )
                result[level, 6, row, column] = np.clip(
                    min(value[4] for value in values) / 4.0, 0.0, 1.0
                )
        return result

    def _gnomonic_lattice(self) -> np.ndarray:
        """Low-cost pinhole projection used by the structured academy."""

        assert self.blueprint is not None and self._top is not None
        height, width = GNOMONIC_SHAPE
        result = np.zeros(
            (len(VIEW_LATTICE_CHANNELS), height, width), dtype=np.float32
        )
        px, pz = self._position
        eye_y = float(self._top[self._position]) + 1.62
        for column in range(width):
            horizontal = (column + 0.5) / width - 0.5
            angle = math.radians(self._yaw + horizontal * 90.0)
            for distance in np.linspace(0.75, 16.0, 32):
                x = int(math.floor(px + 0.5 - math.sin(angle) * distance))
                z = int(math.floor(pz + 0.5 + math.cos(angle) * distance))
                if not self._inside(x, z):
                    continue
                top = int(self._top[x, z])
                if top < 0:
                    continue
                elevation = math.degrees(math.atan2((top + 1.0) - eye_y, distance))
                elevation -= self._pitch
                row = int(np.clip(round((0.5 - elevation / 60.0) * (height - 1)), 0, height - 1))
                result[0, row:, column] = 1.0
                result[1, row:, column] = 1.0 - min(1.0, distance / 16.0)
                result[2, row:, column] = np.clip((top + 1.0) / 4.0, 0.0, 1.0)
                if (x, z) == self.blueprint.goal:
                    result[3, max(0, row - 1) : row + 1, column] = 1.0
                if any((x, y, z) in self._breakable for y in range(4)):
                    material = 0.75
                elif top > int(self.blueprint.base_y[x, z]):
                    material = 1.0
                else:
                    material = {
                        SurfaceMaterial.NORMAL: 0.4,
                        SurfaceMaterial.POWDER_SNOW: 0.25,
                        SurfaceMaterial.WATER: 0.1,
                        SurfaceMaterial.LAVA: 0.0,
                    }[SurfaceMaterial(int(self.blueprint.surface[x, z]))]
                result[4, row:, column] = material
                break
        return result
