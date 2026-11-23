"""Deterministic voxel arenas shared by tests, rewards, and world generation.

The arena representation is intentionally small and declarative.  The fast
Gym environment consumes it directly, while ``world_bundle`` translates the
same columns into Minecraft commands.  This prevents the simulated curriculum
and the real training world from silently becoming different tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import json

import numpy as np


class AcademyStage(IntEnum):
    FLAT = 0
    CLIMB = 1
    BREAK = 2
    BUILD = 3
    COMBINED = 4

    @property
    def slug(self) -> str:
        return self.name.lower()


class NavigationDifficulty(IntEnum):
    """Difficulty bands used by the open target-navigation benchmark."""

    EASY = 0
    MODERATE = 1
    HARD = 2
    EXPERT = 3
    MIXED = 4

    @property
    def slug(self) -> str:
        return self.name.lower()


class NavigationBiome(str, Enum):
    PLAINS = "plains"
    JUNGLE = "jungle"
    SNOWY = "snowy"
    RIVER = "river"
    VOLCANIC = "volcanic"
    MIXED = "mixed"


class NavigationRequirement(str, Enum):
    """Physical affordance that a successful route must exercise."""

    TRAVEL = "travel"
    BREAK = "break"
    PLACE = "place"
    COMBINED = "combined"


class SurfaceMaterial(IntEnum):
    NORMAL = 0
    POWDER_SNOW = 1
    WATER = 2
    LAVA = 3


@dataclass(frozen=True, slots=True)
class ArenaBlueprint:
    stage: AcademyStage
    seed: int
    top_y: np.ndarray
    base_y: np.ndarray
    ceiling_y: np.ndarray
    surface: np.ndarray
    breakable: frozenset[tuple[int, int, int]]
    spawn: tuple[int, int]
    goal: tuple[int, int]
    biome: NavigationBiome
    difficulty: NavigationDifficulty | None
    requirement: NavigationRequirement
    initial_blocks: int
    max_steps: int

    def __post_init__(self) -> None:
        if (
            self.top_y.shape != self.base_y.shape
            or self.top_y.shape != self.ceiling_y.shape
            or self.top_y.shape != self.surface.shape
            or self.top_y.ndim != 2
        ):
            raise ValueError("arena height arrays must be matching 2-D grids")
        if (
            self.top_y.dtype.kind not in "iu"
            or self.base_y.dtype.kind not in "iu"
            or self.ceiling_y.dtype.kind not in "iu"
            or self.surface.dtype.kind not in "iu"
        ):
            raise ValueError("arena height arrays must contain integers")
        for point in (self.spawn, self.goal):
            x, z = point
            if not (0 <= x < self.width and 0 <= z < self.length):
                raise ValueError("spawn and goal must be inside the arena")
            if self.top_y[x, z] < 0:
                raise ValueError("spawn and goal must have solid ground")

    @property
    def width(self) -> int:
        return int(self.top_y.shape[0])

    @property
    def length(self) -> int:
        return int(self.top_y.shape[1])

    @property
    def logical_sha256(self) -> str:
        value = {
            "stage": self.stage.slug,
            "seed": self.seed,
            "top_y": self.top_y.tolist(),
            "base_y": self.base_y.tolist(),
            "ceiling_y": self.ceiling_y.tolist(),
            "surface": self.surface.tolist(),
            "breakable": sorted([list(item) for item in self.breakable]),
            "spawn": list(self.spawn),
            "goal": list(self.goal),
            "biome": self.biome.value,
            "difficulty": self.difficulty.slug if self.difficulty is not None else None,
            "requirement": self.requirement.value,
            "initial_blocks": self.initial_blocks,
            "max_steps": self.max_steps,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_manifest(self) -> dict[str, object]:
        return {
            "interface": "minecraft-navigation-arena/v3",
            "stage": self.stage.slug,
            "seed": self.seed,
            "logical_sha256": self.logical_sha256,
            "shape": [self.width, self.length],
            "spawn": list(self.spawn),
            "goal": list(self.goal),
            "biome": self.biome.value,
            "difficulty": self.difficulty.slug if self.difficulty is not None else None,
            "requirement": self.requirement.value,
            "initial_blocks": self.initial_blocks,
            "max_steps": self.max_steps,
            "policy_channels": [
                "map_window",
                "gnomonic_lattice",
                "orientation",
                "inventory",
                "elapsed_fraction",
                "action_mask",
            ],
            "privileged_reward_channels": [
                "absolute_position",
                "absolute_goal",
                "route_cost",
                "world_delta",
            ],
        }


def _corridor() -> tuple[np.ndarray, np.ndarray]:
    width, length = 9, 14
    top = np.full((width, length), -1, dtype=np.int16)
    top[3:6, :] = 0
    base = top.copy()
    return top, base


def build_arena(stage: AcademyStage | int | str, seed: int = 0) -> ArenaBlueprint:
    """Build one reproducible stage without depending on a live Minecraft save."""

    if isinstance(stage, str):
        stage = AcademyStage[stage.upper()]
    else:
        stage = AcademyStage(stage)
    rng = np.random.default_rng(seed)
    top, base = _corridor()
    ceiling = np.full(top.shape, -1, dtype=np.int16)
    surface = np.full(top.shape, int(SurfaceMaterial.NORMAL), dtype=np.uint8)
    center = 4 + int(rng.integers(-1, 2))
    center = max(3, min(5, center))
    spawn = (center, 1)
    goal = (center, 12)
    breakable: set[tuple[int, int, int]] = set()
    blocks = 0
    max_steps = 180

    if stage is AcademyStage.CLIMB:
        first_step = 4 + seed % 3
        second_step = 8 + seed % 2
        top[3:6, first_step:second_step] = 1
        top[3:6, second_step:] = 2
        base = top.copy()
        max_steps = 240
    elif stage is AcademyStage.BREAK:
        wall_z = 5 + (seed * 3) % 4
        top[3:6, wall_z] = 2
        base[3:6, wall_z] = 0
        breakable.update((x, y, wall_z) for x in range(3, 6) for y in (1, 2))
        max_steps = 300
    elif stage is AcademyStage.BUILD:
        gap_z = 4 + (seed * 2) % 4
        rise_z = 9 + seed % 3
        top[3:6, gap_z] = -1
        base = top.copy()
        top[3:6, rise_z:] = 2
        base[3:6, rise_z:] = 2
        blocks = 8
        max_steps = 420
    elif stage is AcademyStage.COMBINED:
        wall_z = 3 + seed % 3
        gap_z = 6 + seed % 2
        rise_z = 10 + seed % 2
        top[3:6, wall_z] = 2
        base[3:6, wall_z] = 0
        breakable.update((x, y, wall_z) for x in range(3, 6) for y in (1, 2))
        top[3:6, gap_z] = -1
        base[3:6, gap_z] = -1
        top[3:6, rise_z:] = 2
        base[3:6, rise_z:] = 2
        blocks = 8
        max_steps = 600

    return ArenaBlueprint(
        stage=stage,
        seed=seed,
        top_y=top,
        base_y=base,
        ceiling_y=ceiling,
        surface=surface,
        breakable=frozenset(breakable),
        spawn=spawn,
        goal=goal,
        biome=NavigationBiome.PLAINS,
        difficulty=None,
        requirement={
            AcademyStage.FLAT: NavigationRequirement.TRAVEL,
            AcademyStage.CLIMB: NavigationRequirement.TRAVEL,
            AcademyStage.BREAK: NavigationRequirement.BREAK,
            AcademyStage.BUILD: NavigationRequirement.PLACE,
            AcademyStage.COMBINED: NavigationRequirement.COMBINED,
        }[stage],
        initial_blocks=blocks,
        max_steps=max_steps,
    )


_BIOMES_BY_DIFFICULTY: dict[NavigationDifficulty, tuple[NavigationBiome, ...]] = {
    NavigationDifficulty.EASY: (
        NavigationBiome.PLAINS,
        NavigationBiome.JUNGLE,
    ),
    NavigationDifficulty.MODERATE: (
        NavigationBiome.JUNGLE,
        NavigationBiome.SNOWY,
        NavigationBiome.RIVER,
    ),
    NavigationDifficulty.HARD: (
        NavigationBiome.JUNGLE,
        NavigationBiome.SNOWY,
        NavigationBiome.RIVER,
        NavigationBiome.VOLCANIC,
    ),
    NavigationDifficulty.EXPERT: (
        NavigationBiome.SNOWY,
        NavigationBiome.RIVER,
        NavigationBiome.VOLCANIC,
        NavigationBiome.MIXED,
    ),
    NavigationDifficulty.MIXED: tuple(NavigationBiome),
}


_REQUIREMENTS_BY_DIFFICULTY: dict[
    NavigationDifficulty, tuple[NavigationRequirement, ...]
] = {
    # Even the introductory distribution contains interaction episodes.  The
    # difference between difficulty bands is distance, clutter and hazards,
    # not whether breaking/placing exists at all.
    NavigationDifficulty.EASY: (
        NavigationRequirement.TRAVEL,
        NavigationRequirement.BREAK,
        NavigationRequirement.PLACE,
    ),
    NavigationDifficulty.MODERATE: tuple(NavigationRequirement),
    NavigationDifficulty.HARD: tuple(NavigationRequirement),
    NavigationDifficulty.EXPERT: tuple(NavigationRequirement),
    NavigationDifficulty.MIXED: tuple(NavigationRequirement),
}


def _safe_route(spawn: tuple[int, int], goal: tuple[int, int]) -> set[tuple[int, int]]:
    """One deterministic traversable reference route used by hazardous arenas."""

    x, z = spawn
    route = {(x, z)}
    while z != goal[1]:
        z += 1 if goal[1] > z else -1
        route.add((x, z))
    while x != goal[0]:
        x += 1 if goal[0] > x else -1
        route.add((x, z))
    return route


def build_target_arena(
    difficulty: NavigationDifficulty | int | str,
    seed: int = 0,
    *,
    biome: NavigationBiome | str | None = None,
    requirement: NavigationRequirement | str | None = None,
) -> ArenaBlueprint:
    """Build a random, reproducible target-navigation challenge.

    Difficulty controls target distance, obstacle density and hazard exposure;
    it never grants the policy privileged coordinates.  Every generated arena
    contains a declared physical requirement.  BREAK and PLACE gates span the
    full arena width, so successful trajectories cannot silently route around
    the interaction being evaluated.
    """

    if isinstance(difficulty, str):
        difficulty = NavigationDifficulty[difficulty.upper()]
    else:
        difficulty = NavigationDifficulty(difficulty)
    rng = np.random.default_rng(seed)
    if biome is None:
        choices = _BIOMES_BY_DIFFICULTY[difficulty]
        biome = choices[int(rng.integers(len(choices)))]
    else:
        biome = NavigationBiome(biome)
    if requirement is None:
        requirements = _REQUIREMENTS_BY_DIFFICULTY[difficulty]
        requirement = requirements[seed % len(requirements)]
    else:
        requirement = NavigationRequirement(requirement)

    width, length = 9, 14
    top = np.zeros((width, length), dtype=np.int16)
    base = top.copy()
    ceiling = np.full(top.shape, -1, dtype=np.int16)
    surface = np.full(top.shape, int(SurfaceMaterial.NORMAL), dtype=np.uint8)
    spawn = (4, 1)
    distance_ranges = {
        NavigationDifficulty.EASY: (4, 7),
        NavigationDifficulty.MODERATE: (6, 10),
        NavigationDifficulty.HARD: (8, 12),
        NavigationDifficulty.EXPERT: (10, 13),
        NavigationDifficulty.MIXED: (5, 13),
    }
    low, high = distance_ranges[difficulty]
    goal_z = int(rng.integers(low, high + 1))
    lateral = int(difficulty)
    goal_x = int(np.clip(4 + rng.integers(-min(3, lateral + 1), min(3, lateral + 1) + 1), 1, 7))
    goal = (goal_x, goal_z)
    protected = {spawn, goal}
    gate_candidates = list(range(3, max(4, goal_z)))
    break_gate_z: int | None = None
    place_gate_z: int | None = None
    if requirement in {NavigationRequirement.BREAK, NavigationRequirement.COMBINED}:
        break_gate_z = gate_candidates[max(0, len(gate_candidates) // 3)]
    if requirement in {NavigationRequirement.PLACE, NavigationRequirement.COMBINED}:
        place_gate_z = gate_candidates[max(0, (2 * len(gate_candidates)) // 3)]
        if place_gate_z == break_gate_z and len(gate_candidates) > 1:
            place_gate_z = gate_candidates[-1]
    gate_rows = {value for value in (break_gate_z, place_gate_z) if value is not None}
    protected.update(
        (x, z)
        for z in gate_rows
        for x in range(width)
    )
    reference_route = _safe_route(spawn, goal)
    breakable: set[tuple[int, int, int]] = set()

    density = (2, 4, 7, 10, 8)[int(difficulty)]

    def add_tree(x: int, z: int) -> None:
        if (x, z) in protected:
            return
        ground = int(base[x, z])
        top[x, z] = ground + 2
        breakable.update({(x, ground + 1, z), (x, ground + 2, z)})

    def scatter_trees(count: int) -> None:
        candidates = [
            (x, z)
            for x in range(1, width - 1)
            for z in range(3, goal_z)
            if (x, z) not in protected
        ]
        rng.shuffle(candidates)
        for x, z in candidates[:count]:
            add_tree(x, z)

    def add_surface_band(material: SurfaceMaterial, *, thickness: int) -> None:
        if goal_z <= 4:
            return
        start = int(rng.integers(3, max(4, goal_z - thickness + 1)))
        surface[1 : width - 1, start : min(goal_z, start + thickness)] = int(material)

    if difficulty >= NavigationDifficulty.MODERATE:
        # Small height changes exercise jumping without turning the benchmark
        # into a construction task. Build terrain before adding trees so their
        # breakable coordinates remain aligned with the elevated ground.
        z = min(goal_z - 1, 4 + seed % max(1, goal_z - 4))
        if z > 2:
            top[1 : width - 1, z:] += 1
            base[1 : width - 1, z:] += 1

    if biome in {NavigationBiome.JUNGLE, NavigationBiome.MIXED}:
        scatter_trees(density)
        # Harder jungle samples contain a breakable shortcut across the direct
        # route, but going around remains possible.
        if difficulty >= NavigationDifficulty.HARD and goal_z > 6:
            tree_z = max(3, goal_z // 2)
            for x in range(2, width - 2):
                if x != 1 + (seed % (width - 2)):
                    add_tree(x, tree_z)

        # Off-route canopy cells exercise vertical topology without removing
        # the deterministic safe route.  top+2 blocks the two-block body;
        # top+3 remains a passable low tunnel.
        if difficulty >= NavigationDifficulty.HARD:
            canopy_candidates = [
                (x, z)
                for x in range(1, width - 1)
                for z in range(3, goal_z)
                if (x, z) not in protected and (x, z) not in reference_route
                and top[x, z] == base[x, z]
            ]
            rng.shuffle(canopy_candidates)
            for index, (x, z) in enumerate(
                canopy_candidates[: max(1, density // 3)]
            ):
                ceiling[x, z] = int(top[x, z]) + (2 if index % 2 == 0 else 3)

    if biome in {NavigationBiome.SNOWY, NavigationBiome.MIXED}:
        add_surface_band(
            SurfaceMaterial.POWDER_SNOW,
            thickness=1 + int(difficulty >= NavigationDifficulty.HARD),
        )

    if biome in {NavigationBiome.RIVER, NavigationBiome.MIXED}:
        add_surface_band(
            SurfaceMaterial.WATER,
            thickness=1 + int(difficulty >= NavigationDifficulty.EXPERT),
        )

    if biome in {NavigationBiome.VOLCANIC, NavigationBiome.MIXED}:
        candidates = [
            (x, z)
            for x in range(1, width - 1)
            for z in range(3, goal_z)
            if (x, z) not in protected and (x, z) not in reference_route
        ]
        rng.shuffle(candidates)
        lava_count = min(len(candidates), density + 2)
        for x, z in candidates[:lava_count]:
            surface[x, z] = int(SurfaceMaterial.LAVA)

    # Apply mandatory gates after terrain and biome decoration.  This makes
    # their semantics invariant: no tree, hazard or elevation pass can open a
    # side route or change the number/type of interactions required.
    if break_gate_z is not None:
        for x in range(width):
            ground = max(0, int(base[x, break_gate_z]))
            base[x, break_gate_z] = ground
            top[x, break_gate_z] = ground + 2
            ceiling[x, break_gate_z] = -1
            surface[x, break_gate_z] = int(SurfaceMaterial.NORMAL)
            breakable.update(
                {
                    (x, ground + 1, break_gate_z),
                    (x, ground + 2, break_gate_z),
                }
            )
    if place_gate_z is not None:
        top[:, place_gate_z] = -1
        base[:, place_gate_z] = -1
        ceiling[:, place_gate_z] = -1
        surface[:, place_gate_z] = int(SurfaceMaterial.NORMAL)

    max_steps = (80, 110, 150, 190, 170)[int(difficulty)]
    return ArenaBlueprint(
        stage=AcademyStage.FLAT,
        seed=seed,
        top_y=top,
        base_y=base,
        ceiling_y=ceiling,
        surface=surface,
        breakable=frozenset(breakable),
        spawn=spawn,
        goal=goal,
        biome=biome,
        difficulty=difficulty,
        requirement=requirement,
        initial_blocks=(
            2
            if requirement is NavigationRequirement.COMBINED
            else 1
            if requirement is NavigationRequirement.PLACE
            else 0
        ),
        max_steps=max_steps,
    )
