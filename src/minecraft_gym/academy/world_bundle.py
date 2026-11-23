"""Generate and safely install the real Minecraft navigation academy datapack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

from minecraft_gym.dataset.worlds import assert_isolated_world

from .arena import AcademyStage, ArenaBlueprint, build_arena
from .curriculum import academy_task_suite


NAMESPACE = "minecraft_gym"
DATAPACK_NAME = "navigation-academy-v1"
TARGET_MINECRAFT_VERSION = "26.3"
DEFAULT_DATA_PACK_FORMAT = 121


def _commands_for_stage(arena: ArenaBlueprint, *, offset_x: int, base_y: int) -> list[str]:
    ox, oz = offset_x, 0
    commands = [
        f"fill {ox - 2} {base_y - 4} {oz - 2} {ox + arena.width + 1} {base_y + 6} {oz + arena.length + 1} air",
        f"fill {ox - 2} {base_y - 4} {oz - 2} {ox + arena.width + 1} {base_y - 4} {oz + arena.length + 1} bedrock",
    ]
    for x in range(arena.width):
        for z in range(arena.length):
            top = int(arena.top_y[x, z])
            base = int(arena.base_y[x, z])
            if top < 0:
                continue
            for y in range(0, top + 1):
                material = "smooth_stone"
                if y > base and (x, y, z) in arena.breakable:
                    material = "sandstone"
                commands.append(f"setblock {ox + x} {base_y + y} {oz + z} {material}")
            ceiling = int(arena.ceiling_y[x, z])
            if ceiling >= 0:
                commands.append(
                    f"setblock {ox + x} {base_y + ceiling} {oz + z} smooth_stone"
                )
    # Invisible side walls keep the task topology identical to the blueprint.
    commands.extend(
        [
            f"fill {ox + 2} {base_y + 1} {oz} {ox + 2} {base_y + 4} {oz + arena.length - 1} barrier",
            f"fill {ox + 6} {base_y + 1} {oz} {ox + 6} {base_y + 4} {oz + arena.length - 1} barrier",
        ]
    )
    gx, gz = arena.goal
    goal_y = base_y + int(arena.top_y[gx, gz])
    commands.append(f"setblock {ox + gx} {goal_y} {oz + gz} lime_concrete")
    sx, sz = arena.spawn
    spawn_y = base_y + int(arena.top_y[sx, sz]) + 1
    selector = "@a[tag=minecraft_gym_trainee]"
    commands.extend(
        [
            f"clear {selector}",
            f"effect clear {selector}",
            f"gamemode survival {selector}",
            f"tp {selector} {ox + sx + 0.5} {spawn_y} {oz + sz + 0.5} 0 0",
            f"spawnpoint {selector} {ox + sx} {spawn_y} {oz + sz}",
            f"give {selector} iron_pickaxe 1",
        ]
    )
    if arena.initial_blocks:
        commands.append(f"give {selector} oak_planks {arena.initial_blocks}")
    commands.extend(
        [
            f"scoreboard players set {selector} gym_stage {int(arena.stage)}",
            f"scoreboard players set {selector} gym_success 0",
            f"title {selector} title {{\"text\":\"Academia: {arena.stage.slug}\",\"color\":\"green\"}}",
        ]
    )
    return commands


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_academy_bundle(
    output: str | Path,
    *,
    seed: int = 0,
    pack_format: int = DEFAULT_DATA_PACK_FORMAT,
) -> dict[str, object]:
    """Create a versioned datapack bundle; never touches an existing world."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"academy bundle already exists: {output}")
    if pack_format < 1:
        raise ValueError("pack_format must be positive")
    datapack = output / "datapack" / DATAPACK_NAME
    function_root = datapack / "data" / NAMESPACE / "function" / "academy"
    function_root.mkdir(parents=True)
    arenas = [build_arena(stage, seed) for stage in AcademyStage]

    (datapack / "pack.mcmeta").write_text(
        json.dumps(
            {
                "pack": {
                    "pack_format": pack_format,
                    "min_format": pack_format,
                    "max_format": pack_format,
                    "description": "Minecraft Gym navigation academy v1 (isolated training only)",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    setup = [
        "gamerule spawn_mobs false",
        "gamerule advance_time false",
        "gamerule advance_weather false",
        "gamerule keep_inventory true",
        "gamerule respawn_radius 0",
        "gamerule random_tick_speed 0",
        "difficulty peaceful",
        "time set day",
        "weather clear",
        "forceload add -2 -2 138 15",
        "setworldspawn 3 65 1",
        "scoreboard objectives add gym_stage dummy",
        "scoreboard objectives add gym_success dummy",
        "tellraw @a [{\"text\":\"Academia criada. Use /function minecraft_gym:academy/attach\",\"color\":\"green\"}]",
    ]
    _write_lines(function_root / "setup.mcfunction", setup)
    _write_lines(
        function_root / "attach.mcfunction",
        [
            "tag @s add minecraft_gym_trainee",
            "function minecraft_gym:academy/reset_flat",
        ],
    )
    for index, arena in enumerate(arenas):
        _write_lines(
            function_root / f"reset_{arena.stage.slug}.mcfunction",
            _commands_for_stage(arena, offset_x=index * 32, base_y=64),
        )

    artifact_hashes = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(datapack.rglob("*"))
        if path.is_file()
    }

    manifest = {
        "interface": "minecraft-navigation-academy-bundle/v1",
        "deployment_mode": "isolated_training_only",
        "target_minecraft_version": TARGET_MINECRAFT_VERSION,
        "seed": seed,
        "pack_format": pack_format,
        "datapack_name": DATAPACK_NAME,
        "live_world_access": False,
        "automatic_world_mutation": False,
        "stages": [arena.to_manifest() for arena in arenas],
        "tasks": [task.model_dump(mode="json") for task in academy_task_suite(seed)],
        "activation": {
            "setup": "/function minecraft_gym:academy/setup",
            "attach_player": "/function minecraft_gym:academy/attach",
            "reset_pattern": "/function minecraft_gym:academy/reset_<stage>",
        },
        "artifacts_sha256": artifact_hashes,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output / "academy-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "Navigation Academy v1\n\n"
        "This bundle is only for a disposable, isolated training world.\n"
        "Install its datapack with the minecraft-gym academy install command.\n"
        "Then run /reload, /function minecraft_gym:academy/setup, and\n"
        "/function minecraft_gym:academy/attach as the trainee player.\n"
        "The family world must never be supplied as the training world.\n",
        encoding="utf-8",
    )
    return {
        "path": str(output),
        "datapack": str(datapack),
        "bundle_sha256": manifest["bundle_sha256"],
        "stages": len(arenas),
    }


def validate_academy_bundle(bundle: str | Path) -> dict[str, object]:
    """Verify identity, hashes, target format, and all stage reset functions."""

    bundle = Path(bundle).expanduser().resolve()
    manifest_path = bundle / "academy-manifest.json"
    datapack = bundle / "datapack" / DATAPACK_NAME
    if not manifest_path.is_file() or not datapack.is_dir():
        raise ValueError(f"invalid academy bundle: {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("interface") != "minecraft-navigation-academy-bundle/v1":
        raise ValueError("academy bundle interface is incompatible")
    if manifest.get("live_world_access") is not False:
        raise ValueError("academy bundle may not claim live-world access")
    recorded_hash = manifest.pop("bundle_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if recorded_hash != actual_hash:
        raise ValueError("academy bundle manifest hash mismatch")
    pack_meta = json.loads((datapack / "pack.mcmeta").read_text(encoding="utf-8"))
    if int(pack_meta["pack"]["pack_format"]) != int(manifest["pack_format"]):
        raise ValueError("datapack format differs from academy manifest")
    artifacts = manifest.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("academy bundle artifact hashes are missing")
    for relative, expected in artifacts.items():
        path = (bundle / str(relative)).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as error:
            raise ValueError("academy bundle artifact path is unsafe") from error
        if not path.is_file():
            raise ValueError(f"academy bundle artifact is missing: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"academy bundle artifact hash mismatch: {relative}")
    seed = int(manifest["seed"])
    recorded_stages = {item["stage"]: item for item in manifest.get("stages", ())}
    function_root = datapack / "data" / NAMESPACE / "function" / "academy"
    for stage in AcademyStage:
        expected = build_arena(stage, seed)
        item = recorded_stages.get(stage.slug)
        if item is None or item.get("logical_sha256") != expected.logical_sha256:
            raise ValueError(f"arena hash mismatch for stage {stage.slug}")
        if not (function_root / f"reset_{stage.slug}.mcfunction").is_file():
            raise ValueError(f"arena reset function missing for stage {stage.slug}")
    return {
        "path": str(bundle),
        "bundle_sha256": actual_hash,
        "target_minecraft_version": manifest.get("target_minecraft_version"),
        "pack_format": int(manifest["pack_format"]),
        "stages": len(recorded_stages),
    }


def install_academy_bundle(
    bundle: str | Path,
    world: str | Path,
    *,
    live_world: str | Path,
) -> dict[str, str]:
    """Install into an existing closed training save with mandatory isolation."""

    bundle = Path(bundle).expanduser().resolve()
    world = Path(world).expanduser().resolve()
    live_world = Path(live_world).expanduser().resolve()
    if not (world / "level.dat").is_file():
        raise ValueError(f"not a Minecraft world: {world}")
    if not (live_world / "level.dat").is_file():
        raise ValueError(f"live family world was not identified: {live_world}")
    assert_isolated_world(world, live_world)
    validate_academy_bundle(bundle)
    source = bundle / "datapack" / DATAPACK_NAME
    destination = world / "datapacks" / DATAPACK_NAME
    if destination.exists():
        raise FileExistsError(f"academy datapack already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return {
        "world": str(world),
        "datapack": str(destination),
        "live_world": str(live_world),
        "live_world_access": "false",
    }
