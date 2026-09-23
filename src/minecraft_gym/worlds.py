"""Safe, reusable Minecraft world snapshots for repeatable Gym episodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


TEMPLATE_METADATA = ".minecraft-gym-template.json"
_SAFE_WORLD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def saves_directory(game_dir: str | Path) -> Path:
    return Path(game_dir).expanduser().resolve() / "saves"


def world_digest(directory: str | Path) -> str:
    """Hash persistent save contents while ignoring Minecraft's lock file."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir() or not (root / "level.dat").is_file():
        raise ValueError(f"Not a Minecraft world save: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"session.lock", TEMPLATE_METADATA}:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@contextmanager
def _try_lock(path: Path) -> Iterator[bool]:
    """Yield whether the Java session lock could be acquired non-blockingly."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
            else:
                try:
                    yield True
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.lockf(handle, fcntl.LOCK_UN)


def _assert_world_closed(world: Path) -> None:
    lock = world / "session.lock"
    with _try_lock(lock) as acquired:
        if not acquired:
            raise RuntimeError(f"Minecraft is using world {world}; close it first")


def _assert_no_open_worlds(saves_dir: Path) -> None:
    if not saves_dir.is_dir():
        raise ValueError(f"Minecraft saves directory does not exist: {saves_dir}")
    for world in saves_dir.iterdir():
        if world.is_dir() and (world / "level.dat").is_file():
            _assert_world_closed(world)


def _validate_world_name(name: str) -> str:
    if not _SAFE_WORLD_NAME.fullmatch(name):
        raise ValueError(
            "World name must start with a letter or digit and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return name


def snapshot_world(source: str | Path, output: str | Path) -> dict[str, str]:
    """Copy one closed save into an immutable, hash-verified template."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_dir() or not (source_path / "level.dat").is_file():
        raise ValueError(f"Not a Minecraft world save: {source_path}")
    _assert_world_closed(source_path)
    if output_path.exists():
        raise FileExistsError(f"Template already exists: {output_path}")
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("Template output must not be inside the source world")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_path,
        output_path,
        ignore=shutil.ignore_patterns("session.lock", TEMPLATE_METADATA),
    )
    digest = world_digest(output_path)
    metadata = {
        "schema_version": 1,
        "source_world": source_path.name,
        "sha256": digest,
        "created_at_unix": time.time(),
    }
    (output_path / TEMPLATE_METADATA).write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"path": str(output_path), "sha256": digest}


def clone_world(
    template: str | Path,
    saves_dir: str | Path,
    *,
    name: str | None = None,
) -> dict[str, str]:
    """Create a fresh verified save from a template without overwriting data."""

    template_path = Path(template).expanduser().resolve()
    target_root = Path(saves_dir).expanduser().resolve()
    metadata_path = template_path / TEMPLATE_METADATA
    if not metadata_path.is_file():
        raise ValueError(
            f"Template metadata is missing: {metadata_path}. "
            "Create it with `minecraft-gym world snapshot`."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = world_digest(template_path)
    if metadata.get("schema_version") != 1 or metadata.get("sha256") != digest:
        raise ValueError("Template contents differ from their recorded hash")
    _assert_no_open_worlds(target_root)
    default_name = f"gym-run-{template_path.name}-{uuid.uuid4().hex[:8]}"
    world_name = _validate_world_name(name or default_name)
    destination = target_root / world_name
    if destination.exists():
        raise FileExistsError(f"World already exists: {destination}")
    shutil.copytree(
        template_path,
        destination,
        ignore=shutil.ignore_patterns("session.lock", TEMPLATE_METADATA),
    )
    if world_digest(destination) != digest:
        shutil.rmtree(destination)
        raise RuntimeError("Cloned world differs from its template; clone was removed")
    return {"name": world_name, "path": str(destination), "sha256": digest}
