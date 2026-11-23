"""Isolation and reproducibility primitives for Minecraft world copies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable


TRANSIENT_WORLD_FILES = {
    "session.lock",
}


def logical_world_hash(root: Path) -> str:
    """Hash persistent world state independent of file metadata and locks."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in TRANSIENT_WORLD_FILES
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def assert_isolated_world(candidate: Path, live_world: Path) -> None:
    candidate = candidate.resolve()
    live_world = live_world.resolve()
    if candidate == live_world:
        raise ValueError("training world cannot be the live family world")
    if candidate.is_relative_to(live_world) or live_world.is_relative_to(candidate):
        raise ValueError("training and live world paths must not overlap")


def verify_reset_reproducibility(
    reset: Callable[[int], Path],
    *,
    attempts: int = 20,
) -> dict[str, object]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    hashes = [logical_world_hash(reset(index)) for index in range(attempts)]
    expected = hashes[0]
    return {
        "attempts": attempts,
        "logical_sha256": expected,
        "reproducible": all(value == expected for value in hashes),
        "distinct_hashes": sorted(set(hashes)),
    }
