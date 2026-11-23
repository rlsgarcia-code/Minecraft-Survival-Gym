from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from minecraft_gym.dataset.worlds import (
    assert_isolated_world,
    logical_world_hash,
    verify_reset_reproducibility,
)


def _world(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "level.dat").write_bytes(b"level-state")
    (path / "region").mkdir()
    (path / "region" / "r.0.0.mca").write_bytes(b"region-state")
    (path / "session.lock").write_bytes(b"transient")
    return path


def test_logical_hash_ignores_lock_but_detects_world_changes(tmp_path: Path) -> None:
    world = _world(tmp_path / "world")
    initial = logical_world_hash(world)
    (world / "session.lock").write_bytes(b"another-lock")
    assert logical_world_hash(world) == initial
    (world / "region" / "r.0.0.mca").write_bytes(b"changed")
    assert logical_world_hash(world) != initial


def test_live_world_can_never_be_used_as_training_world(tmp_path: Path) -> None:
    live = _world(tmp_path / "brincadeiras")
    with pytest.raises(ValueError, match="live family world"):
        assert_isolated_world(live, live)
    with pytest.raises(ValueError, match="must not overlap"):
        assert_isolated_world(live / "clone", live)
    isolated = _world(tmp_path / "research" / "worlds" / "clone")
    assert_isolated_world(isolated, live)


def test_twenty_resets_have_the_same_logical_hash(tmp_path: Path) -> None:
    snapshot = _world(tmp_path / "snapshot")

    def reset(index: int) -> Path:
        destination = tmp_path / "resets" / str(index)
        shutil.copytree(snapshot, destination)
        (destination / "session.lock").write_text(str(index), encoding="utf-8")
        return destination

    report = verify_reset_reproducibility(reset, attempts=20)
    assert report["reproducible"] is True
    assert report["attempts"] == 20
    assert len(report["distinct_hashes"]) == 1
