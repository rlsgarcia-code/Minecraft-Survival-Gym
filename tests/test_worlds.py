from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_gym.worlds import (
    TEMPLATE_METADATA,
    clone_world,
    snapshot_world,
    world_digest,
)


def _world(path: Path, *, payload: bytes = b"level") -> Path:
    path.mkdir(parents=True)
    (path / "level.dat").write_bytes(payload)
    (path / "region").mkdir()
    (path / "region" / "r.0.0.mca").write_bytes(b"region")
    (path / "session.lock").write_bytes(b"lock")
    return path


def test_snapshot_and_clone_preserve_verified_world(tmp_path) -> None:
    source = _world(tmp_path / "saves" / "Survival")
    template = tmp_path / "templates" / "Survival"
    result = snapshot_world(source, template)

    assert result["sha256"] == world_digest(source)
    assert not (template / "session.lock").exists()
    metadata = json.loads((template / TEMPLATE_METADATA).read_text(encoding="utf-8"))
    assert metadata["sha256"] == result["sha256"]

    saves = tmp_path / "other-saves"
    saves.mkdir()
    clone = clone_world(template, saves, name="experiment-001")
    assert clone["name"] == "experiment-001"
    assert clone["sha256"] == result["sha256"]
    assert world_digest(clone["path"]) == result["sha256"]
    assert not (Path(clone["path"]) / TEMPLATE_METADATA).exists()


def test_snapshot_refuses_overwrite(tmp_path) -> None:
    source = _world(tmp_path / "saves" / "Survival")
    output = tmp_path / "template"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_world(source, output)


def test_clone_rejects_changed_template(tmp_path) -> None:
    source = _world(tmp_path / "saves" / "Survival")
    template = tmp_path / "template"
    snapshot_world(source, template)
    (template / "level.dat").write_bytes(b"changed")
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="recorded hash"):
        clone_world(template, target)


def test_clone_rejects_unsafe_name(tmp_path) -> None:
    source = _world(tmp_path / "saves" / "Survival")
    template = tmp_path / "template"
    snapshot_world(source, template)
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="World name"):
        clone_world(template, target, name="../escape")
