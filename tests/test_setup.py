from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import minecraft_gym.cli as cli
import minecraft_gym.setup as installer


def _launcher_dir(path: Path) -> Path:
    path.mkdir()
    (path / "launcher_profiles.json").write_text("{}", encoding="utf-8")
    return path


def _fabric_loader(path: Path) -> None:
    version = path / "versions" / "fabric-loader-0.19.5-1.21"
    version.mkdir(parents=True)
    (version / f"{version.name}.json").write_text("{}", encoding="utf-8")


def test_setup_cli_dry_run_does_not_install(monkeypatch, tmp_path, capsys) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    assert cli.main(["setup", "--game-dir", str(game_dir), "--dry-run"]) == 0
    assert "Dry run" in capsys.readouterr().out
    assert not (game_dir / "mods").exists()


def test_setup_installs_loader_and_mods(monkeypatch, tmp_path) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    downloaded: list[str] = []

    def download(url: str, checksum: str | None = None) -> bytes:
        downloaded.append(url)
        return b"verified jar"

    def run(command, check=False):
        assert command[-4:] == ["-dir", str(game_dir), "-mcversion", "1.21"]
        _fabric_loader(game_dir)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(installer, "_java_21", lambda: Path("/jdk21/bin/java"))
    monkeypatch.setattr(installer, "_verified_download", download)
    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.setup(game_dir)
    assert (game_dir / "mods" / installer.FABRIC_API_NAME).read_bytes() == b"verified jar"
    assert (game_dir / "mods" / installer.BRIDGE_NAME).read_bytes() == b"verified jar"
    assert downloaded == [
        installer.FABRIC_INSTALLER_URL,
        installer.FABRIC_API_URL,
        installer.BRIDGE_URL,
    ]


def test_setup_rejects_conflicting_mod_without_changes(tmp_path) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    mods = game_dir / "mods"
    mods.mkdir()
    (mods / "fabric-api-old.jar").write_bytes(b"old")
    with pytest.raises(RuntimeError, match="Conflicting mod"):
        installer.setup(game_dir)
    assert sorted(p.name for p in mods.iterdir()) == ["fabric-api-old.jar"]


def test_setup_requires_launcher_directory(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Launcher game directory not found"):
        installer.setup(tmp_path / "absent")


def test_setup_is_idempotent_when_matching_files_exist(monkeypatch, tmp_path) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    _fabric_loader(game_dir)
    mods = game_dir / "mods"
    mods.mkdir()
    (mods / installer.FABRIC_API_NAME).write_bytes(b"api")
    (mods / installer.BRIDGE_NAME).write_bytes(b"bridge")
    monkeypatch.setattr(installer, "FABRIC_API_SHA256", hashlib.sha256(b"api").hexdigest())
    monkeypatch.setattr(installer, "BRIDGE_SHA256", hashlib.sha256(b"bridge").hexdigest())
    monkeypatch.setattr(
        installer, "_verified_download",
        lambda *args: pytest.fail("existing components must not be downloaded"),
    )
    installer.setup(game_dir)


def test_setup_rejects_modified_api(tmp_path) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    _fabric_loader(game_dir)
    mods = game_dir / "mods"
    mods.mkdir()
    (mods / installer.FABRIC_API_NAME).write_bytes(b"bad api")
    with pytest.raises(RuntimeError, match="does not match"):
        installer.setup(game_dir)


def test_verified_download_rejects_bad_checksum(monkeypatch) -> None:
    monkeypatch.setattr(installer, "_download", lambda url: b"jar")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        installer._verified_download("https://example.com/mod.jar", "0" * 64)


def test_verified_download_uses_published_checksum(monkeypatch) -> None:
    data = b"jar"
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        installer, "_download", lambda url: checksum.encode() if url.endswith(".sha256") else data
    )
    assert installer._verified_download("https://example.com/mod.jar") == data


@pytest.mark.skipif(
    os.environ.get("MINECRAFT_GYM_SETUP_E2E") != "1",
    reason="requires network and Java 21",
)
def test_real_setup_in_temporary_launcher_directory(tmp_path) -> None:
    game_dir = _launcher_dir(tmp_path / "minecraft")
    installer.setup(game_dir)
    assert installer._has_fabric_loader(game_dir)
    assert (game_dir / "mods" / installer.FABRIC_API_NAME).is_file()
    assert (game_dir / "mods" / installer.BRIDGE_NAME).is_file()
