from __future__ import annotations

import argparse

import gymnasium as gym
import pytest

import minecraft_gym.cli as cli


def test_start_record_waits_for_world_and_records(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        cli, "_launch_minecraft", lambda launcher: events.append("launch")
    )
    monkeypatch.setattr(
        cli, "_wait_for_bridge", lambda host, port, timeout: events.append("bridge")
    )
    monkeypatch.setattr("builtins.input", lambda prompt: events.append("world"))
    monkeypatch.setattr(cli, "_record", lambda args: events.append("record"))

    result = cli.main(["start", "record", "--output", str(tmp_path), "--steps", "2"])

    assert result == 0
    assert events == ["launch", "bridge", "world", "record"]


def test_start_agent_can_use_existing_minecraft(monkeypatch, tmp_path) -> None:
    script = tmp_path / "agent.py"
    script.write_text("print('agent')\n", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        cli, "_launch_minecraft", lambda launcher: events.append("launch")
    )
    monkeypatch.setattr(
        cli, "_wait_for_bridge", lambda host, port, timeout: events.append("bridge")
    )
    monkeypatch.setattr("builtins.input", lambda prompt: events.append("world"))
    monkeypatch.setattr(
        cli.subprocess,
        "call",
        lambda command: events.append("agent") or 7,
    )

    result = cli.main(["start", "--no-launch", "agent", str(script), "--", "--test"])

    assert result == 7
    assert events == ["bridge", "world", "agent"]


def test_interactive_start_selects_recording(monkeypatch) -> None:
    prompts = iter(("", "1"))
    monkeypatch.setattr(cli, "_launch_minecraft", lambda launcher: None)
    monkeypatch.setattr(
        cli, "_wait_for_bridge", lambda host, port, timeout: None
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    selected: list[str] = []
    monkeypatch.setattr(
        cli, "_record", lambda args: selected.append(args.mode)
    )

    assert cli.main(["start"]) == 0
    assert selected == ["record"]


def test_record_mode_writes_episode_with_mock_backend(monkeypatch, tmp_path) -> None:
    real_make = gym.make
    monkeypatch.setattr(
        cli.gym,
        "make",
        lambda env_id, **kwargs: real_make(
            env_id,
            backend="mock",
            width=8,
            height=6,
            control_mode=kwargs["control_mode"],
            max_episode_steps=kwargs["max_episode_steps"],
        ),
    )
    args = argparse.Namespace(
        steps=2, seed=42, host="127.0.0.1", port=25570, output=tmp_path
    )
    cli._record(args)
    assert len(list(tmp_path.glob("episode-*.npz"))) == 1


def test_launcher_requires_explicit_command_on_unknown_platform(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Unknown")
    with pytest.raises(RuntimeError, match="--launcher"):
        cli._launcher_command(None)


def test_macos_launcher_uses_minecraft_app(monkeypatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    launched: list[list[str]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, check: launched.append(command),
    )
    cli._launch_minecraft(None)
    assert launched == [["open", "-a", "Minecraft"]]


def test_bridge_timeout_is_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        cli._wait_for_bridge("127.0.0.1", 25570, 0)
