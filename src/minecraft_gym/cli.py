"""Command-line entry point for starting a Minecraft Gym session."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import gymnasium as gym

import minecraft_gym
from minecraft_gym.actions import noop_action
from minecraft_gym.recording import TrajectoryRecorder
from minecraft_gym.setup import (
    BRIDGE_NAME,
    FABRIC_API_NAME,
    MINECRAFT_VERSION,
    _has_fabric_loader,
    default_game_dir,
    setup,
)
from minecraft_gym.transport import (
    BridgeProtocolError,
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    LengthPrefixedJsonConnection,
)
from minecraft_gym.worlds import clone_world, saves_directory, snapshot_world


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minecraft-gym")
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser(
        "setup", help="Install Fabric Loader and the bridge mods for the official Launcher"
    )
    install.add_argument(
        "--game-dir", type=Path,
        help="Minecraft Launcher game directory (defaults to the OS standard location)",
    )
    install.add_argument(
        "--dry-run", action="store_true", help="Show the target without changing files"
    )
    install.add_argument(
        "--bridge-jar",
        type=Path,
        help="Install a verified locally built bridge JAR instead of downloading it",
    )
    doctor = commands.add_parser(
        "doctor", help="Inspect the local Launcher, mods, saves, and live bridge"
    )
    doctor.add_argument(
        "--game-dir", type=Path,
        help="Minecraft Launcher game directory (defaults to the OS standard location)",
    )
    doctor.add_argument("--host", default=DEFAULT_HOST)
    doctor.add_argument("--port", type=int, default=DEFAULT_PORT)

    world = commands.add_parser(
        "world", help="Create reproducible world templates and fresh clones"
    )
    world_commands = world.add_subparsers(dest="world_command", required=True)
    snapshot = world_commands.add_parser(
        "snapshot", help="Copy a closed save into a verified template"
    )
    snapshot.add_argument("source", type=Path)
    snapshot.add_argument("--output", type=Path, required=True)
    clone = world_commands.add_parser(
        "clone", help="Create a fresh Launcher save from a verified template"
    )
    clone.add_argument("template", type=Path)
    clone.add_argument(
        "--saves-dir", type=Path,
        help="Minecraft saves directory (defaults to the official Launcher)",
    )
    clone.add_argument("--name", help="Optional exact name for the cloned save")

    start = commands.add_parser(
        "start", help="Open Minecraft and start recording or an agent"
    )
    start.add_argument(
        "--launcher",
        help="Launcher application path/name on macOS, or executable command elsewhere",
    )
    start.add_argument(
        "--no-launch",
        action="store_true",
        help="Connect to a Minecraft client that is already running",
    )
    start.add_argument(
        "--bridge-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the Fabric bridge (default: 180)",
    )
    start.add_argument(
        "--host", default=DEFAULT_HOST, help="Fabric bridge host"
    )
    start.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Fabric bridge port"
    )
    start.set_defaults(
        output=Path("datasets/demonstrations"),
        steps=9000,
        seed=42,
        width=128,
        height=128,
        frame_skip=4,
        metadata=[],
        script_args=[],
    )
    modes = start.add_subparsers(dest="mode")

    record = modes.add_parser("record", help="Record keyboard/mouse demonstrations")
    record.add_argument(
        "--output", type=Path, default=Path("datasets/demonstrations")
    )
    record.add_argument("--steps", type=int, default=9000)
    record.add_argument("--seed", type=int, default=42)
    record.add_argument("--width", type=int, default=128)
    record.add_argument("--height", type=int, default=128)
    record.add_argument("--frame-skip", type=int, default=4)
    record.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable string metadata stored with the demonstration",
    )

    agent = modes.add_parser("agent", help="Run a Python agent script")
    agent.add_argument("script", type=Path)
    agent.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def _launcher_command(launcher: str | None) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["open", "-a", launcher or "Minecraft"]
    if launcher:
        return shlex.split(launcher)
    if system == "Linux" and shutil.which("minecraft-launcher"):
        return ["minecraft-launcher"]
    raise RuntimeError(
        "Minecraft Launcher was not found automatically. Pass --launcher "
        "with its executable command, or use --no-launch after starting it yourself."
    )


def _launch_minecraft(launcher: str | None) -> None:
    command = _launcher_command(launcher)
    print(f"Opening Minecraft Launcher: {shlex.join(command)}", flush=True)
    try:
        if platform.system() == "Darwin":
            subprocess.run(command, check=True)
        else:
            subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not open Minecraft Launcher. Use --launcher with its path "
            "or --no-launch after opening Minecraft manually."
        ) from error


def _bridge_responds(host: str, port: int) -> bool:
    connection = LengthPrefixedJsonConnection(host, port, timeout=2.0)
    try:
        response = connection.request(
            {"type": "hello", "protocol_version": PROTOCOL_VERSION}
        )
        return int(response.get("protocol_version", -1)) == PROTOCOL_VERSION
    except (OSError, TimeoutError, ValueError, BridgeProtocolError):
        return False
    finally:
        connection.close()


def _wait_for_bridge(host: str, port: int, timeout: float) -> None:
    if timeout <= 0:
        raise ValueError("--bridge-timeout must be positive")
    deadline = time.monotonic() + timeout
    print(
        f"Select the Minecraft 1.21 Fabric profile with the bridge mod installed. "
        f"Waiting for the bridge at {host}:{port}...",
        flush=True,
    )
    while True:
        if _bridge_responds(host, port):
            print("Fabric bridge connected.", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Fabric bridge did not start in time. Check the Fabric profile, "
                "mod installation, and Minecraft logs."
            )
        time.sleep(1)


def _metadata(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip():
            raise ValueError("--metadata values must use KEY=VALUE")
        result[key.strip()] = item
    return result


def _doctor(args: argparse.Namespace) -> None:
    game_dir = (args.game_dir or default_game_dir()).expanduser().resolve()
    mods_dir = game_dir / "mods"
    options = game_dir / "options.txt"
    pause_setting = None
    if options.is_file():
        for line in options.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("pauseOnLostFocus:"):
                pause_setting = line.partition(":")[2].strip().lower() == "true"
    report = {
        "package_version": getattr(minecraft_gym, "__version__", "unknown"),
        "minecraft_version": MINECRAFT_VERSION,
        "game_dir": str(game_dir),
        "launcher_initialized": (game_dir / "launcher_profiles.json").is_file(),
        "fabric_loader": _has_fabric_loader(game_dir),
        "fabric_api": (mods_dir / FABRIC_API_NAME).is_file(),
        "bridge_mod": (mods_dir / BRIDGE_NAME).is_file(),
        "saves_dir": str(saves_directory(game_dir)),
        "pause_on_lost_focus": pause_setting,
        "bridge_listening": _bridge_responds(args.host, args.port),
        "bridge_address": f"{args.host}:{args.port}",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _world(args: argparse.Namespace) -> None:
    if args.world_command == "snapshot":
        result = snapshot_world(args.source, args.output)
    else:
        target = args.saves_dir or saves_directory(default_game_dir())
        result = clone_world(args.template, target, name=args.name)
    print(json.dumps(result, indent=2, sort_keys=True))


def _record(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.width <= 0 or args.height <= 0 or args.frame_skip <= 0:
        raise ValueError("--width, --height, and --frame-skip must be positive")
    base = gym.make(
        minecraft_gym.ENV_ID,
        backend="socket",
        host=args.host,
        port=args.port,
        control_mode="human",
        max_episode_steps=args.steps,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
        frame_skip=args.frame_skip,
    )
    env = TrajectoryRecorder(
        base,
        args.output,
        metadata={
            "input_device": "keyboard_mouse",
            "control_mode": "human",
            "width": args.width,
            "height": args.height,
            "frame_skip": args.frame_skip,
            **_metadata(args.metadata),
        },
    )
    interrupted = False
    try:
        _, info = env.reset(seed=args.seed)
        print(
            f"Recording started at tick {info['tick']}. Keep Minecraft focused; "
            f"data will be saved to {args.output}. Press Ctrl+C to stop.",
            flush=True,
        )
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(noop_action())
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        env.close()
    print(f"Recording saved to {args.output}.", flush=True)
    if interrupted:
        print("Recording stopped by user.", flush=True)


def _run_agent(args: argparse.Namespace) -> int:
    if not args.script.is_file():
        raise FileNotFoundError(f"Agent script not found: {args.script}")
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args.pop(0)
    command = [sys.executable, str(args.script), *script_args]
    print(f"Running agent: {shlex.join(command)}", flush=True)
    return subprocess.call(command)


def _choose_mode(args: argparse.Namespace) -> None:
    if args.mode is not None:
        return
    selection = input(
        "Choose a task: [1] record keyboard/mouse, [2] run an agent: "
    ).strip()
    if selection == "1":
        args.mode = "record"
    elif selection == "2":
        args.mode = "agent"
        script = input("Path to the Python agent script: ").strip()
        if not script:
            raise ValueError("An agent script path is required")
        args.script = Path(script)
    else:
        raise ValueError("Choose 1 for recording or 2 for an agent")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            setup(
                args.game_dir or default_game_dir(),
                dry_run=args.dry_run,
                bridge_jar=args.bridge_jar,
            )
            return 0
        if args.command == "doctor":
            _doctor(args)
            return 0
        if args.command == "world":
            _world(args)
            return 0
        if not args.no_launch:
            _launch_minecraft(args.launcher)
        _wait_for_bridge(args.host, args.port, args.bridge_timeout)
        input(
            "Enter a single-player Survival world, wait for the terrain and HUD, "
            "then press Enter here to continue: "
        )
        _choose_mode(args)
        if args.mode == "record":
            _record(args)
            return 0
        return _run_agent(args)
    except (RuntimeError, TimeoutError, ValueError, FileNotFoundError, socket.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nSession cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
