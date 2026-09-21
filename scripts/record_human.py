#!/usr/bin/env python3
"""Record a keyboard/mouse demonstration through the Fabric bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import minecraft_gym
from gymnasium import make

from minecraft_gym.actions import noop_action
from minecraft_gym.recording import TrajectoryRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/demonstrations"))
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25570)
    parser.add_argument("--mode", choices=("human", "dagger"), default="human")
    args = parser.parse_args()

    base = make(
        minecraft_gym.ENV_ID,
        backend="socket",
        host=args.host,
        port=args.port,
        control_mode=args.mode,
        max_episode_steps=args.steps,
        render_mode="rgb_array",
    )
    env = TrajectoryRecorder(
        base,
        args.output,
        metadata={"input_device": "keyboard_mouse", "control_mode": args.mode},
    )
    try:
        _, info = env.reset(seed=args.seed)
        print(f"Recording started at tick {info['tick']}. Control Minecraft normally.")
        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step(noop_action())
            if terminated or truncated:
                print(f"Recording finished at tick {info['tick']}.")
                break
    except KeyboardInterrupt:
        print("Recording interrupted; committing captured transitions.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
