#!/usr/bin/env python3
"""Run a short real or mock Minecraft Gym smoke test."""

from __future__ import annotations

import argparse

import minecraft_gym
from gymnasium import make

from minecraft_gym.actions import noop_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("socket", "mock"), default="socket")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25570)
    args = parser.parse_args()

    env = make(
        minecraft_gym.ENV_ID,
        backend=args.backend,
        host=args.host,
        port=args.port,
        render_mode="rgb_array",
    )
    try:
        observation, info = env.reset(seed=args.seed)
        print(
            f"reset tick={info['tick']} seed={info['world_seed']} "
            f"rgb={observation['rgb'].shape}"
        )
        for index in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(noop_action())
            print(
                f"step={index + 1} tick={info['tick']} reward={reward:.6f} "
                f"terminated={terminated} truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
