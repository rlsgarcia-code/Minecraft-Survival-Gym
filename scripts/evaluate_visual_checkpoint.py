#!/usr/bin/env python3
"""Evaluate a frozen visual actor-critic checkpoint on unseen proxy seeds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import torch

from minecraft_gym.academy.visual_actor_critic import VisualHybridActorCriticAgent
from minecraft_gym.academy.visual_ppo import VisualHybridPpoAgent

from run_rgbd_ablation import run_episode, summarize, visible_task_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_002_600)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--balanced-difficulties", action="store_true")
    args = parser.parse_args()

    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    checkpoint_header = torch.load(
        args.checkpoint.expanduser(), map_location="cpu", weights_only=True
    )
    agent_class = (
        VisualHybridPpoAgent
        if checkpoint_header.get("interface") == VisualHybridPpoAgent.interface
        else VisualHybridActorCriticAgent
    )
    agent, checkpoint_metadata = agent_class.load(
        args.checkpoint, device=device, load_optimizer=False
    )
    difficulty_cycle = None
    if args.balanced_difficulties:
        from minecraft_gym.academy import NavigationDifficulty

        difficulty_cycle = (
            NavigationDifficulty.EASY,
            NavigationDifficulty.MODERATE,
        )
    schedule = visible_task_seeds(
        base_seed=args.seed,
        count=args.episodes,
        difficulty_cycle=difficulty_cycle,
    )
    results = [
        run_episode(
            agent,
            arena_seed=arena_seed,
            difficulty=difficulty,
            episode_index=index,
            training=False,
        )
        for index, (arena_seed, difficulty) in enumerate(schedule)
    ]
    payload = {
        "interface": "minecraft-visual-checkpoint-evaluation/v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_metadata": checkpoint_metadata,
        "frozen_policy": True,
        "evaluation_device": device,
        "environment": "procedural_academy_proxy",
        "balanced_difficulties": args.balanced_difficulties,
        "real_minecraft_evaluation": False,
        "schedule": schedule,
        "summary": summarize(results),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
