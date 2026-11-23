#!/usr/bin/env python3
"""Summarize and render a persisted VGG RGB-D training curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def moving(values: list[float], window: int) -> list[float]:
    return [
        statistics.mean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def points(values: list[float], *, low: float, high: float, width: int, height: int) -> str:
    span = max(1e-9, high - low)
    denominator = max(1, len(values) - 1)
    return " ".join(
        f"{40 + index * width / denominator:.2f},{20 + (high - value) * height / span:.2f}"
        for index, value in enumerate(values)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("progress", type=Path)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.progress.read_text(encoding="utf-8"))
    rewards = [float(value) for value in payload["reward_curve"]]
    successes = [1.0 if value else 0.0 for value in payload["success_curve"]]
    reward_average = moving(rewards, args.window)
    success_average = moving(successes, args.window)
    if len(rewards) < args.window:
        raise ValueError("the curve must contain at least one complete window")
    complete_window_indices = range(args.window - 1, len(rewards))
    best_reward_index = max(
        complete_window_indices, key=reward_average.__getitem__
    )
    best_success_index = max(
        range(args.window - 1, len(successes)), key=success_average.__getitem__
    )
    analysis = {
        "interface": "minecraft-vgg-rgbd-curve-analysis/v1",
        "source": str(args.progress),
        "episodes": len(rewards),
        "window": args.window,
        "first_window_reward": statistics.mean(rewards[: args.window]),
        "last_window_reward": statistics.mean(rewards[-args.window :]),
        "best_reward_moving_average": reward_average[best_reward_index],
        "best_reward_episode": best_reward_index + 1,
        "peak_success_moving_average": success_average[best_success_index],
        "peak_success_episode": best_success_index + 1,
        "last_success_moving_average": success_average[-1],
        "collapse_detected": (
            success_average[best_success_index] >= 0.5
            and success_average[-1] <= 0.05
            and reward_average[-1] < reward_average[best_success_index] - 5.0
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    chart_width, chart_height = 900, 320
    reward_low = min(rewards + reward_average)
    reward_high = max(rewards + reward_average)
    reward_raw_points = points(
        rewards, low=reward_low, high=reward_high, width=chart_width, height=chart_height
    )
    reward_ma_points = points(
        reward_average,
        low=reward_low,
        high=reward_high,
        width=chart_width,
        height=chart_height,
    )
    success_points = points(
        success_average, low=0.0, high=1.0, width=chart_width, height=chart_height
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="410" viewBox="0 0 980 410">
<rect width="980" height="410" fill="#fafaf8"/>
<text x="40" y="22" font-family="-apple-system,sans-serif" font-size="16" font-weight="600">VGG RGB-D — reward e sucesso</text>
<text x="40" y="42" font-family="-apple-system,sans-serif" font-size="11" fill="#666">{len(rewards)} episódios · média móvel {args.window}</text>
<line x1="40" y1="60" x2="40" y2="380" stroke="#bbb"/>
<line x1="40" y1="380" x2="940" y2="380" stroke="#bbb"/>
<polyline points="{reward_raw_points}" transform="translate(0 40)" fill="none" stroke="#8eb493" stroke-opacity=".25" stroke-width="1"/>
<polyline points="{reward_ma_points}" transform="translate(0 40)" fill="none" stroke="#347a42" stroke-width="2.5"/>
<polyline points="{success_points}" transform="translate(0 40)" fill="none" stroke="#a56a32" stroke-width="2" stroke-dasharray="5 4"/>
<text x="50" y="400" font-family="-apple-system,sans-serif" font-size="10" fill="#347a42">reward médio</text>
<text x="155" y="400" font-family="-apple-system,sans-serif" font-size="10" fill="#a56a32">sucesso médio</text>
</svg>"""
    (args.output / "curve.svg").write_text(svg, encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
