#!/usr/bin/env python3
"""Render controlled and live minimap evidence for topology audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from minecraft_gym.academy import NavigationAcademyEnv
from minecraft_gym.academy.policy import MAP_LATTICE_CHANNELS, PYRAMID_SCALES


CELL = 10
PANEL = 330
LABEL = 34


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def colorize(values: np.ndarray, *, kind: str) -> Image.Image:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    if kind == "goal":
        rgb = np.stack((40 + values * 20, 55 + values * 200, 55 + values * 35), axis=-1)
    elif kind == "stack":
        rgb = np.stack((35 + values * 220, 38 + values * 35, 42 + values * 20), axis=-1)
    elif kind == "clearance":
        rgb = np.stack((45 + values * 35, 45 + values * 110, 55 + values * 195), axis=-1)
    elif kind == "material":
        rgb = np.stack((45 + values * 190, 65 + values * 145, 65 + (1.0 - values) * 80), axis=-1)
    elif kind == "variation":
        rgb = np.stack((45 + values * 210, 42 + values * 105, 42 + values * 25), axis=-1)
    elif kind == "elevation":
        rgb = np.stack((35 + values * 205, 50 + values * 180, 70 + (1.0 - values) * 120), axis=-1)
    else:
        rgb = np.repeat((25 + values[..., None] * 225), 3, axis=-1)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return image.resize((PANEL, PANEL), Image.Resampling.NEAREST)


def labeled(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL + LABEL), (246, 246, 248))
    canvas.paste(image.resize((PANEL, PANEL)), (0, LABEL))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(20, 20, 24), font=font(15))
    if subtitle:
        draw.text((PANEL - 8, 6), subtitle, fill=(95, 95, 105), font=font(11), anchor="ra")
    return canvas


def ground_truth(env: NavigationAcademyEnv) -> Image.Image:
    assert env.blueprint is not None and env._top is not None
    top = env._top
    width, length = env.blueprint.width, env.blueprint.length
    canvas = Image.new("RGB", (width * CELL, length * CELL), (27, 30, 36))
    draw = ImageDraw.Draw(canvas)
    palette = {
        -1: (28, 32, 40),
        0: (202, 188, 151),
        1: (169, 151, 114),
        2: (130, 109, 82),
        3: (95, 78, 61),
    }
    for x in range(width):
        for z in range(length):
            y = int(top[x, z])
            fill = palette.get(y, palette[3])
            box = (x * CELL, z * CELL, (x + 1) * CELL - 1, (z + 1) * CELL - 1)
            draw.rectangle(box, fill=fill)
            if any((x, yy, z) in env._breakable for yy in range(6)):
                draw.rectangle(box, outline=(220, 70, 185), width=2)
            ceiling = int(env.blueprint.ceiling_y[x, z])
            if ceiling >= 0:
                draw.rectangle(box, outline=(60, 205, 225), width=2)
    gx, gz = env.blueprint.goal
    draw.rectangle((gx * CELL, gz * CELL, (gx + 1) * CELL - 1, (gz + 1) * CELL - 1), fill=(65, 205, 95))
    px, pz = env._position
    draw.ellipse((px * CELL + 1, pz * CELL + 1, (px + 1) * CELL - 2, (pz + 1) * CELL - 2), fill=(55, 115, 235))
    return canvas.resize((PANEL, PANEL), Image.Resampling.NEAREST)


def controlled(output: Path) -> dict[str, object]:
    cases = (
        ("Plano", dict(stage="flat"), 7),
        ("Parede quebrável", dict(stage="break"), 7),
        ("Vão + subida", dict(stage="build"), 7),
        ("Quebrar + colocar", dict(stage="combined"), 7),
        ("Selva + teto baixo", dict(difficulty="hard", biome="jungle", requirement="break"), 31),
    )
    channel_kinds = ("evidence", "elevation", "variation", "goal", "material", "stack", "clearance")
    rows: list[Image.Image] = []
    metrics: dict[str, object] = {}
    for title, kwargs, seed in cases:
        env = NavigationAcademyEnv(**kwargs)
        observation, _ = env.reset(seed=seed)
        local = observation["grid_pyramid"][0]
        panels = [labeled(ground_truth(env), title, "verdade top-down")]
        for index, (name, kind) in enumerate(zip(MAP_LATTICE_CHANNELS, channel_kinds, strict=True)):
            panels.append(labeled(colorize(local[index], kind=kind), name, "escala 1 bloco"))
        row = Image.new("RGB", (len(panels) * PANEL, PANEL + LABEL), (238, 238, 242))
        for index, panel in enumerate(panels):
            row.paste(panel, (index * PANEL, 0))
        rows.append(row)

        observed = local[0] > 0
        stack = local[5] > 0
        low_clearance = observed & (local[6] < 0.5)
        metrics[title] = {
            "observed_bins": int(observed.sum()),
            "stack_bins": int(stack.sum()),
            "low_clearance_bins": int(low_clearance.sum()),
            "goal_mass": float(local[3].sum()),
            "material_values": sorted(float(value) for value in np.unique(local[4][observed])),
        }
        env.close()
    sheet = Image.new("RGB", (rows[0].width, len(rows) * rows[0].height), (238, 238, 242))
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * row.height))
    sheet.save(output / "controlled_topology.png")

    env = NavigationAcademyEnv(difficulty="expert", biome="mixed", requirement="combined")
    observation, _ = env.reset(seed=44)
    panels = []
    for level, scale in enumerate(PYRAMID_SCALES):
        lattice = observation["grid_pyramid"][level]
        # A compact topology view: red=stack, green=goal, blue=low clearance.
        evidence = lattice[0]
        rgb = np.stack((lattice[5], lattice[3], evidence * (1.0 - lattice[6])), axis=-1)
        panels.append(labeled(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((PANEL, PANEL), Image.Resampling.NEAREST), f"Escala {scale}", f"{scale} bloco(s)/bin"))
    scales = Image.new("RGB", (len(panels) * PANEL, PANEL + LABEL), (238, 238, 242))
    for index, panel in enumerate(panels):
        scales.paste(panel, (index * PANEL, 0))
    scales.save(output / "multiscale_topology.png")
    env.close()
    return metrics


def live_map(output: Path, url: str) -> dict[str, object]:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - explicit localhost audit
        payload = json.load(response)
    panels: list[Image.Image] = []
    diagnostics: dict[str, object] = {}
    for level in payload["levels"]:
        cells = level["cells"]
        xs = [int(cell["cell_x"]) for cell in cells]
        zs = [int(cell["cell_z"]) for cell in cells]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        width, height = max_x - min_x + 1, max_z - min_z + 1
        image = Image.new("RGB", (max(1, width) * CELL, max(1, height) * CELL), (235, 235, 239))
        draw = ImageDraw.Draw(image)
        contradictory = 0
        for cell in cells:
            x, z = int(cell["cell_x"]), int(cell["cell_z"])
            blocks = cell.get("block_counts") or {}
            hits = int(cell.get("hit_count") or 0)
            free = int(cell.get("free_count") or 0)
            if hits > 0 and free > 0:
                contradictory += 1
            if blocks:
                name = max(blocks, key=blocks.get).lower()
                color = (205, 185, 125) if "sand" in name else (115, 85, 62) if "wood" in name or "oak" in name else (130, 135, 140)
            elif free > 0:
                color = (214, 229, 237)
            else:
                color = (238, 238, 241)
            box = ((x - min_x) * CELL, (z - min_z) * CELL, (x - min_x + 1) * CELL - 1, (z - min_z + 1) * CELL - 1)
            draw.rectangle(box, fill=color)
            if int(cell.get("visited_count") or 0) > 0:
                draw.rectangle(box, outline=(70, 165, 85), width=2)
            if float(cell.get("head_clearance") or 0) < 2 and hits > 0:
                draw.line((box[0], box[1], box[2], box[3]), fill=(220, 60, 75), width=2)
        current = level.get("current_cell")
        if current:
            cx, cz = int(current[0]) - min_x, int(current[1]) - min_z
            draw.ellipse((cx * CELL + 1, cz * CELL + 1, (cx + 1) * CELL - 2, (cz + 1) * CELL - 2), fill=(55, 115, 235))
        panels.append(labeled(image.resize((PANEL, PANEL), Image.Resampling.NEAREST), level["label"], f'{level["bin_size_blocks"]} bloco(s)/bin'))
        diagnostics[level["key"]] = {
            "cells": len(cells),
            "visited": sum(int(cell.get("visited_count") or 0) > 0 for cell in cells),
            "semantic": sum(bool(cell.get("block_counts")) for cell in cells),
            "hit_and_free": contradictory,
            "low_clearance_with_hits": sum(float(cell.get("head_clearance") or 0) < 2 and int(cell.get("hit_count") or 0) > 0 for cell in cells),
        }
    sheet = Image.new("RGB", (len(panels) * PANEL, PANEL + LABEL), (238, 238, 242))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * PANEL, 0))
    sheet.save(output / "live_accumulated_map.png")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-url", default="http://127.0.0.1:8014/api/spatial/map")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"controlled": controlled(args.output)}
    if not args.skip_live:
        report["live"] = live_map(args.output, args.live_url)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
