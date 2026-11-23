#!/usr/bin/env python3
"""Generate a visual and numeric audit of the sparse voxel experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_gym.dataset import SparseVoxelWorldMemory


def set_block(
    memory: SparseVoxelWorldMemory,
    position: tuple[int, int, int],
    block: str | None,
) -> None:
    memory.set_authoritative_block(position, block, server_tick=1)


def scenario() -> SparseVoxelWorldMemory:
    memory = SparseVoxelWorldMemory("voxel-audit", "actor:audit")
    # Known floor and body volume.
    for x in range(-8, 9):
        for z in range(-5, 6):
            set_block(memory, (x, 63, z), "stone")
            set_block(memory, (x, 64, z), None)
            set_block(memory, (x, 65, z), None)

    # West: two-block wall.
    for z in range(-3, 4):
        set_block(memory, (-5, 64, z), "stone_bricks")
        set_block(memory, (-5, 65, z), "stone_bricks")

    # Center: tunnel at y=64/65 and bridge deck at y=66.
    for x in range(-2, 3):
        set_block(memory, (x, 66, 0), "oak_planks")
        set_block(memory, (x, 67, 0), None)
        set_block(memory, (x, 68, 0), None)

    # East: one-block rise followed by two-block rise.
    for x, top in ((4, 64), (5, 64), (6, 65), (7, 65)):
        for y in range(64, top + 1):
            set_block(memory, (x, y, 2), "quartz_block")
        set_block(memory, (x, top + 1, 2), None)
        set_block(memory, (x, top + 2, 2), None)

    # South: low ceiling, body has only one free voxel.
    for x in range(2, 6):
        set_block(memory, (x, 65, -3), "oak_planks")
    return memory


def _solid_voxels(memory: SparseVoxelWorldMemory):
    result = []
    for position, state in memory.iter_voxels():
        if state.solid_probability >= 0.5:
            result.append((*position, state.dominant_block or "unknown"))
    return result


def _shade(color: tuple[int, int, int], amount: int) -> str:
    return "#" + "".join(f"{max(0, value - amount):02x}" for value in color)


def _points(points: tuple[tuple[int, int], ...]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _cube_svg(
    *,
    x: int,
    y: int,
    z: int,
    color: tuple[int, int, int],
    origin: tuple[int, int],
    scale: int = 19,
) -> str:
    sx = origin[0] + (x - z) * scale
    sy = origin[1] + (x + z) * scale // 2 - (y - 63) * scale
    top = ((sx, sy - scale), (sx + scale, sy - scale // 2), (sx, sy), (sx - scale, sy - scale // 2))
    left = ((sx - scale, sy - scale // 2), (sx, sy), (sx, sy + scale), (sx - scale, sy + scale // 2))
    right = ((sx, sy), (sx + scale, sy - scale // 2), (sx + scale, sy + scale // 2), (sx, sy + scale))
    return "".join(
        (
            f'<polygon points="{_points(left)}" fill="{_shade(color, 42)}" stroke="#2d3034" stroke-width="1"/>',
            f'<polygon points="{_points(right)}" fill="{_shade(color, 20)}" stroke="#2d3034" stroke-width="1"/>',
            f'<polygon points="{_points(top)}" fill="{_shade(color, 0)}" stroke="#2d3034" stroke-width="1"/>',
        )
    )


def render(memory: SparseVoxelWorldMemory, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    minimap = memory.navigation_minimap(center_feet=(0, 64, 0), size=21)
    volume = memory.local_volume(center=(0, 65, 0), shape=(21, 11, 21))
    solids = _solid_voxels(memory)

    elements = [
        '<rect width="1500" height="760" fill="#f6f7f9"/>',
        '<text x="42" y="52" fill="#181b1f" font-size="28" font-family="system-ui,sans-serif">Memória voxel canônica</text>',
        '<text x="42" y="80" fill="#5a5f66" font-size="17" font-family="system-ui,sans-serif">ponte e túnel preservados no mesmo (x,z)</text>',
        '<text x="880" y="52" fill="#181b1f" font-size="28" font-family="system-ui,sans-serif">Minimapa derivado</text>',
        '<text x="880" y="80" fill="#5a5f66" font-size="15" font-family="system-ui,sans-serif">verde=atravessável · vermelho=obstáculo · cinza=desconhecido</text>',
    ]
    colors = {
        "stone": (145, 149, 156),
        "stone_bricks": (82, 91, 103),
        "oak_planks": (168, 111, 58),
        "quartz_block": (230, 225, 211),
    }
    # Painter order for the fixed isometric camera.
    for x, y, z, block in sorted(solids, key=lambda item: (item[0] + item[2], item[1])):
        if y >= 63:
            elements.append(_cube_svg(
                x=x,
                y=y,
                z=z,
                color=colors.get(block, (110, 143, 100)),
                origin=(410, 290),
            ))

    cell = 25
    map_left, map_top = 920, 125
    for row in range(21):
        for column in range(21):
            known = minimap[0, row, column] > 0
            traversable = minimap[4, row, column] > 0.5
            color = (217, 220, 224) if not known else (105, 183, 116) if traversable else (205, 100, 91)
            box = (
                map_left + column * cell,
                map_top + row * cell,
                map_left + (column + 1) * cell - 1,
                map_top + (row + 1) * cell - 1,
            )
            elements.append(
                f'<rect x="{box[0]}" y="{box[1]}" width="{cell - 1}" height="{cell - 1}" fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}"/>'
            )
    center_x = map_left + 10 * cell + cell // 2
    center_y = map_top + 10 * cell + cell // 2
    elements.append(
        f'<circle cx="{center_x}" cy="{center_y}" r="8" fill="#2870b5" stroke="#ffffff" stroke-width="2"/>'
    )
    elements.append(
        '<text x="920" y="680" fill="#464b52" font-size="15" font-family="system-ui,sans-serif">O volume 3D continua disponível; este mapa é apenas uma projeção.</text>'
    )

    figure_path = output / "voxel_topology_audit.svg"
    figure_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="760" viewBox="0 0 1500 760">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )

    bridge_column = [
        {
            "y": y,
            "solid_probability": memory.get((0, y, 0)).solid_probability,
            "block": memory.get((0, y, 0)).dominant_block,
        }
        for y in range(63, 69)
    ]
    report = {
        "map_id": memory.map_id,
        "known_voxels": memory.voxel_count,
        "allocated_chunks": memory.chunk_count,
        "local_volume_shape": list(volume.shape),
        "local_volume_bytes_float32": int(volume.nbytes),
        "minimap_shape": list(minimap.shape),
        "bridge_tunnel_same_xz": bridge_column,
        "unknown_is_traversable": bool(minimap[4, 0, 0]),
        "low_ceiling_is_traversable": bool(minimap[4, 13, 13]),
        "figure": str(figure_path),
    }
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(scenario(), args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
