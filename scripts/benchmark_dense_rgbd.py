#!/usr/bin/env python3
"""Gate dense Prismarine RGB-D before any long-running training starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import time
import urllib.parse
import urllib.request
import zlib


MAGIC = b"MGRD"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def endpoint(base_url: str, *, depth: bool) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.update({"fresh": "1", "depth": "1" if depth else "0"})
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def capture(url: str, *, timeout: float, require_depth: bool) -> dict:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        envelope = response.read()
    wall_ms = (time.perf_counter() - started) * 1000.0
    if len(envelope) < 16 or envelope[:4] != MAGIC:
        raise RuntimeError("RGB-D envelope magic mismatch")
    header_size, rgb_size, depth_size = struct.unpack(">III", envelope[4:16])
    if len(envelope) != 16 + header_size + rgb_size + depth_size:
        raise RuntimeError("RGB-D envelope length mismatch")
    header_end = 16 + header_size
    rgb_end = header_end + rgb_size
    metadata = json.loads(envelope[16:header_end])
    rgb = envelope[header_end:rgb_end]
    depth = envelope[rgb_end:]
    if hashlib.sha256(rgb).hexdigest() != metadata["rgb_sha256"]:
        raise RuntimeError("RGB checksum mismatch")
    if require_depth:
        raw = zlib.decompress(depth)
        expected = int(metadata["width"]) * int(metadata["height"]) * 3
        if len(raw) != expected:
            raise RuntimeError("dense depth size mismatch")
        if hashlib.sha256(depth).hexdigest() != metadata["depth_sha256"]:
            raise RuntimeError("compressed depth checksum mismatch")
        if hashlib.sha256(raw).hexdigest() != metadata["depth_uncompressed_sha256"]:
            raise RuntimeError("uncompressed depth checksum mismatch")
    elif depth:
        raise RuntimeError("RGB control unexpectedly returned depth")
    return {
        "wall_ms": wall_ms,
        "envelope_bytes": len(envelope),
        "metadata": metadata,
    }


def summary(records: list[dict]) -> dict:
    wall = [float(item["wall_ms"]) for item in records]
    renderer = [
        float(item["metadata"].get("capture_timings_ms", {}).get("total", 0.0))
        for item in records
    ]
    sizes = [int(item["envelope_bytes"]) for item in records]
    return {
        "samples": len(records),
        "wall_ms": {
            "median": statistics.median(wall),
            "p95": percentile(wall, 0.95),
            "max": max(wall),
        },
        "renderer_ms": {
            "median": statistics.median(renderer),
            "p95": percentile(renderer, 0.95),
            "max": max(renderer),
        },
        "envelope_bytes": {
            "median": statistics.median(sizes),
            "max": max(sizes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:3008/v1/rgbd/latest"
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--p95-frame-budget-ms", type=float, default=50.0)
    parser.add_argument("--p95-depth-overhead-ms", type=float, default=5.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--planned-hours", type=float, default=8.0)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 5 or args.warmup < 0:
        parser.error("samples must be >= 5 and warmup must be >= 0")

    rgb_url = endpoint(args.url, depth=False)
    rgbd_url = endpoint(args.url, depth=True)
    failures: list[str] = []
    for _ in range(args.warmup):
        capture(rgb_url, timeout=args.timeout, require_depth=False)
        capture(rgbd_url, timeout=args.timeout, require_depth=True)

    rgb_records: list[dict] = []
    rgbd_records: list[dict] = []
    pair_overheads: list[float] = []
    for _ in range(args.samples):
        try:
            rgb = capture(rgb_url, timeout=args.timeout, require_depth=False)
            rgbd = capture(rgbd_url, timeout=args.timeout, require_depth=True)
        except Exception as error:  # a gate report must retain every failure
            failures.append(f"{type(error).__name__}: {error}")
            continue
        rgb_records.append(rgb)
        rgbd_records.append(rgbd)
        pair_overheads.append(rgbd["wall_ms"] - rgb["wall_ms"])

    if not rgb_records or not rgbd_records:
        raise SystemExit("no valid paired benchmark samples were captured")
    rgb_summary = summary(rgb_records)
    rgbd_summary = summary(rgbd_records)
    overhead_p95 = percentile(pair_overheads, 0.95)
    sample_metadata = rgbd_records[-1]["metadata"]
    resolution_ok = (
        int(sample_metadata["width"]) == 640
        and int(sample_metadata["height"]) == 360
    )
    estimated_bytes_per_hour = int(
        float(rgbd_summary["envelope_bytes"]["median"]) * args.target_fps * 3600
    )
    estimated_run_bytes = int(estimated_bytes_per_hour * args.planned_hours)
    disk = os.statvfs(args.output.parent if args.output.parent.exists() else Path.cwd())
    free_bytes = disk.f_frsize * disk.f_bavail
    storage_ok = free_bytes - estimated_run_bytes >= int(args.minimum_free_gib * 1024**3)
    gates = {
        "no_capture_failures": not failures,
        "resolution_is_640x360": resolution_ok,
        "rgbd_wall_p95_inside_frame_budget": (
            rgbd_summary["wall_ms"]["p95"] <= args.p95_frame_budget_ms
        ),
        "depth_added_p95_inside_overhead_budget": (
            overhead_p95 <= args.p95_depth_overhead_ms
        ),
        "planned_run_preserves_free_space": storage_ok,
    }
    report = {
        "interface": "minecraft-dense-rgbd-preflight/v1",
        "status": "passed" if all(gates.values()) else "blocked",
        "captured_at_unix_ns": time.time_ns(),
        "url": args.url,
        "resolution": [int(sample_metadata["width"]), int(sample_metadata["height"])],
        "target_fps": args.target_fps,
        "limits": {
            "p95_frame_budget_ms": args.p95_frame_budget_ms,
            "p95_depth_overhead_ms": args.p95_depth_overhead_ms,
            "minimum_free_gib_after_run": args.minimum_free_gib,
        },
        "rgb_control": rgb_summary,
        "rgbd": rgbd_summary,
        "paired_depth_overhead_ms": {
            "median": statistics.median(pair_overheads),
            "p95": overhead_p95,
            "max": max(pair_overheads),
        },
        "storage": {
            "free_bytes": free_bytes,
            "estimated_bytes_per_hour": estimated_bytes_per_hour,
            "planned_hours": args.planned_hours,
            "estimated_run_bytes": estimated_run_bytes,
            "minimum_free_bytes": int(args.minimum_free_gib * 1024**3),
        },
        "gates": gates,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
