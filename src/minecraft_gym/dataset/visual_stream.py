"""Validated visual stream transport shared by Gym, Observer, and recorders."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass

import numpy as np

from .contracts import (
    DenseDepthFrameRefV1,
    ObservationRefV2,
    SpatialMapUpdateV1,
    ViewpointFrameRefV1,
)


VISUAL_STREAM_PROTOCOL = "minecraft-visual-stream/v1"
DENSE_RGBD_STREAM_PROTOCOL = "minecraft-dense-rgbd-frame/v1"
DENSE_RGBD_MAGIC = b"MGRD"


class VisualStreamError(RuntimeError):
    pass


def native_viewpoint_reference(
    *,
    data: bytes,
    rgb_ref: str,
    actor_id: str,
    server_tick: int,
    capture_sequence: int,
    monotonic_ns: int,
    width: int,
    height: int,
    stream_id: str = "view:demonstrator:egocentric",
    policy_eligible: bool = True,
) -> ViewpointFrameRefV1:
    """Build the exact Fabric viewpoint carried by a Gym transition.

    The checksum covers the persisted representation (normally JPEG), not the
    transient raw RGB buffer, so a dataset audit verifies the actual artifact.
    """

    if not actor_id.startswith("actor:pseudonym:"):
        raise VisualStreamError("native visual actors must use dataset pseudonyms")
    digest = hashlib.sha256(data).hexdigest()
    return ViewpointFrameRefV1(
        stream_id=stream_id,
        actor_id=actor_id,
        viewpoint="egocentric",
        source="fabric_native",
        visual_domain="native_java",
        frame_id=f"frame:native:{capture_sequence}:{digest[:12]}",
        rgb_ref=rgb_ref,
        server_tick=server_tick,
        monotonic_ns=monotonic_ns,
        capture_sequence=capture_sequence,
        width=width,
        height=height,
        sha256=digest,
        alignment_ticks=0,
        alignment_method="exact_client_tick",
        policy_eligible=policy_eligible,
    )


def select_policy_viewpoint(
    viewpoints: tuple[ViewpointFrameRefV1, ...] | list[ViewpointFrameRefV1],
    *,
    action_source: str,
    actor_id: str | None = None,
) -> ViewpointFrameRefV1 | None:
    """Select the camera paired with an action without discarding other views."""

    candidates = list(viewpoints)
    if actor_id is not None:
        actor_matches = [item for item in candidates if item.actor_id == actor_id]
        if actor_matches:
            candidates = actor_matches
    required_source = {
        "java_native_input": "fabric_native",
        "gym_agent": "prismarine_renderer",
        "gym_agent_macro": "prismarine_renderer",
    }.get(action_source)
    if required_source is not None:
        return next(
            (item for item in candidates if item.source == required_source), None
        )
    return candidates[0] if candidates else None


def assemble_multiview_observation(
    *,
    server_tick: int,
    monotonic_ns: int,
    viewpoints: tuple[ViewpointFrameRefV1, ...] | list[ViewpointFrameRefV1],
    action_source: str,
    actor_id: str | None = None,
    structured: dict | None = None,
    privileged: dict | None = None,
    spatial_updates: tuple[SpatialMapUpdateV1, ...] | list[SpatialMapUpdateV1] = (),
) -> ObservationRefV2:
    """Build one policy observation while preserving every synchronized view."""

    frames = tuple(viewpoints)
    primary = select_policy_viewpoint(
        frames, action_source=action_source, actor_id=actor_id
    )
    structured_payload = structured or {}
    return ObservationRefV2(
        server_tick=server_tick,
        monotonic_ns=monotonic_ns,
        frame_id=primary.frame_id if primary else None,
        rgb_ref=primary.rgb_ref if primary else None,
        primary_stream_id=primary.stream_id if primary else None,
        viewpoints=frames,
        structured=structured_payload,
        policy_channels=(
            ("rgb", *tuple(sorted(structured_payload)))
            if primary is not None
            else tuple(sorted(structured_payload))
        ),
        privileged=privileged or {},
        spatial_updates=tuple(spatial_updates),
    )


@dataclass(frozen=True, slots=True)
class CapturedVisualFrame:
    data: bytes
    stream_id: str
    actor_id: str
    viewpoint: str
    source: str
    visual_domain: str
    frame_id: str
    capture_sequence: int
    captured_monotonic_ns: int
    width: int
    height: int
    sha256: str
    nominal_fps: float
    dropped_frame_count: int
    self_entity_excluded: bool = False

    def reference(
        self,
        *,
        rgb_ref: str,
        server_tick: int,
        now_monotonic_ns: int | None = None,
        policy_eligible: bool = True,
    ) -> ViewpointFrameRefV1:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        age_ns = max(0, now - self.captured_monotonic_ns)
        alignment_ticks = int(math.ceil(age_ns / 50_000_000))
        return ViewpointFrameRefV1(
            stream_id=self.stream_id,
            actor_id=self.actor_id,
            viewpoint=self.viewpoint,
            source=self.source,
            visual_domain=self.visual_domain,
            frame_id=self.frame_id,
            rgb_ref=rgb_ref,
            server_tick=server_tick,
            monotonic_ns=self.captured_monotonic_ns,
            capture_sequence=self.capture_sequence,
            width=self.width,
            height=self.height,
            sha256=self.sha256,
            alignment_ticks=alignment_ticks,
            alignment_method="nearest_monotonic",
            policy_eligible=policy_eligible and self.self_entity_excluded,
        )


@dataclass(frozen=True, slots=True)
class CapturedDenseRgbdFrame:
    """One renderer-atomic RGB/depth pair from the local Prismarine stream."""

    visual: CapturedVisualFrame
    depth_compressed: bytes
    depth_sha256: str
    depth_uncompressed_sha256: str
    depth_encoding: str
    max_distance: float
    camera_near: float
    camera_far: float
    vertical_fov_degrees: float
    capture_timings_ms: dict[str, float]

    def depth_reference(
        self,
        *,
        depth_ref: str,
        server_tick: int,
        now_monotonic_ns: int | None = None,
        policy_eligible: bool = True,
    ) -> DenseDepthFrameRefV1:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        age_ns = max(0, now - self.visual.captured_monotonic_ns)
        alignment_ticks = int(math.ceil(age_ns / 50_000_000))
        return DenseDepthFrameRefV1(
            sensor_id="sensor:bot:prismarine-depth-buffer",
            actor_id=self.visual.actor_id,
            server_tick=server_tick,
            monotonic_ns=self.visual.captured_monotonic_ns,
            camera_stream_id=self.visual.stream_id,
            camera_frame_id=self.visual.frame_id,
            depth_ref=depth_ref,
            width=self.visual.width,
            height=self.visual.height,
            max_distance=self.max_distance,
            sha256=self.depth_sha256,
            uncompressed_sha256=self.depth_uncompressed_sha256,
            alignment_ticks=alignment_ticks,
            policy_eligible=policy_eligible and self.visual.self_entity_excluded,
        )

    def unpack_depth(self) -> tuple[tuple[float, ...], tuple[int, ...]]:
        """Decode the portable interleaved payload for audits and training."""

        depth, status = self.depth_arrays()
        return tuple(float(value) for value in depth.flat), tuple(
            int(value) for value in status.flat
        )

    def depth_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Decode depth and status as native rasters without Python pixel loops."""

        payload = zlib.decompress(self.depth_compressed)
        expected = self.visual.width * self.visual.height * 3
        if len(payload) != expected:
            raise VisualStreamError("dense depth payload size does not match frame")
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.visual.height, self.visual.width, 3
        )
        depth = (
            packed[..., 0].astype(np.float32)
            + packed[..., 1].astype(np.float32) * 256.0
        ) / 100.0
        return depth, packed[..., 2].copy()


class HttpVisualStreamClient:
    """Fetch the latest immutable frame from a local visual producer."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:3008/v1/frame/latest?fresh=1",
        *,
        timeout: float = 0.35,
    ) -> None:
        self.url = url
        self.timeout = timeout

    def latest(self) -> CapturedVisualFrame:
        request = urllib.request.Request(self.url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                headers = response.headers
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise VisualStreamError(f"visual stream unavailable: {error}") from error
        if headers.get("X-Visual-Protocol") != VISUAL_STREAM_PROTOCOL:
            raise VisualStreamError("visual stream protocol mismatch")
        supplied = headers.get("X-Frame-SHA256", "")
        expected = hashlib.sha256(data).hexdigest()
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise VisualStreamError("visual frame checksum mismatch")
        try:
            return CapturedVisualFrame(
                data=data,
                stream_id=headers["X-Stream-ID"],
                actor_id=headers["X-Actor-ID"],
                viewpoint=headers["X-Viewpoint"],
                source=headers["X-Visual-Source"],
                visual_domain=headers["X-Visual-Domain"],
                frame_id=headers["X-Frame-ID"],
                capture_sequence=int(headers["X-Capture-Sequence"]),
                captured_monotonic_ns=int(headers["X-Captured-Monotonic-NS"]),
                width=int(headers["X-Frame-Width"]),
                height=int(headers["X-Frame-Height"]),
                sha256=supplied,
                nominal_fps=float(headers["X-Nominal-FPS"]),
                dropped_frame_count=int(headers.get("X-Dropped-Frame-Count", "0")),
                self_entity_excluded=(
                    headers.get("X-Self-Entity-Excluded", "").casefold() == "true"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VisualStreamError("visual frame metadata is incomplete") from error


class HttpDenseRgbdStreamClient:
    """Fetch a renderer-atomic dense RGB-D frame using the binary v1 envelope."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:3008/v1/rgbd/latest?fresh=1",
        *,
        timeout: float = 1.0,
    ) -> None:
        self.url = url
        self.timeout = timeout

    def latest(self) -> CapturedDenseRgbdFrame:
        request = urllib.request.Request(self.url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                envelope = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise VisualStreamError(f"dense RGB-D stream unavailable: {error}") from error
        if len(envelope) < 16 or envelope[:4] != DENSE_RGBD_MAGIC:
            raise VisualStreamError("dense RGB-D envelope magic mismatch")
        header_size, rgb_size, depth_size = struct.unpack(">III", envelope[4:16])
        expected_size = 16 + header_size + rgb_size + depth_size
        if len(envelope) != expected_size:
            raise VisualStreamError("dense RGB-D envelope length mismatch")
        header_start = 16
        rgb_start = header_start + header_size
        depth_start = rgb_start + rgb_size
        try:
            metadata = json.loads(envelope[header_start:rgb_start])
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise VisualStreamError("dense RGB-D header is invalid") from error
        if metadata.get("interface") != DENSE_RGBD_STREAM_PROTOCOL:
            raise VisualStreamError("dense RGB-D protocol mismatch")
        rgb = envelope[rgb_start:depth_start]
        depth = envelope[depth_start:]
        rgb_sha256 = hashlib.sha256(rgb).hexdigest()
        depth_sha256 = hashlib.sha256(depth).hexdigest()
        if not hmac.compare_digest(rgb_sha256, str(metadata.get("rgb_sha256", ""))):
            raise VisualStreamError("dense RGB-D RGB checksum mismatch")
        if not hmac.compare_digest(depth_sha256, str(metadata.get("depth_sha256", ""))):
            raise VisualStreamError("dense RGB-D depth checksum mismatch")
        try:
            uncompressed = zlib.decompress(depth)
        except zlib.error as error:
            raise VisualStreamError("dense RGB-D depth payload is not valid zlib") from error
        expected_depth_size = int(metadata["width"]) * int(metadata["height"]) * 3
        if len(uncompressed) != expected_depth_size:
            raise VisualStreamError("dense RGB-D depth raster has an invalid size")
        uncompressed_sha256 = hashlib.sha256(uncompressed).hexdigest()
        if not hmac.compare_digest(
            uncompressed_sha256,
            str(metadata.get("depth_uncompressed_sha256", "")),
        ):
            raise VisualStreamError("dense RGB-D uncompressed checksum mismatch")
        try:
            visual = CapturedVisualFrame(
                data=rgb,
                stream_id=str(metadata["stream_id"]),
                actor_id=str(metadata["actor_id"]),
                viewpoint=str(metadata["viewpoint"]),
                source=str(metadata["source"]),
                visual_domain=str(metadata["visual_domain"]),
                frame_id=str(metadata["frame_id"]),
                capture_sequence=int(metadata["capture_sequence"]),
                captured_monotonic_ns=int(metadata["captured_monotonic_ns"]),
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                sha256=rgb_sha256,
                nominal_fps=float(metadata["nominal_fps"]),
                dropped_frame_count=int(metadata.get("dropped_frame_count", 0)),
                self_entity_excluded=bool(metadata.get("self_entity_excluded", False)),
            )
            timings = {
                str(key): float(value)
                for key, value in dict(metadata.get("capture_timings_ms", {})).items()
            }
            return CapturedDenseRgbdFrame(
                visual=visual,
                depth_compressed=depth,
                depth_sha256=depth_sha256,
                depth_uncompressed_sha256=uncompressed_sha256,
                depth_encoding=str(metadata["depth_encoding"]),
                max_distance=float(metadata["max_distance"]),
                camera_near=float(metadata["camera_near"]),
                camera_far=float(metadata["camera_far"]),
                vertical_fov_degrees=float(metadata["vertical_fov_degrees"]),
                capture_timings_ms=timings,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VisualStreamError("dense RGB-D metadata is incomplete") from error
