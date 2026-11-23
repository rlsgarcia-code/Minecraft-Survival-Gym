from __future__ import annotations

import hashlib
from io import BytesIO
import json
import struct
import zlib

import numpy as np
from PIL import Image

from minecraft_gym.academy import encode_dense_rgbd_policy_input
from minecraft_gym.academy.visual_actor_critic import VisualHybridActorCriticAgent
from minecraft_gym.dataset import HttpDenseRgbdStreamClient, ObservationRefV2


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _envelope() -> bytes:
    image = Image.fromarray(
        np.asarray([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8), mode="RGB"
    )
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=95)
    rgb = encoded.getvalue()
    # 2.00 blocks measured, then 64.00 blocks censored at max range.
    raw_depth = bytes((200, 0, 1, 0, 25, 2))
    compressed = zlib.compress(raw_depth)
    metadata = {
        "interface": "minecraft-dense-rgbd-frame/v1",
        "stream_id": "view:bot:egocentric",
        "actor_id": "actor:bot",
        "viewpoint": "egocentric",
        "source": "prismarine_renderer",
        "visual_domain": "rendered_prismarine",
        "frame_id": "bot-rgbd:7:test",
        "capture_sequence": 7,
        "captured_monotonic_ns": "100000000",
        "width": 2,
        "height": 1,
        "nominal_fps": 20,
        "dropped_frame_count": 0,
        "self_entity_excluded": True,
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
        "depth_sha256": hashlib.sha256(compressed).hexdigest(),
        "depth_uncompressed_sha256": hashlib.sha256(raw_depth).hexdigest(),
        "depth_encoding": "uint16_le_centiblocks+uint8_status_interleaved",
        "max_distance": 64,
        "camera_near": 0.1,
        "camera_far": 1000,
        "vertical_fov_degrees": 75,
        "capture_timings_ms": {"total": 4.5, "depth_encode_and_read": 1.2},
    }
    header = json.dumps(metadata).encode()
    prefix = b"MGRD" + struct.pack(">III", len(header), len(rgb), len(compressed))
    return prefix + header + rgb + compressed


def test_dense_rgbd_binary_transport_validates_and_decodes(monkeypatch) -> None:
    monkeypatch.setattr(
        "minecraft_gym.dataset.visual_stream.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_envelope()),
    )
    frame = HttpDenseRgbdStreamClient().latest()

    assert frame.visual.width == 2
    assert frame.visual.height == 1
    assert frame.capture_timings_ms["total"] == 4.5
    depth, status = frame.unpack_depth()
    assert depth == (2.0, 64.0)
    assert status == (1, 2)


def test_dense_depth_reference_is_frame_aligned(monkeypatch) -> None:
    monkeypatch.setattr(
        "minecraft_gym.dataset.visual_stream.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_envelope()),
    )
    captured = HttpDenseRgbdStreamClient().latest()
    viewpoint = captured.visual.reference(
        rgb_ref="video/view_bot/frame.jpg",
        server_tick=9,
        now_monotonic_ns=100_000_000,
    )
    depth = captured.depth_reference(
        depth_ref="depth/view_bot/frame.depth.zlib",
        server_tick=9,
        now_monotonic_ns=100_000_000,
    )
    observation = ObservationRefV2(
        server_tick=9,
        monotonic_ns=100_000_000,
        frame_id=viewpoint.frame_id,
        rgb_ref=viewpoint.rgb_ref,
        primary_stream_id=viewpoint.stream_id,
        viewpoints=(viewpoint,),
        policy_channels=("rgb", "depth"),
        dense_depth=depth,
    )

    assert observation.dense_depth is not None
    assert observation.dense_depth.width == viewpoint.width
    assert observation.dense_depth.camera_frame_id == viewpoint.frame_id


def test_dense_frame_becomes_five_aligned_policy_planes(monkeypatch) -> None:
    monkeypatch.setattr(
        "minecraft_gym.dataset.visual_stream.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_envelope()),
    )
    captured = HttpDenseRgbdStreamClient().latest()
    encoded = encode_dense_rgbd_policy_input(
        captured, output_width=2, output_height=1
    )

    assert encoded.shape == (5, 1, 2)
    assert encoded[3, 0].tolist() == [2.0 / 64.0, 1.0]
    assert encoded[4, 0].tolist() == [1.0, 1.0]


def test_visual_actor_critic_consumes_live_dense_frame(monkeypatch) -> None:
    monkeypatch.setattr(
        "minecraft_gym.dataset.visual_stream.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(_envelope()),
    )
    captured = HttpDenseRgbdStreamClient().latest()
    agent = VisualHybridActorCriticAgent(modality="rgbd", seed=7)

    state = agent._policy_state({"_dense_rgbd_frame": captured})

    assert state.joint_lattice.shape == (5, 1, 2)
    assert np.all(state.joint_lattice[4] == 1.0)
    assert agent.architecture == "vgg-rgbd-shared-flatten-mlp/v1"
