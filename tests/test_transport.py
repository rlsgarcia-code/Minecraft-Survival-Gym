from __future__ import annotations

import base64
import json
import socket
import struct
import threading

import numpy as np

from minecraft_gym.actions import ControlState
from minecraft_gym.transport import PROTOCOL_VERSION, SocketMinecraftBackend


def test_decode_transition_validates_and_decodes_rgb() -> None:
    backend = SocketMinecraftBackend()
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    response = {
        "ok": True,
        "observation": {
            "rgb": {
                "encoding": "raw_rgb8",
                "width": 3,
                "height": 2,
                "data": base64.b64encode(rgb.tobytes()).decode("ascii"),
            },
            "vitals": [1.0] * 8,
            "pose": [0.0] * 10,
            "inventory_ids": [0] * 36,
            "inventory_counts": [0] * 36,
            "equipped_slot": 0,
            "ui_mode": 0,
            "tick": 4,
            "world_seed": 7,
        },
        "events": [],
        "info": {},
    }
    transition = backend._decode_transition(response)
    np.testing.assert_array_equal(transition.observation.rgb, rgb)
    assert transition.observation.tick == 4


def test_socket_backend_round_trip() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    requests: list[dict] = []

    def observation(tick: int) -> dict:
        rgb = np.full((2, 3, 3), tick, dtype=np.uint8)
        return {
            "rgb": {
                "encoding": "raw_rgb8",
                "width": 3,
                "height": 2,
                "data": base64.b64encode(rgb.tobytes()).decode("ascii"),
            },
            "vitals": [1.0] * 8,
            "pose": [0.0] * 10,
            "inventory_ids": [0] * 36,
            "inventory_counts": [0] * 36,
            "equipped_slot": 0,
            "ui_mode": 0,
            "tick": tick,
            "world_seed": 17,
        }

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            while True:
                header = connection.recv(4)
                if not header:
                    return
                size = struct.unpack(">I", header)[0]
                body = bytearray()
                while len(body) < size:
                    body.extend(connection.recv(size - len(body)))
                request = json.loads(body)
                requests.append(request)
                if request["type"] == "hello":
                    response = {"ok": True, "protocol_version": PROTOCOL_VERSION}
                elif request["type"] == "close":
                    response = {"ok": True}
                else:
                    tick = 0 if request["type"] == "reset" else 4
                    response = {
                        "ok": True,
                        "observation": observation(tick),
                        "terminated": False,
                        "truncated": False,
                        "events": [],
                        "info": {},
                    }
                encoded = json.dumps(response, separators=(",", ":")).encode()
                connection.sendall(struct.pack(">I", len(encoded)) + encoded)
                if request["type"] == "close":
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    backend = SocketMinecraftBackend(port=port, timeout=2)
    try:
        reset = backend.reset(seed=17, options={}, width=3, height=2)
        assert reset.observation.tick == 0
        step = backend.step(ControlState(move_z=1), frame_skip=4, control_mode="agent")
        assert step.observation.tick == 4
        assert requests[-1]["action"]["move_z"] == 1
        np.testing.assert_array_equal(backend.render(), np.full((2, 3, 3), 4, dtype=np.uint8))
    finally:
        backend.close()
        thread.join(timeout=2)
        listener.close()
    assert [request["type"] for request in requests] == ["hello", "reset", "step", "close"]
