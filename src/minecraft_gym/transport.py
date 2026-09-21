"""Length-prefixed JSON bridge used by Python and the Fabric client mod."""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any

import numpy as np

from minecraft_gym.actions import ControlState
from minecraft_gym.backend import (
    BackendObservation,
    BackendTransition,
    ControlMode,
    MinecraftBackend,
)


PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 25570
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class BridgeProtocolError(RuntimeError):
    pass


class LengthPrefixedJsonConnection:
    """One request at a time over a persistent TCP connection."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.create_connection((self.host, self.port), self.timeout)
        self._socket.settimeout(self.timeout)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        assert self._socket is not None
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise BridgeProtocolError("Outgoing bridge message is too large")
        self._socket.sendall(struct.pack(">I", len(encoded)) + encoded)
        length = struct.unpack(">I", self._read_exact(4))[0]
        if length > MAX_MESSAGE_BYTES:
            raise BridgeProtocolError(f"Incoming bridge message is too large: {length}")
        response = json.loads(self._read_exact(length).decode("utf-8"))
        if not isinstance(response, dict):
            raise BridgeProtocolError("Bridge response must be a JSON object")
        if response.get("ok") is False:
            raise BridgeProtocolError(str(response.get("error", "Minecraft bridge error")))
        return response

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def _read_exact(self, size: int) -> bytes:
        assert self._socket is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise BridgeProtocolError("Minecraft bridge closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class SocketMinecraftBackend(MinecraftBackend):
    """Backend connected to the Fabric mod over localhost TCP."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.connection = LengthPrefixedJsonConnection(host, port, timeout=timeout)
        self._last_frame: np.ndarray | None = None
        self._hello_complete = False

    def reset(
        self, *, seed: int, options: dict[str, Any], width: int, height: int
    ) -> BackendTransition:
        self._hello()
        response = self.connection.request(
            {
                "type": "reset",
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "options": options,
                "image": {"width": width, "height": height, "format": "raw_rgb8"},
            }
        )
        return self._decode_transition(response)

    def step(
        self,
        control: ControlState,
        *,
        frame_skip: int,
        control_mode: ControlMode,
    ) -> BackendTransition:
        response = self.connection.request(
            {
                "type": "step",
                "protocol_version": PROTOCOL_VERSION,
                "frame_skip": frame_skip,
                "control_mode": control_mode,
                "action": control.to_dict(),
            }
        )
        return self._decode_transition(response)

    def render(self) -> np.ndarray | None:
        return None if self._last_frame is None else self._last_frame.copy()

    def close(self) -> None:
        if self._hello_complete:
            try:
                self.connection.request(
                    {"type": "close", "protocol_version": PROTOCOL_VERSION}
                )
            except (OSError, BridgeProtocolError, TimeoutError):
                pass
        self.connection.close()
        self._hello_complete = False
        self._last_frame = None

    def _hello(self) -> None:
        if self._hello_complete:
            return
        response = self.connection.request(
            {"type": "hello", "protocol_version": PROTOCOL_VERSION}
        )
        remote_version = int(response.get("protocol_version", -1))
        if remote_version != PROTOCOL_VERSION:
            raise BridgeProtocolError(
                f"Protocol mismatch: Python={PROTOCOL_VERSION}, bridge={remote_version}"
            )
        self._hello_complete = True

    def _decode_transition(self, response: dict[str, Any]) -> BackendTransition:
        raw = response.get("observation")
        if not isinstance(raw, dict):
            raise BridgeProtocolError("Bridge response has no observation object")
        observation = self._decode_observation(raw)
        self._last_frame = observation.rgb.copy()
        events = response.get("events", [])
        info = response.get("info", {})
        if not isinstance(events, list) or not isinstance(info, dict):
            raise BridgeProtocolError("Invalid events or info in bridge response")
        return BackendTransition(
            observation=observation,
            terminated=bool(response.get("terminated", False)),
            truncated=bool(response.get("truncated", False)),
            events=events,
            info=info,
        )

    def _decode_observation(self, raw: dict[str, Any]) -> BackendObservation:
        image = raw.get("rgb")
        if not isinstance(image, dict) or image.get("encoding") != "raw_rgb8":
            raise BridgeProtocolError("Only raw_rgb8 observations are supported")
        width = int(image["width"])
        height = int(image["height"])
        pixels = base64.b64decode(image["data"], validate=True)
        expected = width * height * 3
        if len(pixels) != expected:
            raise BridgeProtocolError(
                f"RGB payload has {len(pixels)} bytes; expected {expected}"
            )
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
        return BackendObservation(
            rgb=rgb,
            vitals=np.asarray(raw["vitals"], dtype=np.float32),
            pose=np.asarray(raw["pose"], dtype=np.float32),
            inventory_ids=np.asarray(raw["inventory_ids"], dtype=np.int32),
            inventory_counts=np.asarray(raw["inventory_counts"], dtype=np.int32),
            equipped_slot=int(raw["equipped_slot"]),
            ui_mode=int(raw["ui_mode"]),
            tick=int(raw["tick"]),
            world_seed=int(raw["world_seed"]),
        )
