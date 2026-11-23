"""Length-prefixed JSON bridge used by Python and the Fabric client mod."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import time
from typing import Any

import numpy as np

from minecraft_gym.actions import ControlState
from minecraft_gym.backend import (
    BackendObservation,
    BackendTransition,
    ControlMode,
    MinecraftBackend,
)


PROTOCOL_VERSION = 2
SUPPORTED_PROTOCOL_VERSIONS = (2, 1)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 25570
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class BridgeProtocolError(RuntimeError):
    pass


def probe_bridge(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    timeout: float = 0.8,
) -> dict[str, Any]:
    """Read bridge identity and capabilities without changing a world."""

    connection = LengthPrefixedJsonConnection(host, port, timeout=timeout)
    try:
        response = _negotiate(connection)
    finally:
        connection.close()
    remote_version = int(response.get("protocol_version", -1))
    if remote_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise BridgeProtocolError(
            f"Protocol mismatch: Python={SUPPORTED_PROTOCOL_VERSIONS}, bridge={remote_version}"
        )
    capabilities = response.get("capabilities", [])
    if not isinstance(capabilities, list):
        raise BridgeProtocolError("Bridge capabilities must be a list")
    return {
        "available": True,
        "host": host,
        "port": port,
        "protocol_version": remote_version,
        "bridge_version": response.get("bridge_version"),
        "minecraft_version": response.get("minecraft_version"),
        "capabilities": [str(item) for item in capabilities],
        "action_contract": response.get("action_contract"),
        "observation_contract": response.get("observation_contract"),
        "tick_rate_hz": response.get("tick_rate_hz"),
    }


def _negotiate(connection: "LengthPrefixedJsonConnection") -> dict[str, Any]:
    last_error: Exception | None = None
    for version in SUPPORTED_PROTOCOL_VERSIONS:
        payload: dict[str, Any] = {"type": "hello", "protocol_version": version}
        if version == 2:
            payload.update(
                {
                    "action_contract": "minecraft-native-control/v2",
                    "observation_contract": "minecraft-observation/v2",
                }
            )
        try:
            return connection.request(payload)
        except BridgeProtocolError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


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
        visual_actor_id: str = "actor:pseudonym:java-client",
    ) -> None:
        if not visual_actor_id.startswith("actor:pseudonym:"):
            raise ValueError("visual_actor_id must be a dataset pseudonym")
        self.connection = LengthPrefixedJsonConnection(host, port, timeout=timeout)
        self._visual_actor_id = visual_actor_id
        self._visual_capture_sequence = 0
        self._last_frame: np.ndarray | None = None
        self._hello_complete = False
        self._protocol_version = PROTOCOL_VERSION
        self._session_id: str | None = None
        self._episode_id: str | None = None
        self._next_step_id = 0

    def reset(
        self, *, seed: int, options: dict[str, Any], width: int, height: int
    ) -> BackendTransition:
        self._hello()
        response = self._request(
            "reset",
            transition=True,
            values={
                "seed": seed,
                "options": options,
                "image": {"width": width, "height": height, "format": "raw_rgb8"},
            },
        )
        return self._decode_transition(response)

    def attach(self, *, width: int, height: int) -> BackendTransition:
        """Attach passively to the current world without resetting it."""

        response = self._request(
            "observe_start",
            transition=True,
            values={
                "image": {"width": width, "height": height, "format": "raw_rgb8"},
            },
        )
        return self._decode_transition(response)

    def observe(self) -> BackendTransition:
        """Sample the current world without taking the control lease."""

        return self._decode_transition(self._request("observe", transition=True))

    def start_macro(self, control: ControlState) -> dict[str, Any]:
        return self._request("macro_start", values={"action": control.to_dict()})

    def stop_macro(self) -> BackendTransition:
        return self._decode_transition(self._request("macro_stop", transition=True))

    def keepalive(self) -> dict[str, Any]:
        """Keep a passive persistent bridge session alive without a Gym step."""

        return self._request("keepalive")

    def step(
        self,
        control: ControlState,
        *,
        frame_skip: int,
        control_mode: ControlMode,
    ) -> BackendTransition:
        response = self._request(
            "step",
            transition=True,
            values={
                "frame_skip": frame_skip,
                "control_mode": control_mode,
                "action": control.to_dict(),
            },
        )
        return self._decode_transition(response)

    def render(self) -> np.ndarray | None:
        return None if self._last_frame is None else self._last_frame.copy()

    def close(self) -> None:
        if self._hello_complete:
            try:
                self._request("close")
            except (OSError, BridgeProtocolError, TimeoutError):
                pass
        self.connection.close()
        self._hello_complete = False
        self._session_id = None
        self._episode_id = None
        self._next_step_id = 0
        self._visual_capture_sequence = 0
        self._last_frame = None

    def _hello(self) -> None:
        if self._hello_complete:
            return
        response = _negotiate(self.connection)
        remote_version = int(response.get("protocol_version", -1))
        if remote_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise BridgeProtocolError(
                f"Protocol mismatch: Python={SUPPORTED_PROTOCOL_VERSIONS}, bridge={remote_version}"
            )
        self._protocol_version = remote_version
        self._session_id = response.get("session_id")
        self._episode_id = response.get("episode_id")
        self._next_step_id = 0
        self._hello_complete = True

    def _request(
        self,
        request_type: str,
        *,
        transition: bool = False,
        values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._hello()
        payload: dict[str, Any] = {
            "type": request_type,
            "protocol_version": self._protocol_version,
            **(values or {}),
        }
        if self._protocol_version == 2:
            payload["session_id"] = self._session_id
            if transition:
                payload["step_id"] = self._next_step_id
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        response = self.connection.request(payload)
        if self._protocol_version == 2 and transition:
            if response.get("session_id") != self._session_id:
                raise BridgeProtocolError("Bridge response changed session_id")
            if int(response.get("step_id", -1)) != self._next_step_id:
                raise BridgeProtocolError("Bridge response step_id is not contiguous")
            self._episode_id = response.get("episode_id")
            self._next_step_id += 1
        return response

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
        # The Fabric image and observation tick are produced by the same
        # client transition. Publish that provenance at the Gym boundary so
        # consumers never have to infer which actor or renderer made a frame.
        info = dict(info)
        info.setdefault(
            "visual_frame",
            {
                "interface": "minecraft-native-visual-frame/v1",
                "stream_id": "view:demonstrator:egocentric",
                "actor_id": self._visual_actor_id,
                "viewpoint": "egocentric",
                "source": "fabric_native",
                "visual_domain": "native_java",
                "server_tick": observation.tick,
                "captured_monotonic_ns": time.monotonic_ns(),
                "capture_sequence": self._visual_capture_sequence,
                "width": int(observation.rgb.shape[1]),
                "height": int(observation.rgb.shape[0]),
                "alignment_method": "exact_client_tick",
                "policy_eligible": True,
            },
        )
        self._visual_capture_sequence += 1
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
