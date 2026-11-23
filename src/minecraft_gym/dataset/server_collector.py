"""Client for the passive Minecraft 26.3 authoritative server collector."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

from minecraft_gym.transport import BridgeProtocolError, LengthPrefixedJsonConnection

from .contracts import (
    ActionLabelV2,
    AuthoritativeTickV1,
    ObservationRefV2,
    StepRecordV2,
)


DEFAULT_COLLECTOR_HOST = "127.0.0.1"
DEFAULT_COLLECTOR_PORT = 25571
COLLECTOR_PROTOCOL_VERSION = 2


class CollectorProtocolError(BridgeProtocolError):
    pass


class CollectorGapError(CollectorProtocolError):
    def __init__(self, earliest_sequence: int, requested_after: int) -> None:
        super().__init__(
            "authoritative collector lost records: "
            f"requested after {requested_after}, earliest is {earliest_sequence}"
        )
        self.earliest_sequence = earliest_sequence
        self.requested_after = requested_after


@dataclass(frozen=True, slots=True)
class CollectorBatch:
    earliest_sequence: int
    latest_sequence: int
    records: tuple[AuthoritativeTickV1, ...]


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _signed(payload: dict[str, Any]) -> dict[str, Any]:
    payload_json = _canonical(payload).decode("utf-8")
    return {
        "payload_json": payload_json,
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
    }


def _verify_response(response: dict[str, Any]) -> dict[str, Any]:
    payload_json = response.get("payload_json")
    supplied = response.get("payload_sha256")
    if not isinstance(payload_json, str) or not isinstance(supplied, str):
        raise CollectorProtocolError("collector response has no signed payload envelope")
    expected = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise CollectorProtocolError("collector response checksum mismatch")
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError) as error:
        raise CollectorProtocolError("collector response payload is not JSON") from error
    if not isinstance(payload, dict):
        raise CollectorProtocolError("collector response payload must be an object")
    if payload.get("ok") is False:
        raise CollectorProtocolError(str(payload.get("error", "collector error")))
    return payload


class AuthoritativeCollectorClient:
    """Read-only resumable client; it exposes no world-control operations."""

    def __init__(
        self,
        host: str = DEFAULT_COLLECTOR_HOST,
        port: int = DEFAULT_COLLECTOR_PORT,
        *,
        timeout: float = 5.0,
    ) -> None:
        self.connection = LengthPrefixedJsonConnection(host, port, timeout=timeout)
        self.session_id: str | None = None
        self.capabilities: frozenset[str] = frozenset()
        self.minecraft_version: str | None = None
        self.last_sequence = -1

    def connect(self) -> dict[str, Any]:
        envelope = self.connection.request(
            {"type": "hello", "protocol_version": COLLECTOR_PROTOCOL_VERSION}
        )
        response = _verify_response(envelope)
        if int(response.get("protocol_version", -1)) != COLLECTOR_PROTOCOL_VERSION:
            raise CollectorProtocolError("authoritative collector protocol mismatch")
        session_id = response.get("session_id")
        capabilities = response.get("capabilities")
        if not isinstance(session_id, str) or not isinstance(capabilities, list):
            raise CollectorProtocolError("collector handshake is incomplete")
        required = {
            "authoritative_tick",
            "pseudonymous_actors",
            "resume_after_sequence",
            "gap_detection",
            "response_checksums",
        }
        available = frozenset(str(value) for value in capabilities)
        missing = required.difference(available)
        if missing:
            raise CollectorProtocolError(
                "collector is missing required capabilities: "
                + ", ".join(sorted(missing))
            )
        self.session_id = session_id
        self.capabilities = available
        self.minecraft_version = str(response.get("minecraft_version", ""))
        return response

    def snapshot(
        self,
        *,
        after_sequence: int | None = None,
        limit: int = 200,
    ) -> CollectorBatch:
        if self.session_id is None:
            self.connect()
        requested_after = self.last_sequence if after_sequence is None else after_sequence
        response = self._request(
            {
                "type": "snapshot",
                "after_sequence": int(requested_after),
                "limit": max(1, min(int(limit), 400)),
            }
        )
        earliest = int(response.get("earliest_sequence", 0))
        latest = int(response.get("latest_sequence", -1))
        if bool(response.get("gap", False)):
            raise CollectorGapError(earliest, requested_after)
        raw_records = response.get("records")
        if not isinstance(raw_records, list):
            raise CollectorProtocolError("collector snapshot has no record list")
        records = tuple(AuthoritativeTickV1.model_validate(item) for item in raw_records)
        expected = requested_after + 1
        for record in records:
            if record.sequence != expected:
                raise CollectorProtocolError(
                    f"non-contiguous collector sequence: expected {expected}, "
                    f"received {record.sequence}"
                )
            expected += 1
        if records:
            self.last_sequence = records[-1].sequence
        return CollectorBatch(earliest, latest, records)

    def ack(self, sequence: int | None = None) -> int:
        value = self.last_sequence if sequence is None else int(sequence)
        response = self._request({"type": "ack", "sequence": value})
        acknowledged = int(response.get("acked_sequence", -2))
        if acknowledged != value:
            raise CollectorProtocolError("collector acknowledged a different sequence")
        return acknowledged

    def close(self) -> None:
        self.connection.close()
        self.session_id = None

    def _request(self, values: dict[str, Any]) -> dict[str, Any]:
        if self.session_id is None:
            raise CollectorProtocolError("collector handshake has not completed")
        payload = _signed(
            {
                **values,
                "protocol_version": COLLECTOR_PROTOCOL_VERSION,
                "session_id": self.session_id,
            }
        )
        envelope = self.connection.request(payload)
        response = _verify_response(envelope)
        if response.get("session_id") != self.session_id:
            raise CollectorProtocolError("collector response changed session_id")
        return response


def semantic_steps_from_ticks(
    records: tuple[AuthoritativeTickV1, ...],
    *,
    episode_id: str,
    actor_id: str,
) -> tuple[StepRecordV2, ...]:
    """Project authoritative movement into weak labels without inventing input."""

    steps: list[StepRecordV2] = []
    for current, following in zip(records, records[1:]):
        before = _actor(current, actor_id)
        after = _actor(following, actor_id)
        if before is None or after is None:
            continue
        events = tuple(
            event
            for event in following.events
            if event.get("actor_id") in {None, actor_id}
        )
        terminated = any(event.get("type") == "death" for event in events)
        values = {
            "displacement": [
                float(after.get("x", 0.0)) - float(before.get("x", 0.0)),
                float(after.get("y", 0.0)) - float(before.get("y", 0.0)),
                float(after.get("z", 0.0)) - float(before.get("z", 0.0)),
            ],
            "yaw_delta": float(after.get("yaw", 0.0))
            - float(before.get("yaw", 0.0)),
            "pitch_delta": float(after.get("pitch", 0.0))
            - float(before.get("pitch", 0.0)),
        }
        steps.append(
            StepRecordV2(
                episode_id=episode_id,
                step_id=len(steps),
                actor_id=actor_id,
                data_tier="semantic_weak",
                observation=_semantic_observation(current, before),
                action=ActionLabelV2(
                    values=values,
                    source="server_reconstructed",
                    known_fields=("displacement", "yaw_delta", "pitch_delta"),
                    unknown_fields=(
                        "native_keys",
                        "native_mouse",
                        "attack",
                        "use",
                        "inventory_input",
                        "touch",
                    ),
                    confidence=0.6,
                ),
                next_observation=_semantic_observation(following, after),
                events=events,
                terminated=terminated,
                allowed_consumers=("semantic_imitation",),
                provenance={
                    "collector": "minecraft-gym-server-collector/0.1",
                    "sequence_start": current.sequence,
                    "sequence_end": following.sequence,
                    "label_limit": "server movement does not reveal native input",
                },
            )
        )
    return tuple(steps)


def _actor(record: AuthoritativeTickV1, actor_id: str) -> dict[str, Any] | None:
    return next(
        (dict(actor) for actor in record.actors if actor.get("actor_id") == actor_id),
        None,
    )


def _semantic_observation(
    record: AuthoritativeTickV1, actor: dict[str, Any]
) -> ObservationRefV2:
    policy_fields = {
        key: value
        for key, value in actor.items()
        if key
        not in {
            "actor_id",
            "dimension",
            "x",
            "y",
            "z",
            "yaw",
            "pitch",
        }
    }
    return ObservationRefV2(
        server_tick=record.server_tick,
        monotonic_ns=record.monotonic_ns,
        structured=policy_fields,
        policy_channels=tuple(sorted(policy_fields)),
        privileged={
            "dimension": actor.get("dimension"),
            "absolute_position": [
                actor.get("x"),
                actor.get("y"),
                actor.get("z"),
            ],
            "absolute_rotation": [actor.get("yaw"), actor.get("pitch")],
        },
    )
