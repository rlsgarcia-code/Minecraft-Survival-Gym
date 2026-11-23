from __future__ import annotations

import json
import socket
import struct
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from pydantic import ValidationError

from minecraft_gym.dataset.contracts import AuthoritativeTickV1
from minecraft_gym.dataset.server_collector import (
    AuthoritativeCollectorClient,
    CollectorGapError,
    CollectorProtocolError,
    _signed,
    semantic_steps_from_ticks,
)


def tick(sequence: int) -> dict[str, Any]:
    return {
        "interface": "minecraft-authoritative-tick/v1",
        "sequence": sequence,
        "server_tick": 100 + sequence,
        "monotonic_ns": 1_000_000 + sequence,
        "actors": [
            {
                "actor_id": "actor:pseudonym:0011223344556677",
                "x": 1.0,
                "y": 64.0,
                "z": 2.0,
                "health": 20.0,
            }
        ],
        "events": [],
    }


def read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


@contextmanager
def fake_collector(*, gap: bool = False, corrupt_hello: bool = False) -> Iterator[int]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    finished = threading.Event()

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                while True:
                    try:
                        size = struct.unpack(">I", read_exact(connection, 4))[0]
                        wire_request = json.loads(read_exact(connection, size))
                    except EOFError:
                        break
                    request = (
                        json.loads(wire_request["payload_json"])
                        if "payload_json" in wire_request
                        else wire_request
                    )
                    if request["type"] == "hello":
                        response = _signed(
                            {
                                "ok": True,
                                "protocol_version": 2,
                                "session_id": "collector-session:test",
                                "minecraft_version": "26.3",
                                "capabilities": [
                                    "authoritative_tick",
                                    "pseudonymous_actors",
                                    "resume_after_sequence",
                                    "gap_detection",
                                    "response_checksums",
                                ],
                            }
                        )
                        if corrupt_hello:
                            response["payload_json"] = response["payload_json"].replace(
                                "26.3", "tampered"
                            )
                    elif request["type"] == "snapshot":
                        response = _signed(
                            {
                                "ok": True,
                                "protocol_version": 2,
                                "session_id": "collector-session:test",
                                "earliest_sequence": 10 if gap else 0,
                                "latest_sequence": 1,
                                "gap": gap,
                                "records": [] if gap else [tick(0), tick(1)],
                            }
                        )
                    else:
                        response = _signed(
                            {
                                "ok": True,
                                "protocol_version": 2,
                                "session_id": "collector-session:test",
                                "acked_sequence": request["sequence"],
                            }
                        )
                    encoded = json.dumps(response, separators=(",", ":")).encode()
                    connection.sendall(struct.pack(">I", len(encoded)) + encoded)
        finally:
            finished.set()
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        listener.close()
        finished.wait(timeout=2)


def test_authoritative_collector_client_validates_and_acks_contiguous_ticks() -> None:
    with fake_collector() as port:
        client = AuthoritativeCollectorClient(port=port)
        hello = client.connect()
        assert hello["minecraft_version"] == "26.3"
        batch = client.snapshot()
        assert [record.sequence for record in batch.records] == [0, 1]
        assert batch.records[0].actors[0]["actor_id"].startswith("actor:pseudonym:")
        assert client.ack() == 1
        client.close()


def test_authoritative_collector_client_rejects_gap() -> None:
    with fake_collector(gap=True) as port:
        client = AuthoritativeCollectorClient(port=port)
        with pytest.raises(CollectorGapError, match="lost records"):
            client.snapshot()
        client.close()


def test_authoritative_collector_client_rejects_tampered_response() -> None:
    with fake_collector(corrupt_hello=True) as port:
        client = AuthoritativeCollectorClient(port=port)
        with pytest.raises(CollectorProtocolError, match="checksum mismatch"):
            client.connect()
        client.close()


def test_authoritative_tick_rejects_names_and_raw_uuid_identity() -> None:
    value = tick(0)
    value["actors"][0]["player_name"] = "PrivateName"
    with pytest.raises(ValidationError, match="privacy-protected"):
        AuthoritativeTickV1.model_validate(value)

    value = tick(0)
    value["actors"][0]["actor_id"] = "raw-player-uuid"
    with pytest.raises(ValidationError, match="pseudonymous"):
        AuthoritativeTickV1.model_validate(value)


def test_authoritative_ticks_become_weak_semantic_steps_only() -> None:
    first = tick(0)
    second = tick(1)
    second["actors"][0].update(x=2.5, yaw=10.0)
    records = (
        AuthoritativeTickV1.model_validate(first),
        AuthoritativeTickV1.model_validate(second),
    )
    steps = semantic_steps_from_ticks(
        records,
        episode_id="episode:semantic",
        actor_id="actor:pseudonym:0011223344556677",
    )

    assert len(steps) == 1
    step = steps[0]
    assert step.data_tier == "semantic_weak"
    assert step.allowed_consumers == ("semantic_imitation",)
    assert step.action.values["displacement"] == [1.5, 0.0, 0.0]
    assert "attack" in step.action.unknown_fields
    assert "absolute_position" not in step.observation.structured
    assert step.observation.privileged["absolute_position"] == [1.0, 64.0, 2.0]
