from __future__ import annotations

import os
import importlib.util
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from minecraft_gym.dataset.contracts import (
    ActionLabelV2,
    ActorDescriptor,
    ConsentRecord,
    DepthMapV1,
    EpisodeManifestV2,
    ObservationRefV2,
    StepRecordV2,
    ViewpointFrameRefV1,
    VisualStreamDescriptorV1,
)
from minecraft_gym.dataset.visual_stream import (
    CapturedVisualFrame,
    assemble_multiview_observation,
)
from minecraft_gym.dataset.recorder import DatasetRecorderV2
from minecraft_gym.dataset.storage import (
    EpisodeWriter,
    ResearchStorage,
    iter_steps,
    verify_episode_artifacts,
)
from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.actions import noop_action


def observation(tick: int, *, visual: bool = False) -> ObservationRefV2:
    return ObservationRefV2(
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        frame_id=f"frame:{tick}" if visual else None,
        rgb_ref=f"video/frames.mp4#{tick}" if visual else None,
        structured={"health": 20.0},
        policy_channels=("rgb",) if visual else ("vitals",),
        privileged={"absolute_position": [1.0, 64.0, 2.0]},
    )


def action(*, source: str = "gym_agent") -> ActionLabelV2:
    return ActionLabelV2(
        values={"forward": True, "camera_dx": 0.0},
        source=source,
        known_fields=("forward", "camera_dx"),
        confidence=1.0,
        policy_projection=(1, 0),
    )


def test_action_label_preserves_hybrid_policy_action_beside_executed_controls() -> None:
    label = ActionLabelV2(
        values={"move_z": 1.0, "yaw_delta": 37.25, "pitch_delta": -11.5},
        source="gym_agent",
        known_fields=("move_z", "yaw_delta", "pitch_delta"),
        confidence=1.0,
        policy_action_contract="minecraft-control-hybrid-camera-body/v1",
        policy_action={
            "delta_longitude": 37.25,
            "delta_latitude": -11.5,
            "body_mode": 2,
        },
    )
    assert label.policy_action["delta_longitude"] == 37.25
    assert label.values["yaw_delta"] == 37.25


def test_hybrid_policy_action_requires_its_three_public_fields() -> None:
    with pytest.raises(ValidationError):
        ActionLabelV2(
            values={"move_z": 1.0},
            source="gym_agent",
            known_fields=("move_z",),
            confidence=1.0,
            policy_action_contract="minecraft-control-hybrid-camera-body/v1",
            policy_action={"delta_longitude": 0.0},
        )


def test_camera_keyboard_policy_action_preserves_independent_controls() -> None:
    policy_action = {
        "delta_longitude": 42.0,
        "delta_latitude": -8.0,
        "strafe": -1,
        "move": 1,
        "jump": True,
        "sprint": True,
        "sneak": False,
        "attack": False,
        "use": True,
    }
    label = ActionLabelV2(
        values={"move_x": -1.0, "move_z": 1.0, "yaw_delta": 42.0},
        source="gym_agent",
        known_fields=("move_x", "move_z", "yaw_delta"),
        confidence=1.0,
        policy_action_contract="minecraft-control-camera-keyboard/v3",
        policy_action=policy_action,
    )
    assert label.policy_action == policy_action


def viewpoint(
    tick: int,
    *,
    stream_id: str,
    actor_id: str,
    source: str,
    domain: str,
    alignment_ticks: int = 0,
) -> ViewpointFrameRefV1:
    digest = hashlib.sha256(f"{stream_id}:{tick}".encode()).hexdigest()
    return ViewpointFrameRefV1(
        stream_id=stream_id,
        actor_id=actor_id,
        viewpoint="egocentric",
        source=source,
        visual_domain=domain,
        frame_id=f"frame:{stream_id}:{tick}",
        rgb_ref=f"video/{stream_id}/frame-{tick}.jpg",
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        capture_sequence=tick,
        width=640,
        height=360,
        sha256=digest,
        alignment_ticks=alignment_ticks,
        alignment_method=(
            "exact_client_tick" if source == "fabric_native" else "nearest_monotonic"
        ),
        policy_eligible=True,
    )


def multiview_observation(tick: int, *, primary: str) -> ObservationRefV2:
    player = viewpoint(
        tick,
        stream_id="view:expert:egocentric",
        actor_id="player:expert01",
        source="fabric_native",
        domain="native_java",
    )
    bot = viewpoint(
        tick,
        stream_id="view:bot:egocentric",
        actor_id="actor:bot",
        source="prismarine_renderer",
        domain="rendered_prismarine",
    )
    selected = player if primary == player.stream_id else bot
    return ObservationRefV2(
        server_tick=tick,
        monotonic_ns=tick * 50_000_000,
        frame_id=selected.frame_id,
        rgb_ref=selected.rgb_ref,
        primary_stream_id=primary,
        viewpoints=(player, bot),
        structured={"health": 20.0},
        policy_channels=("rgb", "health"),
    )


def test_multiview_assembler_keeps_both_frames_and_selects_action_actor() -> None:
    demonstrator = viewpoint(
        1,
        stream_id="view:expert:egocentric",
        actor_id="actor:pseudonym:expert",
        source="fabric_native",
        domain="native_java",
    )
    bot = viewpoint(
        1,
        stream_id="view:bot:egocentric",
        actor_id="actor:bot",
        source="prismarine_renderer",
        domain="rendered_prismarine",
    )

    human_observation = assemble_multiview_observation(
        server_tick=1,
        monotonic_ns=50_000_000,
        viewpoints=(demonstrator, bot),
        action_source="java_native_input",
        actor_id="actor:pseudonym:expert",
        structured={"health": 20},
    )
    bot_observation = assemble_multiview_observation(
        server_tick=1,
        monotonic_ns=50_000_000,
        viewpoints=(demonstrator, bot),
        action_source="gym_agent",
        actor_id="actor:bot",
        structured={"health": 20},
    )

    assert len(human_observation.viewpoints) == 2
    assert human_observation.primary_stream_id == demonstrator.stream_id
    assert bot_observation.primary_stream_id == bot.stream_id


def test_multiview_step_preserves_demonstrator_and_bot_visual_domains() -> None:
    record = StepRecordV2(
        episode_id="episode:multi",
        step_id=0,
        actor_id="player:expert01",
        data_tier="exact_visual",
        observation=multiview_observation(20, primary="view:expert:egocentric"),
        action=action(source="java_native_input"),
        next_observation=multiview_observation(21, primary="view:expert:egocentric"),
        allowed_consumers=("behavior_cloning", "offline_rl", "dreamerv3"),
    )

    assert len(record.observation.viewpoints) == 2
    assert record.observation.primary_viewpoint().visual_domain == "native_java"
    assert {item.visual_domain for item in record.observation.viewpoints} == {
        "native_java",
        "rendered_prismarine",
    }


def test_rendered_bot_view_is_trainable_only_with_exact_gym_actions() -> None:
    record = StepRecordV2(
        episode_id="episode:bot-view",
        step_id=0,
        actor_id="actor:bot",
        data_tier="rendered_visual",
        observation=multiview_observation(30, primary="view:bot:egocentric"),
        action=action(source="gym_agent"),
        next_observation=multiview_observation(31, primary="view:bot:egocentric"),
        allowed_consumers=("behavior_cloning", "offline_rl", "dreamerv3"),
    )
    assert record.observation.primary_viewpoint().source == "prismarine_renderer"

    with pytest.raises(ValidationError, match="exact Gym action"):
        StepRecordV2.model_validate(
            {
                **record.model_dump(),
                "action": action(source="java_native_input").model_dump(),
            }
        )


def test_policy_viewpoint_alignment_is_gated_per_stream() -> None:
    before = multiview_observation(40, primary="view:bot:egocentric")
    stale_bot = before.viewpoints[1].model_copy(update={"alignment_ticks": 2})
    before = before.model_copy(update={"viewpoints": (before.viewpoints[0], stale_bot)})
    with pytest.raises(ValidationError, match="within one server tick"):
        StepRecordV2(
            episode_id="episode:stale",
            step_id=0,
            actor_id="actor:bot",
            data_tier="rendered_visual",
            observation=before,
            action=action(source="gym_agent"),
            next_observation=multiview_observation(41, primary="view:bot:egocentric"),
            allowed_consumers=("dreamerv3",),
        )


def test_episode_manifest_declares_both_visual_streams_and_consent() -> None:
    episode = EpisodeManifestV2(
        episode_id="episode:multi",
        session_id="session:multi",
        environment_id="env:minecraft",
        world_id="brincadeiras",
        environment_mode="live_observe",
        state="recording",
        data_tiers=("exact_visual", "rendered_visual"),
        actors=(
            ActorDescriptor(
                actor_id="player:expert01",
                role="expert_player",
                platform="java",
                identity_scope="dataset_pseudonym",
            ),
            ActorDescriptor(
                actor_id="actor:bot",
                role="bot",
                platform="java",
                identity_scope="public_bot",
            ),
        ),
        consent=(
            ConsentRecord(
                actor_id="player:expert01",
                granted=True,
                scope="visual_and_input",
            ),
        ),
        visual_streams=(
            VisualStreamDescriptorV1(
                stream_id="view:expert:egocentric",
                actor_id="player:expert01",
                viewpoint="egocentric",
                source="fabric_native",
                visual_domain="native_java",
                width=640,
                height=360,
                nominal_fps=20,
                policy_eligible=True,
                input_aligned=True,
            ),
            VisualStreamDescriptorV1(
                stream_id="view:bot:egocentric",
                actor_id="actor:bot",
                viewpoint="egocentric",
                source="prismarine_renderer",
                visual_domain="rendered_prismarine",
                width=640,
                height=360,
                nominal_fps=20,
                policy_eligible=True,
            ),
        ),
        server_version="26.3",
        adapter_version="test",
    )
    assert len(episode.visual_streams) == 2


def test_captured_frame_builds_auditable_nearest_tick_reference() -> None:
    data = b"jpeg-test"
    digest = hashlib.sha256(data).hexdigest()
    frame = CapturedVisualFrame(
        data=data,
        stream_id="view:bot:egocentric",
        actor_id="actor:bot",
        viewpoint="egocentric",
        source="prismarine_renderer",
        visual_domain="rendered_prismarine",
        frame_id="bot-frame:1",
        capture_sequence=1,
        captured_monotonic_ns=1_000_000_000,
        width=640,
        height=360,
        sha256=digest,
        nominal_fps=20,
        dropped_frame_count=0,
    )
    reference = frame.reference(
        rgb_ref="video/bot/frame-1.jpg",
        server_tick=50,
        now_monotonic_ns=1_049_000_000,
    )
    assert reference.alignment_ticks == 1
    assert reference.alignment_method == "nearest_monotonic"
    assert reference.policy_eligible is False

    verified = CapturedVisualFrame(
        **{
            field: getattr(frame, field)
            for field in frame.__dataclass_fields__
            if field != "self_entity_excluded"
        },
        self_entity_excluded=True,
    ).reference(
        rgb_ref="video/bot/frame-1.jpg",
        server_tick=50,
        now_monotonic_ns=1_049_000_000,
    )
    assert verified.policy_eligible is True


def manifest(*, tier: str = "exact_structured") -> EpisodeManifestV2:
    return EpisodeManifestV2(
        episode_id="episode:test",
        session_id="session:test",
        environment_id="env:test",
        world_id="brincadeiras",
        environment_mode="live_observe",
        state="recording",
        data_tiers=(tier,),
        actors=(
            ActorDescriptor(
                actor_id="actor:bot",
                role="bot",
                platform="java",
                identity_scope="public_bot",
            ),
        ),
        consent=(
            ConsentRecord(
                actor_id="actor:bot",
                granted=True,
                scope="visual_and_input" if tier == "exact_visual" else "structured",
            ),
        ),
        server_version="26.3",
        adapter_version="test",
    )


def step(step_id: int, start_tick: int, *, tier: str = "exact_structured") -> StepRecordV2:
    visual = tier == "exact_visual"
    return StepRecordV2(
        episode_id="episode:test",
        step_id=step_id,
        actor_id="actor:bot",
        data_tier=tier,
        observation=observation(start_tick, visual=visual),
        action=action(source="java_native_input" if visual else "gym_agent"),
        next_observation=observation(start_tick + 1, visual=visual),
        allowed_consumers=(
            ("behavior_cloning", "offline_rl", "dreamerv3")
            if visual
            else ("behavior_cloning", "offline_rl")
        ),
    )


def test_semantic_weak_cannot_silently_become_low_level_demonstration() -> None:
    with pytest.raises(ValidationError, match="semantic_weak"):
        StepRecordV2(
            episode_id="episode:weak",
            step_id=0,
            actor_id="actor:android",
            data_tier="semantic_weak",
            observation=observation(1),
            action=ActionLabelV2(
                values={"displacement": [0.1, 0.0, 0.0]},
                source="server_reconstructed",
                known_fields=("displacement",),
                unknown_fields=("touch", "camera"),
                confidence=0.5,
            ),
            next_observation=observation(2),
            allowed_consumers=("behavior_cloning",),
        )


def test_privacy_fields_are_rejected_from_events_and_privileged_state() -> None:
    with pytest.raises(ValidationError, match="privacy-protected"):
        ObservationRefV2(
            server_tick=1,
            monotonic_ns=1,
            structured={"health": 20},
            privileged={"players": ["real-name"]},
        )
    with pytest.raises(ValidationError, match="privacy-protected"):
        StepRecordV2(
            episode_id="episode:privacy",
            step_id=0,
            actor_id="actor:bot",
            data_tier="exact_structured",
            observation=observation(1),
            action=action(),
            next_observation=observation(2),
            events=({"chat": "secret"},),
        )


def test_depth_policy_channel_requires_same_tick_and_frame() -> None:
    depth = DepthMapV1(
        sensor_id="sensor:depth",
        actor_id="actor:bot",
        server_tick=10,
        monotonic_ns=500_000_000,
        camera_stream_id="view:bot:egocentric",
        camera_frame_id="frame:10",
        source_width=640,
        source_height=360,
        width=2,
        height=1,
        max_distance=64.0,
        depth_blocks=(3.0, 64.0),
        status_codes=(1, 2),
        confidence=(0.95, 0.95),
    )
    observation = ObservationRefV2(
        server_tick=10,
        monotonic_ns=500_000_000,
        frame_id="frame:10",
        rgb_ref="video/frame-10.jpg",
        policy_channels=("rgb", "depth"),
        depth_map=depth,
    )
    assert observation.depth_map is not None
    assert observation.depth_map.status_codes == (1, 2)

    with pytest.raises(ValidationError, match="depth map tick"):
        ObservationRefV2(
            server_tick=11,
            monotonic_ns=550_000_000,
            frame_id="frame:10",
            rgb_ref="video/frame-10.jpg",
            policy_channels=("rgb", "depth"),
            depth_map=depth,
        )
def test_gym_recorder_writes_exact_structured_episode_v2(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    base = MinecraftSurvivalEnv(
        backend="mock", width=8, height=6, max_episode_steps=2
    )
    env = DatasetRecorderV2(
        base,
        storage,
        environment_id="env:test",
        world_id="world:test",
        server_version="test",
        require_parquet=False,
    )
    env.reset(seed=7)
    env.step(noop_action())
    env.step(noop_action())

    assert len(env.completed) == 1
    finished, quality, destination = env.completed[0]
    assert finished.state == "accepted"
    assert quality.eligible_consumers == (
        "behavior_cloning",
        "evaluation",
        "offline_rl",
    )
    records = list(iter_steps(destination / "steps.jsonl"))
    assert [record.step_id for record in records] == [0, 1]
    assert all(record.data_tier == "exact_structured" for record in records)
    assert all(record.observation.rgb_ref is None for record in records)
    env.close()


def test_exact_visual_requires_native_input_and_rgb() -> None:
    with pytest.raises(ValidationError, match="RGB"):
        StepRecordV2(
            episode_id="episode:visual",
            step_id=0,
            actor_id="actor:java",
            data_tier="exact_visual",
            observation=observation(1),
            action=action(),
            next_observation=observation(2),
        )


def test_training_manifest_requires_world_snapshot() -> None:
    payload = manifest().model_dump()
    payload.update(environment_mode="isolated_training", world_snapshot_id=None)
    with pytest.raises(ValidationError, match="world snapshot"):
        EpisodeManifestV2.model_validate(payload)


def test_writer_commits_atomically_with_hashes_and_contiguous_ticks(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(storage, manifest(), fsync_every=1, require_parquet=False)
    writer.append_step(step(0, 10))
    writer.append_step(step(1, 11))
    finished, quality, destination = writer.finalize()

    assert destination == external / "raw" / "episode:test"
    assert quality.accepted is True
    assert quality.tick_gap_count == 0
    assert finished.step_count == 2
    assert finished.first_tick == 10
    assert finished.last_tick == 12
    assert all(len(item.sha256) == 64 for item in finished.artifacts)
    assert [item.step_id for item in iter_steps(destination / "steps.jsonl")] == [0, 1]
    assert not (destination / "manifest.partial.json").exists()


def test_writer_quarantines_visual_episode_that_misses_quality_gate(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(
        storage,
        manifest(tier="exact_visual"),
        require_parquet=False,
    )
    writer.append_step(step(0, 30, tier="exact_visual"))
    finished, quality, destination = writer.finalize(
        frame_count=1000,
        dropped_frame_count=1,
        max_alignment_ticks=2,
    )

    assert finished.state == "quarantined"
    assert quality.accepted is False
    assert set(quality.rejection_reasons) == {
        "frame_action_misalignment",
        "frame_drop_rate_exceeded",
    }
    assert destination.parent.name == "quarantine"


def test_absent_external_volume_blocks_visual_but_allows_bounded_structured_spool(
    tmp_path: Path,
) -> None:
    storage = ResearchStorage(
        tmp_path / "missing-volume",
        spool_root=tmp_path / "spool",
        spool_limit_bytes=1024,
        minimum_system_free_bytes=0,
        require_apfs=False,
    )
    status = storage.status()
    assert status.visual_capture_allowed is False
    assert status.structured_capture_allowed is True
    assert "research_volume_missing" in status.reasons
    assert not (tmp_path / "missing-volume").exists()


def test_plain_directory_cannot_impersonate_a_mounted_research_volume(
    tmp_path: Path,
) -> None:
    fake_volume = tmp_path / "MinecraftRL"
    fake_volume.mkdir()
    storage = ResearchStorage(
        fake_volume,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
    )
    status = storage.status()
    assert status.external_mounted is False
    assert status.visual_capture_allowed is False
    assert "research_volume_missing" in status.reasons


def test_writer_rejects_reversed_or_non_contiguous_steps(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(storage, manifest(), require_parquet=False)
    writer.append_step(step(0, 10))
    with pytest.raises(ValueError, match="contiguous"):
        writer.append_step(step(2, 11))
    with pytest.raises(ValueError, match="backwards"):
        writer.append_step(step(1, 9))
    writer.abort("test_complete")


def test_integrity_audit_detects_silent_artifact_corruption(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(storage, manifest(), require_parquet=False)
    writer.append_step(step(0, 1))
    finished, _, destination = writer.finalize()
    assert verify_episode_artifacts(destination, finished) == []
    with (destination / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("corruption")
    assert verify_episode_artifacts(destination, finished) == [
        "artifact_size_mismatch:steps.jsonl"
    ]


def test_crash_recovery_quarantines_stale_partial_without_silent_acceptance(
    tmp_path: Path,
) -> None:
    external = tmp_path / "research"
    partial = external / "raw" / ".partial" / "episode:crashed"
    partial.mkdir(parents=True)
    (partial / "steps.wal.jsonl").write_text("partial\n", encoding="utf-8")
    os.utime(partial, (1, 1))
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    recovered = storage.recover_partials(stale_after_seconds=0)
    assert len(recovered) == 1
    destination = Path(recovered[0])
    assert destination.parent.name == "quarantine"
    assert (destination / "recovery.json").is_file()
    assert not partial.exists()


def test_privacy_protected_fields_never_enter_trainable_observations() -> None:
    with pytest.raises(ValidationError, match="privacy-protected"):
        ObservationRefV2(
            server_tick=1,
            monotonic_ns=1,
            structured={"health": 20, "chat": "secret player message"},
        )


def test_event_wal_redacts_chat_and_player_names(tmp_path: Path) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(storage, manifest(), require_parquet=False)
    writer.append_step(step(0, 1))
    writer.append_event(
        {"type": "chat", "chat": "hello", "player_name": "RealPlayer"}
    )
    _, _, destination = writer.finalize()
    event_text = (destination / "events.jsonl").read_text(encoding="utf-8")
    assert "hello" not in event_text
    assert "RealPlayer" not in event_text
    assert event_text.count("[REDACTED]") == 2


def test_missing_parquet_materialization_quarantines_instead_of_silent_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    monkeypatch.setattr(
        EpisodeWriter,
        "_materialize_parquet",
        staticmethod(lambda source, destination: "test_conversion_failure"),
    )
    writer = EpisodeWriter(storage, manifest())
    writer.append_step(step(0, 1))
    finished, quality, destination = writer.finalize()

    assert finished.state == "quarantined"
    assert quality.rejection_reasons == ("parquet_materialization_failed",)
    assert destination.parent.name == "quarantine"


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow is provided by the research extra",
)
def test_parquet_is_canonical_steps_artifact_when_research_extra_is_installed(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as parquet

    external = tmp_path / "research"
    external.mkdir()
    storage = ResearchStorage(
        external,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0,
        require_apfs=False,
    )
    writer = EpisodeWriter(storage, manifest())
    writer.append_step(step(0, 1))
    finished, quality, destination = writer.finalize()

    assert quality.accepted is True
    steps_artifact = next(item for item in finished.artifacts if item.role == "steps")
    assert steps_artifact.path == "steps.parquet"
    table = parquet.read_table(destination / steps_artifact.path)
    assert table.column("episode_id").to_pylist() == ["episode:test"]
    assert table.column("step_id").to_pylist() == [0]
