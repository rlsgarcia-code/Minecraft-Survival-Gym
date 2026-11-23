"""Gymnasium-to-dataset v2 recorder.

This wrapper is deliberately limited to exact structured actions. Visual
demonstrations require the dedicated Java frame/input collector so an RGB
array returned by Gym can never be mislabeled as synchronized native input.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from minecraft_gym.actions import CONTROL_FIELDS, ControlState, decode_action, encode_human_control

from .contracts import (
    ActionLabelV2,
    ActorDescriptor,
    ConsentRecord,
    EpisodeManifestV2,
    ObservationRefV2,
    QualityReportV1,
    SpatialMapUpdateV1,
    StepRecordV2,
    VisualStreamDescriptorV1,
    VisualStreamQualityV1,
)
from .depth import depth_map_from_raycast
from .storage import EpisodeWriter, ResearchStorage
from .spatial import HttpSpatialStreamClient
from .visual_stream import (
    CapturedDenseRgbdFrame,
    CapturedVisualFrame,
    HttpDenseRgbdStreamClient,
    HttpVisualStreamClient,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


class DatasetRecorderV2(gym.Wrapper):
    """Record exact Gym actions using the crash-tolerant episode format.

    Use this wrapper for bot rollouts and structured BC/offline-RL data. It
    intentionally rejects ``exact_visual`` because only the Java collector can
    prove frame/native-input alignment.
    """

    def __init__(
        self,
        env: gym.Env,
        storage: ResearchStorage,
        *,
        environment_id: str,
        world_id: str,
        actor_id: str = "actor:bot",
        server_version: str = "unknown",
        adapter_version: str = "minecraft-gym/0.4",
        environment_mode: str = "live_observe",
        world_snapshot_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        require_parquet: bool = True,
        visual_stream: HttpVisualStreamClient | None = None,
        dense_rgbd_stream: HttpDenseRgbdStreamClient | None = None,
        spatial_stream: HttpSpatialStreamClient | None = None,
    ) -> None:
        super().__init__(env)
        if environment_mode != "live_observe" and not world_snapshot_id:
            raise ValueError("isolated training/evaluation requires world_snapshot_id")
        if visual_stream is not None and dense_rgbd_stream is not None:
            raise ValueError("choose either RGB or dense RGB-D stream, not both")
        if spatial_stream is not None and visual_stream is None and dense_rgbd_stream is None:
            raise ValueError("spatial raycasts require their paired visual stream")
        self.storage = storage
        self.environment_id = environment_id
        self.world_id = world_id
        self.actor_id = actor_id
        self.server_version = server_version
        self.adapter_version = adapter_version
        self.environment_mode = environment_mode
        self.world_snapshot_id = world_snapshot_id
        self.task_id = task_id
        self.session_id = session_id or f"session:{uuid.uuid4()}"
        self.require_parquet = require_parquet
        self.visual_stream = visual_stream
        self.dense_rgbd_stream = dense_rgbd_stream
        self.spatial_stream = spatial_stream
        self.writer: EpisodeWriter | None = None
        self._previous: ObservationRefV2 | None = None
        self._step_id = 0
        self.completed: list[tuple[EpisodeManifestV2, QualityReportV1, Path]] = []
        self._visual_sequences: set[int] = set()
        self._visual_dropped = 0
        self._visual_drop_baseline = 0
        self._visual_max_alignment = 0
        self._visual_descriptor: VisualStreamDescriptorV1 | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._finalize_open_episode()
        observation, info = self.env.reset(**kwargs)
        dense_frame = self.dense_rgbd_stream.latest() if self.dense_rgbd_stream else None
        visual_frame = (
            dense_frame.visual
            if dense_frame is not None
            else self.visual_stream.latest() if self.visual_stream else None
        )
        episode_id = f"episode:{uuid.uuid4()}"
        if visual_frame is not None:
            self._visual_descriptor = VisualStreamDescriptorV1(
                stream_id=visual_frame.stream_id,
                actor_id=visual_frame.actor_id,
                viewpoint=visual_frame.viewpoint,
                source=visual_frame.source,
                visual_domain=visual_frame.visual_domain,
                width=visual_frame.width,
                height=visual_frame.height,
                nominal_fps=visual_frame.nominal_fps,
                policy_eligible=True,
                input_aligned=False,
            )
        manifest = EpisodeManifestV2(
            episode_id=episode_id,
            session_id=self.session_id,
            environment_id=self.environment_id,
            world_id=self.world_id,
            world_snapshot_id=self.world_snapshot_id,
            environment_mode=self.environment_mode,
            task_id=self.task_id,
            state="recording",
            data_tiers=("rendered_visual",) if visual_frame else ("exact_structured",),
            actors=(
                ActorDescriptor(
                    actor_id=self.actor_id,
                    role="bot",
                    platform="java",
                    identity_scope="public_bot",
                ),
            ),
            consent=(
                ConsentRecord(
                    actor_id=self.actor_id,
                    granted=True,
                    scope="visual_and_input" if visual_frame else "structured",
                ),
            ),
            visual_streams=(self._visual_descriptor,) if self._visual_descriptor else (),
            server_version=self.server_version,
            adapter_version=self.adapter_version,
            provenance={"recorder": "minecraft-gym/dataset-recorder-v2"},
        )
        self.writer = EpisodeWriter(
            self.storage,
            manifest,
            require_parquet=self.require_parquet,
        )
        if self.spatial_stream is not None:
            self.spatial_stream.start_episode(episode_id)
        self._visual_sequences.clear()
        self._visual_dropped = 0
        self._visual_drop_baseline = (
            visual_frame.dropped_frame_count if visual_frame is not None else 0
        )
        self._visual_max_alignment = 0
        self._previous = self._observation_ref(
            observation, info, visual_frame, dense_frame=dense_frame
        )
        self._step_id = 0
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.writer is None or self._previous is None:
            raise RuntimeError("reset() must be called before recording step()")
        observation, reward, terminated, truncated, info = self.env.step(action)
        dense_frame = self.dense_rgbd_stream.latest() if self.dense_rgbd_stream else None
        visual_frame = (
            dense_frame.visual
            if dense_frame is not None
            else self.visual_stream.latest() if self.visual_stream else None
        )
        following = self._observation_ref(
            observation, info, visual_frame, dense_frame=dense_frame
        )
        control, source = self._executed_control(action, info)
        projection = encode_human_control(control)
        events = tuple(_json_value(event) for event in info.get("events", ()))
        record = StepRecordV2(
            episode_id=self.writer.manifest.episode_id,
            step_id=self._step_id,
            actor_id=self.actor_id,
            data_tier=(
                "rendered_visual"
                if self.visual_stream or self.dense_rgbd_stream
                else "exact_structured"
            ),
            observation=self._previous,
            action=ActionLabelV2(
                values=control.to_dict(),
                source=source,
                known_fields=CONTROL_FIELDS,
                confidence=1.0,
                policy_projection=tuple(int(value) for value in projection),
            ),
            next_observation=following,
            reward=float(reward),
            reward_terms={
                str(key): float(value)
                for key, value in info.get("reward_terms", {}).items()
            },
            events=events,
            terminated=bool(terminated),
            truncated=bool(truncated),
            allowed_consumers=(
                ("behavior_cloning", "offline_rl", "dreamerv3")
                if self.visual_stream or self.dense_rgbd_stream
                else ("behavior_cloning", "offline_rl")
            ),
            provenance={"control_mode": str(info.get("control_mode", "unknown"))},
        )
        self.writer.append_step(record)
        for event in events:
            self.writer.append_event(event)
        self._previous = following
        self._step_id += 1
        if terminated or truncated:
            self._finalize_open_episode()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._finalize_open_episode()
        self.env.close()

    def _finalize_open_episode(self) -> None:
        if self.writer is None:
            return
        if self._step_id == 0:
            self.writer.abort("episode_reset_without_steps")
        else:
            if self._visual_descriptor is None:
                self.completed.append(self.writer.finalize())
            else:
                frame_count = len(self._visual_sequences)
                reasons: list[str] = []
                if frame_count == 0:
                    reasons.append("no_visual_frames")
                if self._visual_max_alignment > 1:
                    reasons.append("frame_action_misalignment")
                if frame_count and self._visual_dropped / frame_count >= 0.001:
                    reasons.append("frame_drop_rate_exceeded")
                stream_quality = VisualStreamQualityV1(
                    stream_id=self._visual_descriptor.stream_id,
                    actor_id=self._visual_descriptor.actor_id,
                    frame_count=frame_count,
                    dropped_frame_count=self._visual_dropped,
                    max_alignment_ticks=self._visual_max_alignment,
                    checksum_failures=0,
                    policy_eligible=True,
                    accepted=not reasons,
                    rejection_reasons=tuple(reasons),
                )
                self.completed.append(
                    self.writer.finalize(
                        frame_count=frame_count,
                        dropped_frame_count=self._visual_dropped,
                        max_alignment_ticks=self._visual_max_alignment,
                        visual_stream_quality=(stream_quality,),
                    )
                )
        self.writer = None
        self._previous = None
        self._step_id = 0
        self._visual_descriptor = None
        self._visual_sequences.clear()
        self._visual_dropped = 0
        self._visual_drop_baseline = 0
        self._visual_max_alignment = 0

    def _observation_ref(
        self,
        observation: Any,
        info: dict[str, Any],
        visual_frame: CapturedVisualFrame | None,
        *,
        dense_frame: CapturedDenseRgbdFrame | None = None,
    ) -> ObservationRefV2:
        if not isinstance(observation, dict):
            raise TypeError("DatasetRecorderV2 requires dictionary observations")
        tick = info.get("tick")
        if tick is None:
            raise ValueError("backend did not provide an authoritative tick")
        structured = {
            str(key): _json_value(value)
            for key, value in observation.items()
            if key != "rgb"
        }
        privileged = _json_value(info.get("privileged_context", {}))
        spatial_updates = tuple(
            item
            if isinstance(item, SpatialMapUpdateV1)
            else SpatialMapUpdateV1.model_validate(item)
            for item in info.get("spatial_updates", ())
        )
        depth_map = None
        dense_depth = None
        if visual_frame is None:
            return ObservationRefV2(
                server_tick=int(tick),
                monotonic_ns=time.monotonic_ns(),
                structured=structured,
                policy_channels=tuple(sorted(structured)),
                privileged=privileged,
                spatial_updates=spatial_updates,
            )
        if self.writer is None:
            raise RuntimeError("visual observation requires an open episode writer")
        self._visual_dropped = max(
            self._visual_dropped,
            max(0, visual_frame.dropped_frame_count - self._visual_drop_baseline),
        )
        rgb_ref = self.writer.append_visual_frame(
            stream_id=visual_frame.stream_id,
            capture_sequence=visual_frame.capture_sequence,
            jpeg=visual_frame.data,
            expected_sha256=visual_frame.sha256,
        )
        frame_ref = visual_frame.reference(rgb_ref=rgb_ref, server_tick=int(tick))
        if dense_frame is not None:
            depth_ref = self.writer.append_dense_depth_frame(
                stream_id=visual_frame.stream_id,
                capture_sequence=visual_frame.capture_sequence,
                compressed=dense_frame.depth_compressed,
                expected_sha256=dense_frame.depth_sha256,
            )
            dense_depth = dense_frame.depth_reference(
                depth_ref=depth_ref,
                server_tick=int(tick),
            )
        if self.spatial_stream is not None:
            position = privileged.get("absolute_position")
            visited_position = None
            if isinstance(position, (list, tuple)) and len(position) == 3:
                visited_position = tuple(float(value) for value in position)
            spatial_update = self.spatial_stream.latest(
                server_tick=int(tick),
                camera_frame_id=frame_ref.frame_id,
                camera_stream_id=frame_ref.stream_id,
                visited_position=visited_position,
            )
            spatial_updates = (*spatial_updates, spatial_update)
            if dense_frame is None:
                depth_map = depth_map_from_raycast(spatial_update.raycast)
        self._visual_sequences.add(visual_frame.capture_sequence)
        self._visual_max_alignment = max(
            self._visual_max_alignment, frame_ref.alignment_ticks
        )
        return ObservationRefV2(
            server_tick=int(tick),
            monotonic_ns=time.monotonic_ns(),
            frame_id=frame_ref.frame_id,
            rgb_ref=rgb_ref,
            primary_stream_id=frame_ref.stream_id,
            viewpoints=(frame_ref,),
            structured=structured,
            policy_channels=(
                "rgb",
                *(("depth",) if depth_map is not None or dense_depth is not None else ()),
                *tuple(sorted(structured)),
            ),
            privileged=privileged,
            spatial_updates=spatial_updates,
            depth_map=depth_map,
            dense_depth=dense_depth,
        )

    @staticmethod
    def _executed_control(action: Any, info: dict[str, Any]) -> tuple[ControlState, str]:
        value = info.get("executed_action")
        if isinstance(value, dict):
            control = ControlState.from_mapping(value)
        else:
            control = decode_action(np.asarray(action, dtype=np.int64))
        source = (
            "java_native_input"
            if info.get("control_source") == "human"
            else "gym_agent"
        )
        return control, source
