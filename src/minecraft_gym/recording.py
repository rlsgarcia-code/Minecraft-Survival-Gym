"""Episode recording for keyboard/mouse demonstrations and DAgger."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from minecraft_gym.actions import (
    ControlState,
    control_to_array,
    decode_action,
    encode_human_control,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class TrajectoryRecorder(gym.Wrapper):
    """Record complete transitions without changing environment semantics.

    The Fabric bridge may add ``human_action``, ``policy_action`` and
    ``executed_action`` to ``info``. They are kept verbatim to support behavior
    cloning and DAgger. Each episode is committed atomically as one compressed
    NumPy archive plus human-readable metadata.
    """

    def __init__(
        self,
        env: gym.Env,
        directory: str | os.PathLike[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_metadata = dict(metadata or {})
        self._episode_id: str | None = None
        self._episode_metadata: dict[str, Any] = {}
        self._initial_observation: dict[str, Any] | None = None
        self._transitions: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._finalize_episode()
        observation, info = self.env.reset(**kwargs)
        self._episode_id = f"episode-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        self._episode_metadata = {
            **self.session_metadata,
            "episode_id": self._episode_id,
            "started_at_unix": time.time(),
            "reset_info": info,
        }
        self._initial_observation = self._copy_observation(observation)
        self._transitions = []
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._episode_id is None or self._initial_observation is None:
            raise RuntimeError("reset() must be called before recording step()")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._transitions.append(
            {
                "action": np.asarray(action, dtype=np.int64).copy(),
                "observation": self._copy_observation(observation),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )
        if terminated or truncated:
            self._finalize_episode()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._finalize_episode()
        self.env.close()

    def _finalize_episode(self) -> None:
        if self._episode_id is None or self._initial_observation is None:
            return
        episode_id = self._episode_id
        observations = [self._initial_observation]
        observations.extend(t["observation"] for t in self._transitions)
        arrays: dict[str, np.ndarray] = {}
        for key in self._initial_observation:
            arrays[f"obs_{key}"] = np.stack(
                [np.asarray(obs[key]) for obs in observations], axis=0
            )
        action_width = int(np.prod(self.action_space.shape or (0,)))
        arrays["actions"] = (
            np.stack([t["action"] for t in self._transitions], axis=0)
            if self._transitions
            else np.empty((0, action_width), dtype=np.int64)
        )
        self._add_control_arrays(arrays, action_width)
        arrays["rewards"] = np.asarray(
            [t["reward"] for t in self._transitions], dtype=np.float32
        )
        arrays["terminated"] = np.asarray(
            [t["terminated"] for t in self._transitions], dtype=np.bool_
        )
        arrays["truncated"] = np.asarray(
            [t["truncated"] for t in self._transitions], dtype=np.bool_
        )
        arrays["infos_json"] = np.asarray(
            [
                json.dumps(t["info"], default=_json_default, separators=(",", ":"))
                for t in self._transitions
            ],
            dtype=np.str_,
        )

        tmp_path = self.directory / f".{episode_id}.tmp.npz"
        final_path = self.directory / f"{episode_id}.npz"
        with tmp_path.open("wb") as file_handle:
            np.savez_compressed(file_handle, **arrays)
        os.replace(tmp_path, final_path)

        metadata = {
            **self._episode_metadata,
            "finished_at_unix": time.time(),
            "transition_count": len(self._transitions),
            "archive": final_path.name,
        }
        metadata_tmp = self.directory / f".{episode_id}.metadata.tmp"
        metadata_path = self.directory / f"{episode_id}.json"
        metadata_tmp.write_text(
            json.dumps(metadata, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(metadata_tmp, metadata_path)

        self._episode_id = None
        self._initial_observation = None
        self._transitions = []
        self._episode_metadata = {}

    def _add_control_arrays(
        self, arrays: dict[str, np.ndarray], action_width: int
    ) -> None:
        """Expose raw and policy-projected device actions for IL/DAgger."""

        control_width = len(ControlState.__dataclass_fields__)
        raw_controls: dict[str, list[np.ndarray]] = {
            "policy": [],
            "human": [],
            "executed": [],
        }
        projected_actions: dict[str, list[np.ndarray]] = {
            "policy": [],
            "human": [],
            "executed": [],
        }
        human_present: list[bool] = []

        for transition in self._transitions:
            info = transition["info"]
            submitted = transition["action"]
            policy = self._control_from_info(info.get("policy_action"))
            if policy is None:
                policy = decode_action(submitted)
            human = self._control_from_info(info.get("human_action"))
            executed = self._control_from_info(info.get("executed_action")) or policy
            human_present.append(human is not None)

            for name, control in (
                ("policy", policy),
                ("human", human),
                ("executed", executed),
            ):
                if control is None:
                    raw_controls[name].append(
                        np.full(control_width, np.nan, dtype=np.float32)
                    )
                    projected_actions[name].append(
                        np.full(action_width, -1, dtype=np.int64)
                    )
                else:
                    raw_controls[name].append(control_to_array(control))
                    projected_actions[name].append(encode_human_control(control))

        for name in raw_controls:
            arrays[f"{name}_controls"] = (
                np.stack(raw_controls[name])
                if raw_controls[name]
                else np.empty((0, control_width), dtype=np.float32)
            )
            arrays[f"{name}_actions"] = (
                np.stack(projected_actions[name])
                if projected_actions[name]
                else np.empty((0, action_width), dtype=np.int64)
            )
        arrays["human_action_present"] = np.asarray(human_present, dtype=np.bool_)

    @staticmethod
    def _control_from_info(value: Any) -> ControlState | None:
        if not isinstance(value, dict):
            return None
        return ControlState.from_mapping(value)

    @staticmethod
    def _copy_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            raise TypeError("TrajectoryRecorder requires dictionary observations")
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }
