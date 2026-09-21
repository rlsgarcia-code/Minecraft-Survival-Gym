from __future__ import annotations

import json

import numpy as np

from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.actions import noop_action
from minecraft_gym.recording import TrajectoryRecorder


def test_recorder_commits_atomic_episode(tmp_path) -> None:
    base = MinecraftSurvivalEnv(
        backend="mock", width=8, height=6, max_episode_steps=2
    )
    env = TrajectoryRecorder(base, tmp_path, metadata={"operator": "test"})
    env.reset(seed=9)
    env.step(noop_action())
    env.step(noop_action())

    archives = list(tmp_path.glob("episode-*.npz"))
    metadata_files = list(tmp_path.glob("episode-*.json"))
    assert len(archives) == 1
    assert len(metadata_files) == 1
    with np.load(archives[0]) as trajectory:
        assert trajectory["actions"].shape == (2, 15)
        assert trajectory["policy_actions"].shape == (2, 15)
        assert trajectory["executed_actions"].shape == (2, 15)
        assert trajectory["human_actions"].shape == (2, 15)
        assert trajectory["executed_controls"].shape == (2, 16)
        assert not trajectory["human_action_present"].any()
        assert (trajectory["human_actions"] == -1).all()
        assert trajectory["obs_rgb"].shape == (3, 6, 8, 3)
        assert trajectory["terminated"].shape == (2,)
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["transition_count"] == 2
    assert metadata["operator"] == "test"
    env.close()
