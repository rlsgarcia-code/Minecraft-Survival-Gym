from __future__ import annotations

import numpy as np
import pytest

from minecraft_gym.academy import (
    NavigationAcademyEnv,
    academy_visual_pair,
    encode_visual_policy_input,
)
from minecraft_gym.dataset import DepthMapV1


def _depth() -> DepthMapV1:
    return DepthMapV1(
        sensor_id="sensor:depth",
        actor_id="actor:bot",
        server_tick=10,
        monotonic_ns=500_000_000,
        camera_stream_id="view:bot:egocentric",
        camera_frame_id="frame:10",
        source_width=4,
        source_height=2,
        width=2,
        height=1,
        max_distance=10.0,
        depth_blocks=(2.0, 10.0),
        status_codes=(1, 0),
        confidence=(0.95, 0.0),
    )


def test_rgb_and_rgbd_keep_identical_shape_and_model_capacity() -> None:
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    control = encode_visual_policy_input(rgb, _depth(), modality="rgb")
    intervention = encode_visual_policy_input(rgb, _depth(), modality="rgbd")

    assert control.shape == intervention.shape == (5, 2, 4)
    np.testing.assert_array_equal(control[:3], intervention[:3])
    assert not control[3:].any()
    np.testing.assert_allclose(intervention[3, :, :2], 0.2)
    np.testing.assert_allclose(intervention[4, :, :2], 1.0)
    assert not intervention[3, :, 2:].any()
    assert not intervention[4, :, 2:].any()


def test_academy_visual_proxy_keeps_rgb_identical_between_arms() -> None:
    env = NavigationAcademyEnv(difficulty="easy")
    observation, info = env.reset(seed=17, options={"requirement": "travel"})
    rgb, depth = academy_visual_pair(
        observation,
        frame_id="frame:17",
        server_tick=info["tick"],
    )

    control = encode_visual_policy_input(rgb, depth, modality="rgb")
    intervention = encode_visual_policy_input(rgb, depth, modality="rgbd")

    assert rgb.shape == (18, 32, 3)
    assert depth.width == 32
    assert depth.height == 18
    assert control.shape == intervention.shape == (5, 18, 32)
    np.testing.assert_array_equal(control[:3], intervention[:3])
    assert not control[3:].any()
    assert intervention[4].any()
    assert not depth.policy_eligible


def test_visual_actor_critic_arms_have_identical_parameter_count() -> None:
    pytest.importorskip("torch")
    from minecraft_gym.academy.visual_actor_critic import VisualHybridActorCriticAgent

    env = NavigationAcademyEnv(difficulty="easy")
    observation, _ = env.reset(seed=23, options={"requirement": "travel"})
    rgb = VisualHybridActorCriticAgent(modality="rgb", seed=5)
    rgbd = VisualHybridActorCriticAgent(modality="rgbd", seed=5)

    rgb.choose(observation, stochastic=False)
    rgbd.choose(observation, stochastic=False)

    assert rgb.actor is not None and rgb.critic is not None
    assert rgbd.actor is not None and rgbd.critic is not None
    rgb_parameters = sum(item.numel() for item in rgb.actor.parameters()) + sum(
        item.numel() for item in rgb.critic.parameters()
    )
    rgbd_parameters = sum(item.numel() for item in rgbd.actor.parameters()) + sum(
        item.numel() for item in rgbd.critic.parameters()
    )
    assert rgb_parameters == rgbd_parameters
    state, _, _ = rgb._state_tensors(observation)
    assert state.joint_lattice.shape == (5, 18, 32)
    assert rgb.architecture == rgbd.architecture == "vgg-rgbd-shared-flatten-mlp/v1"
    assert rgb.actor is rgb.critic
    assert rgbd.actor is rgbd.critic


def test_vgg_visual_policy_checkpoint_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")
    from minecraft_gym.academy.visual_actor_critic import VisualHybridActorCriticAgent

    env = NavigationAcademyEnv(difficulty="easy")
    observation, _ = env.reset(seed=29, options={"requirement": "travel"})
    agent = VisualHybridActorCriticAgent(
        modality="rgbd", seed=9, input_width=64, input_height=36
    )
    expected = agent.choose(observation, stochastic=False)
    path = agent.save(tmp_path / "vgg-rgbd.pt", metadata={"test": True})

    loaded, metadata = VisualHybridActorCriticAgent.load(path)
    actual = loaded.choose(observation, stochastic=False)

    assert metadata == {"test": True}
    assert loaded.architecture == "vgg-rgbd-shared-flatten-mlp/v1"
    assert actual.dataset_value() == expected.dataset_value()


def _short_ppo_trajectory(agent, *, seed: int, difficulty: str):
    from minecraft_gym.academy import hybrid_control_action

    env = NavigationAcademyEnv(difficulty=difficulty, max_episode_steps=80)
    observation, _ = env.reset(seed=seed, options={"requirement": "travel"})
    trajectory = []
    for step in range(3):
        action = agent.choose(observation, stochastic=True)
        next_observation, reward, terminated, truncated, _ = env.step_control(
            hybrid_control_action(action)
        )
        if step == 2 and not terminated:
            truncated = True
        trajectory.append(
            (
                observation,
                action,
                float(reward),
                next_observation,
                bool(terminated),
                bool(truncated),
            )
        )
        observation = next_observation
        if terminated or truncated:
            break
    env.close()
    return trajectory


def test_visual_ppo_uses_disposable_rollout_and_round_trips(tmp_path) -> None:
    pytest.importorskip("torch")
    from minecraft_gym.academy.visual_ppo import VisualHybridPpoAgent

    agent = VisualHybridPpoAgent(
        modality="rgbd",
        seed=11,
        input_width=32,
        input_height=18,
        minibatch_size=3,
        rollout_episodes=2,
        ppo_epochs=2,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-4,
    )
    first = agent.update_episode(
        _short_ppo_trajectory(agent, seed=31, difficulty="easy")
    )
    assert first["ppo_update_applied"] == 0.0
    assert agent.pending_rollout_episodes == 1

    second = agent.update_episode(
        _short_ppo_trajectory(agent, seed=37, difficulty="moderate")
    )
    assert second["ppo_update_applied"] == 1.0
    assert second["rollout_episodes"] == 2.0
    assert second["ppo_epochs_completed"] <= 2.0
    assert second["approximate_kl"] >= 0.0
    assert 0.0 <= second["clip_fraction"] <= 1.0
    assert agent.pending_rollout_episodes == 0

    env = NavigationAcademyEnv(difficulty="easy")
    observation, _ = env.reset(seed=41, options={"requirement": "travel"})
    expected = agent.choose(observation, stochastic=False)
    path = agent.save(tmp_path / "ppo.pt", metadata={"algorithm": "ppo"})
    loaded, metadata = VisualHybridPpoAgent.load(path, load_optimizer=False)
    actual = loaded.choose(observation, stochastic=False)

    assert metadata == {"algorithm": "ppo"}
    assert loaded.rollout_episodes == 2
    assert loaded.ppo_epochs == 2
    assert actual.dataset_value() == expected.dataset_value()


def test_visual_ppo_preserves_positive_terminal_return() -> None:
    torch = pytest.importorskip("torch")
    from minecraft_gym.academy.visual_ppo import VisualHybridPpoAgent

    agent = VisualHybridPpoAgent(
        modality="rgbd",
        seed=13,
        input_width=32,
        input_height=18,
        rollout_episodes=1,
    )
    # The terminal reward contract can produce +20.  The critic/GAE boundary
    # must preserve that signal instead of clipping it to the old zero ceiling.
    agent._episode_buffer = [[(None, None, 20.0, None, True, False)]]
    _, returns = agent._gae_targets(
        torch.as_tensor([-10.0], dtype=torch.float32, device=agent.device)
    )

    assert returns.item() == pytest.approx(20.0)
