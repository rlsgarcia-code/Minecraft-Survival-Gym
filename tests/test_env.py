from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import minecraft_gym
from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.actions import ControlState, decode_action, encode_human_control, noop_action


def test_environment_passes_gymnasium_checker() -> None:
    env = MinecraftSurvivalEnv(backend="mock", width=32, height=24)
    check_env(env, skip_render_check=False)
    env.close()


def test_registered_environment_can_be_created() -> None:
    env = gym.make(minecraft_gym.ENV_ID, backend="mock", width=8, height=6)
    observation, _ = env.reset(seed=4)
    assert observation["rgb"].shape == (6, 8, 3)
    env.close()


def test_namespaced_and_legacy_environment_ids_are_registered() -> None:
    assert minecraft_gym.ENV_ID == "minecraft_gym/MinecraftSurvival-v0"
    assert minecraft_gym.ENV_ID in gym.registry
    assert minecraft_gym.LEGACY_ENV_ID in gym.registry

    legacy_env = gym.make(
        minecraft_gym.LEGACY_ENV_ID,
        backend="mock",
        width=8,
        height=6,
    )
    observation, _ = legacy_env.reset(seed=4)
    assert observation["rgb"].shape == (6, 8, 3)
    legacy_env.close()


def test_reset_is_seed_reproducible_and_step_advances_exact_ticks() -> None:
    env = MinecraftSurvivalEnv(
        backend="mock", width=16, height=12, frame_skip=4, render_mode="rgb_array"
    )
    first, first_info = env.reset(seed=123)
    second, second_info = env.reset(seed=123)
    assert first_info["world_seed"] == second_info["world_seed"] == 123
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])

    observation, reward, terminated, truncated, info = env.step(noop_action())
    assert env.observation_space.contains(observation)
    assert info["tick"] == 4
    assert reward > 0
    assert not terminated
    assert not truncated
    assert env.render().shape == (12, 16, 3)
    env.close()


def test_human_control_round_trip_projection_is_valid() -> None:
    control = ControlState(
        move_x=-1,
        move_z=1,
        jump=True,
        sprint=True,
        attack=True,
        yaw_delta=20,
        pitch_delta=-3,
        hotbar_slot=4,
        primary_click=True,
    )
    action = encode_human_control(control)
    decoded = decode_action(action)
    assert decoded.move_x == -1
    assert decoded.move_z == 1
    assert decoded.jump
    assert decoded.sprint
    assert decoded.attack
    assert decoded.hotbar_slot == 4
    assert decoded.primary_click


def test_internal_episode_limit_sets_truncation_reason() -> None:
    env = MinecraftSurvivalEnv(backend="mock", max_episode_steps=2)
    env.reset(seed=1)
    _, _, _, truncated, _ = env.step(noop_action())
    assert not truncated
    _, _, _, truncated, info = env.step(noop_action())
    assert truncated
    assert info["truncation_reason"] == "max_episode_steps"
    env.close()
