from __future__ import annotations

import json
import math
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import minecraft_gym
from minecraft_gym.academy import (
    AcademyStage,
    CurriculumController,
    NavigationAcademyEnv,
    NavigationBiome,
    NavigationDifficulty,
    NavigationRequirement,
    SurfaceMaterial,
    academy_task_suite,
    build_arena,
    build_target_arena,
    create_academy_bundle,
    compare_strategies,
    target_navigation_task_suite,
    validate_academy_bundle,
)
from minecraft_gym.academy.world_bundle import install_academy_bundle
from minecraft_gym.academy.training import (
    CURRICULUM_PHASES,
    TabularAcademyAgent,
    _curriculum_target_schedule,
    train_curriculum,
)
from minecraft_gym.academy.learning import (
    BranchingExpectedSarsaCnnAgent,
    ExpectedSarsaCnnAgent,
    JointControlExpectedSarsaCnnAgent,
    LinearSarsaLambdaAgent,
)
from minecraft_gym.academy.policy import (
    ACTION_COUNT,
    AcademyAction,
    HybridNavigationAction,
    JOINT_CONTROL_ACTIONS,
    JOINT_CONTROL_LABELS,
    abstract_action,
    encode_policy_state,
    factorized_control_action,
    hybrid_control_action,
    joint_control_action,
    make_hybrid_navigation_action_space,
    project_sparse_policy_lattices,
    project_world_map_window,
    WorldMapWindowCache,
)
from minecraft_gym.actions import noop_action
from minecraft_gym.dataset import DatasetRecorderV2, ResearchStorage, iter_steps


def _action(
    *,
    forward: bool = False,
    jump: bool = False,
    sprint: bool = False,
    attack: bool = False,
    use: bool = False,
) -> np.ndarray:
    action = noop_action()
    action[1] = int(forward)
    action[2] = int(jump)
    action[3] = int(sprint)
    action[5] = int(attack)
    action[6] = int(use)
    return action


def _advance(env: NavigationAcademyEnv, *, jump: bool = False) -> bool:
    _, _, terminated, _, _ = env.step(_action(forward=True, jump=jump))
    return terminated


def _solve(stage: AcademyStage) -> tuple[NavigationAcademyEnv, float]:
    env = NavigationAcademyEnv(stage=stage)
    _, info = env.reset(seed=7)
    total = 0.0

    def step(action: np.ndarray) -> bool:
        nonlocal total
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        assert not truncated
        return terminated

    if stage is AcademyStage.FLAT:
        for _ in range(11):
            if step(_action(forward=True)):
                break
    elif stage is AcademyStage.CLIMB:
        for z in range(1, 12):
            if step(_action(forward=True, jump=z in {4, 8})):
                break
    elif stage is AcademyStage.BREAK:
        for _ in range(4):
            step(_action(forward=True))
        step(_action(attack=True))
        step(_action(attack=True))
        for _ in range(7):
            if step(_action(forward=True)):
                break
    elif stage is AcademyStage.BUILD:
        for _ in range(4):
            step(_action(forward=True))
        step(_action(use=True))
        for _ in range(3):
            step(_action(forward=True))
        step(_action(use=True))
        step(_action(forward=True, jump=True))
        for _ in range(3):
            if step(_action(forward=True, jump=True)):
                break
    else:
        for _ in range(2):
            step(_action(forward=True))
        step(_action(attack=True))
        step(_action(attack=True))
        for _ in range(3):
            step(_action(forward=True))
        step(_action(use=True))
        for _ in range(3):
            step(_action(forward=True))
        step(_action(use=True))
        step(_action(forward=True, jump=True))
        for _ in range(2):
            if step(_action(forward=True, jump=True)):
                break

    assert info["arena_sha256"] == build_arena(stage, 7).logical_sha256
    return env, total


def test_academy_environment_passes_gymnasium_checker() -> None:
    env = NavigationAcademyEnv(stage="flat", render_mode="rgb_array")
    check_env(env)
    env.close()


def test_registered_academy_environment_is_deterministic() -> None:
    env = gym.make(minecraft_gym.ACADEMY_ENV_ID, stage="combined")
    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)
    assert first_info["arena_sha256"] == second_info["arena_sha256"]
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    env.close()


@pytest.mark.parametrize("stage", list(AcademyStage))
def test_every_curriculum_stage_is_solvable(stage: AcademyStage) -> None:
    env, total = _solve(stage)
    assert env._position == env.blueprint.goal
    # A solved route receives the v2 terminal success bonus while retaining
    # the elapsed-time ordering below the theoretical +20 ceiling.
    assert 0 < total < 20
    env.close()


def test_privileged_route_state_never_enters_policy_observation() -> None:
    env = NavigationAcademyEnv(stage="build")
    observation, info = env.reset(seed=3)
    assert "absolute_position" not in observation
    assert "absolute_goal" not in observation
    assert "route_cost" not in observation
    assert set(info["privileged_evaluator"]) == {
        "absolute_position",
        "absolute_goal",
        "route_cost",
        "minimum_expected_travel_time",
        "world_delta_columns",
    }
    env.close()


def test_world_map_window_is_metric_and_has_no_evaluator_channels() -> None:
    env = NavigationAcademyEnv(stage="combined", camera_fast_degrees=90)
    observation, info = env.reset(seed=19)
    state = encode_policy_state(observation)
    assert state.grid.shape == (8, 14, 9)
    assert observation["grid_pyramid"].shape == (3, 7, 33, 33)
    assert observation["map_window"].shape == (8, 65, 65)
    assert observation["gnomonic_lattice"].shape == (5, 18, 32)
    assert state.joint_lattice.shape == (13, 65, 65)
    assert state.auxiliary.shape == (6,)
    assert state.action_mask.shape == (ACTION_COUNT,)
    assert observation["orientation"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # The agent is the implicit centre of the north-up metric crop.  The
    # historical pyramid is emitted only for checkpoint compatibility.
    assert observation["map_window"][1, 32, 32] == 1.0
    assert observation["map_window"][7].sum() == 1.0
    assert np.all(observation["grid_pyramid"][:, 0, 16, 16] > 0)
    assert observation["grid_pyramid"][:, 3].sum() > 0
    for forbidden in info["privileged_evaluator"]:
        assert forbidden not in observation
    env.step(abstract_action(AcademyAction.TURN_RIGHT_90))
    observation, *_ = env.step(abstract_action(AcademyAction.WAIT))
    assert observation["orientation"].tolist() == [0.0, 1.0, 0.0, 0.0]
    env.close()


def test_runtime_sparse_projection_uses_the_same_world_map_encoder() -> None:
    observation = project_sparse_policy_lattices(
        (
            {
                "cell_x": 0,
                "cell_z": 0,
                "free_count": 4,
                "hit_count": 0,
                "surface_y_max": 64.0,
                "block_counts": {},
            },
            {
                "cell_x": 0,
                "cell_z": 8,
                "free_count": 0,
                "hit_count": 3,
                "surface_y_max": 65.0,
                "solid_stack_height": 2.0,
                "head_clearance": 1.0,
                "block_counts": {"stone": 3},
            },
        ),
        current_cell=(0, 0),
        target_cell=(0, 8),
        inventory=np.asarray([0, 1], dtype=np.int32),
        action_mask=np.ones(ACTION_COUNT, dtype=np.int8),
        current_y=65.0,
    )
    state = encode_policy_state(observation)
    assert state.joint_lattice.shape == (13, 65, 65)
    assert observation["map_window"].shape == (8, 65, 65)
    assert observation["map_window"][0, 32, 32] == pytest.approx(
        1.0 - math.exp(-4.0 / 8.0)
    )
    assert observation["map_window"][4, 24, 32] == pytest.approx(0.5)
    assert observation["map_window"][5, 24, 32] == pytest.approx(0.25)
    assert observation["map_window"][7, 24, 32] == 1.0
    assert observation["grid_pyramid"][:, 3].sum() > 0
    assert observation["grid_pyramid"][:, 5].sum() > 0
    assert observation["grid_pyramid"][:, 6].min() == 0
    assert observation["grid_pyramid"][0, 1, 16, 16] == pytest.approx(0.25)
    assert observation["gnomonic_lattice"][3].sum() > 0


def test_world_map_memory_survives_outside_crop_and_target_clips_to_border() -> None:
    cells = (
        {
            "cell_x": 0,
            "cell_z": 0,
            "visited_count": 1,
            "free_count": 3,
            "hit_count": 0,
            "surface_y_max": 64.0,
            "block_counts": {},
        },
        {
            "cell_x": 40,
            "cell_z": 0,
            "visited_count": 0,
            "free_count": 0,
            "hit_count": 2,
            "surface_y_max": 65.0,
            "solid_stack_height": 1.0,
            "head_clearance": 4.0,
            "block_counts": {"stone": 2},
        },
    )
    first = project_world_map_window(
        cells, current_cell=(0, 0), target_cell=(100, 0), current_y=65.0
    )
    assert first[0, 32, 32] == pytest.approx(1.0 - math.exp(-4.0 / 8.0))
    assert first[0].sum() == pytest.approx(1.0 - math.exp(-4.0 / 8.0))
    assert first[7, 32, 64] == 1.0

    # The source world memory still contains x=40. Moving the crop exposes it
    # at its exact one-block position instead of reconstructing a coarse bin.
    second = project_world_map_window(
        cells, current_cell=(40, 0), target_cell=(100, 0), current_y=66.0
    )
    assert second[0, 32, 32] == pytest.approx(1.0 - math.exp(-2.0 / 8.0))
    assert second[3, 32, 32] == 1.0
    assert second[7, 32, 64] == 1.0


def test_world_map_cache_reuses_stable_core_and_only_reprojects_changes() -> None:
    cache = WorldMapWindowCache()
    stable = {
        "cell_x": 0,
        "cell_z": 0,
        "visited_count": 4,
        "free_count": 100,
        "hit_count": 0,
        "surface_y_max": 64.0,
        "solid_stack_height": 0.0,
        "head_clearance": 4.0,
        "block_counts": {},
    }
    cache.update((stable,))
    first = cache.render(current_cell=(0, 0), target_cell=(80, 0), current_y=65.0)
    assert cache.world_cell_count == 1
    assert first[1, 32, 32] == 1.0

    # More identical evidence raises counters but topology is already highly
    # confident, so the projected core is reused.
    repeated = {**stable, "free_count": 120}
    cache.update((repeated,))
    second = cache.render(current_cell=(1, 0), target_cell=(80, 0), current_y=65.0)
    assert cache.last_metrics["updated"] == 0
    assert cache.last_metrics["reused"] == 1
    assert cache.last_metrics["window_cells_written"] == 0
    assert second[1, 32, 31] == 1.0

    changed = {**repeated, "solid_stack_height": 2.0}
    cache.update((changed,))
    third = cache.render(current_cell=(1, 0), target_cell=(80, 0), current_y=65.0)
    assert cache.last_metrics["updated"] == 1
    assert cache.last_metrics["window_cells_written"] == 1
    assert third[4, 32, 31] == pytest.approx(0.5)


def test_topology_channels_distinguish_wall_stack_and_low_ceiling() -> None:
    arena = build_target_arena(
        NavigationDifficulty.HARD,
        31,
        biome=NavigationBiome.JUNGLE,
    )
    assert np.any(arena.top_y > arena.base_y)
    low_ceiling_cells = np.argwhere(
        (arena.ceiling_y >= 0) & (arena.ceiling_y - arena.top_y - 1 < 2)
    )
    assert len(low_ceiling_cells) > 0
    env = NavigationAcademyEnv(
        difficulty=NavigationDifficulty.HARD,
        biome=NavigationBiome.JUNGLE,
    )
    observation, _ = env.reset(seed=31)
    local = observation["grid_pyramid"][0]
    assert local[5].max() > 0.0
    observed = local[0] > 0.0
    assert np.any(observed & (local[6] < 0.5))

    target = tuple(int(value) for value in low_ceiling_cells[0])
    neighbor = next(
        (target[0] + dx, target[1] + dz)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if env._inside(target[0] + dx, target[1] + dz)
        and env._top[target[0] + dx, target[1] + dz] >= 0
    )
    env._position = neighbor
    dx, dz = target[0] - neighbor[0], target[1] - neighbor[1]
    env._yaw = math.degrees(math.atan2(-dx, dz))
    events: list[dict[str, object]] = []
    assert not env._move(0.0, 1.0, False, events)
    assert events[-1]["reason"] == "insufficient_head_clearance"
    env.close()


def test_obstacle_layout_generalizes_across_deterministic_seeds() -> None:
    first = build_arena("combined", 3)
    repeated = build_arena("combined", 3)
    held_out = build_arena("combined", 10_003)
    assert first.logical_sha256 == repeated.logical_sha256
    assert first.logical_sha256 != held_out.logical_sha256
    assert not np.array_equal(first.top_y, held_out.top_y)


def test_linear_sarsa_lambda_updates_and_round_trips(tmp_path: Path) -> None:
    env = NavigationAcademyEnv(stage="flat", camera_fast_degrees=90)
    observation, _ = env.reset(seed=4)
    agent = LinearSarsaLambdaAgent()
    agent.begin_episode()
    action = int(AcademyAction.FORWARD)
    next_observation, reward, *_ = env.step(abstract_action(action))
    before = agent.values(observation).copy()
    agent.update(observation, action, reward, next_observation, action, done=False)
    assert not np.array_equal(before, agent.values(observation))
    checkpoint = agent.save(tmp_path / "linear.npz", metadata={"test": True})
    loaded, metadata = LinearSarsaLambdaAgent.load(checkpoint)
    np.testing.assert_allclose(agent.values(observation), loaded.values(observation))
    assert metadata == {"test": True}
    env.close()


def test_expected_sarsa_cnn_updates_spatial_estimator_and_round_trips(tmp_path: Path) -> None:
    env = NavigationAcademyEnv(stage="flat", camera_fast_degrees=90)
    observation, _ = env.reset(seed=5)
    next_observation, reward, *_ = env.step(abstract_action(AcademyAction.FORWARD))
    agent = ExpectedSarsaCnnAgent(seed=5)
    before = agent.values(observation).copy()
    agent.update(
        observation,
        int(AcademyAction.FORWARD),
        reward,
        next_observation,
        done=False,
        epsilon=0.2,
    )
    after = agent.values(observation)
    assert not np.array_equal(before, after)
    checkpoint = agent.save(tmp_path / "cnn.npz", metadata={"test": True})
    loaded, metadata = ExpectedSarsaCnnAgent.load(checkpoint)
    np.testing.assert_allclose(agent.values(observation), loaded.values(observation), atol=1e-5)
    assert metadata == {"test": True}
    env.close()


def test_expected_sarsa_cnn_clears_frame_cache_between_episodes() -> None:
    env = NavigationAcademyEnv(stage="flat", camera_fast_degrees=90)
    observation, _ = env.reset(seed=5)
    agent = ExpectedSarsaCnnAgent(seed=5)
    agent.values(observation)
    assert agent._cache
    agent.begin_episode()
    assert not agent._cache
    env.close()


def test_factorized_control_combines_camera_and_keyboard_buttons() -> None:
    action = factorized_control_action(
        np.asarray([0, 1, 4, 2, 1, 0, 1, 0, 0], dtype=np.int64)
    )
    assert action[1] == 1  # forward
    assert action[2] == 1  # jump
    assert action[3] == 1  # sprint
    assert action[4] == 0  # sneak
    assert action[9] == 4  # fast positive longitude
    assert action[10] == 2  # slow positive latitude


def test_joint_control_catalogue_contains_25_unique_valid_chords() -> None:
    env = NavigationAcademyEnv(stage="flat")
    assert len(JOINT_CONTROL_LABELS) == len(JOINT_CONTROL_ACTIONS) == 25
    assert len({tuple(action.tolist()) for action in JOINT_CONTROL_ACTIONS}) == 25
    assert all(env.action_space.contains(action) for action in JOINT_CONTROL_ACTIONS)
    fast_left = joint_control_action(JOINT_CONTROL_LABELS.index("w_ctrl_a"))
    assert (fast_left[0], fast_left[1], fast_left[3]) == (1, 1, 1)
    fast_right = joint_control_action(JOINT_CONTROL_LABELS.index("w_ctrl_d"))
    assert (fast_right[0], fast_right[1], fast_right[3]) == (2, 1, 1)
    env.close()


def test_diagonal_chord_moves_forward_and_left_in_the_simulator() -> None:
    env = NavigationAcademyEnv(stage="flat")
    env.reset(seed=4)
    before = env._position
    env.step(joint_control_action(JOINT_CONTROL_LABELS.index("w_a")))
    after = env._position
    assert after[0] == before[0] - 1
    assert after[1] == before[1] + 1
    env.close()


def test_hybrid_action_decouples_continuous_camera_and_keyboard() -> None:
    action = HybridNavigationAction(
        delta_longitude=37.25,
        delta_latitude=-11.5,
        strafe=-1,
        move=1,
        jump=True,
        sprint=True,
        use=True,
    )
    space = make_hybrid_navigation_action_space()
    assert space.contains(
        {
            "camera_delta": np.asarray([37.25, -11.5], dtype=np.float32),
            "movement_axes": np.asarray([0, 2], dtype=np.int64),
            "buttons": np.asarray([1, 1, 0, 0, 1], dtype=np.int8),
        }
    )
    control = hybrid_control_action(action)
    assert control.yaw_delta == 37.25
    assert control.pitch_delta == -11.5
    assert control.move_x == -1.0
    assert control.move_z == 1.0
    assert control.sprint and control.jump and control.use


def test_hybrid_step_does_not_quantize_camera_displacement() -> None:
    env = NavigationAcademyEnv(stage="flat")
    env.reset(seed=11)
    action = HybridNavigationAction(37.25, -11.5)
    _, _, _, _, info = env.step_control(hybrid_control_action(action))
    assert env._yaw == pytest.approx(37.25)
    assert env._pitch == pytest.approx(-11.5)
    assert info["executed_action"]["yaw_delta"] == pytest.approx(37.25)
    assert info["elapsed_ticks"] >= 4
    env.close()


def test_hybrid_forward_is_projected_from_continuous_camera_heading() -> None:
    env = NavigationAcademyEnv(difficulty="easy")
    env.reset(seed=11)
    before = env._position
    action = HybridNavigationAction(45.0, 0.0, move=1)
    env.step_control(hybrid_control_action(action))
    assert env._position == (before[0] - 1, before[1] + 1)
    env.close()


def test_hybrid_actor_critic_updates_bounded_value_and_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from minecraft_gym.academy.actor_critic import HybridActorCriticAgent

    env = NavigationAcademyEnv(difficulty="easy")
    observation, _ = env.reset(seed=21)
    agent = HybridActorCriticAgent(seed=21)
    action = agent.choose(observation, stochastic=True)
    next_observation, reward, terminated, truncated, _ = env.step_control(
        hybrid_control_action(action)
    )
    done = terminated or truncated
    next_action = None if done else agent.choose(next_observation, stochastic=True)
    metrics = agent.update(
        observation,
        action,
        reward,
        next_observation,
        next_action,
        done=done,
    )
    assert set(metrics) == {
        "actor_loss",
        "critic_loss",
        "advantage",
        "raw_advantage",
        "state_value",
        "q_value",
        "entropy",
        "actor_gradient_norm",
        "critic_gradient_norm",
        "td_target",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert -40.0 <= metrics["state_value"] <= 20.0
    assert -40.0 <= metrics["td_target"] <= 20.0
    assert -10.0 <= metrics["advantage"] <= 10.0
    episode_metrics = agent.update_episode(
        [(observation, action, reward), (next_observation, next_action or action, -0.1)]
    )
    assert all(np.isfinite(value) for value in episode_metrics.values())
    assert -40.0 <= episode_metrics["state_value"] <= 20.0
    assert -40.0 <= episode_metrics["td_target"] <= 20.0
    checkpoint = agent.save(tmp_path / "hybrid.pt", metadata={"test": True})
    expected_stochastic = agent.choose(observation, stochastic=True)
    loaded, metadata = HybridActorCriticAgent.load(checkpoint)
    restored_stochastic = loaded.choose(observation, stochastic=True)
    np.testing.assert_allclose(
        expected_stochastic.vector(), restored_stochastic.vector(), atol=1e-5
    )
    deterministic = agent.choose(observation, stochastic=False)
    restored = loaded.choose(observation, stochastic=False)
    np.testing.assert_allclose(deterministic.vector(), restored.vector(), atol=1e-5)
    assert metadata == {"test": True}
    env.close()


def test_hybrid_training_resumes_from_atomic_episode_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from minecraft_gym.academy.actor_critic import HybridActorCriticAgent
    from minecraft_gym.academy.training import _train_one_strategy

    output = tmp_path / "hybrid_actor_critic"

    def interrupt_after_checkpoint(*args, **kwargs) -> None:
        del kwargs
        completed = int(args[2])
        if completed == 2:
            raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        _train_one_strategy(
            "hybrid_actor_critic",
            output,
            episodes_per_stage=2,
            seed=31,
            epsilon_start=0.4,
            epsilon_end=0.05,
            checkpoint_interval=1,
            on_episode=interrupt_after_checkpoint,
        )

    checkpoint = output / "hybrid-actor-critic.pt"
    _, interrupted_metadata = HybridActorCriticAgent.load(checkpoint)
    assert interrupted_metadata["training_state"]["status"] == "running"
    assert interrupted_metadata["training_state"]["completed_episodes"] == 1

    _, stages, saved = _train_one_strategy(
        "hybrid_actor_critic",
        output,
        episodes_per_stage=2,
        seed=31,
        epsilon_start=0.4,
        epsilon_end=0.05,
        checkpoint_interval=1,
    )
    _, completed_metadata = HybridActorCriticAgent.load(saved)
    assert sum(stage["episodes"] for stage in stages.values()) == 10
    assert completed_metadata["training_state"]["status"] == "complete"
    assert completed_metadata["training_state"]["completed_episodes"] == 10


def test_branching_expected_sarsa_updates_and_round_trips(tmp_path: Path) -> None:
    env = NavigationAcademyEnv(difficulty="easy", camera_fast_degrees=90)
    observation, _ = env.reset(seed=8)
    agent = BranchingExpectedSarsaCnnAgent(seed=8)
    rng = np.random.default_rng(8)
    action = agent.choose(observation, epsilon=0.0, rng=rng)
    assert action.shape == (9,)
    next_observation, reward, *_ = env.step(factorized_control_action(action))
    before = agent.values(observation).copy()
    agent.update(
        observation,
        action,
        reward,
        next_observation,
        done=False,
        epsilon=0.2,
    )
    assert not np.array_equal(before, agent.values(observation))
    checkpoint = agent.save(tmp_path / "branching.npz", metadata={"test": True})
    loaded, metadata = BranchingExpectedSarsaCnnAgent.load(checkpoint)
    np.testing.assert_allclose(agent.values(observation), loaded.values(observation), atol=1e-5)
    assert metadata == {"test": True}
    env.close()


def test_joint_control_expected_sarsa_updates_and_round_trips(tmp_path: Path) -> None:
    env = NavigationAcademyEnv(difficulty="easy", camera_fast_degrees=90)
    observation, _ = env.reset(seed=18)
    agent = JointControlExpectedSarsaCnnAgent(seed=18)
    assert agent.discount == 1.0
    assert encode_policy_state(observation, include_elapsed=True).auxiliary.shape == (7,)
    rng = np.random.default_rng(18)
    action = agent.choose(observation, epsilon=0.0, rng=rng)
    assert 0 <= action < 25
    next_observation, reward, *_ = env.step(joint_control_action(action))
    before = agent.values(observation).copy()
    agent.update(
        observation,
        action,
        reward,
        next_observation,
        done=False,
        epsilon=0.2,
    )
    assert not np.array_equal(before, agent.values(observation))
    checkpoint = agent.save(tmp_path / "joint.npz", metadata={"test": True})
    loaded, metadata = JointControlExpectedSarsaCnnAgent.load(checkpoint)
    np.testing.assert_allclose(agent.values(observation), loaded.values(observation), atol=1e-5)
    assert metadata == {"test": True}
    env.close()


def test_sprint_stops_on_goal_instead_of_overshooting_success() -> None:
    env = NavigationAcademyEnv(stage="flat")
    env.reset(seed=3)
    for _ in range(5):
        _, _, terminated, _, _ = env.step(_action(forward=True, sprint=True))
        assert not terminated
    _, _, terminated, _, _ = env.step(_action(forward=True, sprint=True))
    assert terminated
    assert env._position == env.blueprint.goal
    env.close()


def test_curriculum_requires_sustained_success_before_promotion() -> None:
    curriculum = CurriculumController(window_size=5, promotion_success_rate=0.8)
    for value in (True, True, False, True):
        assert curriculum.record(value) is AcademyStage.FLAT
    assert curriculum.record(True) is AcademyStage.CLIMB
    assert curriculum.report()["stage"] == "climb"


def test_training_checkpoint_is_atomic_and_round_trips(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "smoke.json"
    progress = tmp_path / "runs" / "progress.json"
    report = train_curriculum(
        checkpoint,
        episodes_per_stage=1,
        seed=2,
        progress_path=progress,
        evaluation_episodes=1,
    )
    assert report["checkpoint"] == str(checkpoint.resolve())
    assert not checkpoint.with_suffix(".json.partial").exists()
    agent, metadata = TabularAcademyAgent.load(checkpoint)
    assert agent.q
    assert metadata["policy_input"] == "structured_academy_only"
    assert metadata["deployable_on_rgb"] is False
    progress_payload = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_payload["interface"] == "minecraft-academy-training-progress/v1"
    assert progress_payload["status"] == "succeeded"
    assert progress_payload["progress"] == 1.0
    assert set(progress_payload["evaluation"]["stages"]) == {
        stage.slug for stage in AcademyStage
    }


def test_academy_rollout_uses_the_canonical_dataset_contract(tmp_path: Path) -> None:
    volume = tmp_path / "research-volume"
    volume.mkdir()
    storage = ResearchStorage(
        volume,
        spool_root=tmp_path / "spool",
        minimum_system_free_bytes=0,
        minimum_external_free_ratio=0.0,
        require_apfs=False,
        require_mountpoint=False,
    )
    base = NavigationAcademyEnv(stage="flat")
    env = DatasetRecorderV2(
        base,
        storage,
        environment_id=minecraft_gym.ACADEMY_ENV_ID,
        world_id="navigation-academy:test",
        environment_mode="isolated_training",
        world_snapshot_id="navigation-academy:test-snapshot",
        task_id="navigation-academy/flat-v2",
        require_parquet=False,
    )
    env.reset(seed=1)
    env.step(_action(forward=True))
    env.close()
    assert len(env.completed) == 1
    manifest, quality, episode_path = env.completed[0]
    records = list(iter_steps(episode_path / "steps.jsonl"))
    assert manifest.environment_mode == "isolated_training"
    assert quality.accepted
    assert len(records) == 1
    assert records[0].data_tier == "exact_structured"
    assert "absolute_position" not in records[0].observation.policy_channels
    assert records[0].observation.privileged["absolute_position"]


def test_task_suite_covers_all_stages_with_privileged_reward_only() -> None:
    tasks = academy_task_suite(seed=9)
    assert len(tasks) == len(AcademyStage)
    assert {task.task_id for task in tasks} == {
        f"navigation-academy/{stage.slug}-v2" for stage in AcademyStage
    }
    for task in tasks:
        assert task.privileged_reward_channels == ("elapsed_ticks", "task_success")
        assert "absolute_position" not in task.policy_channels


def _world(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "level.dat").write_bytes(b"level")
    return path


def test_bundle_and_real_arena_use_the_same_hashes(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    result = create_academy_bundle(output, seed=17, pack_format=48)
    manifest = json.loads((output / "academy-manifest.json").read_text())
    assert result["stages"] == 5
    assert manifest["live_world_access"] is False
    assert manifest["automatic_world_mutation"] is False
    assert [item["logical_sha256"] for item in manifest["stages"]] == [
        build_arena(stage, 17).logical_sha256 for stage in AcademyStage
    ]
    assert {item["interface"] for item in manifest["stages"]} == {
        "minecraft-navigation-arena/v3"
    }
    assert (output / "datapack" / "navigation-academy-v1" / "pack.mcmeta").is_file()
    validation = validate_academy_bundle(output)
    assert validation["target_minecraft_version"] == "26.3"
    assert validation["pack_format"] == 48


def test_bundle_tampering_is_rejected_before_install(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    create_academy_bundle(output)
    reset = (
        output
        / "datapack"
        / "navigation-academy-v1"
        / "data"
        / "minecraft_gym"
        / "function"
        / "academy"
        / "reset_flat.mcfunction"
    )
    reset.unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        validate_academy_bundle(output)


def test_install_requires_a_world_distinct_from_family_world(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    create_academy_bundle(bundle)
    family = _world(tmp_path / "brincadeiras")
    with pytest.raises(ValueError, match="live family world"):
        install_academy_bundle(bundle, family, live_world=family)

    training = _world(tmp_path / "isolated" / "academy-run")
    result = install_academy_bundle(bundle, training, live_world=family)
    assert result["live_world_access"] == "false"
    assert (training / "datapacks" / "navigation-academy-v1" / "pack.mcmeta").is_file()


def test_strategy_comparison_separates_online_learning_from_evaluation(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress.json"
    result = compare_strategies(
        tmp_path / "checkpoints",
        strategies=["linear_sarsa_lambda"],
        episodes_per_stage=2,
        evaluation_episodes=1,
        seed=31,
        epsilon_start=0.4,
        epsilon_end=0.05,
        progress_path=progress,
    )

    assert result["learning_mode"] == "online_on_policy"
    assert result["benchmark_contract"].endswith("random-biomes-interactions-v3")
    report = result["reports"]["linear_sarsa_lambda"]
    assert report["training"]["learning_enabled"] is True
    assert report["training_seed_evaluation"]["held_out"] is False
    assert report["evaluation"]["held_out"] is True
    easy = report["training"]["stages"]["easy"]
    assert len(easy["reward_curve"]) == 4
    assert len(easy["reward_moving_average"]) == 4
    assert set(report["training"]["stages"]) == {
        difficulty.slug for difficulty in NavigationDifficulty
    }
    comparison = result["comparison"][0]
    assert "recent_online_success_rate" in comparison
    assert "trained_seed_success_rate" in comparison
    assert "held_out_success_rate" in comparison
    assert progress.is_file()
    timeline = report["training"]["timeline"]
    assert timeline["interface"] == "minecraft-academy-chronology/v1"
    assert len(timeline["episodes"]) == 10
    assert [item["index"] for item in timeline["episodes"]] == list(range(1, 11))
    assert sum(
        item["target"] for item in timeline["difficulty_progress"].values()
    ) == 10


def test_target_curriculum_preserves_phase_distributions_and_affordances() -> None:
    schedule = _curriculum_target_schedule(seed=37, episodes_per_phase=100)
    assert len(schedule) == 500
    expected = [
        {"easy": 80, "moderate": 20},
        {"easy": 20, "moderate": 60, "hard": 20},
        {"easy": 10, "moderate": 20, "hard": 50, "expert": 20},
        {"easy": 10, "moderate": 15, "hard": 25, "expert": 25, "mixed": 25},
        {"easy": 20, "moderate": 20, "hard": 20, "expert": 20, "mixed": 20},
    ]
    for phase_index, phase in enumerate(CURRICULUM_PHASES, start=1):
        rows = [row for row in schedule if row[0] == phase_index]
        assert len(rows) == 100
        assert {row[1] for row in rows} == {phase["slug"]}
        counts = {
            difficulty.slug: sum(1 for row in rows if row[2] is difficulty)
            for difficulty in NavigationDifficulty
            if any(row[2] is difficulty for row in rows)
        }
        assert counts == expected[phase_index - 1]
        if phase_index == 1:
            assert {row[4] for row in rows} == {NavigationRequirement.TRAVEL}
        elif phase_index == 2:
            assert {row[4] for row in rows} == {
                NavigationRequirement.TRAVEL,
                NavigationRequirement.BREAK,
                NavigationRequirement.PLACE,
            }
    for difficulty in NavigationDifficulty:
        requirements = {
            row[4] for row in schedule if row[2] is difficulty
        }
        expected_requirements = (
            {
                NavigationRequirement.TRAVEL,
                NavigationRequirement.BREAK,
                NavigationRequirement.PLACE,
            }
            if difficulty is NavigationDifficulty.EASY
            else set(NavigationRequirement)
        )
        assert requirements == expected_requirements


def test_target_arena_is_random_reproducible_and_stratified() -> None:
    first = build_target_arena("hard", 91)
    repeated = build_target_arena("hard", 91)
    assert first.logical_sha256 == repeated.logical_sha256
    assert first.difficulty is NavigationDifficulty.HARD
    assert first.biome in {
        NavigationBiome.JUNGLE,
        NavigationBiome.SNOWY,
        NavigationBiome.RIVER,
        NavigationBiome.VOLCANIC,
    }
    samples = [build_target_arena("mixed", seed) for seed in range(30, 45)]
    assert len({arena.logical_sha256 for arena in samples}) == len(samples)
    assert len({arena.biome for arena in samples}) >= 3
    assert len({arena.goal for arena in samples}) >= 3
    assert {arena.requirement for arena in samples} == set(NavigationRequirement)


def test_target_navigation_penalizes_elapsed_time_and_exposes_construction() -> None:
    env = NavigationAcademyEnv(
        difficulty="moderate",
        biome="river",
        camera_fast_degrees=90,
    )
    observation, info = env.reset(seed=7)
    assert info["difficulty"] == "moderate"
    assert info["biome"] == "river"
    assert info["requirement"] == NavigationRequirement.COMBINED.value
    assert observation["inventory"][0] == 2
    _, _, _, _, step_info = env.step(abstract_action(AcademyAction.WAIT))
    assert step_info["reward_terms"]["elapsed_time"] < 0
    assert step_info["elapsed_ticks"] == 4
    env.close()


def test_bounded_asymptotic_time_cost_grows_and_stationary_time_is_multiplied() -> None:
    moving = NavigationAcademyEnv(stage="flat", max_episode_steps=20)
    stationary = NavigationAcademyEnv(stage="flat", max_episode_steps=20)
    moving.reset(seed=9)
    stationary.reset(seed=9)
    _, moving_reward, *_ = moving.step(_action(forward=True))
    _, stationary_reward, *_ = stationary.step(abstract_action(AcademyAction.WAIT))
    assert stationary_reward == pytest.approx(1.5 * moving_reward)
    _, later_stationary_reward, *_ = stationary.step(abstract_action(AcademyAction.WAIT))
    assert abs(later_stationary_reward) > abs(stationary_reward)
    assert stationary._cumulative_time_cost(80) == pytest.approx(
        20.0 * (1.0 - np.exp(-3.0))
    )
    assert stationary._cumulative_time_cost(8_000) <= 20.0
    moving.close()
    stationary.close()


def test_terminal_reward_separates_success_from_timeout() -> None:
    successful = NavigationAcademyEnv(stage="flat", max_episode_steps=1)
    timed_out = NavigationAcademyEnv(stage="flat", max_episode_steps=1)
    successful.reset(seed=9)
    timed_out.reset(seed=9)

    # Isolate the terminal contract from path planning: the policy has already
    # arrived when the final environment decision is accounted for.
    successful._position = successful.blueprint.goal
    _, success_reward, terminated, truncated, success_info = successful.step(
        abstract_action(AcademyAction.WAIT)
    )
    _, timeout_reward, timeout_terminated, timeout_truncated, timeout_info = (
        timed_out.step(abstract_action(AcademyAction.WAIT))
    )

    assert terminated and not truncated
    assert success_info["is_success"]
    assert success_info["reward_terms"]["terminal_success"] == 20.0
    assert success_info["reward_terms"]["terminal_timeout"] == 0.0
    assert not timeout_terminated and timeout_truncated
    assert not timeout_info["is_success"]
    assert timeout_info["reward_terms"]["terminal_success"] == 0.0
    assert timeout_info["reward_terms"]["terminal_timeout"] == -5.0
    assert success_reward > timeout_reward

    successful.close()
    timed_out.close()


def test_hazard_surfaces_are_encoded_without_new_policy_channels() -> None:
    arena = build_target_arena(
        NavigationDifficulty.EXPERT,
        4,
        biome=NavigationBiome.VOLCANIC,
    )
    assert np.any(arena.surface == int(SurfaceMaterial.LAVA))
    env = NavigationAcademyEnv(
        difficulty=NavigationDifficulty.EXPERT,
        biome=NavigationBiome.VOLCANIC,
    )
    observation, _ = env.reset(seed=4)
    assert observation["grid_pyramid"].shape[1] == 7
    env.close()


def test_target_navigation_contract_includes_required_interactions_and_prices_time() -> None:
    tasks = target_navigation_task_suite(seed=12)
    assert len(tasks) == len(NavigationDifficulty)
    for task in tasks:
        assert task.family == "navigate"
        assert task.initial_state["construction_allowed"] is True
        assert task.initial_state["required_affordance"] in {
            item.value for item in NavigationRequirement
        }
        assert "required_affordance_satisfied" in task.success_predicates
        assert task.reward_terms == {
            "time_cost_max": 20.0,
            "time_cost_beta": 3.0,
            "stationary_time_multiplier": 1.5,
            "terminal_success_bonus": 20.0,
            "terminal_timeout_penalty": 5.0,
        }
        assert "elapsed_fraction" in task.policy_channels
        assert task.privileged_reward_channels == ("elapsed_ticks", "task_success")
        assert "absolute_position" not in task.policy_channels


@pytest.mark.parametrize(
    ("requirement", "initial_blocks"),
    [
        (NavigationRequirement.BREAK, 0),
        (NavigationRequirement.PLACE, 1),
        (NavigationRequirement.COMBINED, 2),
    ],
)
def test_target_interaction_gates_span_the_arena_and_cannot_be_bypassed(
    requirement: NavigationRequirement,
    initial_blocks: int,
) -> None:
    arena = build_target_arena(
        NavigationDifficulty.HARD,
        44,
        biome=NavigationBiome.PLAINS,
        requirement=requirement,
    )
    assert arena.requirement is requirement
    assert arena.initial_blocks == initial_blocks
    if requirement in {NavigationRequirement.BREAK, NavigationRequirement.COMBINED}:
        wall_rows = [
            z
            for z in range(arena.length)
            if all(arena.top_y[x, z] - arena.base_y[x, z] == 2 for x in range(arena.width))
        ]
        assert wall_rows
    if requirement in {NavigationRequirement.PLACE, NavigationRequirement.COMBINED}:
        gap_rows = [
            z for z in range(arena.length) if np.all(arena.top_y[:, z] < 0)
        ]
        assert gap_rows


def test_target_env_records_break_and_place_as_required_competencies() -> None:
    break_env = NavigationAcademyEnv(
        difficulty="hard",
        biome="plains",
        requirement="break",
    )
    break_env.reset(seed=44)
    wall_z = next(
        z
        for z in range(break_env.blueprint.length)
        if all(
            break_env._top[x, z] - break_env.blueprint.base_y[x, z] == 2
            for x in range(break_env.blueprint.width)
        )
    )
    break_env._position = (4, wall_z - 1)
    _, _, _, _, break_info = break_env.step_control(
        hybrid_control_action(HybridNavigationAction(0.0, 0.0, attack=True))
    )
    assert break_info["interaction_counts"]["blocks_broken"] == 1
    assert break_info["requirement_satisfied"] is True
    break_env.close()

    place_env = NavigationAcademyEnv(
        difficulty="hard",
        biome="plains",
        requirement="place",
    )
    place_env.reset(seed=44)
    gap_z = next(
        z
        for z in range(place_env.blueprint.length)
        if np.all(place_env._top[:, z] < 0)
    )
    place_env._position = (4, gap_z - 1)
    _, _, _, _, place_info = place_env.step_control(
        hybrid_control_action(HybridNavigationAction(0.0, 0.0, use=True))
    )
    assert place_info["interaction_counts"]["blocks_placed"] == 1
    assert place_info["requirement_satisfied"] is True
    assert place_env._top[4, gap_z] == 0
    place_env.close()


def test_combined_interaction_requires_distinct_break_and_place_decisions() -> None:
    env = NavigationAcademyEnv(
        difficulty="expert",
        biome="plains",
        requirement="combined",
    )
    env.reset(seed=44)
    wall_z = next(
        z
        for z in range(env.blueprint.length)
        if all(
            env._top[x, z] - env.blueprint.base_y[x, z] == 2
            for x in range(env.blueprint.width)
        )
    )
    env._position = (4, wall_z - 1)
    _, _, _, _, info = env.step_control(
        hybrid_control_action(
            HybridNavigationAction(0.0, 0.0, attack=True, use=True)
        )
    )
    assert info["interaction_counts"] == {"blocks_broken": 1, "blocks_placed": 0}
    assert info["requirement_satisfied"] is False
    assert any(event["type"] == "use_suppressed" for event in info["events"])
    env.close()
