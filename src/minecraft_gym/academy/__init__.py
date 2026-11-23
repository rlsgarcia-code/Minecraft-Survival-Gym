"""Controlled locomotion curriculum for isolated Minecraft training worlds."""

from .arena import (
    AcademyStage,
    ArenaBlueprint,
    NavigationBiome,
    NavigationDifficulty,
    NavigationRequirement,
    SurfaceMaterial,
    build_arena,
    build_target_arena,
)
from .curriculum import (
    CurriculumController,
    academy_task_suite,
    target_navigation_task_suite,
)
from .env import NavigationAcademyEnv
from .learning import (
    BranchingExpectedSarsaCnnAgent,
    ExpectedSarsaCnnAgent,
    JointControlExpectedSarsaCnnAgent,
    LinearSarsaLambdaAgent,
)
from .policy import (
    AcademyAction,
    HybridBodyMode,
    HybridNavigationAction,
    JOINT_CONTROL_ACTIONS,
    JOINT_CONTROL_LABELS,
    WorldMapWindowCache,
    encode_policy_state,
    hybrid_control_action,
    joint_control_action,
    make_hybrid_navigation_action_space,
    project_sparse_policy_lattices,
    project_world_map_window,
)
from .training import (
    STRATEGY_LABELS,
    TabularAcademyAgent,
    compare_strategies,
    evaluate_checkpoint,
    train_curriculum,
)
from .world_bundle import create_academy_bundle, validate_academy_bundle
from .visual import (
    VISUAL_POLICY_CHANNELS,
    academy_visual_pair,
    encode_dense_rgbd_policy_input,
    encode_visual_policy_input,
    resample_depth_planes,
)


def __getattr__(name: str):
    """Load optional PyTorch policies only when callers request them."""

    if name == "VisualHybridPpoAgent":
        try:
            from .visual_ppo import VisualHybridPpoAgent
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                "VisualHybridPpoAgent requires the 'learning' extra: "
                "pip install 'minecraft-gym[learning]'"
            ) from exc
        return VisualHybridPpoAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AcademyStage",
    "ArenaBlueprint",
    "CurriculumController",
    "NavigationAcademyEnv",
    "NavigationBiome",
    "NavigationDifficulty",
    "NavigationRequirement",
    "SurfaceMaterial",
    "AcademyAction",
    "HybridBodyMode",
    "HybridNavigationAction",
    "ExpectedSarsaCnnAgent",
    "BranchingExpectedSarsaCnnAgent",
    "JointControlExpectedSarsaCnnAgent",
    "LinearSarsaLambdaAgent",
    "STRATEGY_LABELS",
    "TabularAcademyAgent",
    "academy_task_suite",
    "target_navigation_task_suite",
    "build_arena",
    "build_target_arena",
    "create_academy_bundle",
    "compare_strategies",
    "encode_policy_state",
    "JOINT_CONTROL_ACTIONS",
    "JOINT_CONTROL_LABELS",
    "WorldMapWindowCache",
    "joint_control_action",
    "hybrid_control_action",
    "make_hybrid_navigation_action_space",
    "project_sparse_policy_lattices",
    "project_world_map_window",
    "evaluate_checkpoint",
    "train_curriculum",
    "validate_academy_bundle",
    "VISUAL_POLICY_CHANNELS",
    "academy_visual_pair",
    "encode_dense_rgbd_policy_input",
    "encode_visual_policy_input",
    "resample_depth_planes",
    "VisualHybridPpoAgent",
]
