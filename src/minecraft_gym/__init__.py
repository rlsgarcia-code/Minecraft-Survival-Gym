"""Gymnasium registration for the Minecraft survival environment."""

from importlib.metadata import PackageNotFoundError, version

from gymnasium.envs.registration import register, registry

from minecraft_gym.academy.env import NavigationAcademyEnv
from minecraft_gym.env import MinecraftSurvivalEnv


ENV_ID = "minecraft_gym/MinecraftSurvival-v0"
LEGACY_ENV_ID = "MinecraftSurvival-v0"
ACADEMY_ENV_ID = "minecraft_gym/NavigationAcademy-v0"

try:
    __version__ = version("minecraft-gym")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

for environment_id in (ENV_ID, LEGACY_ENV_ID):
    if environment_id not in registry:
        register(
            id=environment_id,
            entry_point="minecraft_gym.env:MinecraftSurvivalEnv",
            nondeterministic=True,
        )

if ACADEMY_ENV_ID not in registry:
    register(
        id=ACADEMY_ENV_ID,
        entry_point="minecraft_gym.academy.env:NavigationAcademyEnv",
        nondeterministic=False,
    )

__all__ = [
    "ACADEMY_ENV_ID",
    "ENV_ID",
    "LEGACY_ENV_ID",
    "MinecraftSurvivalEnv",
    "NavigationAcademyEnv",
    "__version__",
]
