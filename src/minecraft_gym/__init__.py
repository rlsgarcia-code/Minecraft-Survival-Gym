"""Gymnasium registration for the Minecraft survival environment."""

from gymnasium.envs.registration import register, registry

from minecraft_gym.env import MinecraftSurvivalEnv


ENV_ID = "minecraft_gym/MinecraftSurvival-v0"
LEGACY_ENV_ID = "MinecraftSurvival-v0"

for environment_id in (ENV_ID, LEGACY_ENV_ID):
    if environment_id not in registry:
        register(
            id=environment_id,
            entry_point="minecraft_gym.env:MinecraftSurvivalEnv",
            nondeterministic=True,
        )

__all__ = ["ENV_ID", "LEGACY_ENV_ID", "MinecraftSurvivalEnv"]
