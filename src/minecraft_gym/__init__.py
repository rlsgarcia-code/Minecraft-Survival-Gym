"""Gymnasium registration for the Minecraft survival environment."""

from gymnasium.envs.registration import register, registry

from minecraft_gym.env import MinecraftSurvivalEnv


ENV_ID = "MinecraftSurvival-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="minecraft_gym.env:MinecraftSurvivalEnv",
    )

__all__ = ["ENV_ID", "MinecraftSurvivalEnv"]
