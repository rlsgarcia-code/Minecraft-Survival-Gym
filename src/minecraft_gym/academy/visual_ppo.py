"""Clipped PPO for the shared VGG RGB-D hybrid policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .actor_critic import VALUE_RETURN_MAX, VALUE_RETURN_MIN
from .policy import HybridNavigationAction
from .visual_actor_critic import VisualHybridActorCriticAgent


class VisualHybridPpoAgent(VisualHybridActorCriticAgent):
    """On-policy PPO with a short, disposable multi-episode rollout buffer."""

    interface = "minecraft-vgg-rgbd-ppo/v1"
    algorithm = "ppo-clip-gae/v1"

    def __init__(
        self,
        *,
        rollout_episodes: int = 10,
        ppo_epochs: int = 4,
        clip_ratio: float = 0.2,
        value_clip_range: float = 0.2,
        gae_lambda: float = 0.95,
        target_kl: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if rollout_episodes < 1:
            raise ValueError("rollout_episodes must be positive")
        if ppo_epochs < 1:
            raise ValueError("ppo_epochs must be positive")
        if not 0.0 < clip_ratio < 1.0:
            raise ValueError("clip_ratio must be between zero and one")
        if not 0.0 < value_clip_range:
            raise ValueError("value_clip_range must be positive")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between zero and one")
        if target_kl <= 0.0:
            raise ValueError("target_kl must be positive")
        self.rollout_episodes = int(rollout_episodes)
        self.ppo_epochs = int(ppo_epochs)
        self.clip_ratio = float(clip_ratio)
        self.value_clip_range = float(value_clip_range)
        self.gae_lambda = float(gae_lambda)
        self.target_kl = float(target_kl)
        self._episode_buffer: list[list[tuple[Any, ...]]] = []

    @property
    def pending_rollout_episodes(self) -> int:
        return len(self._episode_buffer)

    def update_episode(
        self,
        trajectory: list[tuple[Any, ...]],
    ) -> dict[str, float]:
        if not trajectory:
            raise ValueError("episode trajectory cannot be empty")
        self._episode_buffer.append(trajectory)
        if len(self._episode_buffer) < self.rollout_episodes:
            return {
                "ppo_update_applied": 0.0,
                "rollout_buffer_episodes": float(len(self._episode_buffer)),
                "rollout_buffer_steps": float(
                    sum(len(episode) for episode in self._episode_buffer)
                ),
            }
        return self._optimize_rollout()

    def flush_rollout(self) -> dict[str, float]:
        if not self._episode_buffer:
            return {}
        return self._optimize_rollout()

    def _old_policy_statistics(
        self,
        observations: list[dict[str, Any]],
        actions: list[HybridNavigationAction],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(observations), self.minibatch_size):
                stop = start + self.minibatch_size
                log_probability, _, value = self._evaluate_actions(
                    observations[start:stop], actions[start:stop]
                )
                log_probabilities.append(log_probability)
                values.append(value)
        return torch.cat(log_probabilities), torch.cat(values)

    def _gae_targets(
        self,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(old_values)
        cursor = 0
        for episode in self._episode_buffer:
            episode_values = old_values[cursor : cursor + len(episode)]
            next_advantage = torch.zeros((), dtype=torch.float32, device=self.device)
            for local_index in range(len(episode) - 1, -1, -1):
                transition = episode[local_index]
                reward = float(transition[2])
                is_last = local_index == len(episode) - 1
                terminated = bool(transition[4]) if len(transition) > 4 else is_last
                truncated = bool(transition[5]) if len(transition) > 5 else False
                if not is_last:
                    next_value = episode_values[local_index + 1]
                elif not terminated and truncated and len(transition) > 3:
                    next_value = torch.as_tensor(
                        self.state_value(transition[3]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    next_value = torch.zeros((), dtype=torch.float32, device=self.device)
                bootstrap = 0.0 if terminated else 1.0
                delta = reward + self.discount * bootstrap * next_value - episode_values[
                    local_index
                ]
                if is_last:
                    next_advantage = delta
                else:
                    next_advantage = (
                        delta
                        + self.discount * self.gae_lambda * next_advantage
                    )
                advantages[cursor + local_index] = next_advantage
            cursor += len(episode)
        returns = (advantages + old_values).clamp(
            VALUE_RETURN_MIN, VALUE_RETURN_MAX
        )
        normalized = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-6)
        return normalized.clamp(-5.0, 5.0), returns

    def _optimize_rollout(self) -> dict[str, float]:
        episodes = self._episode_buffer
        transitions = [item for episode in episodes for item in episode]
        observations = [item[0] for item in transitions]
        actions = [item[1] for item in transitions]
        old_log_probability, old_values = self._old_policy_statistics(
            observations, actions
        )
        advantages, returns = self._gae_targets(old_values)
        raw_advantage_mean = float((returns - old_values).mean().cpu().item())
        totals: dict[str, float] = {}
        samples = 0
        completed_epochs = 0
        stopped_for_kl = False
        assert self.network is not None and self.optimizer is not None
        for epoch in range(self.ppo_epochs):
            order = torch.randperm(len(transitions)).tolist()
            epoch_kl_samples = 0
            epoch_kl_total = 0.0
            for start in range(0, len(order), self.minibatch_size):
                indices = order[start : start + self.minibatch_size]
                batch_observations = [observations[index] for index in indices]
                batch_actions = [actions[index] for index in indices]
                index_tensor = torch.as_tensor(
                    indices, dtype=torch.int64, device=self.device
                )
                new_log_probability, entropy, values = self._evaluate_actions(
                    batch_observations, batch_actions
                )
                log_ratio = new_log_probability - old_log_probability[index_tensor]
                ratio = log_ratio.clamp(-20.0, 20.0).exp()
                batch_advantage = advantages[index_tensor]
                unclipped = ratio * batch_advantage
                clipped = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantage
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                old_batch_value = old_values[index_tensor]
                batch_return = returns[index_tensor]
                clipped_value = old_batch_value + (values - old_batch_value).clamp(
                    -self.value_clip_range, self.value_clip_range
                )
                value_loss = 0.5 * torch.maximum(
                    (values - batch_return).square(),
                    (clipped_value - batch_return).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = (
                    actor_loss
                    + self.value_loss_coefficient * value_loss
                    - self.entropy_coefficient * entropy_mean
                )
                approximate_kl = (ratio - 1.0 - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > self.clip_ratio
                ).float().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_gradient_norm
                )
                self.optimizer.step()
                count = len(indices)
                samples += count
                epoch_kl_samples += count
                epoch_kl_total += float(approximate_kl.detach().cpu().item()) * count
                metrics = {
                    "actor_loss": actor_loss,
                    "critic_loss": value_loss,
                    "entropy": entropy_mean,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "policy_ratio": ratio.mean(),
                    "state_value": values.mean(),
                    "actor_gradient_norm": gradient_norm,
                }
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(
                        value.detach().cpu().item()
                    ) * count
            completed_epochs = epoch + 1
            if epoch_kl_total / max(1, epoch_kl_samples) > self.target_kl * 1.5:
                stopped_for_kl = True
                break
        self.update_count += len(transitions)
        self._episode_buffer = []
        result = {name: value / samples for name, value in totals.items()}
        result.update(
            {
                "raw_advantage": raw_advantage_mean,
                "q_value": result["state_value"],
                "critic_gradient_norm": result["actor_gradient_norm"],
                "ppo_update_applied": 1.0,
                "ppo_epochs_completed": float(completed_epochs),
                "ppo_early_stop_kl": float(stopped_for_kl),
                "rollout_episodes": float(len(episodes)),
                "rollout_steps": float(len(transitions)),
            }
        )
        return result

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "rollout_episodes": self.rollout_episodes,
            "ppo_epochs": self.ppo_epochs,
            "clip_ratio": self.clip_ratio,
            "value_clip_range": self.value_clip_range,
            "gae_lambda": self.gae_lambda,
            "target_kl": self.target_kl,
        }

    @classmethod
    def _checkpoint_constructor_kwargs(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "rollout_episodes": int(payload.get("rollout_episodes", 10)),
            "ppo_epochs": int(payload.get("ppo_epochs", 4)),
            "clip_ratio": float(payload.get("clip_ratio", 0.2)),
            "value_clip_range": float(payload.get("value_clip_range", 0.2)),
            "gae_lambda": float(payload.get("gae_lambda", 0.95)),
            "target_kl": float(payload.get("target_kl", 0.02)),
        }
