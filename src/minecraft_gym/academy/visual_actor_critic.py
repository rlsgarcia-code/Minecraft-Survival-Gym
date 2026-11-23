"""VGG-style shared-backbone actor-critic for native RGB/RGB-D observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal, VonMises
from torch.nn import functional as F

from minecraft_gym.dataset import CapturedDenseRgbdFrame

from .actor_critic import (
    ADVANTAGE_LIMIT,
    VALUE_RETURN_MAX,
    VALUE_RETURN_MIN,
    HybridActorCriticAgent,
)
from .policy import HYBRID_LATITUDE_LIMIT_DEGREES, HybridNavigationAction, PolicyState
from .visual import (
    VisualModality,
    academy_visual_pair,
    encode_dense_rgbd_policy_input,
    encode_visual_policy_input,
)


def _group_count(channels: int) -> int:
    for candidate in (16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class _VggBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = _group_count(output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _VggRgbdActorCritic(nn.Module):
    """Shared visual trunk, explicit flatten and separate policy/value heads."""

    def __init__(self, input_channels: int, auxiliary_size: int) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 256)
        blocks: list[nn.Module] = []
        previous = input_channels
        for width in widths:
            blocks.append(_VggBlock(previous, width))
            previous = width
        self.features = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.spatial_projection = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + auxiliary_size, 256),
            nn.ReLU(inplace=True),
        )
        self.longitude_location = nn.Linear(256, 1)
        self.longitude_log_concentration = nn.Linear(256, 1)
        self.latitude_location = nn.Linear(256, 1)
        self.latitude_log_scale = nn.Linear(256, 1)
        self.strafe_logits = nn.Linear(256, 3)
        self.move_logits = nn.Linear(256, 3)
        self.button_logits = nn.Linear(256, 5)
        self.value_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.longitude_location.bias)
        nn.init.constant_(self.longitude_log_concentration.bias, -0.3)
        nn.init.zeros_(self.latitude_location.bias)
        nn.init.constant_(self.latitude_log_scale.bias, -0.7)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def encode(
        self, lattice: torch.Tensor, auxiliary: torch.Tensor
    ) -> torch.Tensor:
        spatial = self.spatial_projection(self.flatten(self.features(lattice)))
        return self.fusion(torch.cat((spatial, auxiliary), dim=1))

    def distributions_and_value(
        self, lattice: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[VonMises, Normal, Categorical, Categorical, Bernoulli, torch.Tensor]:
        features = self.encode(lattice, auxiliary)
        longitude = VonMises(
            self.longitude_location(features).squeeze(1),
            F.softplus(
                self.longitude_log_concentration(features).clamp(-4.0, 4.0)
            ).squeeze(1)
            + 0.05,
        )
        latitude = Normal(
            self.latitude_location(features).clamp(-3.0, 3.0).squeeze(1),
            self.latitude_log_scale(features).clamp(-2.5, 0.35).exp().squeeze(1),
        )
        value = VALUE_RETURN_MIN + (
            VALUE_RETURN_MAX - VALUE_RETURN_MIN
        ) * torch.sigmoid(self.value_head(features).squeeze(1))
        return (
            longitude,
            latitude,
            Categorical(logits=self.strafe_logits(features)),
            Categorical(logits=self.move_logits(features)),
            Bernoulli(logits=self.button_logits(features)),
            value,
        )


class VisualHybridActorCriticAgent(HybridActorCriticAgent):
    """Native-resolution visual AC with a shared VGG-style representation."""

    interface = "minecraft-vgg-rgbd-actor-critic/v1"
    observation_contract = "minecraft-visual-policy-input/native-rgbd-v2"
    architecture = "vgg-rgbd-shared-flatten-mlp/v1"

    def __init__(
        self,
        *,
        modality: VisualModality,
        input_width: int | None = None,
        input_height: int | None = None,
        minibatch_size: int = 8,
        value_loss_coefficient: float = 0.5,
        backbone_learning_rate: float = 1e-5,
        representation_learning_rate: float = 3e-5,
        max_gradient_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if modality not in ("rgb", "rgbd"):
            raise ValueError(f"unsupported visual modality: {modality}")
        if (input_width is None) != (input_height is None):
            raise ValueError("visual proxy dimensions must be both set or both omitted")
        if input_width is not None and (input_width <= 0 or input_height <= 0):
            raise ValueError("visual proxy dimensions must be positive")
        if minibatch_size < 1:
            raise ValueError("minibatch_size must be positive")
        self.modality: VisualModality = modality
        self.input_width = None if input_width is None else int(input_width)
        self.input_height = None if input_height is None else int(input_height)
        self.minibatch_size = int(minibatch_size)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.backbone_learning_rate = float(backbone_learning_rate)
        self.representation_learning_rate = float(representation_learning_rate)
        self.max_gradient_norm = float(max_gradient_norm)
        if min(
            self.backbone_learning_rate,
            self.representation_learning_rate,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.max_gradient_norm,
        ) <= 0:
            raise ValueError("learning rates and max_gradient_norm must be positive")
        self.network: _VggRgbdActorCritic | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_shape: tuple[int, int, int] | None = None

    def _policy_state(self, observation: dict[str, Any]) -> PolicyState:
        dense_frame = observation.get("_dense_rgbd_frame")
        if dense_frame is not None:
            if not isinstance(dense_frame, CapturedDenseRgbdFrame):
                raise TypeError("_dense_rgbd_frame must be CapturedDenseRgbdFrame")
            planes = encode_dense_rgbd_policy_input(
                dense_frame,
                modality=self.modality,
                output_width=self.input_width,
                output_height=self.input_height,
            )
        else:
            tick = int(observation.get("_visual_tick", 0))
            frame_id = str(observation.get("_visual_frame_id", f"academy-frame:{tick}"))
            rgb, depth = academy_visual_pair(
                observation,
                frame_id=frame_id,
                server_tick=tick,
            )
            planes = encode_visual_policy_input(
                rgb,
                depth,
                modality=self.modality,
                output_width=self.input_width,
                output_height=self.input_height,
            )
        return PolicyState(
            grid=np.zeros((1, 1, 1), dtype=np.float32),
            joint_lattice=planes,
            auxiliary=np.zeros(0, dtype=np.float32),
            action_mask=np.asarray(
                observation.get("action_mask", np.zeros(1, dtype=np.int8)),
                dtype=np.int8,
            ),
        )

    def _ensure(self, state: PolicyState) -> None:
        shape = tuple(int(value) for value in state.joint_lattice.shape)
        if self.network is not None:
            if shape != self.input_shape:
                raise ValueError(
                    f"visual checkpoint expects {self.input_shape}, received {shape}"
                )
            return
        self.input_shape = shape
        self.network = _VggRgbdActorCritic(shape[0], int(state.auxiliary.size)).to(
            self.device
        )
        with torch.no_grad():
            sample_lattice = torch.as_tensor(
                state.joint_lattice[None], dtype=torch.float32, device=self.device
            )
            sample_auxiliary = torch.as_tensor(
                state.auxiliary[None], dtype=torch.float32, device=self.device
            )
            self.network.distributions_and_value(sample_lattice, sample_auxiliary)
        actor_heads = (
            self.network.longitude_location,
            self.network.longitude_log_concentration,
            self.network.latitude_location,
            self.network.latitude_log_scale,
            self.network.strafe_logits,
            self.network.move_logits,
            self.network.button_logits,
        )
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.features.parameters(),
                    "lr": self.backbone_learning_rate,
                    "name": "visual_backbone",
                },
                {
                    "params": [
                        *self.network.spatial_projection.parameters(),
                        *self.network.fusion.parameters(),
                    ],
                    "lr": self.representation_learning_rate,
                    "name": "shared_representation",
                },
                {
                    "params": [
                        parameter
                        for module in actor_heads
                        for parameter in module.parameters()
                    ],
                    "lr": self.actor_learning_rate,
                    "name": "actor_heads",
                },
                {
                    "params": self.network.value_head.parameters(),
                    "lr": self.critic_learning_rate,
                    "name": "value_head",
                },
            ]
        )
        self.actor = self.network
        self.critic = self.network
        self.actor_optimizer = self.optimizer
        self.critic_optimizer = self.optimizer

    def _distribution(
        self, lattice: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[VonMises, Normal, Categorical, Categorical, Bernoulli]:
        assert self.network is not None
        return self.network.distributions_and_value(lattice, auxiliary)[:5]

    def choose(
        self,
        observation: dict[str, Any],
        *,
        stochastic: bool,
        rng: np.random.Generator | None = None,
    ) -> HybridNavigationAction:
        del rng
        _, lattice, auxiliary = self._state_tensors(observation)
        longitude_d, latitude_d, strafe_d, move_d, button_d = self._distribution(
            lattice, auxiliary
        )
        with torch.no_grad():
            if stochastic:
                # PyTorch's VonMises sampler promotes to float64 internally,
                # which MPS cannot represent. Sampling this scalar on CPU keeps
                # the exact distribution while all learned computation stays on MPS.
                longitude = VonMises(
                    longitude_d.loc.cpu(), longitude_d.concentration.cpu()
                ).sample()[0].to(self.device)
                latitude = torch.tanh(latitude_d.sample())[0]
                strafe = int(strafe_d.sample().item()) - 1
                move = int(move_d.sample().item()) - 1
                buttons = button_d.sample()[0].to(dtype=torch.bool)
            else:
                longitude = longitude_d.loc[0]
                latitude = torch.tanh(latitude_d.loc)[0]
                strafe = int(torch.argmax(strafe_d.logits, dim=1).item()) - 1
                move = int(torch.argmax(move_d.logits, dim=1).item()) - 1
                buttons = button_d.probs[0] > 0.5
        longitude_value = float(longitude.detach().cpu().item())
        return HybridNavigationAction(
            delta_longitude=math.degrees(
                math.atan2(math.sin(longitude_value), math.cos(longitude_value))
            ),
            delta_latitude=float(latitude.item() * HYBRID_LATITUDE_LIMIT_DEGREES),
            strafe=strafe,
            move=move,
            jump=bool(buttons[0].item()),
            sprint=bool(buttons[1].item()),
            sneak=bool(buttons[2].item()),
            attack=bool(buttons[3].item()),
            use=bool(buttons[4].item()),
        )

    def state_value(self, observation: dict[str, Any]) -> float:
        _, lattice, auxiliary = self._state_tensors(observation)
        assert self.network is not None
        with torch.no_grad():
            return float(
                self.network.distributions_and_value(lattice, auxiliary)[5].item()
            )

    def _evaluate_actions(
        self,
        observations: list[dict[str, Any]],
        actions: list[HybridNavigationAction],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return joint log-probability, entropy and value for a batch."""
        states = [self._policy_state(observation) for observation in observations]
        self._ensure(states[0])
        lattice = torch.as_tensor(
            np.stack([state.joint_lattice for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        auxiliary = torch.as_tensor(
            np.stack([state.auxiliary for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        assert self.network is not None
        longitude_d, latitude_d, strafe_d, move_d, button_d, values = (
            self.network.distributions_and_value(lattice, auxiliary)
        )
        longitude = torch.as_tensor(
            [math.radians(action.delta_longitude) for action in actions],
            dtype=torch.float32,
            device=self.device,
        )
        normalized_latitude = torch.as_tensor(
            [
                action.delta_latitude / HYBRID_LATITUDE_LIMIT_DEGREES
                for action in actions
            ],
            dtype=torch.float32,
            device=self.device,
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_latitude = self._inverse_squash(normalized_latitude)
        latitude_correction = torch.log(1.0 - normalized_latitude.square() + 1e-6)
        strafe = torch.as_tensor(
            [action.strafe + 1 for action in actions],
            dtype=torch.int64,
            device=self.device,
        )
        move = torch.as_tensor(
            [action.move + 1 for action in actions],
            dtype=torch.int64,
            device=self.device,
        )
        buttons = torch.as_tensor(
            [
                [action.jump, action.sprint, action.sneak, action.attack, action.use]
                for action in actions
            ],
            dtype=torch.float32,
            device=self.device,
        )
        log_probability = longitude_d.log_prob(longitude)
        log_probability += latitude_d.log_prob(raw_latitude) - latitude_correction
        log_probability += strafe_d.log_prob(strafe)
        log_probability += move_d.log_prob(move)
        log_probability += button_d.log_prob(buttons).sum(dim=1)
        entropy = (
            self._von_mises_entropy(longitude_d)
            + latitude_d.entropy()
            + strafe_d.entropy()
            + move_d.entropy()
            + button_d.entropy().sum(dim=1)
        )
        return log_probability, entropy, values

    def _batch_loss(
        self,
        observations: list[dict[str, Any]],
        actions: list[HybridNavigationAction],
        targets: torch.Tensor,
        prescribed_advantages: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        log_probability, entropy, values = self._evaluate_actions(
            observations, actions
        )
        raw_advantage = targets - values.detach()
        advantage = (
            raw_advantage.clamp(-ADVANTAGE_LIMIT, ADVANTAGE_LIMIT)
            if prescribed_advantages is None
            else prescribed_advantages
        )
        actor_loss = -(
            log_probability * advantage + self.entropy_coefficient * entropy
        ).mean()
        critic_loss = F.smooth_l1_loss(values, targets)
        loss = actor_loss + self.value_loss_coefficient * critic_loss
        return loss, {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "advantage": advantage.mean(),
            "raw_advantage": raw_advantage.mean(),
            "state_value": values.mean(),
            "entropy": entropy.mean(),
            "td_target": targets.mean(),
        }

    def update(
        self,
        observation: dict[str, Any],
        action: HybridNavigationAction,
        reward: float,
        next_observation: dict[str, Any],
        next_action: HybridNavigationAction | None,
        *,
        done: bool,
    ) -> dict[str, float]:
        if not done and next_action is None:
            raise ValueError("on-policy actor-critic update requires next_action")
        future = 0.0 if done else self.state_value(next_observation)
        target = torch.as_tensor(
            [reward + self.discount * future],
            dtype=torch.float32,
            device=self.device,
        ).clamp(VALUE_RETURN_MIN, VALUE_RETURN_MAX)
        loss, metrics = self._batch_loss([observation], [action], target)
        assert self.network is not None and self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.network.parameters(), self.max_gradient_norm
        )
        self.optimizer.step()
        self.update_count += 1
        result = {
            name: float(value.detach().cpu().item())
            for name, value in metrics.items()
        }
        result["q_value"] = result["state_value"]
        result["actor_gradient_norm"] = float(gradient_norm.detach().cpu().item())
        result["critic_gradient_norm"] = result["actor_gradient_norm"]
        return result

    def update_episode(
        self,
        trajectory: list[tuple[dict[str, Any], HybridNavigationAction, float]],
    ) -> dict[str, float]:
        if not trajectory:
            raise ValueError("episode trajectory cannot be empty")
        rewards = [float(item[2]) for item in trajectory]
        returns: list[float] = []
        accumulated = 0.0
        for reward in reversed(rewards):
            accumulated = reward + self.discount * accumulated
            returns.append(accumulated)
        returns.reverse()
        return_tensor = torch.as_tensor(
            returns, dtype=torch.float32, device=self.device
        ).clamp(VALUE_RETURN_MIN, VALUE_RETURN_MAX)
        baseline_values: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(trajectory), self.minibatch_size):
                observations = [
                    item[0]
                    for item in trajectory[start : start + self.minibatch_size]
                ]
                states = [self._policy_state(observation) for observation in observations]
                self._ensure(states[0])
                lattice = torch.as_tensor(
                    np.stack([state.joint_lattice for state in states]),
                    dtype=torch.float32,
                    device=self.device,
                )
                auxiliary = torch.as_tensor(
                    np.stack([state.auxiliary for state in states]),
                    dtype=torch.float32,
                    device=self.device,
                )
                assert self.network is not None
                baseline_values.append(
                    self.network.distributions_and_value(lattice, auxiliary)[5]
                )
        baseline = torch.cat(baseline_values)
        raw_episode_advantage = return_tensor - baseline
        if len(trajectory) > 1:
            normalized_advantage = (
                raw_episode_advantage - raw_episode_advantage.mean()
            ) / raw_episode_advantage.std(unbiased=False).clamp_min(1e-6)
        else:
            normalized_advantage = raw_episode_advantage
        normalized_advantage = normalized_advantage.clamp(-5.0, 5.0)
        order = torch.randperm(len(trajectory)).tolist()
        totals: dict[str, float] = {}
        samples = 0
        gradient_norm = 0.0
        for start in range(0, len(order), self.minibatch_size):
            indices = order[start : start + self.minibatch_size]
            observations = [trajectory[index][0] for index in indices]
            actions = [trajectory[index][1] for index in indices]
            targets = return_tensor[indices]
            loss, metrics = self._batch_loss(
                observations,
                actions,
                targets,
                prescribed_advantages=normalized_advantage[indices],
            )
            assert self.network is not None and self.optimizer is not None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_gradient_norm
            )
            self.optimizer.step()
            count = len(indices)
            samples += count
            gradient_norm += float(norm.detach().cpu().item()) * count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().cpu().item()
                ) * count
        self.update_count += len(trajectory)
        result = {name: value / samples for name, value in totals.items()}
        result["q_value"] = result["state_value"]
        result["actor_gradient_norm"] = gradient_norm / samples
        result["critic_gradient_norm"] = gradient_norm / samples
        return result

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        if self.network is None or self.optimizer is None or self.input_shape is None:
            raise RuntimeError("cannot save an uninitialized visual actor-critic")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        payload = {
            "interface": self.interface,
            "architecture": self.architecture,
            "modality": self.modality,
            "input_shape": self.input_shape,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "minibatch_size": self.minibatch_size,
            "value_loss_coefficient": self.value_loss_coefficient,
            "backbone_learning_rate": self.backbone_learning_rate,
            "representation_learning_rate": self.representation_learning_rate,
            "max_gradient_norm": self.max_gradient_norm,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "discount": self.discount,
            "entropy_coefficient": self.entropy_coefficient,
            "seed": self.seed,
            "update_count": self.update_count,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "metadata_json": json.dumps(metadata, sort_keys=True),
            **self._checkpoint_extra_payload(),
        }
        torch.save(payload, temporary)
        temporary.replace(target)
        return target

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _checkpoint_constructor_kwargs(cls, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        load_optimizer: bool = True,
    ) -> tuple["VisualHybridActorCriticAgent", dict[str, Any]]:
        # Loading a large checkpoint directly onto MPS can crash inside the
        # native backend.  Deserialize on CPU and let load_state_dict perform
        # the controlled device transfer instead.
        payload = torch.load(
            Path(path).expanduser(), map_location="cpu", weights_only=True
        )
        if payload.get("interface") != cls.interface:
            raise ValueError("not a VGG RGB-D actor-critic checkpoint")
        agent = cls(
            modality=str(payload["modality"]),
            input_width=payload.get("input_width"),
            input_height=payload.get("input_height"),
            minibatch_size=int(payload.get("minibatch_size", 8)),
            value_loss_coefficient=float(payload.get("value_loss_coefficient", 0.5)),
            backbone_learning_rate=float(payload.get("backbone_learning_rate", 1e-5)),
            representation_learning_rate=float(
                payload.get("representation_learning_rate", 3e-5)
            ),
            max_gradient_norm=float(payload.get("max_gradient_norm", 1.0)),
            actor_learning_rate=float(payload["actor_learning_rate"]),
            critic_learning_rate=float(payload["critic_learning_rate"]),
            discount=float(payload["discount"]),
            entropy_coefficient=float(payload["entropy_coefficient"]),
            seed=int(payload["seed"]),
            device=device,
            **cls._checkpoint_constructor_kwargs(payload),
        )
        shape = tuple(int(value) for value in payload["input_shape"])
        dummy = PolicyState(
            grid=np.zeros((1, 1, 1), dtype=np.float32),
            joint_lattice=np.zeros(shape, dtype=np.float32),
            auxiliary=np.zeros(0, dtype=np.float32),
            action_mask=np.zeros(1, dtype=np.int8),
        )
        agent._ensure(dummy)
        assert agent.network is not None and agent.optimizer is not None
        agent.network.load_state_dict(payload["network_state"])
        if load_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer_state"])
            for state in agent.optimizer.state.values():
                for name, value in state.items():
                    if torch.is_tensor(value):
                        state[name] = value.to(agent.device)
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        agent.update_count = int(payload.get("update_count", 0))
        return agent, json.loads(str(payload["metadata_json"]))
