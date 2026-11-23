"""On-policy hybrid actor-critic for target navigation.

The actor chooses continuous relative camera displacement independently from
W/S, A/D and button factors.  The critic is a bounded state-value baseline. Keeping the value
inside the physically possible episodic cost range prevents bootstrapping from
inventing returns that the bounded time reward can never produce.

PyTorch is intentionally an optional training dependency.  Importing the base
Minecraft Gym package does not import this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Categorical, Normal, VonMises
    from torch.nn import functional as F
except ImportError as error:  # pragma: no cover - depends on optional extra
    raise RuntimeError(
        "hybrid actor-critic training requires the optional 'learning' dependencies"
    ) from error

from .policy import (
    HYBRID_LATITUDE_LIMIT_DEGREES,
    HybridNavigationAction,
    PolicyState,
    WORLD_MAP_WINDOW_SIZE,
    encode_policy_state,
)


ACTION_CONTRACT = "minecraft-control-camera-keyboard/v3"
OBSERVATION_CONTRACT = "minecraft-academy-world-map-window-time/v4"
VALUE_RETURN_MIN = -40.0
VALUE_RETURN_MAX = 20.0
ADVANTAGE_LIMIT = 10.0


class _SpatialEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        auxiliary_size: int,
        hidden: int = 96,
        input_stride: int = 1,
        input_pool: int = 1,
    ) -> None:
        super().__init__()
        if input_pool < 1:
            raise ValueError("input_pool must be positive")
        self.input_pool = nn.AvgPool2d(
            kernel_size=input_pool, stride=input_pool
        )
        self.convolution = nn.Sequential(
            nn.Conv2d(
                channels, 16, kernel_size=3, stride=input_stride, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.projection = nn.Sequential(
            nn.Linear(24 * 5 * 5 + auxiliary_size, hidden),
            nn.ReLU(),
        )

    def forward(self, lattice: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        spatial = self.convolution(self.input_pool(lattice)).flatten(start_dim=1)
        return self.projection(torch.cat((spatial, auxiliary), dim=1))


class _HybridActor(nn.Module):
    def __init__(
        self,
        channels: int,
        auxiliary_size: int,
        *,
        input_stride: int = 1,
        input_pool: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = _SpatialEncoder(
            channels,
            auxiliary_size,
            input_stride=input_stride,
            input_pool=input_pool,
        )
        self.longitude_location = nn.Linear(96, 1)
        self.longitude_log_concentration = nn.Linear(96, 1)
        self.latitude_location = nn.Linear(96, 1)
        self.latitude_log_scale = nn.Linear(96, 1)
        self.strafe_logits = nn.Linear(96, 3)
        self.move_logits = nn.Linear(96, 3)
        self.button_logits = nn.Linear(96, 5)
        nn.init.zeros_(self.longitude_location.bias)
        nn.init.constant_(self.longitude_log_concentration.bias, -0.3)
        nn.init.zeros_(self.latitude_location.bias)
        nn.init.constant_(self.latitude_log_scale.bias, -0.7)

    def forward(
        self, lattice: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        features = self.encoder(lattice, auxiliary)
        longitude_location = self.longitude_location(features)
        longitude_concentration = F.softplus(
            self.longitude_log_concentration(features).clamp(-4.0, 4.0)
        ) + 0.05
        latitude_location = self.latitude_location(features).clamp(-3.0, 3.0)
        latitude_log_scale = self.latitude_log_scale(features).clamp(-2.5, 0.35)
        return (
            longitude_location,
            longitude_concentration,
            latitude_location,
            latitude_log_scale,
            self.strafe_logits(features),
            self.move_logits(features),
            self.button_logits(features),
        )


class _StateValueCritic(nn.Module):
    def __init__(
        self,
        channels: int,
        auxiliary_size: int,
        *,
        input_stride: int = 1,
        input_pool: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = _SpatialEncoder(
            channels,
            auxiliary_size,
            input_stride=input_stride,
            input_pool=input_pool,
        )
        self.value = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        # A zero raw output starts at the midpoint (-10) of the bounded
        # terminal-reward interval and avoids arbitrary bootstrap targets.
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(
        self,
        lattice: torch.Tensor,
        auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encoder(lattice, auxiliary)
        raw = self.value(state).squeeze(1)
        return VALUE_RETURN_MIN + (
            VALUE_RETURN_MAX - VALUE_RETURN_MIN
        ) * torch.sigmoid(raw)


class HybridActorCriticAgent:
    """Stochastic actor with a bounded on-policy state-value baseline."""

    interface = "minecraft-academy-hybrid-actor-critic/v4"
    action_contract = ACTION_CONTRACT
    observation_contract = OBSERVATION_CONTRACT

    def __init__(
        self,
        *,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        discount: float = 1.0,
        entropy_coefficient: float = 0.01,
        seed: int = 0,
        device: str = "cpu",
        encoder_stride: int = 1,
        encoder_pool: int = 1,
    ) -> None:
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.discount = float(discount)
        self.entropy_coefficient = float(entropy_coefficient)
        self.seed = int(seed)
        self.device = torch.device(device)
        if int(encoder_stride) < 1:
            raise ValueError("encoder_stride must be positive")
        self.encoder_stride = int(encoder_stride)
        if int(encoder_pool) < 1:
            raise ValueError("encoder_pool must be positive")
        self.encoder_pool = int(encoder_pool)
        torch.manual_seed(self.seed)
        self.actor: _HybridActor | None = None
        self.critic: _StateValueCritic | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.update_count = 0

    def _ensure(self, state: PolicyState) -> None:
        if self.actor is not None:
            return
        channels = int(state.joint_lattice.shape[0])
        auxiliary_size = int(state.auxiliary.size)
        self.actor = _HybridActor(
            channels,
            auxiliary_size,
            input_stride=self.encoder_stride,
            input_pool=self.encoder_pool,
        ).to(self.device)
        self.critic = _StateValueCritic(
            channels,
            auxiliary_size,
            input_stride=self.encoder_stride,
            input_pool=self.encoder_pool,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_learning_rate
        )

    def begin_episode(self) -> None:
        """Kept for the common academy-agent lifecycle."""

    def _state_tensors(
        self, observation: dict[str, Any]
    ) -> tuple[PolicyState, torch.Tensor, torch.Tensor]:
        state = self._policy_state(observation)
        self._ensure(state)
        lattice = torch.as_tensor(
            state.joint_lattice[None], dtype=torch.float32, device=self.device
        )
        auxiliary = torch.as_tensor(
            state.auxiliary[None], dtype=torch.float32, device=self.device
        )
        return state, lattice, auxiliary

    def _policy_state(self, observation: dict[str, Any]) -> PolicyState:
        """Encode the public observation consumed by this policy.

        Visual ablations override this boundary while retaining the exact
        actor, critic, action distribution and optimization code. Keeping the
        intervention here prevents the visual experiment from becoming a
        second, silently divergent actor-critic implementation.
        """

        return encode_policy_state(observation, include_elapsed=True)

    @staticmethod
    def _inverse_squash(value: torch.Tensor) -> torch.Tensor:
        clipped = value.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

    @staticmethod
    def _von_mises_entropy(distribution: VonMises) -> torch.Tensor:
        """Differentiable entropy unavailable on older PyTorch releases."""

        concentration = distribution.concentration
        i0 = torch.special.i0(concentration).clamp_min(1e-8)
        i1 = torch.special.i1(concentration)
        return math.log(2.0 * math.pi) + torch.log(i0) - concentration * i1 / i0

    def _distribution(
        self, lattice: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[VonMises, Normal, Categorical, Categorical, Bernoulli]:
        assert self.actor is not None
        (
            longitude_location,
            longitude_concentration,
            latitude_location,
            latitude_log_scale,
            strafe_logits,
            move_logits,
            button_logits,
        ) = self.actor(lattice, auxiliary)
        return (
            VonMises(longitude_location.squeeze(1), longitude_concentration.squeeze(1)),
            Normal(latitude_location.squeeze(1), latitude_log_scale.squeeze(1).exp()),
            Categorical(logits=strafe_logits),
            Categorical(logits=move_logits),
            Bernoulli(logits=button_logits),
        )

    def _log_probability(
        self,
        longitude_distribution: VonMises,
        latitude_distribution: Normal,
        strafe_distribution: Categorical,
        move_distribution: Categorical,
        button_distribution: Bernoulli,
        action: HybridNavigationAction,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        longitude = torch.as_tensor(
            [math.radians(action.delta_longitude)],
            dtype=torch.float32,
            device=self.device,
        )
        normalized_latitude = torch.as_tensor(
            [action.delta_latitude / HYBRID_LATITUDE_LIMIT_DEGREES],
            dtype=torch.float32,
            device=self.device,
        )
        raw_latitude = self._inverse_squash(normalized_latitude)
        latitude_correction = torch.log(1.0 - normalized_latitude.square() + 1e-6)
        camera_log_probability = longitude_distribution.log_prob(longitude)
        camera_log_probability += latitude_distribution.log_prob(
            raw_latitude
        ) - latitude_correction
        strafe = torch.as_tensor(
            [action.strafe + 1], dtype=torch.int64, device=self.device
        )
        move = torch.as_tensor(
            [action.move + 1], dtype=torch.int64, device=self.device
        )
        buttons = torch.as_tensor(
            [[action.jump, action.sprint, action.sneak, action.attack, action.use]],
            dtype=torch.float32,
            device=self.device,
        )
        return (
            camera_log_probability
            + strafe_distribution.log_prob(strafe)
            + move_distribution.log_prob(move)
            + button_distribution.log_prob(buttons).sum(dim=1),
            self._von_mises_entropy(longitude_distribution)
            + latitude_distribution.entropy()
            + strafe_distribution.entropy()
            + move_distribution.entropy()
            + button_distribution.entropy().sum(dim=1),
        )

    def choose(
        self,
        observation: dict[str, Any],
        *,
        stochastic: bool,
        rng: np.random.Generator | None = None,
    ) -> HybridNavigationAction:
        del rng  # torch owns the checkpointed policy RNG.
        _, lattice, auxiliary = self._state_tensors(observation)
        (
            longitude_distribution,
            latitude_distribution,
            strafe_distribution,
            move_distribution,
            button_distribution,
        ) = self._distribution(lattice, auxiliary)
        with torch.no_grad():
            if stochastic:
                longitude = longitude_distribution.sample()[0]
                latitude = torch.tanh(latitude_distribution.sample())[0]
                strafe = int(strafe_distribution.sample().item()) - 1
                move = int(move_distribution.sample().item()) - 1
                buttons = button_distribution.sample()[0].to(dtype=torch.bool)
            else:
                longitude = longitude_distribution.loc[0]
                latitude = torch.tanh(latitude_distribution.loc)[0]
                strafe = int(torch.argmax(strafe_distribution.logits, dim=1).item()) - 1
                move = int(torch.argmax(move_distribution.logits, dim=1).item()) - 1
                buttons = button_distribution.probs[0] > 0.5
        longitude_value = float(longitude.detach().cpu().item())
        longitude_degrees = math.degrees(
            math.atan2(math.sin(longitude_value), math.cos(longitude_value))
        )
        return HybridNavigationAction(
            delta_longitude=longitude_degrees,
            delta_latitude=float(latitude.item() * HYBRID_LATITUDE_LIMIT_DEGREES),
            strafe=strafe,
            move=move,
            jump=bool(buttons[0].item()),
            sprint=bool(buttons[1].item()),
            sneak=bool(buttons[2].item()),
            attack=bool(buttons[3].item()),
            use=bool(buttons[4].item()),
        )

    def q_value(
        self, observation: dict[str, Any], action: HybridNavigationAction
    ) -> float:
        """Backward-compatible alias for the state-value diagnostic.

        The v2 critic is V(s), so ``action`` intentionally does not affect this
        value.  The method remains available for older Observer consumers.
        """
        del action
        return self.state_value(observation)

    def state_value(self, observation: dict[str, Any]) -> float:
        _, lattice, auxiliary = self._state_tensors(observation)
        assert self.critic is not None
        with torch.no_grad():
            return float(self.critic(lattice, auxiliary).item())

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
        _, lattice, auxiliary = self._state_tensors(observation)
        assert self.actor is not None and self.critic is not None
        assert self.actor_optimizer is not None and self.critic_optimizer is not None
        current_value = self.critic(lattice, auxiliary)
        with torch.no_grad():
            future_value = torch.zeros(1, device=self.device)
            if not done:
                if next_action is None:
                    raise ValueError("on-policy actor-critic update requires next_action")
                _, next_lattice, next_auxiliary = self._state_tensors(next_observation)
                future_value = self.critic(next_lattice, next_auxiliary)
            target = torch.as_tensor([float(reward)], device=self.device)
            target = (target + self.discount * future_value).clamp(
                VALUE_RETURN_MIN, VALUE_RETURN_MAX
            )
        raw_advantage = (target - current_value).detach()
        advantage = raw_advantage.clamp(-ADVANTAGE_LIMIT, ADVANTAGE_LIMIT)

        critic_loss = F.smooth_l1_loss(current_value, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        (
            longitude_distribution,
            latitude_distribution,
            strafe_distribution,
            move_distribution,
            button_distribution,
        ) = self._distribution(lattice, auxiliary)
        log_probability, entropy = self._log_probability(
            longitude_distribution,
            latitude_distribution,
            strafe_distribution,
            move_distribution,
            button_distribution,
            action,
        )
        actor_loss = -(
            log_probability * advantage
            + self.entropy_coefficient * entropy
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        self.update_count += 1
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "advantage": float(advantage.detach().cpu().item()),
            "raw_advantage": float(raw_advantage.detach().cpu().item()),
            "state_value": float(current_value.detach().cpu().item()),
            # Compatibility name retained until all dataset consumers migrate.
            "q_value": float(current_value.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "actor_gradient_norm": float(actor_gradient_norm.detach().cpu().item()),
            "critic_gradient_norm": float(critic_gradient_norm.detach().cpu().item()),
            "td_target": float(target.detach().cpu().item()),
        }

    def update_episode(
        self,
        trajectory: list[
            tuple[dict[str, Any], HybridNavigationAction, float]
        ],
    ) -> dict[str, float]:
        """Update actor and critic from observed on-policy episodic returns.

        The academy reward is an undiscounted bounded time cost.  Using the
        realized return avoids the one-step bootstrap feedback loop that can
        drive a nonlinear critic toward an arbitrary fixed point even while
        every environment return is bounded.
        """

        if not trajectory:
            raise ValueError("episode trajectory cannot be empty")
        states = [self._policy_state(item[0]) for item in trajectory]
        self._ensure(states[0])
        assert self.actor is not None and self.critic is not None
        assert self.actor_optimizer is not None and self.critic_optimizer is not None
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
        actions = [item[1] for item in trajectory]
        rewards = [float(item[2]) for item in trajectory]
        realized_returns: list[float] = []
        accumulated = 0.0
        for reward in reversed(rewards):
            accumulated = reward + self.discount * accumulated
            realized_returns.append(accumulated)
        realized_returns.reverse()
        target = torch.as_tensor(
            realized_returns, dtype=torch.float32, device=self.device
        ).clamp(VALUE_RETURN_MIN, VALUE_RETURN_MAX)

        current_value = self.critic(lattice, auxiliary)
        critic_loss = F.smooth_l1_loss(current_value, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        (
            longitude_distribution,
            latitude_distribution,
            strafe_distribution,
            move_distribution,
            button_distribution,
        ) = self._distribution(lattice, auxiliary)
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
        )
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
        log_probability = longitude_distribution.log_prob(longitude)
        log_probability += latitude_distribution.log_prob(raw_latitude) - latitude_correction
        log_probability += strafe_distribution.log_prob(strafe)
        log_probability += move_distribution.log_prob(move)
        log_probability += button_distribution.log_prob(buttons).sum(dim=1)
        entropy = (
            self._von_mises_entropy(longitude_distribution)
            + latitude_distribution.entropy()
            + strafe_distribution.entropy()
            + move_distribution.entropy()
            + button_distribution.entropy().sum(dim=1)
        )
        raw_advantage = (target - current_value.detach())
        advantage = raw_advantage.clamp(-ADVANTAGE_LIMIT, ADVANTAGE_LIMIT)
        actor_loss = -(
            log_probability * advantage + self.entropy_coefficient * entropy
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        self.update_count += len(trajectory)
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "advantage": float(advantage.detach().mean().cpu().item()),
            "raw_advantage": float(raw_advantage.detach().mean().cpu().item()),
            "state_value": float(current_value.detach().mean().cpu().item()),
            "q_value": float(current_value.detach().mean().cpu().item()),
            "entropy": float(entropy.detach().mean().cpu().item()),
            "actor_gradient_norm": float(actor_gradient_norm.detach().cpu().item()),
            "critic_gradient_norm": float(critic_gradient_norm.detach().cpu().item()),
            "td_target": float(target.detach().mean().cpu().item()),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        if self.actor is None or self.critic is None:
            raise RuntimeError("cannot save an uninitialized actor-critic")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        payload = {
            "interface": self.interface,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "discount": self.discount,
            "entropy_coefficient": self.entropy_coefficient,
            "seed": self.seed,
            "encoder_stride": self.encoder_stride,
            "encoder_pool": self.encoder_pool,
            "update_count": self.update_count,
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }
        torch.save(payload, temporary)
        temporary.replace(target)
        return target

    @classmethod
    def load(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> tuple["HybridActorCriticAgent", dict[str, Any]]:
        payload = torch.load(
            Path(path).expanduser(), map_location=device, weights_only=True
        )
        if payload.get("interface") != cls.interface:
            raise ValueError("not a hybrid actor-critic academy checkpoint")
        agent = cls(
            actor_learning_rate=float(payload["actor_learning_rate"]),
            critic_learning_rate=float(payload["critic_learning_rate"]),
            discount=float(payload["discount"]),
            entropy_coefficient=float(payload["entropy_coefficient"]),
            seed=int(payload["seed"]),
            device=device,
            encoder_stride=int(payload.get("encoder_stride", 1)),
            encoder_pool=int(payload.get("encoder_pool", 1)),
        )
        actor_state = payload["actor_state"]
        first_weight = actor_state["encoder.convolution.0.weight"]
        channels = int(first_weight.shape[1])
        projection_weight = actor_state["encoder.projection.0.weight"]
        auxiliary_size = int(projection_weight.shape[1] - 24 * 5 * 5)
        dummy = PolicyState(
            grid=np.zeros((1, 1, 1), dtype=np.float32),
            joint_lattice=np.zeros(
                (channels, WORLD_MAP_WINDOW_SIZE, WORLD_MAP_WINDOW_SIZE),
                dtype=np.float32,
            ),
            auxiliary=np.zeros(auxiliary_size, dtype=np.float32),
            action_mask=np.zeros(1, dtype=np.int8),
        )
        agent._ensure(dummy)
        assert agent.actor is not None and agent.critic is not None
        agent.actor.load_state_dict(actor_state)
        agent.critic.load_state_dict(payload["critic_state"])
        if payload.get("actor_optimizer_state") is not None:
            assert agent.actor_optimizer is not None
            agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
        if payload.get("critic_optimizer_state") is not None:
            assert agent.critic_optimizer is not None
            agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        agent.update_count = int(payload.get("update_count", 0))
        metadata = json.loads(str(payload["metadata_json"]))
        return agent, metadata
