"""Function-approximation policies for the Navigation Academy.

Both agents consume the same ``academy-world-map-window/v1`` state and 12-action
body contract.  No evaluator information is accepted by these APIs.  Loaders
retain read compatibility with the earlier single-grid checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .policy import (
    ACTION_COUNT,
    CONTROL_BRANCH_SIZES,
    CONTROL_LOGIT_COUNT,
    JOINT_CONTROL_ACTION_COUNT,
    PolicyState,
    choose_epsilon_greedy,
    encode_policy_state,
    epsilon_greedy_probabilities,
)


class LinearSarsaLambdaAgent:
    """On-policy linear SARSA with replacing eligibility traces."""

    interface = "minecraft-academy-linear-sarsa-lambda/v1"

    def __init__(
        self,
        *,
        learning_rate: float = 0.012,
        discount: float = 0.98,
        trace_decay: float = 0.85,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.trace_decay = trace_decay
        self.weights: np.ndarray | None = None
        self.traces: np.ndarray | None = None

    def _ensure(self, state: PolicyState) -> None:
        if self.weights is None:
            self.weights = np.zeros((ACTION_COUNT, state.vector().size), dtype=np.float32)
            self.traces = np.zeros_like(self.weights)

    def _features(self, state: PolicyState) -> np.ndarray:
        modern = state.vector()
        if self.weights is None or self.weights.shape[1] == modern.size:
            return modern
        legacy = state.legacy_vector()
        if self.weights.shape[1] == legacy.size:
            return legacy
        raise ValueError("SARSA checkpoint observation contract is incompatible")

    def begin_episode(self) -> None:
        if self.traces is not None:
            self.traces.fill(0.0)

    def values(self, observation: dict[str, Any]) -> np.ndarray:
        state = encode_policy_state(observation)
        self._ensure(state)
        assert self.weights is not None
        return self.weights @ self._features(state)

    def choose(
        self,
        observation: dict[str, Any],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        state = encode_policy_state(observation)
        return choose_epsilon_greedy(
            self.values(observation), state.action_mask, epsilon=epsilon, rng=rng
        )

    def update(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        next_action: int | None,
        *,
        done: bool,
    ) -> float:
        state = encode_policy_state(observation)
        self._ensure(state)
        assert self.weights is not None and self.traces is not None
        features = self._features(state)
        current = float(self.weights[action] @ features)
        future = 0.0 if done else float(self.values(next_observation)[int(next_action)])
        delta = float(reward + self.discount * future - current)
        self.traces *= self.discount * self.trace_decay
        self.traces[action] = features
        self.weights += self.learning_rate * np.clip(delta, -10.0, 10.0) * self.traces
        np.clip(self.weights, -25.0, 25.0, out=self.weights)
        return delta

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        if self.weights is None:
            raise RuntimeError("cannot save an uninitialized SARSA agent")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                interface=self.interface,
                weights=self.weights,
                learning_rate=self.learning_rate,
                discount=self.discount,
                trace_decay=self.trace_decay,
                metadata=json.dumps(metadata, sort_keys=True),
            )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinearSarsaLambdaAgent", dict[str, Any]]:
        with np.load(Path(path).expanduser(), allow_pickle=False) as payload:
            if str(payload["interface"]) != cls.interface:
                raise ValueError("not a linear SARSA(lambda) academy checkpoint")
            agent = cls(
                learning_rate=float(payload["learning_rate"]),
                discount=float(payload["discount"]),
                trace_decay=float(payload["trace_decay"]),
            )
            agent.weights = np.asarray(payload["weights"], dtype=np.float32)
            agent.traces = np.zeros_like(agent.weights)
            metadata = json.loads(str(payload["metadata"]))
        return agent, metadata


class TinyConvQNetwork:
    """One trainable convolution over the fused metric-window/view lattice."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        auxiliary_size: int,
        *,
        seed: int,
        filters: int = 6,
        hidden: int = 48,
        learning_rate: float = 7e-4,
        legacy_grid: bool = False,
        output_size: int = ACTION_COUNT,
    ) -> None:
        channels, height, width = grid_shape
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        self.grid_shape = grid_shape
        self.legacy_grid = legacy_grid
        self.output_size = int(output_size)
        self.step = 0
        self.parameters = {
            "conv_w": rng.normal(0.0, np.sqrt(2 / (channels * 9)), (filters, channels, 3, 3)).astype(np.float32),
            "conv_b": np.zeros(filters, dtype=np.float32),
            "dense_w": rng.normal(0.0, np.sqrt(2 / (filters * height * width + auxiliary_size)), (filters * height * width + auxiliary_size, hidden)).astype(np.float32),
            "dense_b": np.zeros(hidden, dtype=np.float32),
            "out_w": rng.normal(0.0, np.sqrt(2 / hidden), (hidden, self.output_size)).astype(np.float32),
            "out_b": np.zeros(self.output_size, dtype=np.float32),
        }
        self.first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.parameters.items()}

    @staticmethod
    def _windows(grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, ((0, 0), (1, 1), (1, 1)))
        return np.lib.stride_tricks.sliding_window_view(
            padded, (3, 3), axis=(1, 2)
        )

    def forward(self, state: PolicyState) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        grid = state.grid if self.legacy_grid else state.joint_lattice
        if grid.shape != self.grid_shape:
            raise ValueError(
                f"policy lattice shape {grid.shape} does not match checkpoint {self.grid_shape}"
            )
        windows = self._windows(grid)
        z = np.einsum("fckl,chwkl->fhw", self.parameters["conv_w"], windows)
        z += self.parameters["conv_b"][:, None, None]
        activation = np.maximum(z, 0.0)
        features = np.concatenate((activation.reshape(-1), state.auxiliary))
        hidden_pre = features @ self.parameters["dense_w"] + self.parameters["dense_b"]
        hidden = np.maximum(hidden_pre, 0.0)
        q = hidden @ self.parameters["out_w"] + self.parameters["out_b"]
        return q.astype(np.float32), {
            "windows": windows,
            "z": z,
            "features": features,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
        }

    def update(
        self,
        state: PolicyState,
        action: int,
        target: float,
        *,
        forward_result: tuple[np.ndarray, dict[str, np.ndarray]] | None = None,
    ) -> float:
        q, cache = forward_result if forward_result is not None else self.forward(state)
        error = float(q[action] - target)
        derivative = float(np.clip(error, -1.0, 1.0))
        dq = np.zeros(self.output_size, dtype=np.float32)
        dq[action] = derivative
        self._apply_gradients(cache, dq)
        return error

    def update_branches(
        self,
        state: PolicyState,
        branch_actions: np.ndarray,
        target: float,
        *,
        forward_result: tuple[np.ndarray, dict[str, np.ndarray]] | None = None,
    ) -> float:
        q, cache = forward_result if forward_result is not None else self.forward(state)
        offsets = np.cumsum((0, *CONTROL_BRANCH_SIZES))
        selected = np.asarray(
            [offsets[index] + int(action) for index, action in enumerate(branch_actions)],
            dtype=np.int64,
        )
        current = float(np.mean(q[selected]))
        error = current - target
        derivative = float(np.clip(error, -1.0, 1.0)) / len(selected)
        dq = np.zeros(self.output_size, dtype=np.float32)
        dq[selected] = derivative
        self._apply_gradients(cache, dq)
        return float(error)

    def _apply_gradients(
        self, cache: dict[str, np.ndarray], dq: np.ndarray
    ) -> None:
        gradients: dict[str, np.ndarray] = {}
        gradients["out_w"] = np.outer(cache["hidden"], dq)
        gradients["out_b"] = dq
        dh = (self.parameters["out_w"] @ dq) * (cache["hidden_pre"] > 0)
        gradients["dense_w"] = np.outer(cache["features"], dh)
        gradients["dense_b"] = dh
        dfeatures = self.parameters["dense_w"] @ dh
        dz = dfeatures[: cache["z"].size].reshape(cache["z"].shape)
        dz *= cache["z"] > 0
        gradients["conv_w"] = np.einsum("fhw,chwkl->fckl", dz, cache["windows"])
        gradients["conv_b"] = dz.sum(axis=(1, 2))
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for name, parameter in self.parameters.items():
            gradient = np.clip(gradients[name], -5.0, 5.0)
            self.first[name] = beta1 * self.first[name] + (1 - beta1) * gradient
            self.second[name] = beta2 * self.second[name] + (1 - beta2) * gradient * gradient
            first = self.first[name] / (1 - beta1**self.step)
            second = self.second[name] / (1 - beta2**self.step)
            parameter -= self.learning_rate * first / (np.sqrt(second) + 1e-8)


class ExpectedSarsaCnnAgent:
    """Expected SARSA using a small trainable convolutional Q estimator."""

    interface = "minecraft-academy-expected-sarsa-cnn/v1"

    def __init__(self, *, discount: float = 0.98, seed: int = 0) -> None:
        self.discount = discount
        self.seed = seed
        self.network: TinyConvQNetwork | None = None
        self._cache: dict[
            int,
            tuple[
                dict[str, Any],
                tuple[np.ndarray, dict[str, np.ndarray]],
            ],
        ] = {}

    def _ensure(self, state: PolicyState) -> None:
        if self.network is None:
            self.network = TinyConvQNetwork(
                state.joint_lattice.shape, state.auxiliary.size, seed=self.seed
            )

    def values(self, observation: dict[str, Any]) -> np.ndarray:
        state = encode_policy_state(observation)
        self._ensure(state)
        assert self.network is not None
        key = id(observation)
        cached = self._cache.get(key)
        if cached is None or cached[0] is not observation:
            if len(self._cache) > 3:
                self._cache.clear()
            cached = (observation, self.network.forward(state))
            self._cache[key] = cached
        return cached[1][0]

    def begin_episode(self) -> None:
        """Discard forward-pass caches at an environment reset boundary."""

        self._cache.clear()

    def choose(
        self,
        observation: dict[str, Any],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        state = encode_policy_state(observation)
        return choose_epsilon_greedy(
            self.values(observation), state.action_mask, epsilon=epsilon, rng=rng
        )

    def update(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        *,
        done: bool,
        epsilon: float,
    ) -> float:
        state = encode_policy_state(observation)
        self._ensure(state)
        assert self.network is not None
        expected = 0.0
        if not done:
            next_state = encode_policy_state(next_observation)
            next_values = self.values(next_observation)
            probabilities = epsilon_greedy_probabilities(
                next_values, next_state.action_mask, epsilon
            )
            expected = float(probabilities @ next_values)
        target = float(reward + self.discount * expected)
        cached = self._cache.get(id(observation))
        current = cached[1] if cached is not None and cached[0] is observation else None
        error = self.network.update(state, action, target, forward_result=current)
        self._cache.clear()
        return error

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> Path:
        if self.network is None:
            raise RuntimeError("cannot save an uninitialized CNN agent")
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                interface=self.interface,
                discount=self.discount,
                seed=self.seed,
                network_step=self.network.step,
                network_grid_shape=np.asarray(self.network.grid_shape, dtype=np.int32),
                network_legacy_grid=int(self.network.legacy_grid),
                metadata=json.dumps(metadata, sort_keys=True),
                **self.network.parameters,
            )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> tuple["ExpectedSarsaCnnAgent", dict[str, Any]]:
        with np.load(Path(path).expanduser(), allow_pickle=False) as payload:
            if str(payload["interface"]) != cls.interface:
                raise ValueError("not an Expected SARSA CNN academy checkpoint")
            agent = cls(discount=float(payload["discount"]), seed=int(payload["seed"]))
            conv = np.asarray(payload["conv_w"], dtype=np.float32)
            dense = np.asarray(payload["dense_w"], dtype=np.float32)
            if "network_grid_shape" in payload:
                shape = tuple(int(value) for value in payload["network_grid_shape"])
                legacy_grid = bool(int(payload["network_legacy_grid"]))
            else:
                shape = (conv.shape[1], 14, 9)
                legacy_grid = True
            auxiliary_size = dense.shape[0] - conv.shape[0] * shape[1] * shape[2]
            network = TinyConvQNetwork(
                shape,
                auxiliary_size,
                seed=agent.seed,
                filters=conv.shape[0],
                hidden=dense.shape[1],
                legacy_grid=legacy_grid,
                output_size=int(np.asarray(payload["out_b"]).size),
            )
            for name in network.parameters:
                network.parameters[name] = np.asarray(payload[name], dtype=np.float32)
            network.step = int(payload["network_step"])
            agent.network = network
            metadata = json.loads(str(payload["metadata"]))
        return agent, metadata


class BranchingExpectedSarsaCnnAgent(ExpectedSarsaCnnAgent):
    """Expected SARSA over independent Minecraft keyboard/camera heads."""

    interface = "minecraft-academy-branching-expected-sarsa-cnn/v1"

    def _ensure(self, state: PolicyState) -> None:
        if self.network is None:
            self.network = TinyConvQNetwork(
                state.joint_lattice.shape,
                state.auxiliary.size,
                seed=self.seed,
                output_size=CONTROL_LOGIT_COUNT,
            )

    @staticmethod
    def _segments(values: np.ndarray) -> list[np.ndarray]:
        offsets = np.cumsum((0, *CONTROL_BRANCH_SIZES))
        return [
            values[offsets[index] : offsets[index + 1]]
            for index in range(len(CONTROL_BRANCH_SIZES))
        ]

    def choose(
        self,
        observation: dict[str, Any],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        branches: list[int] = []
        explore = rng.random() < epsilon
        for values in self._segments(self.values(observation)):
            if explore:
                branches.append(int(rng.integers(values.size)))
            else:
                maximum = float(values.max())
                candidates = np.flatnonzero(np.isclose(values, maximum))
                branches.append(int(rng.choice(candidates)))
        return np.asarray(branches, dtype=np.int64)

    def update(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, Any],
        *,
        done: bool,
        epsilon: float,
    ) -> float:
        state = encode_policy_state(observation)
        self._ensure(state)
        assert self.network is not None
        expected = 0.0
        if not done:
            next_values = self.values(next_observation)
            branch_expectations: list[float] = []
            for values in self._segments(next_values):
                probabilities = epsilon_greedy_probabilities(
                    values,
                    np.ones(values.size, dtype=np.int8),
                    epsilon,
                )
                branch_expectations.append(float(probabilities @ values))
            expected = float(np.mean(branch_expectations))
        target = float(reward + self.discount * expected)
        cached = self._cache.get(id(observation))
        current = cached[1] if cached is not None and cached[0] is observation else None
        error = self.network.update_branches(
            state,
            np.asarray(action, dtype=np.int64),
            target,
            forward_result=current,
        )
        self._cache.clear()
        return error

    @classmethod
    def load(
        cls, path: str | Path
    ) -> tuple["BranchingExpectedSarsaCnnAgent", dict[str, Any]]:
        with np.load(Path(path).expanduser(), allow_pickle=False) as payload:
            if str(payload["interface"]) != cls.interface:
                raise ValueError("not a branching Expected SARSA CNN checkpoint")
            agent = cls(discount=float(payload["discount"]), seed=int(payload["seed"]))
            conv = np.asarray(payload["conv_w"], dtype=np.float32)
            dense = np.asarray(payload["dense_w"], dtype=np.float32)
            shape = tuple(int(value) for value in payload["network_grid_shape"])
            auxiliary_size = dense.shape[0] - conv.shape[0] * shape[1] * shape[2]
            network = TinyConvQNetwork(
                shape,
                auxiliary_size,
                seed=agent.seed,
                filters=conv.shape[0],
                hidden=dense.shape[1],
                legacy_grid=bool(int(payload["network_legacy_grid"])),
                output_size=CONTROL_LOGIT_COUNT,
            )
            for name in network.parameters:
                network.parameters[name] = np.asarray(payload[name], dtype=np.float32)
            network.step = int(payload["network_step"])
            agent.network = network
            metadata = json.loads(str(payload["metadata"]))
        return agent, metadata


class JointControlExpectedSarsaCnnAgent(ExpectedSarsaCnnAgent):
    """Expected SARSA over 25 complete keyboard/mouse navigation chords."""

    interface = "minecraft-academy-joint-control-expected-sarsa-cnn/v1"

    def __init__(self, *, discount: float = 1.0, seed: int = 0) -> None:
        super().__init__(discount=discount, seed=seed)

    def _ensure(self, state: PolicyState) -> None:
        if self.network is None:
            self.network = TinyConvQNetwork(
                state.joint_lattice.shape,
                state.auxiliary.size,
                seed=self.seed,
                output_size=JOINT_CONTROL_ACTION_COUNT,
            )

    def choose(
        self,
        observation: dict[str, Any],
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        return choose_epsilon_greedy(
            self.values(observation),
            np.ones(JOINT_CONTROL_ACTION_COUNT, dtype=np.int8),
            epsilon=epsilon,
            rng=rng,
        )

    def values(self, observation: dict[str, Any]) -> np.ndarray:
        state = encode_policy_state(observation, include_elapsed=True)
        self._ensure(state)
        assert self.network is not None
        key = id(observation)
        cached = self._cache.get(key)
        if cached is None or cached[0] is not observation:
            if len(self._cache) > 3:
                self._cache.clear()
            cached = (observation, self.network.forward(state))
            self._cache[key] = cached
        return cached[1][0]

    def update(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        *,
        done: bool,
        epsilon: float,
    ) -> float:
        state = encode_policy_state(observation, include_elapsed=True)
        self._ensure(state)
        assert self.network is not None
        expected = 0.0
        if not done:
            next_values = self.values(next_observation)
            probabilities = epsilon_greedy_probabilities(
                next_values,
                np.ones(JOINT_CONTROL_ACTION_COUNT, dtype=np.int8),
                epsilon,
            )
            expected = float(probabilities @ next_values)
        target = float(reward + self.discount * expected)
        cached = self._cache.get(id(observation))
        current = cached[1] if cached is not None and cached[0] is observation else None
        error = self.network.update(state, int(action), target, forward_result=current)
        self._cache.clear()
        return error
