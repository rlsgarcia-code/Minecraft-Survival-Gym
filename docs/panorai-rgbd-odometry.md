# PanorAi RGB-D odometry experiment

This experiment is an explicit integration between Minecraft Gym and the
external PanorAi library.  PanorAi remains the owner of spherical/gnomonic
geometry, robust metric relative pose and pose-quality evidence. Minecraft Gym
owns Minecraft depth semantics, action priors and trajectory state.

## Contract

- Camera frame: PanorAi canonical `+X right, +Y up, +Z forward`.
- Relative pose: `point_b = R_b_from_a @ point_a + t_b_from_a`.
- Input depth: camera-Z in blocks, as produced by the current Prismarine depth
  stream.
- PanorAi depth: radial range. The adapter converts with
  `radial = camera_z / bearing_z`.
- Metric pose: Minecraft Gym calls the dedicated Experimental public surface
  `panorai.estimators.metric_rgbd_pose`. PanorAi optionally uses epipolar
  consensus, then robustly estimates the complete 3D–3D SE(3) transform and
  reports covariance, residuals and an acceptance decision.
- Policy input never receives the server's true pose. Authoritative pose is an
  evaluator label only.

The adapter deliberately consumes externally matched pixels. This keeps three
questions measurable rather than conflating them:

1. Does PanorAi recover robust camera geometry from valid correspondences?
2. Can RGB-D recover metric translation scale and reject inconsistent depth?
3. Can a Minecraft image frontend produce enough stable correspondences?

The first two are covered by the isolated benchmark. The passive live audit
checks the third against the running renderer. A static run validates the real
RGB-D transport and frontend, but is deliberately not labelled as a
moving-camera odometry validation.

## Run

Until the Experimental surface is released, expose the clean PR checkout
without copying it into the RL repository:

```bash
PYTHONPATH=/path/to/PanorAi \
  .venv/bin/python scripts/evaluate_panorai_odometry.py \
  --output runs/panorai-odometry/isolated-v1
```

Focused regression tests use the same optional dependency:

```bash
PYTHONPATH=/path/to/PanorAi \
  .venv/bin/pytest -q tests/test_panorai_odometry.py
```

For a bounded, read-only audit of the running local renderer:

```bash
PYTHONPATH=src:/path/to/PanorAi \
  python3 scripts/evaluate_panorai_live_odometry.py \
  --frames 8 \
  --output runs/panorai-odometry/live-passive-v1
```

This command never sends gameplay, camera, or operator-control commands. It
persists the exact JPEG, compressed camera-Z depth, hashes, pair diagnostics,
and a machine-readable `report.json`.

## PanorAi extension under validation

PR [PanorAi #233](https://github.com/rlsgarcia-code/PanorAi/pull/233)
publishes the application-neutral metric pair estimator under
`panorai-metric-rgbd-relative-pose/v1-experimental`. The Gym adapter now only
converts camera-Z depth to radial range and maps the generic result into the
trajectory contract. It no longer contains a second rigid-alignment
implementation.

Sequence-level work remains outside that pair estimator and may later add:

- a typed `RgbdCorrespondenceSet` carrying pixels/bearings, radial depth,
  validity and uncertainty;
- a motion-prior interface that accepts a relative-pose mean and covariance,
  not keyboard semantics;
- an incremental `MetricRgbdOdometry` facade with explicit `tracking`,
  `degraded`, `rotation_only` and `lost` states;
- quality fields for scale support, metric residual, direction agreement and
  drift, while preserving the existing visual-only API unchanged.

Minecraft-specific action encoding, collision events, target memory and Gym
datasets must remain in Minecraft Gym.

O uso da pose em uma observação treinável, a separação entre mapa implantável e
mapa privilegiado e a montagem do vetor de estado estão especificados em
[`odometric-belief-state-v1.md`](odometric-belief-state-v1.md).
