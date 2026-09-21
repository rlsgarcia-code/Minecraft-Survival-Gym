# Contributing to Minecraft Survival Gym

Thank you for helping improve Minecraft Survival Gym. Contributions to the
Gymnasium API, Fabric bridge, documentation, tests, and research tooling are
welcome.

## Development setup

Install the Python development environment from the repository root:

```bash
uv sync --extra dev
```

Run the test suite and the mock smoke test:

```bash
uv run pytest
uv run python scripts/smoke_env.py --backend mock --steps 20
```

The mock backend is the default choice for automated tests because it does not
require Minecraft. Changes to the real bridge should also be validated with
Minecraft Java 1.21, Fabric, and JDK 21.

Build the bridge with:

```bash
cd fabric
./gradlew build
```

## Pull requests

- Keep changes focused and include tests for observable behavior.
- Preserve the Gymnasium `reset()` and `step()` contracts.
- Update the protocol documentation when a bridge message changes.
- Do not commit Minecraft binaries, game assets, worlds, credentials, or
  recorded data containing private information.
- Explain whether the change was tested with the mock backend, the real game,
  or both.

## Bug reports

Include minimal reproduction steps and the versions of Python, Gymnasium,
Minecraft, Fabric Loader, Fabric API, Java, and the operating system. Bridge
problems should include the relevant Minecraft and Python logs with secrets and
personal paths removed.

## Environment compatibility

The canonical environment ID is `minecraft_gym/MinecraftSurvival-v0`.
`MinecraftSurvival-v0` is retained as a compatibility alias. Behavioral changes
that could invalidate benchmark comparisons require a new environment version.
