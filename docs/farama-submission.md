# Farama external-environment submission

Minecraft Survival Gym is intended to be listed in Gymnasium's official
third-party environment directory. Gymnasium does not accept new environments
into its core package, so the project remains independently maintained.

## Submission checklist

- [x] Gymnasium `reset()` and `step()` API
- [x] Namespaced and versioned environment ID
- [x] Gymnasium environment checker test
- [x] Deterministic mock backend for CI
- [x] Python package metadata and MIT license
- [x] Python and Fabric continuous integration
- [x] Publish the Python distribution to [PyPI](https://pypi.org/project/minecraft-gym/)
- [x] Publish the Fabric bridge JAR in the [v0.1.0 GitHub release](https://github.com/rlsgarcia-code/Minecraft-Survival-Gym/releases/tag/v0.1.0)
- [ ] Add a short demonstration video or GIF
- [ ] Confirm the proposed entry with the Farama community on Discord
- [ ] Submit a PR to Gymnasium's third-party environments page

## Proposed directory entry

Place this entry alphabetically under **Game environments** in
`docs/environments/third_party_environments.md`:

```markdown
- [Minecraft Survival Gym](https://github.com/rlsgarcia-code/Minecraft-Survival-Gym)

  ![Gymnasium dependency](https://img.shields.io/badge/Gymnasium-%3E%3D1.1-blue)
  ![GitHub stars](https://img.shields.io/github/stars/rlsgarcia-code/Minecraft-Survival-Gym)

  Minecraft Survival Gym exposes Minecraft Java 1.21 survival gameplay through
  the Gymnasium API. It provides visual and structured observations, full player
  controls, deterministic test infrastructure, and keyboard/mouse demonstration
  recording for reinforcement and imitation learning research.
```

The upstream pull request should be titled:

```text
Add external environment Minecraft Survival Gym
```
