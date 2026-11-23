# Minecraft 26.3 authoritative server collector

This server-only Fabric mod captures authoritative state for dataset v2. It is
passive: it has no command capable of moving a player, modifying the world,
resetting a server, or executing an action. It listens only on
`127.0.0.1:25571`.

## Compatibility

- Minecraft Java 26.3
- Fabric Loader 0.19.5
- Fabric API 0.163.0+26.3
- Java 25

The adjacent `fabric/` module remains the interactive Minecraft 1.21 client
bridge. The two artifacts are intentionally separate because they have
different game and Java compatibility contracts.

## Build

```bash
cd fabric-server-26.3
./gradlew clean build
```

On the development Mac the wrapper selects Prism Launcher's Java 25 runtime.
On another machine, select a Java 25 executable explicitly:

```bash
MINECRAFT_GYM_JAVA_25=/path/to/java ./gradlew build
```

The output is
`build/libs/minecraft-gym-server-collector-0.1.0.jar`.

## Data boundary

The collector emits monotonically sequenced server ticks, pseudonymous actor
IDs, pose, velocity, vitals, inventory, effects, death, inventory deltas and
block interactions. A bounded 2,400-tick ring buffer supports resumable reads
and reports an explicit gap if a client falls behind.

Block use records both the intent and any block-type mutation observed by the
end of that server tick. Block break events include the before and after block.

Checksums use a signed JSON envelope: `payload_json` contains the exact UTF-8
text covered by `payload_sha256`. This avoids cross-language float formatting
differences between Java and Python.

Raw player names, chat and UUIDs are never emitted. Actor IDs are derived with
HMAC-SHA-256 from a local secret stored in Fabric's config directory. The
Python client validates every response checksum and rejects gaps or malformed
records.

Server-derived trajectories are always `semantic_weak`: they cannot reveal the
actual keyboard, mouse or touch input and are therefore ineligible for
low-level behavioral cloning, offline RL or DreamerV3. Exact visual data still
requires the separate Java client collector.

## Installation safety

Building or running the isolated development server does not install this mod
in a family server. A live installation requires a backup, copying the JAR to
that server's `mods/` directory and one planned restart. Do not copy it into a
server running a different Minecraft version.
