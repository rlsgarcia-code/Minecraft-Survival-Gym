"""Install the published Fabric bridge into an official Minecraft Launcher game directory."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MINECRAFT_VERSION = "1.21"
FABRIC_API_VERSION = "0.102.0+1.21"
FABRIC_INSTALLER_VERSION = "1.1.2"
BRIDGE_VERSION = "0.1.1"
FABRIC_MAVEN = "https://maven.fabricmc.net"
FABRIC_API_NAME = f"fabric-api-{FABRIC_API_VERSION}.jar"
BRIDGE_NAME = f"minecraft-gym-bridge-{BRIDGE_VERSION}.jar"
FABRIC_API_URL = (
    f"{FABRIC_MAVEN}/net/fabricmc/fabric-api/fabric-api/"
    f"{FABRIC_API_VERSION}/{FABRIC_API_NAME}"
)
FABRIC_INSTALLER_URL = (
    f"{FABRIC_MAVEN}/net/fabricmc/fabric-installer/"
    f"{FABRIC_INSTALLER_VERSION}/fabric-installer-{FABRIC_INSTALLER_VERSION}.jar"
)
BRIDGE_URL = (
    "https://github.com/rlsgarcia-code/Minecraft-Survival-Gym/releases/"
    f"download/v{BRIDGE_VERSION}/{BRIDGE_NAME}"
)
BRIDGE_SHA256 = "71ff55d9b8500e5103a9d5aec22d7e21b32725e869432d290d80abf301e590cc"
FABRIC_API_SHA256 = "7ec0e5a11e77957fe1ed0328487921a8211bb7eace14298cd74635bec61a3f26"
FABRIC_INSTALLER_SHA256 = "61e035bf7bf70153e127440ce34de47c9036f0a2d0c65d1529454bd35ceefe4f"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


def default_game_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/minecraft"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / ".minecraft"
    if system == "Linux":
        return Path.home() / ".minecraft"
    raise RuntimeError("Unsupported OS; pass --game-dir explicitly.")


def _java_21() -> Path:
    candidates: list[Path] = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java_name = "java.exe" if platform.system() == "Windows" else "java"
        candidates.append(Path(java_home) / "bin" / java_name)
    if platform.system() == "Darwin":
        registry = Path("/usr/libexec/java_home")
        if registry.is_file():
            found = subprocess.run(
                [str(registry), "-v", "21"], capture_output=True, text=True, check=False
            )
            if found.returncode == 0:
                candidates.append(Path(found.stdout.strip()) / "bin" / "java")
        for prefix in ("/opt/homebrew", "/usr/local"):
            candidates.append(
                Path(prefix) / "opt/openjdk@21/libexec/openjdk.jdk/Contents/Home/bin/java"
            )
    on_path = shutil.which("java")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "-version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and re.search(
            r'\bversion "21(?:[.\+"]|$)', result.stderr + result.stdout
        ):
            return candidate
    raise RuntimeError(
        "Java 21 is required to install Fabric for Minecraft 1.21. "
        "Install JDK 21, then rerun minecraft-gym setup."
    )


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "minecraft-gym-setup"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error
    if not data or len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"Download is empty or unexpectedly large: {url}")
    return data


def _verified_download(url: str, expected_sha256: str | None = None) -> bytes:
    if expected_sha256 is None:
        checksum = _download(url + ".sha256").decode("ascii").strip().split()[0]
        if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            raise RuntimeError(f"Invalid SHA-256 checksum published for {url}")
        expected_sha256 = checksum
    data = _download(url)
    if hashlib.sha256(data).hexdigest() != expected_sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {url}; nothing was installed from it")
    return data


def _has_fabric_loader(game_dir: Path) -> bool:
    versions = game_dir / "versions"
    return any(
        version.is_dir() and (version / f"{version.name}.json").is_file()
        for version in versions.glob(f"fabric-loader-*-{MINECRAFT_VERSION}")
    )


def _backup_obsolete(paths: list[Path], backup_dir: Path) -> None:
    if not paths:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        backup = backup_dir / path.name
        if backup.exists():
            raise RuntimeError(f"Backup already exists; move it before setup: {backup}")
        path.replace(backup)
        print(f"Replaced; previous version preserved at: {backup}", flush=True)


def _install_mod(
    mods_dir: Path,
    filename: str,
    url: str,
    checksum: str | None,
    *,
    obsolete: list[Path] | None = None,
    backup_dir: Path | None = None,
) -> None:
    target = mods_dir / filename
    if target.exists():
        if not target.is_file():
            raise RuntimeError(f"Not a regular file: {target}")
        if checksum and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Existing file does not match the expected release: {target}")
        _backup_obsolete(obsolete or [], backup_dir or mods_dir.parent / ".minecraft-gym-backup")
        print(f"Already present: {target}", flush=True)
        return
    data = _verified_download(url, checksum)
    mods_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=mods_dir, prefix=".minecraft-gym-", delete=False) as temp:
        temp.write(data)
        staged = Path(temp.name)
    try:
        _backup_obsolete(obsolete or [], backup_dir or mods_dir.parent / ".minecraft-gym-backup")
        if target.exists():
            raise RuntimeError(f"File appeared during setup; refusing to overwrite: {target}")
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    print(f"Installed: {target}", flush=True)


def setup(game_dir: Path, *, dry_run: bool = False) -> None:
    game_dir = game_dir.expanduser().resolve()
    if not game_dir.is_dir() or not (game_dir / "launcher_profiles.json").is_file():
        raise RuntimeError(
            f"Minecraft Launcher game directory not found at {game_dir}. "
            "Install and open the official Launcher once, or pass --game-dir."
        )
    mods_dir = game_dir / "mods"
    upgradeable_bridge = {f"minecraft-gym-bridge-0.1.0.jar"}
    obsolete_bridge: list[Path] = []
    for pattern, target in (("fabric-api-*.jar", FABRIC_API_NAME),
                            ("minecraft-gym-bridge-*.jar", BRIDGE_NAME)):
        conflicts = [p.name for p in mods_dir.glob(pattern) if p.name != target]
        if pattern.startswith("minecraft-gym-bridge"):
            obsolete_bridge = [mods_dir / name for name in conflicts if name in upgradeable_bridge]
            conflicts = [name for name in conflicts if name not in upgradeable_bridge]
        if conflicts:
            raise RuntimeError(
                f"Conflicting mod in {mods_dir}: {', '.join(conflicts)}. "
                "Move it out of this directory before running setup."
            )
    has_loader = _has_fabric_loader(game_dir)
    print(f"Minecraft directory: {game_dir}", flush=True)
    print(f"Fabric Loader for {MINECRAFT_VERSION}: {'found' if has_loader else 'will install'}", flush=True)
    print(f"Mods directory: {mods_dir}", flush=True)
    if dry_run:
        print("Dry run: no files downloaded or changed.", flush=True)
        return
    if not has_loader:
        java = _java_21()
        installer = _verified_download(FABRIC_INSTALLER_URL, FABRIC_INSTALLER_SHA256)
        with tempfile.TemporaryDirectory(prefix="minecraft-gym-setup-") as temp_dir:
            installer_path = Path(temp_dir) / "fabric-installer.jar"
            installer_path.write_bytes(installer)
            print("Installing Fabric Loader; close Minecraft and its Launcher first.", flush=True)
            result = subprocess.run(
                [str(java), "-jar", str(installer_path), "client", "-dir", str(game_dir),
                 "-mcversion", MINECRAFT_VERSION],
                check=False,
            )
        if result.returncode != 0 or not _has_fabric_loader(game_dir):
            raise RuntimeError("Fabric Installer failed; no mods were installed.")
    _install_mod(mods_dir, FABRIC_API_NAME, FABRIC_API_URL, FABRIC_API_SHA256)
    _install_mod(
        mods_dir,
        BRIDGE_NAME,
        BRIDGE_URL,
        BRIDGE_SHA256,
        obsolete=obsolete_bridge,
        backup_dir=game_dir / ".minecraft-gym-backup",
    )
    print(
        "Setup complete. Run `minecraft-gym start`, select the Minecraft 1.21 "
        "Fabric profile, click Play, and enter a single-player Survival world.",
        flush=True,
    )
