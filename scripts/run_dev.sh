#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_dir="${MINECRAFT_GYM_STATE_DIR:-$repo_root/.minecraft-gym}"
log_dir="$state_dir/logs"
minecraft_pid_file="$state_dir/minecraft.pid"
jupyter_pid_file="$state_dir/jupyter.pid"
minecraft_log="$log_dir/minecraft.log"
jupyter_log="$log_dir/jupyter.log"
notebook="$repo_root/output/jupyter-notebook/minecraft-survival-gym-quickstart.ipynb"
dry_run=false

usage() {
    cat <<'EOF'
Usage: ./scripts/run_dev.sh [--dry-run]

Starts both development services in the background:
  1. Minecraft Fabric development client
  2. JupyterLab with the quickstart notebook

Environment variables:
  JAVA_HOME                         JDK 21 installation to use
  MINECRAFT_GYM_QUICK_PLAY_WORLD   Optional Fabric dev-world directory name
  MINECRAFT_GYM_STATE_DIR          Optional PID/log directory override
EOF
}

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

pid_is_running() {
    local pid_file="$1"
    local pid
    [[ -f "$pid_file" ]] || return 1
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

java_is_21() {
    local candidate="$1"
    [[ -n "$candidate" ]] || return 1
    [[ -x "$candidate/bin/java" ]] || return 1
    "$candidate/bin/java" -version 2>&1 | head -n 1 | grep -Eq 'version "21([.]|\")'
}

find_java_21() {
    local candidate="${JAVA_HOME:-}"
    if java_is_21 "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi

    if [[ -x /usr/libexec/java_home ]]; then
        candidate="$(/usr/libexec/java_home -v 21 2>/dev/null || true)"
        if java_is_21 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    if command -v brew >/dev/null 2>&1; then
        candidate="$(brew --prefix openjdk@21 2>/dev/null || true)"
        candidate="${candidate:+$candidate/libexec/openjdk.jdk/Contents/Home}"
        if java_is_21 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    for candidate in \
        /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
        /usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home; do
        if java_is_21 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not available on PATH." >&2
    exit 1
fi
if [[ ! -f "$notebook" ]]; then
    echo "Error: notebook not found: $notebook" >&2
    exit 1
fi

java_home_path="$(find_java_21 || true)"
if [[ -z "$java_home_path" ]]; then
    cat >&2 <<'EOF'
Error: JDK 21 was not found.

On macOS with Homebrew:
  brew install openjdk@21
  export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
  export PATH="$JAVA_HOME/bin:$PATH"
EOF
    exit 1
fi

if pid_is_running "$minecraft_pid_file" || pid_is_running "$jupyter_pid_file"; then
    cat >&2 <<EOF
Error: a Minecraft Gym development session is already active.
Run $repo_root/scripts/stop_dev.sh before starting another one.
EOF
    exit 1
fi

if $dry_run; then
    cat <<EOF
Dry run; no processes were started.

Repository: $repo_root
JDK 21:     $java_home_path
Minecraft:  (cd "$repo_root/fabric" && ./gradlew --no-daemon runClient)
Jupyter:    uv run --extra notebook jupyter lab "$notebook"
State:      $state_dir
EOF
    exit 0
fi

mkdir -p "$log_dir"
rm -f "$minecraft_pid_file" "$jupyter_pid_file"

(
    cd "$repo_root/fabric"
    nohup env \
        "JAVA_HOME=$java_home_path" \
        "PATH=$java_home_path/bin:$PATH" \
        ./gradlew --no-daemon runClient \
        >"$minecraft_log" 2>&1 &
    printf '%s\n' "$!" >"$minecraft_pid_file"
)

(
    cd "$repo_root"
    nohup uv run --extra notebook jupyter lab "$notebook" \
        --ServerApp.ip=127.0.0.1 \
        >"$jupyter_log" 2>&1 &
    printf '%s\n' "$!" >"$jupyter_pid_file"
)

sleep 2

if ! pid_is_running "$minecraft_pid_file" || ! pid_is_running "$jupyter_pid_file"; then
    echo "One of the services failed to start. Recent logs:" >&2
    tail -n 20 "$minecraft_log" "$jupyter_log" >&2 || true
    "$script_dir/stop_dev.sh" >/dev/null 2>&1 || true
    exit 1
fi

cat <<EOF
Minecraft Survival Gym development session started.

Minecraft PID: $(<"$minecraft_pid_file")
Jupyter PID:   $(<"$jupyter_pid_file")
Minecraft log: $minecraft_log
Jupyter log:   $jupyter_log

Next steps:
  1. Wait for the Minecraft window to open.
  2. Click Singleplayer and enter a Survival world.
  3. Wait until the terrain and player HUD are visible.
  4. Use the notebook opened by JupyterLab.
  5. Set REAL_BACKEND = True only after the world is loaded.

Stop both services with:
  ./scripts/stop_dev.sh
EOF
