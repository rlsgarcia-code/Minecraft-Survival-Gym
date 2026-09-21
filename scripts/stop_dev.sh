#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
state_dir="${MINECRAFT_GYM_STATE_DIR:-$repo_root/.minecraft-gym}"
dry_run=false

usage() {
    cat <<'EOF'
Usage: ./scripts/stop_dev.sh [--dry-run]

Stops the Minecraft and JupyterLab processes started by scripts/run_dev.sh.
Logs are preserved under .minecraft-gym/logs/.
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

terminate_tree() {
    local pid="$1"
    local child
    while IFS= read -r child; do
        [[ -n "$child" ]] && terminate_tree "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -TERM "$pid" 2>/dev/null || true
}

stop_service() {
    local name="$1"
    local pid_file="$2"
    local expected_pattern="$3"
    local pid command_line

    if [[ ! -f "$pid_file" ]]; then
        echo "$name: no PID file; already stopped."
        return 0
    fi

    pid="$(<"$pid_file")"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "$name: invalid PID file; refusing to signal a process." >&2
        return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name: process $pid is no longer running."
        $dry_run || rm -f "$pid_file"
        return 0
    fi

    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ ! "$command_line" =~ $expected_pattern ]]; then
        echo "$name: PID $pid no longer matches the expected command; refusing to stop it." >&2
        echo "Observed command: $command_line" >&2
        return 1
    fi

    if $dry_run; then
        echo "$name: would stop PID $pid ($command_line)"
        return 0
    fi

    echo "$name: stopping PID $pid..."
    terminate_tree "$pid"

    for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$name: graceful shutdown timed out; forcing PID $pid." >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

status=0
stop_service \
    "JupyterLab" \
    "$state_dir/jupyter.pid" \
    '(^|[ /])(uv|jupyter)( |$)|jupyter-lab' || status=1
stop_service \
    "Minecraft" \
    "$state_dir/minecraft.pid" \
    'GradleWrapperMain|gradlew|runClient' || status=1

if (( status == 0 )); then
    echo "Minecraft Survival Gym development session stopped."
else
    echo "One or more processes were not stopped automatically; inspect the messages above." >&2
fi
exit "$status"
