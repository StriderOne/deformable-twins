#!/usr/bin/env bash
# Run first-party training or Newton visualization with host-owned artifacts.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
kind=${1:-}
case "$kind" in
    train|viewer) shift ;;
    *) echo 'Usage: container.sh train|viewer [--x11] [COMMAND ...]' >&2; exit 2 ;;
esac
image="deformable-twins/$kind:cu128-v1"
data=${PHYSTWIN_ROOT:-"$HOME/PhysTwin"}
cache="$HOME/.cache/deformable-twins/container-$kind"
artifacts="$HOME/.local/share/deformable-twins"
[[ -d "$data/data" ]] || { echo "Download PhysTwin first: python3 tools/download_phystwin.py --root '$data'" >&2; exit 1; }
mkdir -p "$cache" "$artifacts"
args=(docker run --rm --gpus all --shm-size=2g --init
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$repo,dst=/workspace"
    --mount "type=bind,src=$data,dst=/home/homa/PhysTwin,readonly"
    --mount "type=bind,src=$cache,dst=/home/homa/.cache"
    --mount "type=bind,src=$artifacts,dst=/home/homa/.local/share/deformable-twins")
[[ -t 0 && -t 1 ]] && args+=(-it)
if [[ ${1:-} == --x11 ]]; then
    shift
    : "${DISPLAY:?Set DISPLAY in a graphical session or use ssh -Y}"
    auth=$(mktemp)
    chmod 600 "$auth"
    trap 'rm -f -- "$auth"' EXIT
    xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth" nmerge -
    [[ -s "$auth" ]] || { echo 'No X11 authorization cookie for DISPLAY' >&2; exit 1; }
    args+=(--env DISPLAY --env XAUTHORITY=/tmp/container.xauth
        --mount "type=bind,src=$auth,dst=/tmp/container.xauth,readonly"
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display)
    if [[ "$DISPLAY" == localhost:* || "$DISPLAY" == 127.0.0.1:* ]]; then
        args+=(--network host)
    else
        args+=(--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly)
    fi
elif [[ "$kind" == viewer ]]; then
    args+=(-p 127.0.0.1:8080:8080)
fi
"${args[@]}" "$image" "$@"
