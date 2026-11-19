#!/usr/bin/env bash
# Separate Isaac RTX window; source/data mounts are read-only.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cache="$HOME/.cache/deformable-twins/isaac-sim"
image=deformable-twins/isaac-sim:6.0.0
ply=${PHYSTWIN_PLY:-"$HOME/PhysTwin/gaussian_output/double_stretch_sloth/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply"}
graph=${PHYSTWIN_EXPORT:-"$HOME/.cache/deformable-twins/phystwin-lesson/exports/double_stretch_sloth"}
[[ -f "$ply" && -f "$graph/graph.npz" && -f "$graph/provenance.json" ]]
[[ -n "${DISPLAY:-}" ]]
[[ $(id -u) == 1000 && $(id -g) == 1000 ]] || { echo 'This image expects host UID/GID 1000:1000' >&2; exit 1; }
mkdir -p "$cache"/{cache,compute,logs,config,data,pkg,kit-cache,kit-data,kit-logs}
[[ $(df -B1 --output=avail "$cache" | tail -1) -ge 10737418240 ]] || {
    echo 'Need at least 10 GiB of free disk space to run.' >&2; exit 1;
}
if ! docker image inspect "$image" >/dev/null 2>&1; then
    [[ $(df -B1 --output=avail "$cache" | tail -1) -ge 64424509440 ]] || {
        echo 'First build needs at least 60 GiB free for image extraction and cache.' >&2; exit 1;
    }
    docker build -t "$image" - < "$repo/environments/Dockerfile.isaac-sim"
fi
[[ $(df -B1 --output=avail "$cache" | tail -1) -ge 10737418240 ]] || {
    echo 'Build finished, but less than 10 GiB remains; not starting the viewer.' >&2; exit 1;
}
image_id=$(docker image inspect "$image" --format '{{.Id}}')
auth=$(mktemp "${XDG_RUNTIME_DIR:-/tmp}/isaac-sim-xauth.XXXXXX")
chmod 600 "$auth"
trap 'rm -f -- "$auth"' EXIT
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth" nmerge -
[[ -s "$auth" ]] || { echo 'No X11 authorization cookie' >&2; exit 1; }
docker run --rm --name phystwin-isaac-rtx --gpus all --shm-size=2g \
    --user 1000:1000 --env DISPLAY --env XAUTHORITY=/tmp/isaac.xauth \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=N \
    --env PYTHONUNBUFFERED=1 \
    --env "PHYSTWIN_CONTAINER_IMAGE_ID=$image_id" \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --mount "type=bind,src=$auth,dst=/tmp/isaac.xauth,readonly" \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
    --mount "type=bind,src=$repo,dst=/workspace,readonly" \
    --mount "type=bind,src=$ply,dst=/data/point_cloud.ply,readonly" \
    --mount "type=bind,src=$graph,dst=/data/export,readonly" \
    --mount "type=bind,src=$cache/cache,dst=/isaac-sim/.cache" \
    --mount "type=bind,src=$cache/compute,dst=/isaac-sim/.nv/ComputeCache" \
    --mount "type=bind,src=$cache/logs,dst=/isaac-sim/.nvidia-omniverse/logs" \
    --mount "type=bind,src=$cache/config,dst=/isaac-sim/.nvidia-omniverse/config" \
    --mount "type=bind,src=$cache/data,dst=/isaac-sim/.local/share/ov/data" \
    --mount "type=bind,src=$cache/pkg,dst=/isaac-sim/.local/share/ov/pkg" \
    --mount "type=bind,src=$cache/kit-cache,dst=/isaac-sim/kit/cache" \
    --mount "type=bind,src=$cache/kit-data,dst=/isaac-sim/kit/data" \
    --mount "type=bind,src=$cache/kit-logs,dst=/isaac-sim/kit/logs" \
    "$image" /data/point_cloud.ply --graph /data/export --viewer isaac "$@"
