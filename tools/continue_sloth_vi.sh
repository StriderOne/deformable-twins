#!/usr/bin/env bash
set -uo pipefail
cd /home/homa/deformable-twins
export PYTHONPATH=src
python_bin=/home/homa/.venvs/phystwin-vi/bin/python
artifact_root=/home/homa/.local/share/deformable-twins/artifacts
"$python_bin" -m lab.calibration.phystwin_vi --config experiments/E-0003.yaml --run-dir runs/R-0006 --resume-checkpoint "$artifact_root/R-0002/checkpoint.pt" &
narrow_pid=$!
"$python_bin" -m lab.calibration.phystwin_vi --config experiments/E-0004.yaml --run-dir runs/R-0007 --resume-checkpoint "$artifact_root/R-0004/checkpoint.pt" &
wide_pid=$!
wait "$narrow_pid"; narrow_status=$?
wait "$wide_pid"; wide_status=$?
printf 'Training exit codes: narrow=%s wide=%s\n' "$narrow_status" "$wide_status"
if (( narrow_status != 0 || wide_status != 0 )); then exit 1; fi
"$python_bin" -m lab.calibration.phystwin_vi --mode evaluate --config experiments/E-0003.yaml --run-dir runs/R-0008 --map-checkpoint "$artifact_root/R-0001/checkpoint.pt" --vi-checkpoint "$artifact_root/R-0006/checkpoint.pt" || exit $?
"$python_bin" -m lab.calibration.phystwin_vi --mode evaluate --config experiments/E-0004.yaml --run-dir runs/R-0009 --map-checkpoint "$artifact_root/R-0001/checkpoint.pt" --vi-checkpoint "$artifact_root/R-0007/checkpoint.pt"
