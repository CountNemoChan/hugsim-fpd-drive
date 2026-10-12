#!/usr/bin/env bash
set -euo pipefail
: "${FPD_DRIVE_PYTHON:?Set FPD_DRIVE_PYTHON to the inference environment Python}"
: "${FPD_DRIVE_ROOT:?Set FPD_DRIVE_ROOT to the FPD-Drive checkout}"
: "${FPD_DRIVE_CKPT:?Set FPD_DRIVE_CKPT to the checkpoint}"
: "${FPD_DRIVE_CFG:?Set FPD_DRIVE_CFG to the inference config}"
export CUDA_VISIBLE_DEVICES="$1"
exec "$FPD_DRIVE_PYTHON" "${HUGSIM_ROOT}/ad_clients/FPD_Drive_closed_loop.py" \
  --output "$2" --checkpoint "$FPD_DRIVE_CKPT" --config "$FPD_DRIVE_CFG"
