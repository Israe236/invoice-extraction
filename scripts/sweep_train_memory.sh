#!/usr/bin/env bash
# Find the largest image resolution that will train on this GPU.
#
# One subprocess per resolution, because a CUDA OOM leaves the context unusable
# for the rest of the process. Prints one RESULT line per setting.
#
#   bash scripts/sweep_train_memory.sh 401408 262144 200704 150528 100352

set -u
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
RESOLUTIONS=${*:-"401408 262144 200704 150528 100352"}
# Must stay BELOW the physically free VRAM, not below the card's total. Windows
# holds ~1.1 GB of the 6 GB for the desktop, and if the allocator is allowed
# past what is actually free, WSL2's WDDM passthrough spills into host RAM and
# the run dies with "CUDA driver error: device not ready" instead of a clean
# out-of-memory error. 0.75 of 6 GB = 4.5 GB, comfortably under the ~4.9 GB free.
FRACTION=${FRACTION:-0.75}

mkdir -p logs
echo "resolution sweep on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

for max_pixels in $RESOLUTIONS; do
    log="logs/probe_${max_pixels}.log"
    PYTHONUNBUFFERED=1 "$PYTHON" scripts/probe_train_memory.py \
        --max-pixels "$max_pixels" --fraction "$FRACTION" >"$log" 2>&1
    status=$?
    if grep -q "^RESULT" "$log"; then
        grep "^RESULT" "$log"
    else
        # No RESULT line means the process died before it could report --
        # usually the host OOM-killer, since WDDM lets CUDA spill into RAM.
        echo "RESULT max_pixels=${max_pixels} verdict=KILLED exit=${status} (see ${log})"
    fi
done
