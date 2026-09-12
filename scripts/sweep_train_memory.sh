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

mkdir -p logs
echo "resolution sweep on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

for max_pixels in $RESOLUTIONS; do
    log="logs/probe_${max_pixels}.log"
    PYTHONUNBUFFERED=1 "$PYTHON" scripts/probe_train_memory.py "$max_pixels" >"$log" 2>&1
    status=$?
    if grep -q "^RESULT" "$log"; then
        grep "^RESULT" "$log"
    else
        # No RESULT line means the process died before it could report --
        # usually the host OOM-killer, since WDDM lets CUDA spill into RAM.
        echo "RESULT max_pixels=${max_pixels} verdict=KILLED exit=${status} (see ${log})"
    fi
done
