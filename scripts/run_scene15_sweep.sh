#!/usr/bin/env bash
set -u

ROOT="/root/code/NEWSTRIDE"
PYTHON="/root/miniconda3/envs/stride/bin/python"
LOG_DIR="$ROOT/tuning_logs/overnight_scene15_mr08"
mkdir -p "$LOG_DIR"

# Wait for the currently running seed-1 screening sequence to finish.
while pgrep -f '[m]ain_train.py' >/dev/null 2>&1; do
    sleep 30
done

for cfg in "$ROOT"/config/overnight_scene15_mr08/*.yaml; do
    name="$(basename "$cfg" .yaml)"
    if [[ "$name" == "manifest" ]]; then
        continue
    fi
    "$PYTHON" -u "$ROOT/main_train.py" \
        --config "overnight_scene15_mr08/${name}.yaml" \
        2>&1 | tee "$LOG_DIR/${name}.log"
done
