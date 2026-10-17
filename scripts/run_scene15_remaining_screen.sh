#!/usr/bin/env bash
set -u

ROOT="/root/code/NEWSTRIDE"
PYTHON="/root/miniconda3/envs/stride/bin/python"
LOG_ROOT="$ROOT/tuning_logs"

# Wait for the MR=0.8 sweep runner itself, then process high noise first.
while pgrep -f '[r]un_scene15_sweep.sh' >/dev/null 2>&1; do
    sleep 30
done

for ratio in 0.5 0.2 0; do
    cfg_dir="$ROOT/config/overnight_scene15_mr${ratio}"
    log_dir="$LOG_ROOT/overnight_scene15_mr${ratio}"
    mkdir -p "$log_dir"
    for cfg in "$cfg_dir"/*.yaml; do
        name="$(basename "$cfg" .yaml)"
        if [[ "$name" == "manifest" ]]; then
            continue
        fi
        "$PYTHON" -u "$ROOT/main_train.py" \
            --config "overnight_scene15_mr${ratio}/${name}.yaml" \
            2>&1 | tee "$log_dir/${name}.log"
    done
done
