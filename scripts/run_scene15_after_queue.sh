#!/usr/bin/env bash
set -u
ROOT="/root/code/NEWSTRIDE"
PYTHON="/root/miniconda3/envs/stride/bin/python"
while pgrep -f '[r]un_scene15_resumable_queue.py' >/dev/null 2>&1; do
    sleep 30
done
exec "$PYTHON" "$ROOT/scripts/run_scene15_multiseed_validation.py"
