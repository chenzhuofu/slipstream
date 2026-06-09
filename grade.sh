#!/bin/bash

export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export TMPDIR=$SCRATCH/dotfiles/.tmp
export APPTAINER_TMPDIR=$SCRATCH/dotfiles/.apptainer/tmp
export APPTAINER_CACHEDIR=$SCRATCH/dotfiles/.apptainer/cache

# Set PROJECT_ROOT and APPTAINER_SIF_DIR before running.
PROJ_DIR="${PROJECT_ROOT:-/path/to/project}"
SIF_DIR="${APPTAINER_SIF_DIR:-/path/to/apptainer/sifs}"
cd "$PROJ_DIR"

PREDICTIONS_PATH=${PREDICTIONS_PATH:-./results/eval/preds.json}
RUN_ID=${RUN_ID:-eval_01}
WORKERS=${WORKERS:-16}

uv run --no-sync eval_apptainer.py \
    --predictions-path "$PREDICTIONS_PATH" \
    --subset verified \
    --split test \
    --sif-dir "$SIF_DIR" \
    --max-workers "$WORKERS" \
    --run-id "$RUN_ID" \
    --timeout 1800
