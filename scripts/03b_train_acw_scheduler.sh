#!/usr/bin/env bash
# scripts/03b_train_acw_scheduler.sh — A-CW Stage 2: scheduler training (freeze base)
# Generated from the same template as scripts/02_train_multitask.sh.
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_DEFAULT="$REPO_ROOT/configs/acw/03_train_adaptive_scheduler_pretrain.py"
CONFIG="${1:-$CONFIG_DEFAULT}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

if [[ -n "${MMSEG_ROOT:-}" && -f "$MMSEG_ROOT/tools/train.py" ]]; then
    cd "$MMSEG_ROOT"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python tools/train.py "$CONFIG"
else
    cd "$REPO_ROOT"
    PYTHONPATH=".:${PYTHONPATH:-}" mim train mmseg "$CONFIG"
fi
