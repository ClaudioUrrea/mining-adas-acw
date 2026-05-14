#!/usr/bin/env bash
# =============================================================================
#  scripts/02_train_multitask.sh
#
#  Trains the multitask BiSeNetV1 (segmentation + classification) from scratch.
#  Produces the Best-Classification (BC) and Best-Segmentation (BS) checkpoints
#  used to seed every temporal-inference experiment.
#
#  Usage:
#      bash scripts/02_train_multitask.sh                 # default config
#      bash scripts/02_train_multitask.sh path/to/cfg.py  # override config
#
#  Requires the GitHub repo to be installed (Option A or Option B in
#  docs/INSTALLATION.md). When using Option B, MMSEG_ROOT must be set.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CONFIG_DEFAULT="$REPO_ROOT/configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py"
CONFIG="${1:-$CONFIG_DEFAULT}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

if [[ -n "${MMSEG_ROOT:-}" && -f "$MMSEG_ROOT/tools/train.py" ]]; then
    echo "Training via MMSegmentation tools/train.py (Option B install)"
    cd "$MMSEG_ROOT"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python tools/train.py "$CONFIG"
else
    echo "Training via mim (Option A install)"
    cd "$REPO_ROOT"
    PYTHONPATH=".:${PYTHONPATH:-}" mim train mmseg "$CONFIG"
fi
