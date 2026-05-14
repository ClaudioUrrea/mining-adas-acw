#!/usr/bin/env bash
# =============================================================================
#  scripts/04_eval_acw_stateful.sh
#
#  Runs the stateful Adaptive Clockwork (A-CW) evaluator on the route videos
#  by executing the canonical evaluation notebook end to end. Outputs the
#  per-frame CSV/Parquet under runs/eval_acw_stateful/.
#
#  Usage:
#      bash scripts/04_eval_acw_stateful.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

NB="notebooks/eval_stateful_adaptive_scheduler_fullseq_tau003_1a_code_only_EN_comments_unexecuted.ipynb"

if [[ ! -f "$NB" ]]; then
    echo "ERROR: notebook not found: $NB" >&2
    exit 1
fi

mkdir -p runs/eval_acw_stateful
PYTHONPATH=".:${PYTHONPATH:-}" jupyter nbconvert \
    --to notebook \
    --execute "$NB" \
    --output "runs/eval_acw_stateful/$(basename "$NB" .ipynb)_executed.ipynb" \
    --ExecutePreprocessor.timeout=3600

echo ""
echo "Done. Outputs under runs/eval_acw_stateful/"
