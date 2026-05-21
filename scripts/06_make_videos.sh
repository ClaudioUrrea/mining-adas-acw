#!/usr/bin/env bash
# =============================================================================
#  scripts/06_make_videos.sh
#
#  Renders the simulated HMI overlay videos (1080x720) used for the qualitative
#  panels in the paper, by executing the corresponding video-generation
#  notebooks.
#
#  Usage:
#      bash scripts/06_make_videos.sh acw      # adaptive clockwork only
#      bash scripts/06_make_videos.sh fcw-k1
#      bash scripts/06_make_videos.sh fcw-k30
#      bash scripts/06_make_videos.sh fcw-adap
#      bash scripts/06_make_videos.sh all
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

declare -A NOTEBOOKS=(
    [fcw-k1]="notebooks/videos_multitask_CLOCKWORK_FIX_1a_k1.ipynb"
    [fcw-k30]="notebooks/videos_multitask_CLOCKWORK_FIX_1a_k30.ipynb"
    [fcw-adap]="notebooks/videos_multitask_CLOCKWORK_ADAPT_1a_kc30kl100_code_only_EN_comments_unexecuted_v3.ipynb"
    [acw]="notebooks/videos_multitask_ADAPTIVE_SCHEDULER_realista_tau003_1a_2__code_only_EN_comments_unexecuted.ipynb"
)

METHOD="${1:-acw}"

run_one() {
    local key="$1"
    local nb="${NOTEBOOKS[$key]:-}"
    if [[ -z "$nb" || ! -f "$nb" ]]; then
        echo "WARNING: notebook for '$key' not found, skipping" >&2
        return
    fi
    local out_dir="runs/videos_${key}"
    mkdir -p "$out_dir"
    echo ""
    echo "==> Rendering video: $key"
    PYTHONPATH=".:${PYTHONPATH:-}" jupyter nbconvert \
        --to notebook \
        --execute "$nb" \
        --output "$out_dir/$(basename "$nb" .ipynb)_executed.ipynb" \
        --ExecutePreprocessor.timeout=7200
}

if [[ "$METHOD" == "all" ]]; then
    for k in "${!NOTEBOOKS[@]}"; do
        run_one "$k"
    done
else
    run_one "$METHOD"
fi
