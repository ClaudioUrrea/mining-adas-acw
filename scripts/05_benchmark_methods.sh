#!/usr/bin/env bash
# =============================================================================
#  scripts/05_benchmark_methods.sh
#
#  Runs the GPU-only and end-to-end FPS benchmark notebooks. Picks the right
#  notebook for the method passed as argument:
#
#      bash scripts/05_benchmark_methods.sh fbf      # frame-by-frame
#      bash scripts/05_benchmark_methods.sh fcw-k1   # Fixed Clockwork k=1
#      bash scripts/05_benchmark_methods.sh fcw-k30  # Fixed Clockwork k=30
#      bash scripts/05_benchmark_methods.sh acw      # Adaptive Clockwork
#      bash scripts/05_benchmark_methods.sh all      # run every method
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

declare -A NOTEBOOKS=(
    [fcw-k1]="notebooks/benchmark_clockwork_1a_k1_fixed_gpu_only.ipynb"
    [fcw-k30]="notebooks/benchmark_clockwork_1a_k30_fixed_gpu_only.ipynb"
    [fcw-adap]="notebooks/benchmark_clockwork_adaptive_1a_kc30kl100_gpu_only_1__code_only_EN_comments_unexecuted_v3.ipynb"
    [acw]="notebooks/benchmark_adaptive_scheduler_fullseq_tau003_1a_gpu_only_match_realista_code_only_EN_comments_unexecuted.ipynb"
)

METHOD="${1:-acw}"

run_one() {
    local key="$1"
    local nb="${NOTEBOOKS[$key]:-}"
    if [[ -z "$nb" || ! -f "$nb" ]]; then
        echo "WARNING: notebook for '$key' not found, skipping" >&2
        return
    fi
    local out_dir="runs/bench_${key}"
    mkdir -p "$out_dir"
    echo ""
    echo "==> Running benchmark: $key"
    PYTHONPATH=".:${PYTHONPATH:-}" jupyter nbconvert \
        --to notebook \
        --execute "$nb" \
        --output "$out_dir/$(basename "$nb" .ipynb)_executed.ipynb" \
        --ExecutePreprocessor.timeout=3600
}

if [[ "$METHOD" == "all" ]]; then
    for k in "${!NOTEBOOKS[@]}"; do
        run_one "$k"
    done
else
    run_one "$METHOD"
fi
