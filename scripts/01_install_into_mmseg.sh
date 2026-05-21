#!/usr/bin/env bash
# =============================================================================
#  scripts/01_install_into_mmseg.sh
#
#  Copies every custom Python file from src/ into the canonical paths inside
#  a local MMSegmentation checkout. After running this, the configs under
#  configs/acw/*.py work directly with `mim train mmseg ...` because the
#  custom_imports lines (e.g. mmseg.models.video_modules.adaptive_feature_propagation)
#  resolve to real files on disk.
#
#  Usage:
#      export MMSEG_ROOT=/path/to/mmsegmentation
#      bash scripts/01_install_into_mmseg.sh             # do the copy
#      bash scripts/01_install_into_mmseg.sh --dry-run   # only print actions
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; exit 2 ;;
    esac
done

if [[ -z "${MMSEG_ROOT:-}" ]]; then
    echo "ERROR: please export MMSEG_ROOT=/path/to/mmsegmentation first" >&2
    exit 1
fi

if [[ ! -d "$MMSEG_ROOT/mmseg" ]]; then
    echo "ERROR: $MMSEG_ROOT does not look like an MMSegmentation checkout" >&2
    echo "       (no 'mmseg/' subdirectory found)" >&2
    exit 1
fi

# (source-in-src/ → destination-under-MMSEG_ROOT/mmseg/) tuples
MAPPINGS=(
    "src/models/bisenetv1.py:mmseg/models/backbones/bisenetv1.py"
    "src/models/resnet.py:mmseg/models/backbones/resnet.py"
    "src/models/simple_cls_head.py:mmseg/models/cls_heads/simple_cls_head.py"
    "src/models/dual_task_segmentor.py:mmseg/models/segmentors/dual_task_segmentor.py"
    "src/models/cls_only_segmentor.py:mmseg/models/segmentors/cls_only_segmentor.py"
    "src/models/bisenet_adaptive_video_segmentor.py:mmseg/models/segmentors/bisenet_adaptive_video_segmentor.py"
    "src/models/adaptive_feature_propagation.py:mmseg/models/video_modules/adaptive_feature_propagation.py"
    "src/models/adaptive_keyframe_selector.py:mmseg/models/video_modules/adaptive_keyframe_selector.py"

    "src/preprocessors/pair_seg_data_preprocessor.py:mmseg/models/data_preprocessors/pair_seg_data_preprocessor.py"

    "src/structures/dual_task_seg_data_sample.py:mmseg/structures/dual_task_seg_data_sample.py"

    "src/data/dual_task_dataset.py:mmseg/datasets/dual_task_dataset.py"
    "src/data/video_pair_dual_task_dataset.py:mmseg/datasets/video_pair_dual_task_dataset.py"
    "src/data/scheduler_video_pair_dual_task_dataset.py:mmseg/datasets/scheduler_video_pair_dual_task_dataset.py"

    "src/data/transforms/pack_seg_inputs_with_label.py:mmseg/datasets/transforms/pack_seg_inputs_with_label.py"
    "src/data/transforms/pack_video_pair_inputs.py:mmseg/datasets/transforms/pack_video_pair_inputs.py"
    "src/data/transforms/pack_scheduler_video_pair_inputs.py:mmseg/datasets/transforms/pack_scheduler_video_pair_inputs.py"
    "src/data/transforms/load_video_pair.py:mmseg/datasets/transforms/load_video_pair.py"
    "src/data/transforms/resize_video_pair.py:mmseg/datasets/transforms/resize_video_pair.py"
    "src/data/transforms/random_flip_video_pair.py:mmseg/datasets/transforms/random_flip_video_pair.py"
    "src/data/transforms/random_crop_video_pair.py:mmseg/datasets/transforms/random_crop_video_pair.py"
    "src/data/transforms/pair_random_clahe_video_pair.py:mmseg/datasets/transforms/pair_random_clahe_video_pair.py"
    "src/data/transforms/pair_photometric_distortion_video_pair.py:mmseg/datasets/transforms/pair_photometric_distortion_video_pair.py"
    "src/data/transforms/map_cls_after_flip.py:mmseg/datasets/transforms/map_cls_after_flip.py"
    "src/data/transforms/preresize.py:mmseg/datasets/transforms/preresize.py"
)

run_copy() {
    local src="$1"
    local dst="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] $src -> $dst"
    else
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
        echo "  copied:   $src -> $dst"
    fi
}

# Apply the inverse patches so the files behave as canonical mmseg modules.
# Specifically: dual_task_dataset.py / pack_seg_inputs_with_label.py have
# absolute imports for standalone use; convert them back to relative imports
# when installing into mmseg.
patch_for_mmseg() {
    local dst="$1"
    [[ $DRY_RUN -eq 1 ]] && return
    case "$dst" in
        *mmseg/datasets/dual_task_dataset.py)
            sed -i 's|from mmseg.datasets.basesegdataset import BaseSegDataset|from .basesegdataset import BaseSegDataset|' "$dst"
            ;;
        *mmseg/datasets/transforms/pack_seg_inputs_with_label.py)
            sed -i 's|from mmseg.datasets.transforms.formatting import PackSegInputs as _Pack|from .formatting import PackSegInputs as _Pack|' "$dst"
            ;;
    esac
}

echo "Installing mining-adas-acw into MMSegmentation at: $MMSEG_ROOT"
echo "Mode: $([[ $DRY_RUN -eq 1 ]] && echo dry-run || echo apply)"
echo ""

cd "$REPO_ROOT"
for entry in "${MAPPINGS[@]}"; do
    IFS=':' read -r src rel_dst <<< "$entry"
    if [[ ! -f "$src" ]]; then
        echo "WARNING: source missing: $src (skipping)" >&2
        continue
    fi
    dst="$MMSEG_ROOT/$rel_dst"
    run_copy "$src" "$dst"
    patch_for_mmseg "$dst"
done

echo ""
echo "Done. Now try training with:"
echo ""
echo "    cd \"$MMSEG_ROOT\""
echo "    PYTHONPATH=. python tools/train.py \\"
echo "        \"$REPO_ROOT/configs/acw/03_train_adaptive_scheduler_pretrain.py\""
