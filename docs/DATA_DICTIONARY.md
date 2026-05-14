# Data dictionary

Authoritative description of every file released on the companion **Figshare** deposit and every pointer file in the **GitHub** repository.

> Figshare DOI: <https://doi.org/10.6084/m9.figshare.32274630>
> Source code: <https://github.com/ClaudioUrrea/mining-adas-acw>

Units, dtypes and accepted value ranges are listed for every field so a third party can re-implement readers and re-validate the deposit without consulting the paper.

---

## 1. Bundle layout

```
figshare-deposit/
├── README.md
├── LICENSE.txt                 # CC BY 4.0
├── CHANGELOG.md
├── metadata.json               # DataCite-style FAIR metadata
├── MANIFEST.sha256             # per-file SHA-256, sorted by path
│
├── dataset/
│   ├── annotations/            # 100 manually drawn binary masks (1-bit PNG)
│   ├── splits/                 # train/val/test split text files (8-bit ASCII)
│   ├── routes/                 # IMU-derived per-frame metadata (CSV + Parquet)
│   ├── train/                  # Manifest of training-frame relative paths
│   ├── test_route_a/           # Manifest of Route-A test-frame paths
│   └── test_route_b/           # Manifest of Route-B test-frame paths
│
├── checkpoints/                # PyTorch .pth (1 model per file)
│   └── *_metadata.json         # Sibling metadata file for every checkpoint
│
├── results/                    # Per-frame inference outputs
│   ├── frame_by_frame/
│   ├── fixed_clockwork/
│   ├── adaptive_clockwork/
│   ├── dff/
│   ├── llm/
│   └── multitask_validation/
│
├── figures_source/             # Numeric data + regen_*.py for every figure
│
├── statistical_analysis/       # Aggregated CSV summaries (boxplots, FPS, etc.)
│
└── docs/                       # Self-contained copies of these docs
```

---

## 2. `dataset/annotations/` — manually drawn binary masks

- **Filenames**: `<unix_ts>.png` (e.g. `1661922815.000000.png`). The `<unix_ts>` matches the AutoMine frame timestamp.
- **Format**: 1-bit PNG, no alpha, palette `[0, 0, 0]` (background), `[0, 128, 0]` (road).
- **Dimensions**: 1080 × 1920, the native AutoMine resolution. Network input is 512 × 512 after a single `Resize` with `keep_ratio=False`.
- **Count**: 100 masks total (60 train / 20 val_aug / 20 test split across Routes A and B).
- **Provenance**: drawn manually frame by frame in a polygon editor and reviewed for self-consistency along straight segments.
- **License**: CC BY 4.0.
- **Companion**: each filename appears in exactly one of `splits/train.txt`, `splits/val.txt`, `splits/test_route_a.txt`, `splits/test_route_b.txt`.

---

## 3. `dataset/splits/` — train / val / test partition files

| File | Lines | Format | Purpose |
|---|---|---|---|
| `train.txt` | 60 | one `<unix_ts>.png` per line, no header | training set |
| `val_aug.txt` | 20 | same | validation set (augmented copy of route A) |
| `test_route_a.txt` | 20 | same | held-out Route A test |
| `test_route_b.txt` | 20 | same | held-out Route B test |
| `cls_labels_train.txt` | 60 | `<unix_ts>.png <LABEL>` per line | training classification labels |
| `cls_labels_val_aug_fixed.txt` | 20 | same | validation classification labels |
| `cls_labels_val_mixed.txt` | ~80 | same | mixed-augmentation validation set |

`<LABEL>` ∈ `{LEFT, STRAIGHT, RIGHT}` (canonical) or `{izquierda, recta, derecha}` (Spanish aliases used in the training configs; both forms map to the same integer in `VideoPairDualTaskDataset.label_map`).

Integer mapping used in checkpoints: `LEFT=0`, `STRAIGHT=1`, `RIGHT=2`.

---

## 4. `dataset/routes/` — IMU-derived per-frame metadata

CSV + Parquet (identical content). One row per frame of each route, sorted by `frame_idx`. The yaw-derived steering label is the GT used for Top-1 Accuracy and for the A-CW classification head.

| Column | Type | Unit | Range | Description |
|---|---|---|---|---|
| `route` | str | – | `{A, B}` | Sequence identifier |
| `frame_idx` | int | – | ≥ 0 | 0-based frame index inside the route |
| `unix_ts` | float | s | – | AutoMine timestamp |
| `qx`, `qy`, `qz`, `qw` | float | – | – | IMU orientation quaternion |
| `roll`, `pitch`, `yaw` | float | rad | $[-\pi, \pi]$ | Euler angles (XYZ convention) derived from the quaternion |
| `yaw_rate` | float | rad/s | – | Numerical derivative of `yaw` (3-point central) |
| `cls_gt` | str | – | `{LEFT, STRAIGHT, RIGHT}` | Sparse steering label from rule on `yaw_rate` |
| `cls_gt_int` | int | – | `{0, 1, 2}` | Integer encoding of `cls_gt` |

`cls_gt` is derived by thresholding `|yaw_rate|` against an empirically chosen ±0.025 rad/s window; samples in the dead-zone are labelled `STRAIGHT`.

---

## 5. `checkpoints/` — trained model weights

Naming convention follows MMSegmentation's `CheckpointHook`:

```
best_<metric>_<value>_iter_<iter>.pth
```

Released checkpoints:

| File | What it is | Purpose | Approx size |
|---|---|---|---|
| `multitask_BC_iter_25600.pth` | Multitask, best Top-1 (BC) | Drop-in for paper Fig. 7 BC row | ≈ 320 MB |
| `multitask_BS_iter_33400.pth` | Multitask, best mIoU (BS) | Drop-in for paper Fig. 7 BS row | ≈ 320 MB |
| `seg_only_best_miou.pth` | Single-task seg baseline | Table 3 row | ≈ 310 MB |
| `cls_only_best_top1.pth` | Single-task cls baseline | Table 3 row | ≈ 310 MB |
| `acw_propagation_iter_2200.pth` | A-CW Stage 1 (propagation only) | Reproduces `cfg/acw/02_train_*` | ≈ 360 MB |
| `acw_scheduler_iter_16200.pth` | A-CW Stage 2 (final A-CW) | Reproduces `cfg/acw/03_train_*` + Stateful eval | ≈ 360 MB |
| `dff_iter_XXXX.pth` | DFF baseline | FPS / mIoU comparison row | ≈ 320 MB |
| `llm_iter_XXXX.pth` | LLM baseline | FPS / mIoU comparison row | ≈ 320 MB |

Each `.pth` ships next to a `<basename>_metadata.json` with the following schema:

```json
{
  "file": "multitask_BC_iter_25600.pth",
  "sha256": "0e7c…",
  "size_bytes": 327680000,
  "config_relpath": "configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py",
  "framework": "pytorch 2.1.2 / mmsegmentation 1.2.2 / mmcv 2.1.0",
  "trained_on": "NVIDIA RTX 3090",
  "training_iters": 25600,
  "expected_metrics": {
    "Top-1 Acc": 0.9625,
    "mIoU": 0.9733
  },
  "license": "CC BY 4.0"
}
```

Use `tools/validate_checkpoint.py` to confirm a checkpoint loads cleanly and matches its declared SHA-256 before opening an issue.

---

## 6. `results/` — per-frame inference outputs

For each method directory (`frame_by_frame`, `fixed_clockwork`, `adaptive_clockwork`, `dff`, `llm`, `multitask_validation`) and each route (`A`, `B`), two files share the same content:

```
results/<method>/<route>__per_frame.csv
results/<method>/<route>__per_frame.parquet
```

Schema (one row per frame, sorted by `frame_idx`):

| Column | Type | Unit | Range | Description |
|---|---|---|---|---|
| `route` | str | – | `{A, B}` | Sequence identifier |
| `frame_idx` | int | – | ≥ 0 | 0-based index inside the route |
| `t_seconds` | float | s | ≥ 0 | Frame time (frame_idx / fps) |
| `miou` | float | – | $[0, 1]$ | Per-frame mIoU vs GT mask (sparse — only at annotated frames; `NaN` otherwise) |
| `top1_correct` | int | – | `{0, 1}` | 1 if predicted class equals `cls_gt` |
| `fire` | int | – | `{0, 1}` | 1 if the context path ran this frame (always 1 for FBF; periodic for F-CW; learned for A-CW) |
| `keyframe_age` | int | frames | ≥ 0 | Frames since last `fire` |
| `time_spatial_ms` | float | ms | ≥ 0 | Wall-clock time for the spatial-path forward |
| `time_context_ms` | float | ms | ≥ 0 | Wall-clock time for the context-path forward (0 if not fired) |
| `time_total_ms` | float | ms | ≥ 0 | Total per-frame inference time (sum of paths + heads) |
| `dev_pred` | float | – | $[0, 1]$ | A-CW only: scheduler output before thresholding (`NaN` for other methods) |
| `gt_cls` | int | – | `{0, 1, 2}` | Ground-truth class (`LEFT=0`, `STRAIGHT=1`, `RIGHT=2`) |
| `pred_cls` | int | – | `{0, 1, 2}` | Predicted class |

The two formats are byte-identical in content; Parquet is provided for fast columnar reads in pandas / polars / DuckDB.

---

## 7. `figures_source/` — regenerate every paper figure

Each subdirectory matches a paper figure and contains:
- the **numeric data** used to draw it (`.csv` / `.parquet`);
- a **regen script** (`regen_*.py`) that reads the data and writes the `.png` at ≥600 dpi.

```
figures_source/
├── fig7_training/                  loss_per_iter.csv, top1_per_iter.csv, miou_per_iter.csv, regen_fig7.py
├── fig8_feature_variation/         feature_variation_per_frame.csv,      regen_fig8.py
├── fig9_accuracy_fps/              fps_per_method_per_route.csv,         regen_fig9.py
├── fig10_miou/                     miou_per_method_per_route.csv,        regen_fig10.py
├── fig11_keyframes/                keyframe_stats_per_method.csv,        regen_fig11.py
├── confusion_matrices/             cm_<route>_<method>.csv,              regen_cm.py
├── qualitative_overlays/           overlay_<route>_<frame>_<method>.png  (snapshot frames)
├── trajectory_graphs/              trajectory_<route>.csv,               regen_trajectory.py
├── fig3_annotations/               (qualitative crops, no regen)
├── fig4_architecture/              architecture.tex                      (TikZ)
├── fig5_clockwork/                 clockwork.tex                         (TikZ)
└── fig6_pipeline/                  pipeline.tex                          (TikZ)
```

Reproducing every PNG:

```bash
cd figures_source
for d in */; do
    if [ -f "$d/regen_$(basename $d | sed s/^[0-9]*_//).py" ]; then
        python -m "$d/regen_*"
    fi
done
```

---

## 8. `statistical_analysis/` — aggregated summaries

Single-file rollups used to produce the headline numbers in Table 4 and Table 5:

| File | Description |
|---|---|
| `summary_per_method.csv` | One row per (method, route): mean mIoU, std mIoU, mean Top-1, mean GPU-only FPS, mean E2E FPS, % FIRE frames |
| `keyframes_summary.csv` | One row per A-CW config: `tau`, total `fire` count, mean keyframe interval, std keyframe interval |
| `miou_distribution_per_method.parquet` | All raw per-frame mIoU values pooled by (method, route) for the boxplots in Fig. 10 |
| `fps_distribution_per_method.parquet` | Same for FPS — used by Fig. 9 |

---

## 9. `MANIFEST.sha256` — integrity verification

Sorted, deterministic SHA-256 for every file in the deposit (excluding itself):

```bash
cd figshare-deposit
find . -type f ! -name 'MANIFEST.sha256' -print0 | sort -z \
  | xargs -0 sha256sum > MANIFEST.sha256
```

Verify after download:
```bash
sha256sum --check MANIFEST.sha256
```

The exact same recipe (with a `--quiet` flag) lives in `tools/compute_sha256_manifest.sh`.

---

## 10. License summary

| Item | License | Reference |
|---|---|---|
| Source code (this repo) | MIT | `LICENSE` |
| Derived masks, splits, IMU labels, checkpoints, per-frame results, figure sources, stats | CC BY 4.0 | `figshare-deposit/LICENSE.txt` |
| Original AutoMine RGB frames and IMU streams | AutoMine project license | `https://github.com/AutoMine/AutoMine` |

If you redistribute any artefact from the Figshare deposit you must keep the CC BY 4.0 attribution and cite the paper.
