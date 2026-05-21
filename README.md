# Mining ADAS — Adaptive Clockwork Multitask Perception (BiSeNetV1 + A-CW)

Official source code accompanying the paper:


> **Vélez, M.; Urrea, C.** (2026)
> *Efficient Multitask Onboard Vision Sensing for Open-Pit Mining ADAS with Classification-Guided Adaptive Temporal Inference.*
> **Sensors**, MDPI. <https://doi.org/10.3390/s1010000>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Figshare DOI](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.32274630-blue)](https://doi.org/10.6084/m9.figshare.32274630)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)

---

## Overview

This repository ships the complete implementation of a video-based multitask scene-perception pipeline for open-pit mining ADAS, combining drivable-area semantic segmentation and steering-direction classification within a single BiSeNetV1-based architecture. On top of the multitask backbone we propose **Adaptive Clockwork (A-CW)**: a classification-guided inference policy that updates the slow Context Path only on keyframes whose cadence is decided online by a learned scheduler.

**Headline numbers (from the paper)**

| Metric | Frame-by-Frame | Fixed Clockwork (k=30) | DFF | LLM | **A-CW (ours)** |
|---|---:|---:|---:|---:|---:|
| GPU-only FPS, Route A | 55.1 | 168.4 | 51.5 | 47.7 | **183.6** |
| End-to-end FPS, Route A | 32.7 | 38.0 | 31.4 | 27.8 | **38.5** |
| mIoU, Route A | 91.6 % | 88.2 % | 79.7 % | 83.5 % | **90.7 %** |
| Top-1 Acc, Route A | 95.0 % | 93.8 % | 92.5 % | 92.5 % | **96.2 %** |

A-CW more than triples GPU-only throughput vs the frame-by-frame baseline while keeping segmentation quality within ≈1 mIoU point and improving classification accuracy.

---

## Key features

- **Multitask BiSeNetV1 + ResNet-50** backbone with two heads: a segmentation head (FCN) for the drivable area and a `SimpleClsHead` (GAP→Dropout→FC) for the three steering classes (`LEFT`, `STRAIGHT`, `RIGHT`).
- **Classification-guided Adaptive Clockwork** (`src/models/bisenet_adaptive_video_segmentor.py`): the spatial path runs on every frame; the context path runs only when the learned scheduler `AdaptiveKeyFrameSelector` fires.
- **Adaptive feature propagation** (`src/models/adaptive_feature_propagation.py`): a `KernelPredictor` produces locally adaptive 3×3 weights and a `SpatiallyVariantConv2d` warps the cached context from the last keyframe.
- **No densely annotated video** is required for training: the entire pipeline is supervised from 100 manually annotated isolated frames plus IMU-derived sparse classification labels.
- **Reproducible baselines**: clean MMSeg-compatible implementations of Frame-by-Frame, Fixed Clockwork (k=1, k=30), Deep Feature Flow (DFF) and Low-Latency Method (LLM) on the same BiSeNet backbone.
- **End-to-end FPS benchmarks** (`notebooks/benchmark_*.ipynb`): GPU-only and full-pipeline including I/O, decode, normalisation, inference, post-processing and the simulated 1080×720 RGB HMI overlay.
- **FAIR-compliant data release**: companion Figshare deposit ships derived masks, IMU-derived labels, train/val/test splits, every trained checkpoint, every per-frame inference result (CSV + Parquet), the statistical analysis CSVs and a per-file SHA-256 manifest.

---

## Repository layout

```
mining-adas-acw/
├── README.md                              # This file
├── LICENSE                                # MIT (source code)
├── CITATION.cff                           # Machine-readable citation
├── requirements.txt                       # Pinned pip dependencies
├── environment.yml                        # Conda environment (alternative)
├── .gitignore
│
├── configs/                               # MMSegmentation-style config files
│   ├── multitask_bisenet/                 #   Multitask (BiSeNetV1 + cls head)
│   ├── singletask/                        #   Seg-only and Cls-only ablations
│   ├── acw/                               #   Adaptive Clockwork (proposed)
│   ├── fcw/                               #   Fixed Clockwork baseline
│   ├── dff/                               #   Deep Feature Flow baseline
│   └── llm/                               #   Low-Latency Method baseline
│
├── src/
│   ├── __init__.py                        # Side-effect import: registers
│   ├── registry.py                        #   every custom MMSeg module
│   ├── models/                            # Backbones, heads, segmentors
│   ├── data/                              # Datasets + transforms
│   ├── preprocessors/                     # Pair-aware data preprocessor
│   ├── structures/                        # DualTaskSegDataSample
│   └── utils/                             # Thin shim to mmseg.models.utils
│
├── notebooks/                             # All training / eval / benchmark
├── scripts/                               # Top-level reproducibility shells
├── tools/                                 # Maintenance helpers
├── data/                                  # Pointer dirs (populated from Figshare)
├── checkpoints/                           # Pointer dir (populated from Figshare)
├── assets/                                # Paper figures (after extraction)
└── docs/
    ├── INSTALLATION.md
    ├── REPRODUCIBILITY.md
    ├── DATA_DICTIONARY.md
    ├── FIGURES_INDEX.md
    └── THIRD_PARTY_NOTICES.md
```

---

## Installation

Two supported workflows; pick whichever fits your setup. Full step-by-step instructions live in [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

### Option A — Standalone (recommended for inference / notebooks)

```bash
git clone https://github.com/ClaudioUrrea/mining-adas-acw.git
cd mining-adas-acw
conda env create -f environment.yml
conda activate mining-adas-acw
mim install "mmengine>=0.10,<0.11" "mmcv==2.1.0" "mmsegmentation==1.2.2"
```

Quick sanity check:
```bash
python -c "import src; print('Registered OK')"
```

### Option B — Drop-in to an existing MMSegmentation tree

```bash
MMSEG_ROOT=/path/to/mmsegmentation bash scripts/01_install_into_mmseg.sh
```

---

## Data

The original AutoMine RGB frames and IMU streams are **not** redistributed here; they are governed by the AutoMine project's own license and must be requested from the original authors:

> Li, Y. *et al.* "AutoMine: An Unmanned Mine Dataset." *CVPR 2022*.

All material **derived** from AutoMine in this work is on **Figshare** under CC BY 4.0:

> **Mining-ADAS-ACW: Dataset, Checkpoints, and Reproducibility Bundle** (2026).
> Figshare. <https://doi.org/10.6084/m9.figshare.32274630>

After downloading, place the contents at the corresponding paths under `data/` and `checkpoints/`, or set `MINING_ADAS_DATA_ROOT` to the Figshare extraction directory.

---

## Quick start

```bash
python tools/validate_checkpoint.py checkpoints/best_cls_acc_cls_top1_iter_25600.pth
bash scripts/05_benchmark_methods.sh acw
```

---

## Reproducing the paper

The mapping from every paper figure/table to its generating notebook is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## License

- **Code** in this repository is licensed under the **MIT License** ([`LICENSE`](LICENSE)).
- **Data, checkpoints and figure sources** on the companion Figshare deposit are licensed under **CC BY 4.0**.
- The **AutoMine** source imagery is governed by its own license; refer to the AutoMine project for terms.

---

## Citation

```bibtex
@article{Urrea2026MiningADAS,
  title   = {Efficient Multitask Onboard Vision Sensing for Open-Pit Mining
             ADAS with Classification-Guided Adaptive Temporal Inference},
  author  = {V{\'e}lez, Maximiliano and Urrea, Claudio},
  journal = {Sensors},
  year    = {2026},
  volume  = {1},
  number  = {1},
  pages   = {0},
  doi     = {10.3390/s1010000},
  url     = {https://doi.org/10.3390/s1010000}
}

@misc{Urrea2026MiningADASFigshare,
  title     = {Mining-ADAS-ACW: Dataset, Checkpoints, and Reproducibility Bundle},
  author    = {Urrea, Claudio and V{\'e}lez, Maximiliano},
  year      = {2026},
  doi       = {10.6084/m9.figshare.32274630},
  url       = {https://doi.org/10.6084/m9.figshare.32274630},
  publisher = {Figshare}
}
```

---

## Acknowledgments

This research was supported by **ANID FONDEQUIP Mediano EQM230160**, **ANID BECAS/DOCTORADO NACIONAL 2025**, and the **University of Santiago of Chile**.

---

## Contact

- Claudio Urrea — `claudio.urrea@usach.cl` — Electrical Engineering Department, Faculty of Engineering, University of Santiago of Chile.
