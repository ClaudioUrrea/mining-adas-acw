# Installation guide

Two supported workflows. Read both, pick whichever fits your setup.

---

## Tested environment

| Component | Tested version |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.13 |
| NVIDIA driver | 535.x |
| CUDA toolkit | 12.1 |
| PyTorch | 2.1.2+cu121 |
| MMEngine | 0.10.x |
| MMCV | 2.1.0 |
| MMSegmentation | 1.2.2 |
| GPU | NVIDIA RTX 3090 (24 GB) used for FPS benchmarks |

Other Ubuntu / CUDA combinations work but the **MMCV wheel must be installed through `openmim`** so it picks the wheel that matches your torch + CUDA combination. Mismatched MMCV is the single most common cause of crashes at import time.

---

## Option A — Standalone (recommended for inference, notebooks, FPS benchmarks)

This installs the repo as a self-contained Python package: all the custom MMSeg modules live under `src/` and register into MMSeg's global registry when you `import src`.

```bash
git clone https://github.com/ClaudioUrrea/mining-adas-acw.git
cd mining-adas-acw

# Strongly recommended: conda environment (matches the published versions)
conda env create -f environment.yml
conda activate mining-adas-acw

# Pip-only alternative (more brittle; only do this if you can't use conda)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# MMCV + MMSegmentation MUST come through mim, in this exact order
mim install "mmengine>=0.10,<0.11"
mim install "mmcv==2.1.0"
mim install "mmsegmentation==1.2.2"
```

Sanity check — every module is now in the MMSeg registry:

```bash
python -c "import src; print('Registered OK')"
python -c "import src; from mmseg.registry import MODELS; \
print('BiSeNetV1' in MODELS); \
print('BiSeNetAdaptiveVideoSegmentor' in MODELS); \
print('AdaptiveKeyFrameSelector' in MODELS)"
```

Expected:
```
Registered OK
True
True
True
```

---

## Option B — Drop-in to an existing MMSegmentation tree

Some training scripts (under `configs/acw/*.py`) reference modules by their **canonical MMSeg paths** (e.g. `mmseg.models.backbones.bisenetv1`). For those, the cleanest setup is to copy each `src/*.py` file to its native MMSeg path. The helper script does this for you:

```bash
# Set the path to your local MMSegmentation checkout
export MMSEG_ROOT=/path/to/mmsegmentation

# Optional: dry-run first
bash scripts/01_install_into_mmseg.sh --dry-run

# Perform the copies (idempotent — re-running just overwrites)
bash scripts/01_install_into_mmseg.sh
```

After this, the configs under `configs/acw/` work directly with `mim train mmseg ...` exactly as they did in the original training environment.

---

## Verifying the installation with the released checkpoints

After downloading the Figshare bundle (see `docs/DATA_DICTIONARY.md`):

```bash
# 1. Validate that a checkpoint loads cleanly
python tools/validate_checkpoint.py \
    checkpoints/best_cls_acc_cls_top1_iter_25600.pth \
    --config configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py

# 2. Confirm the per-file SHA-256 of the Figshare bundle
cd /path/to/figshare-bundle
sha256sum --check MANIFEST.sha256
```

---

## Troubleshooting

### `OSError: CUDA error: invalid device function`
Your MMCV wheel was compiled for a different CUDA version than your torch. Reinstall through mim:
```bash
pip uninstall mmcv mmcv-full -y
mim install "mmcv==2.1.0"
```

### `ImportError: cannot import name 'resize' from 'mmseg.models.utils'`
You installed MMSegmentation < 1.0. This repo requires **MMSegmentation ≥ 1.2.2**:
```bash
mim install "mmsegmentation==1.2.2"
```

### `KeyError: 'BiSeNetAdaptiveVideoSegmentor'` when calling `Config.fromfile(...)`
The config's `custom_imports` block expects the modules to live at canonical MMSeg paths. Either:
- run `scripts/01_install_into_mmseg.sh` (Option B above), or
- add `import src` before reading the config.

### Notebooks can't find the modules
Run jupyter from the repo root so `src/` is on `sys.path`:
```bash
cd mining-adas-acw
jupyter lab
```
The first cell of each notebook should be `import src  # registers MMSeg modules`. If the notebook was authored before `src/` existed, append it manually at the top.

### Disk / GPU memory budget
- Training the multitask model from scratch needs ≈11 GB GPU memory at batch size 4, 512×512.
- A-CW propagation pre-training adds ≈2 GB on top of the multitask model.
- The Figshare bundle is ≈4 GB compressed; budget ≈12 GB unpacked.

---

If anything else breaks, please open a GitHub issue with the full traceback and the output of:

```bash
python -c "import torch, mmengine, mmcv, mmseg; \
print(torch.__version__, mmengine.__version__, mmcv.__version__, mmseg.__version__, torch.version.cuda)"
```
