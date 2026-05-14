# Third-party notices

This repository contains material derived from third-party projects. Their
original licenses are reproduced or referenced below. Nothing in this file
overrides the project's primary `LICENSE` (MIT); it documents the upstream
provenance and required attributions for the derived files.

---

## OpenMMLab — MMSegmentation

Project: <https://github.com/open-mmlab/mmsegmentation>
License: **Apache License 2.0** — <https://www.apache.org/licenses/LICENSE-2.0>
Copyright (c) OpenMMLab. All rights reserved.

### Files derived from MMSegmentation

| File in this repo | Upstream origin (approximate) | Nature of modification |
|---|---|---|
| `src/models/bisenetv1.py` | `mmseg/models/backbones/bisenetv1.py` | Added explicit `x_spatial` output (`out_indices` includes `3`) so the Spatial Path can be reused as the keyframe-selector input. Original docstrings and Apache 2.0 header preserved. |
| `src/models/resnet.py` | `mmseg/models/backbones/resnet.py` | Unmodified copy. |
| `src/data/dual_task_dataset.py` | `mmseg/datasets/basesegdataset.py` (as parent) | Subclasses `BaseSegDataset` to inject per-image classification labels read from a separate text file. |
| `src/data/transforms/pack_seg_inputs_with_label.py` | `mmseg/datasets/transforms/formatting.py` (`PackSegInputs`) | Subclasses `PackSegInputs` to also pack `gt_label` as `LabelData`. |

The other files in `src/` (the dual-task segmentor, A-CW video segmentor, adaptive feature propagation, keyframe selector, pair-aware transforms and preprocessors) are **original work** authored for this paper. They register into MMSegmentation's registry but they are not derivatives of MMSegmentation source files.

### Reproduced Apache 2.0 header

Every file in `src/models/bisenetv1.py` and `src/models/resnet.py` keeps the
upstream header line:

```
# Copyright (c) OpenMMLab. All rights reserved.
```

A copy of the Apache License 2.0 is available at the URL above.

---

## PyTorch

Project: <https://github.com/pytorch/pytorch>
License: BSD 3-Clause.

No PyTorch source is bundled in this repository — only an import-time
dependency. Standard pip / conda install.

---

## AutoMine

Project: <https://github.com/AutoMine/AutoMine>
Paper: Li, Y. *et al.* "AutoMine: An Unmanned Mine Dataset." *CVPR 2022*.

The original AutoMine RGB frames and IMU streams are **not** redistributed by
this repository or by the companion Figshare deposit. Access to them is
governed by the AutoMine project's own license; please obtain them directly
from the AutoMine authors.

Only the following **derived** artefacts are released by this work, under
**CC BY 4.0**:

- manually drawn binary masks (`figshare-deposit/dataset/annotations/`),
- IMU-derived per-frame straight/curve labels (`figshare-deposit/dataset/routes/`),
- train / val / test split files (`figshare-deposit/dataset/splits/`).

---

## How to extend this file

If you add a new file under `src/` that is derived from a third-party project,
add a row to the table above and reference the upstream license. Same rule for
any new figure-source script that incorporates third-party code.
