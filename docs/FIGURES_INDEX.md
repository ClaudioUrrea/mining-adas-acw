# Figures index

Canonical mapping of every figure file referenced by the paper's `\includegraphics{Figures/...}` calls to where its source data and final PNG live.

> All final PNGs target **≥600 dpi RGB** as required by the Remote Sensing (MDPI) submission system. The verification helper `tools/verify_figures.py` reads `tools/expected_figures.txt` and checks each file's dpi, mode, and minimum size before zipping for submission.

---

## Where the PNGs live

After populating the repository from the Figshare deposit:

```
mining-adas-acw/assets/
├── figures_paper/            # Body figures (fig1 … fig11)
├── figures_supp/             # Appendix and supplementary
├── confusion_matrices/       # cm_<route>_<method>.png
└── qualitative/              # qual_<route>_<frame>_<method>.png
```

---

## Body figures

| Filename | Caption (short) | Source |
|---|---|---|
| `fig1_adas_prototype_a.png` | DSS-type ADAS prototype on a mining haul truck (3D view) | `figshare-deposit/figures_source/fig1_adas_prototype/` |
| `fig1_adas_prototype_b.png` | DSS-type ADAS prototype on a mining haul truck (system block) | same |
| `fig2_mtl_architectures_a.png` | Hard-parameter-sharing MTL architecture | `figshare-deposit/figures_source/fig2_mtl_architectures/` |
| `fig2_mtl_architectures_b.png` | Soft-parameter-sharing MTL architecture | same |
| `fig3_annotations_a.png` | Annotation example — train+test sequence | `figshare-deposit/figures_source/fig3_annotations/` |
| `fig3_annotations_b.png` | Annotation example — train-only sequence | same |
| `fig3_annotations_c.png` | Annotation example — test-only sequence | same |
| `fig4_architecture_a.png` | Multitask BiSeNetV1 architecture overview | `figshare-deposit/figures_source/fig4_architecture/` |
| `fig4_architecture_b.png` | Semantic Segmentation head (SS Head) detail | same |
| `fig4_architecture_c.png` | Classification head (Cls Head) detail | same |
| `fig5_clockwork.png` | Clockwork timing diagram (FBF vs F-CW vs A-CW) | `figshare-deposit/figures_source/fig5_clockwork/` |
| `fig6_pipeline.png` | Inference pipeline diagram | `figshare-deposit/figures_source/fig6_pipeline/` |
| `fig7_training_curves.png` | Multitask training: loss / Top-1 / mIoU vs iteration | `figshare-deposit/figures_source/fig7_training/regen_fig7.py` |
| `fig8_feature_variation.png` | Feature variation between consecutive frames | `figshare-deposit/figures_source/fig8_feature_variation/regen_fig8.py` |
| `fig9_accuracy_fps_a.png` | Accuracy vs FPS bar chart, Route A | `figshare-deposit/figures_source/fig9_accuracy_fps/regen_fig9.py` |
| `fig9_accuracy_fps_b.png` | Accuracy vs FPS bar chart, Route B | same |
| `fig10_miou_boxplot.png` | Per-frame mIoU boxplots across methods | `figshare-deposit/figures_source/fig10_miou/regen_fig10.py` |
| `trajectory_graph_A.png` | IMU-derived yaw trajectory, Route A | `figshare-deposit/figures_source/trajectory_graphs/regen_trajectory.py` |
| `trajectory_graph_B.png` | IMU-derived yaw trajectory, Route B | same |

---

## Confusion matrices

Naming: `cm_<route>_<method>.png`, with `<route> ∈ {A, B}` and `<method> ∈ {fcw, dff, llm, acw}`.

8 files total:

```
cm_A_acw.png   cm_A_dff.png   cm_A_fcw.png   cm_A_llm.png
cm_B_acw.png   cm_B_dff.png   cm_B_fcw.png   cm_B_llm.png
```

Source: `figshare-deposit/figures_source/confusion_matrices/cm_<route>_<method>.csv` + `regen_cm.py`.

---

## Qualitative overlays

Naming: `qual_<route>_<frame>_<method>.png`.

24 files total (Routes A and B × 3 sample frames each × 4 methods):

```
qual_A_10_acw.png    qual_A_10_dff.png    qual_A_10_fcw.png    qual_A_10_llm.png
qual_A_35_acw.png    qual_A_35_dff.png    qual_A_35_fcw.png    qual_A_35_llm.png
qual_A_90_acw.png    qual_A_90_dff.png    qual_A_90_fcw.png    qual_A_90_llm.png
qual_B_15_acw.png    qual_B_15_dff.png    qual_B_15_fcw.png    qual_B_15_llm.png
qual_B_30_acw.png    qual_B_30_dff.png    qual_B_30_fcw.png    qual_B_30_llm.png
qual_B_60_acw.png    qual_B_60_dff.png    qual_B_60_fcw.png    qual_B_60_llm.png
```

Source: snapshot frames from `notebooks/videos_multitask_*.ipynb`, copied to `figshare-deposit/figures_source/qualitative_overlays/`.

---

## Total figure count

| Group | Count |
|---|---|
| Body figures (Fig. 1 – Fig. 11) | 17 |
| Trajectory graphs | 2 |
| Confusion matrices | 8 |
| Qualitative overlays | 24 |
| **Total** | **51** |

The single ZIP for the MDPI submission is built with:

```bash
bash tools/build_figures_zip.sh
```

This:
1. runs `tools/verify_figures.py` against `tools/expected_figures.txt`,
2. fails if any file is missing, < 50 KB, < 600 dpi, or non-RGB,
3. assembles `figures_zip_for_submission.zip` (release asset for the GitHub tag), and
4. emits `figures_zip_for_submission.sha256`.
