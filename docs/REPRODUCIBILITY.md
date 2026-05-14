# Reproducing the paper

This document is a step-by-step map from every published table and figure to the exact script or notebook that regenerates it. Numbering follows the paper.

> **Notation.** `nb/X` means `notebooks/X` in this repository. `cfg/X` means `configs/X`. `cp/X` means `checkpoints/X` after the Figshare bundle has been extracted to `checkpoints/`.

---

## End-to-end workflow

```
Step 1 — Single-task training (ablation reference)
    cfg/singletask/bisenet_seg_only_strict.py    ── seg-only baseline
    cfg/singletask/bisenet_cls_only_strict.py    ── cls-only baseline

Step 2 — Multitask training (paper main model)
    cfg/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py

Step 3 — A-CW Stage 1: feature propagation pre-training
    cfg/acw/02_train_adaptive_propagation_pretrain.py

Step 4 — A-CW Stage 2: scheduler training (freeze base)
    cfg/acw/03_train_adaptive_scheduler_pretrain.py

Step 5 — Stateful evaluation of A-CW on Routes A and B
    cfg/acw/04_test_adaptive_scheduler_stateful.py

Step 6 — Baseline temporal methods
    Fixed Clockwork k=1   ── nb/eval_clockwork_1a_k1.ipynb
    Fixed Clockwork k=30  ── nb/eval_clockwork_1a_k30.ipynb
    DFF                   ── (checkpoint + per-frame results on Figshare)
    LLM                   ── nb/E_02_Low_latency_Method_Pretrain_Freez.ipynb
                             nb/E_03_Low_latency_Method_Pretrain.ipynb

Step 7 — End-to-end FPS benchmarks (GPU-only + E2E with HMI)
    nb/benchmark_*.ipynb

Step 8 — Statistical analysis (boxplots, keyframe counts)
    Numbers live in figshare-deposit/statistical_analysis/
```

---

## Figure-by-figure

| Figure | Paper title | Generator |
|---|---|---|
| Fig. 1 | ADAS prototype | hand-drawn (CAD); source files in `figshare-deposit/figures_source/fig1_adas_prototype/` |
| Fig. 2 | MTL architecture diagrams | hand-drawn (TikZ); source `.tex` in `figshare-deposit/figures_source/fig2_mtl_architectures/` |
| Fig. 3 | Annotation examples | `nb/eval_BiSeNetSegOnly_val_sin_secuencia.ipynb` cell "qualitative samples" |
| Fig. 4 | Architecture overview | hand-drawn (TikZ); source `.tex` in `figshare-deposit/figures_source/fig4_architecture/` |
| Fig. 5 | Clockwork timing diagram | hand-drawn (TikZ); source `.tex` in `figshare-deposit/figures_source/fig5_clockwork/` |
| Fig. 6 | Inference pipeline diagram | hand-drawn (TikZ); source `.tex` in `figshare-deposit/figures_source/fig6_pipeline/` |
| Fig. 7 | Training curves (loss, Top-1, mIoU) | `nb/E_multitask_val_fix_wloss.ipynb`; CSV in `figshare-deposit/figures_source/fig7_training/` |
| Fig. 8 | Feature variation across frames | `nb/eval_clockwork_adap_1a_kc30kl100_nextframe_code_only_EN_comments_unexecuted_v3.ipynb` |
| Fig. 9a | GPU-only FPS bars, Route A | `nb/benchmark_adaptive_scheduler_fullseq_tau003_1a_gpu_only_match_realista_code_only_EN_comments_unexecuted.ipynb` |
| Fig. 9b | End-to-end FPS bars, Route A | same notebook, second figure cell |
| Fig. 10 | mIoU boxplots | `nb/eval_stateful_adaptive_scheduler_fullseq_tau003_1a_code_only_EN_comments_unexecuted.ipynb` |
| Fig. 11 | Adaptive keyframe statistics | same notebook, "keyframe stats" cell |
| Confusion matrices | `cm_A_acw.png`, `cm_A_fcw.png`, … | `nb/eval_clockwork_*.ipynb` and `nb/eval_stateful_*.ipynb` |
| Qualitative overlays | `qual_A_10_acw.png`, etc. | `nb/videos_multitask_*.ipynb` (snapshot frames) |
| Trajectory graphs | `trajectory_graph_A.png`, `trajectory_graph_B.png` | `figshare-deposit/figures_source/trajectory_graphs/` (per-frame IMU CSV + matplotlib script) |

---

## Table-by-table

| Table | Title | Generator |
|---|---|---|
| 1 | Sequence description (Routes A and B) | manual; CSV in `figshare-deposit/dataset/routes/` |
| 2 | Multitask training summary (BC + BS) | `nb/E_multitask_val_fix_wloss.ipynb` final cells |
| 3 | Single-task ablation | `nb/E_Bisenet_solo_cls.ipynb`, `nb/E_Bisenet_solo_seg.ipynb` |
| 4 | Per-route mIoU, Top-1 Accuracy | `nb/eval_*.ipynb` aggregated CSVs |
| 5 | Per-route FPS (GPU-only + E2E) | `nb/benchmark_*.ipynb` aggregated CSVs |
| 6 | Parameter count by method | `nb/count_params_mmseg_models_detailed.ipynb` |
| Appendix A | Method-vs-method differences | manual prose; no script |

---

## Per-frame inference outputs (Figshare ↔ paper)

For every method (`frame_by_frame`, `fixed_clockwork`, `adaptive_clockwork`, `dff`, `llm`) and every route (`A`, `B`), the Figshare bundle ships:

```
figshare-deposit/results/<method>/<route>__per_frame.csv
figshare-deposit/results/<method>/<route>__per_frame.parquet
```

Schema (one row per frame): `frame_idx, t_seconds, miou, top1_correct, fire (0/1), keyframe_age, time_spatial_ms, time_context_ms, time_total_ms, dev_pred (A-CW only), gt_cls, pred_cls`. Full descriptions and units in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

---

## End-to-end smoke test

After installing the repo and unpacking the Figshare bundle:

```bash
# 1. Re-run the A-CW stateful evaluator on Route A
jupyter nbconvert --to notebook --execute \
  notebooks/eval_stateful_adaptive_scheduler_fullseq_tau003_1a_code_only_EN_comments_unexecuted.ipynb \
  --output ../runs/test_smoke.ipynb

# 2. Diff the produced CSVs against the released ones
python tools/compare_results.py \
  --reference figshare-deposit/results/adaptive_clockwork/A__per_frame.csv \
  --candidate  runs/test_smoke/results_routeA.csv \
  --tolerance 1e-4
```

A clean re-run should reproduce per-frame mIoU and Top-1 within 1e-4 (floating-point noise) and the FPS within ≈5 % (depends on GPU).

---

## A note on hardware-dependent numbers

The FPS values in Fig. 9 and Table 5 depend on the GPU model and the OS scheduler. We measured on an idle workstation; on a shared cluster expect ≈10–15 % variance. The mIoU and Top-1 columns are deterministic given the released seeds and should match within floating-point tolerance.
