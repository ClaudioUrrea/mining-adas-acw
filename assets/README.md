# Assets — paper figures

The high-resolution PNGs referenced by the paper's LaTeX source live here.
They are **not** stored in Git; download them from the companion Figshare
deposit and drop them in:

```
assets/
├── figures_paper/           # fig1 … fig11, trajectory_graph_A, trajectory_graph_B
├── figures_supp/            # (empty — supplementary material not used in v1.0.0)
├── confusion_matrices/      # cm_<route>_<method>.png
└── qualitative/             # qual_<route>_<frame>_<method>.png
```

Canonical filename list: [`../tools/expected_figures.txt`](../tools/expected_figures.txt).
Mapping from filename to source data and regen script: [`../docs/FIGURES_INDEX.md`](../docs/FIGURES_INDEX.md).

After populating, build the submission ZIP with:

```bash
bash tools/build_figures_zip.sh
```
