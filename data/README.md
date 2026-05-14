# Data (pointer directory)

The drivable-area masks, train/val/test splits, and IMU-derived per-frame
metadata are released on Figshare under CC BY 4.0:

> <https://doi.org/10.6084/m9.figshare.32274630>

The expected layout inside this directory is:

```
data/
├── annotations/             # 100 binary masks (1-bit PNG)
├── splits/                  # train.txt, val_aug.txt, test_route_a.txt, test_route_b.txt,
│                            # plus the cls_labels_*.txt classification labels
└── routes/                  # IMU-derived per-frame metadata (CSV + Parquet)
```

The original AutoMine RGB frames and IMU streams must be requested from the
AutoMine authors and placed under your own path (see
`docs/INSTALLATION.md` and `docs/DATA_DICTIONARY.md`). Once both are in place,
either symlink the AutoMine frames into the appropriate subdirectory or set:

```bash
export MINING_ADAS_DATA_ROOT=/path/to/figshare-bundle
export AUTOMINE_FRAMES_ROOT=/path/to/automine_frames
```

and the training configs will resolve the locations through these variables.
