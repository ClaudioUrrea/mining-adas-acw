# Checkpoints (pointer directory)

Trained model weights are **not** stored in this Git repository. Download them
from the companion Figshare deposit and drop the `.pth` files here:

> <https://doi.org/10.6084/m9.figshare.32274630>

The expected filenames are listed in [`EXPECTED.txt`](EXPECTED.txt). Each
`.pth` ships next to a `<basename>_metadata.json` sidecar with framework
versions, training config path, iteration, expected metrics and SHA-256.

To validate a checkpoint after download:

```bash
python tools/validate_checkpoint.py checkpoints/multitask_BC_iter_25600.pth \
    --config configs/multitask_bisenet/multitask_bisenet_v1_resnet50_512x512_40k.py
```
