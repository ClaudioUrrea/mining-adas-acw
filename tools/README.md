# Tools

Small utilities used to package and verify the artefacts that accompany this repository. None of them touch model weights, training, or evaluation logic — they only check / hash / package things.

## When to run each one

| Helper | When | Reads | Writes |
|---|---|---|---|
| `validate_checkpoint.py` | Before opening any "can't load this .pth" issue, and as a sanity check after downloading the Figshare bundle. | a `.pth` checkpoint, optionally a config file | nothing (just prints) |
| `package_checkpoints.py` | After local training, before uploading a `.pth` to Figshare. | your `.pth` files | `<basename>_metadata.json` sidecars, optionally copies into `figshare-deposit/checkpoints/` |
| `verify_figures.py` | Before submitting the figures ZIP to MDPI. | `tools/expected_figures.txt` plus every PNG it references | nothing (just prints; non-zero exit on failure) |
| `build_figures_zip.sh` | After `verify_figures.py` passes, to actually produce `figures_zip_for_submission.zip` for the MDPI portal and as the GitHub release asset. | every PNG in `tools/expected_figures.txt` | `figures_zip_for_submission.zip` + `.sha256` next to it |
| `compute_sha256_manifest.sh` | Just before uploading `figshare-deposit/`. | every file under `figshare-deposit/` (except the manifest itself) | `figshare-deposit/MANIFEST.sha256` |
| `compare_results.py` | After re-executing an evaluation notebook, to make sure your numbers reproduce the released ones. | two per-frame CSVs | nothing (just prints; non-zero exit on diff) |

## End-to-end submission flow

```bash
# (1) Run the local training and copy checkpoints to figshare-deposit/checkpoints/
python tools/package_checkpoints.py \
    --source-dir runs/multitask \
    --staging-dir figshare-deposit/checkpoints

# (2) Verify every figure meets MDPI submission rules
python tools/verify_figures.py

# (3) Build the figures ZIP that goes into the MDPI portal AND as a GitHub release asset
bash tools/build_figures_zip.sh

# (4) Generate the per-file integrity manifest for Figshare
bash tools/compute_sha256_manifest.sh figshare-deposit
```

## Reading the exit codes

Every helper returns 0 on success and a non-zero code on failure. The code is meaningful and stable:

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | argument error / file missing |
| 2 | structural mismatch (columns, schema, etc.) |
| 3 | content mismatch (rows, hash, dpi) |
| 4 | strict-mode violation |
| 5+ | helper-specific (see docstring) |

This makes CI integration straightforward.
