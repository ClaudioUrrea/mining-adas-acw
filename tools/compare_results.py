#!/usr/bin/env python3
"""Compare two per-frame inference-result CSV files within numerical tolerance.

Verifies that a re-run of any evaluation notebook reproduces the released
per-frame outputs from the Figshare deposit. Used by the smoke test described
in docs/REPRODUCIBILITY.md.

Comparison rules
----------------
* The two files must have the same set of columns and the same row count.
* Integer columns must match exactly.
* Float columns must agree within ``--tolerance`` element-wise (absolute).
* ``NaN`` values must agree on which rows are NaN.

Exit code 0 on full agreement, non-zero on any failure (with a unified diff
printed for the first 20 offending rows).

Usage:
    python tools/compare_results.py \\
        --reference figshare-deposit/results/adaptive_clockwork/A__per_frame.csv \\
        --candidate runs/test_smoke/results_routeA.csv \\
        --tolerance 1e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple


def _load(path: Path):
    import pandas as pd
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compare(ref: Path, cand: Path, tol: float) -> Tuple[int, str]:
    import numpy as np
    import pandas as pd

    df_r = _load(ref)
    df_c = _load(cand)

    if list(df_r.columns) != list(df_c.columns):
        only_r = set(df_r.columns) - set(df_c.columns)
        only_c = set(df_c.columns) - set(df_r.columns)
        return 2, (f"column set differs.  only in reference: {sorted(only_r)}  "
                   f"only in candidate: {sorted(only_c)}")

    if len(df_r) != len(df_c):
        return 3, f"row count differs: ref={len(df_r)} cand={len(df_c)}"

    fail_rows = []
    for col in df_r.columns:
        s_r = df_r[col]
        s_c = df_c[col]

        if pd.api.types.is_float_dtype(s_r) or pd.api.types.is_float_dtype(s_c):
            r_nan = s_r.isna()
            c_nan = s_c.isna()
            nan_mismatch = r_nan != c_nan
            if nan_mismatch.any():
                for idx in nan_mismatch[nan_mismatch].index[:20]:
                    fail_rows.append((idx, col, "NaN mismatch",
                                      None if r_nan.loc[idx] else float(s_r.loc[idx]),
                                      None if c_nan.loc[idx] else float(s_c.loc[idx])))
                continue
            common = ~r_nan
            diff = np.abs(s_r[common].astype(float).values
                          - s_c[common].astype(float).values)
            bad = diff > tol
            if bad.any():
                for off, idx in enumerate(common[common].index[bad][:20]):
                    fail_rows.append((idx, col,
                                      f"|diff|={diff[bad][off]:.3e} > {tol:.0e}",
                                      float(s_r.loc[idx]), float(s_c.loc[idx])))
        else:
            ne = (s_r.fillna("__NA__") != s_c.fillna("__NA__"))
            if ne.any():
                for idx in ne[ne].index[:20]:
                    fail_rows.append((idx, col, "exact mismatch",
                                      s_r.loc[idx], s_c.loc[idx]))

    if fail_rows:
        lines = [f"  row={r} col={c!r}  reason={why}  ref={v_r!r}  cand={v_c!r}"
                 for (r, c, why, v_r, v_c) in fail_rows[:20]]
        return 4, ("rows disagree (showing up to 20):\n" + "\n".join(lines))

    return 0, f"OK — {len(df_r)} rows, {len(df_r.columns)} columns, tol={tol}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-4)
    a = p.parse_args()

    code, msg = compare(a.reference, a.candidate, a.tolerance)
    print(f"[compare_results] {msg}")
    return code


if __name__ == "__main__":
    sys.exit(main())
