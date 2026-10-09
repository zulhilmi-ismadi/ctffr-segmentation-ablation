from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import negatives
from ablation import bed_flow

_orig_fit = negatives.fit_global_scaling
CALLS: list[dict] = []

def fit_with_pretune(t, r, terr, targets):
    out = _orig_fit(t, r, terr, targets)
    C2, C_start, nb, at_bound = out
    lo_b, hi_b = np.log10(C_start) - 1.5, np.log10(C_start) + 1.5
    g30 = float(np.linspace(lo_b, hi_b, 61)[30])
    def loss(lc):
        f, _, _, _, _ = t.evaluate(10 ** lc, r)
        q = bed_flow(t, 10 ** lc, f)
        pred = np.array([q[sub].sum() for sub in terr])
        return float(np.mean(((pred - targets) / targets) ** 2))
    f0, _, _, _, _ = t.evaluate(C_start, r)
    q0 = bed_flow(t, C_start, f0)
    pred0 = np.array([q0[sub].sum() for sub in terr])
    CALLS.append(dict(C_start=float(C_start), pretune_resid=float(np.sqrt(loss(np.log10(C_start)))),
                      pretune_resid_grid30=float(np.sqrt(loss(g30))),
                      terr_pred_start_mls=";".join(f"{x * 1e6:.9g}" for x in pred0),
                      terr_target_mls=";".join(f"{x * 1e6:.9g}" for x in targets)))
    return out

negatives.fit_global_scaling = fit_with_pretune

_orig_run = negatives.run_instance
def run_instance(root, row, bed, draws, rng):
    n0 = len(CALLS)
    out, pull, why = _orig_run(root, row, bed, draws, rng)
    calls = CALLS[n0:]
    fitted = [rec for rec in out if "C_clean" in rec]
    if len(calls) != len(fitted):
        raise RuntimeError(f"fit/record mismatch {len(calls)} vs {len(fitted)}")
    for rec, c in zip(fitted, calls):
        rec.update(c)
    return out, pull, why

negatives.run_instance = run_instance

if __name__ == "__main__":
    negatives.main()
