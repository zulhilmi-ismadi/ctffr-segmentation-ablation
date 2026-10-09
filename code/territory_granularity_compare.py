import sys
from pathlib import Path
import numpy as np, pandas as pd

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct", "bed", "error_type", "protocol"]

def compare(new: pd.DataFrame, old: pd.DataFrame, label: str):
    old = old[old.protocol.isin(new.protocol.unique())]
    j = old.merge(new, on=KEY, how="inner", suffixes=("_r", "_n"))
    st = (j.status_r == j.status_n)
    ok = j[(j.status_r == "ok") & (j.status_n == "ok")]
    df = (ok.ffr_r - ok.ffr_n).abs(); dr = (ok.outlet_flow_residual_r - ok.outlet_flow_residual_n).abs()
    print(f"{label}: matched {len(j)} rows; status equal {int(st.sum())}/{len(j)}; both ok {len(ok)}; "
          f"max|dFFR diff| {df.max():.3g}; max|resid diff| {dr.max():.3g}; "
          f"exact ffr {int((df == 0).sum())}/{len(ok)}, exact resid {int((dr == 0).sum())}/{len(ok)}")
    return j

def main():
    out, cfg = Path(sys.argv[1]), sys.argv[2]
    abl = pd.read_csv(sys.argv[3])
    per = pd.read_csv(sys.argv[4])
    na = pd.read_csv(out / f"ablation_{cfg}.csv"); nd = pd.read_csv(out / f"perterritory_{cfg}.csv")
    for p in ("A_fixed", "B_rederived", "C_flowmatched"):
        compare(na[na.protocol == p], abl, p)
    compare(nd, per, "D_perterritory")

if __name__ == "__main__":
    main()
