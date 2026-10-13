import sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
ov_p, ab_p, j14_p, el_p, out = sys.argv[1:6]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
ov = pd.read_csv(ov_p); ab = pd.read_csv(ab_p, low_memory=False); j14 = pd.read_csv(j14_p)
el = pd.read_csv(el_p); el = el[el.eligible.astype(bool)][KEY].assign(_e=1)

clean = ab[ab.error_type == "clean"][KEY + ["bed", "ffr_clean"]].merge(
    ov[ov.error_type == "clean"][KEY + ["bed", "ffr_clean_a5"]], on=KEY + ["bed"])
print("clean FFR reproduced: n", len(clean), "max diff", float((clean.ffr_clean - clean.ffr_clean_a5).abs().max()))

e = ov[(ov.status == "ok") & (ov.error_type != "clean")]
x = e.merge(el, on=KEY, how="left"); x = x[(x.bed == "leaky") | (x._e == 1)].drop(columns="_e")
for p, prot in (("A", "A_fixed"), ("B", "B_rederived")):
    s = ab[(ab.status == "ok") & (ab.protocol == prot)][KEY + ["bed", "error_type", "flip", "dFFR"]]
    x = x.merge(s.rename(columns={"flip": f"flip_{p}", "dFFR": f"dFFR_{p}"}), on=KEY + ["bed", "error_type"], how="left")
    x[f"wrong_{p}"] = np.where(x[f"dFFR_{p}"].notna(), (x[f"dFFR_{p}"].abs() > 0.05).astype(float), np.nan)
x["inst"] = x[KEY].astype(str).agg("_".join, axis=1)
x["host"] = np.where(x.vessel == "RCA", "RCA", "left")
print("T5 rows:", x.groupby(["bed", "error_type"]).size().to_dict(), " missing dFFR_A:", int(x.dFFR_A.isna().sum()))
allj = pd.concat([j14, x], ignore_index=True)
allj.to_csv(out / "overlap_joined_T1-T5.csv", index=False)

def med(v, d=3):
    v = v.dropna(); return f"{v.median():.{d}f} ({v.quantile(.25):.{d}f}--{v.quantile(.75):.{d}f})"

rows = []
for bed, g in x.groupby("bed"):
    gA = g.dropna(subset=["flip_A"]); k, n = int(gA.flip_A.sum()), len(gA); lo, hi = proportion_confint(k, n, method="wilson")
    rows.append(dict(bed=bed, error="T5 pooled", n=len(g), dsc_tree=med(g.net_dsc_tree), dsc_scan=med(g.net_dsc_scan),
                     cldice=med(g.net_cldice), cldice_min=g.net_cldice.min(), flow_lost=med(100 * g.flow_lost, 1),
                     b0_changed=int((g.b0_corr != g.b0_clean).sum()), dsc_scan_min=g.net_dsc_scan.min(),
                     flipA=f"{100*k/n:.0f} ({100*lo:.0f}--{100*hi:.0f})", dFFR_A_abs_med=g.dFFR_A.abs().median(),
                     dFFR_A_range=f"{g.dFFR_A.min():.3f} to {g.dFFR_A.max():.3f}"))
m = pd.DataFrame(rows); m.to_csv(out / "medians_T5.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(m.to_string(index=False))
