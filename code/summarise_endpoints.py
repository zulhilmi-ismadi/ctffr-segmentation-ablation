import sys, argparse
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.proportion import proportion_confint

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
TOPO = ("T1_missed_branch", "T2_truncation")
PROTOS = ("A_fixed", "B_rederived", "C_flowmatched", "D_perterritory")

def ci(k, n):
    if n == 0: return "--"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{100*k/n:.0f} ({100*lo:.0f}--{100*hi:.0f})"

def load(abl, per, elig):
    el = pd.read_csv(elig); el = el[el.eligible.astype(bool)][KEY].assign(_e=1)
    x = pd.concat([pd.read_csv(abl), pd.read_csv(per)], ignore_index=True)
    x = x[(x.status == "ok") & (x.error_type != "clean")].merge(el, on=KEY, how="left")
    x = x[(x.bed == "leaky") | (x._e == 1)].drop(columns="_e")
    x["pw"] = ((x.outlet_flow_residual < 0.10) & (x.dFFR.abs() > 0.05)).astype(int)
    x["wrong"] = (x.dFFR.abs() > 0.05).astype(int)
    x["cls"] = np.where(x.error_type.isin(TOPO), "topo", "cal")
    return x

def mcnemar(w, p, q):
    h = w.dropna(subset=[p, q]); n01 = int(((h[p] == 0) & (h[q] == 1)).sum()); n10 = int(((h[p] == 1) & (h[q] == 0)).sum())
    pv = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    return len(h), int(h[p].sum()), int(h[q].sum()), n01, n10, pv

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("abl"); ap.add_argument("per"); ap.add_argument("elig")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    x = load(a.abl, a.per, a.elig)
    print("instances per bed:", x.groupby("bed").apply(lambda g: g[KEY].drop_duplicates().shape[0]).to_dict())
    rows = []
    print(f"\n{'bed':9}{'err':20}{'proto':16}{'n':>4}  {'flip %':16}{'pass&wrong %':16}{'med|dFFR|':>10}")
    for (bed, e, p), g in x.groupby(["bed", "error_type", "protocol"]):
        r = dict(bed=bed, error_type=e, protocol=p, n=len(g), flips=int(g.flip.sum()), pw=int(g.pw.sum()),
                 flip_ci=ci(int(g.flip.sum()), len(g)), pw_ci=ci(int(g.pw.sum()), len(g)),
                 med_abs_dffr=g.dFFR.abs().median())
        rows.append(r)
        print(f"{bed:9}{e:20}{p:16}{len(g):>4}  {r['flip_ci']:16}{r['pw_ci']:16}{r['med_abs_dffr']:>10.4f}")
    print("\nclass level")
    for (bed, c, p), g in x.groupby(["bed", "cls", "protocol"]):
        print(f"  {bed:9}{c:5}{p:16} n={len(g):4} flip {ci(int(g.flip.sum()), len(g)):14} pass&wrong {ci(int(g.pw.sum()), len(g))}")
    print("\npaired passes-and-wrong (exact McNemar): n, k_first, k_second, +new, -lost, p")
    w = x.pivot_table(index=KEY + ["bed", "error_type", "cls"], columns="protocol", values="pw").reset_index()
    for bed in ("discrete", "leaky"):
        for c in ("topo", "cal"):
            g = w[(w.bed == bed) & (w.cls == c)]
            for p, q in (("A_fixed", "C_flowmatched"), ("B_rederived", "C_flowmatched"), ("B_rederived", "D_perterritory"),
                         ("C_flowmatched", "D_perterritory")):
                print(f"  {bed:9}{c:5}{q[:1]} vs {p[:1]}: " + "n=%d %d->%d (+%d/-%d) p=%.3g" % mcnemar(g, p, q))
    print("\ntopological vs caliber flips under A (paired sign test on per-instance class proportions)")
    for bed in ("discrete", "leaky"):
        for p in PROTOS:
            g = x[(x.bed == bed) & (x.protocol == p)].groupby(KEY + ["cls"]).flip.mean().unstack()
            g = g.dropna(); d = g.topo - g.cal; k = int((d > 0).sum()); n = int((d != 0).sum())
            print(f"  {bed:9}{p:16} topo>cal in {k}/{n} instances, p={binomtest(k, n).pvalue if n else 1:.3g}")
    if a.out: pd.DataFrame(rows).to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
