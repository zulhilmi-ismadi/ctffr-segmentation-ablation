import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
RK = KEY + ["bed", "error_type", "protocol"]
TOPO = ("T1_missed_branch", "T2_truncation")

def ci(k, n):
    if n == 0: return "--"
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{100*k/n:.0f} ({100*lo:.0f}--{100*hi:.0f})"

def prep(x, el):
    x = x[(x.status == "ok") & (x.error_type != "clean") & x.protocol.isin(["C_flowmatched", "D_perterritory"])]
    x = x.merge(el, on=KEY, how="left")
    x = x[(x.bed == "leaky") | (x._e == 1)].drop(columns="_e").copy()
    x["pass_"] = (x.outlet_flow_residual < 0.10).astype(int)
    x["wrong"] = (x.dFFR.abs() > 0.05).astype(int)
    x["pw"] = x.pass_ * x.wrong
    x["cls"] = np.where(x.error_type.isin(TOPO), "topo", "cal")
    return x

def mcn(a, b):
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    return n01, n10, (binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)

def main():
    out = Path(sys.argv[1])
    el = pd.read_csv(sys.argv[4])
    el = el[el.eligible.astype(bool)][KEY].assign(_e=1)
    ref = pd.concat([pd.read_csv(sys.argv[2]), pd.read_csv(sys.argv[3])], ignore_index=True)
    cfgs = {"reference": prep(ref, el)}
    for c in ("main_drop", "main_keep", "finer_keep", "finer_drop"):
        fa, fd = out / f"ablation_{c}.csv", out / f"perterritory_{c}.csv"
        if fa.exists() and fd.exists():
            cfgs[c] = prep(pd.concat([pd.read_csv(fa), pd.read_csv(fd)], ignore_index=True), el)

    tc = out / "territory_counts_finer_keep.csv"
    if tc.exists():
        t = pd.read_csv(tc).merge(el, on=KEY, how="left")
        t = t[(t.bed == "leaky") | (t._e == 1)]
        print("territories with a positive clean target, per instance (main branch -> finer):")
        for bed, g in t.groupby("bed"):
            print(f"  {bed:9} n={len(g)}  main {g.n_terr_main.value_counts().sort_index().to_dict()}  "
                  f"finer {g.n_terr_pos.value_counts().sort_index().to_dict()}  finer median {g.n_terr_pos.median():.0f}"
                  f"  finer incl. zero-target trunks {g.n_terr_all.value_counts().sort_index().to_dict()}")

    rows = []
    print("\nper bed x class x protocol: n, pass %, wrong %, passes-and-wrong %, P(wrong|pass) %, flip %")
    for name, x in cfgs.items():
        for (bed, cls, p), g in x.groupby(["bed", "cls", "protocol"]):
            n, k_p, k_w, k_pw, k_f = len(g), int(g.pass_.sum()), int(g.wrong.sum()), int(g.pw.sum()), int(g.flip.sum())
            r = dict(config=name, bed=bed, cls=cls, protocol=p[0], n=n, passes=k_p, wrong=k_w, pw=k_pw, flips=k_f,
                     pass_ci=ci(k_p, n), wrong_ci=ci(k_w, n), pw_ci=ci(k_pw, n), wgp_ci=ci(k_pw, k_p),
                     flip_ci=ci(k_f, n))
            rows.append(r)
            print(f"  {name:11}{bed:9}{cls:5}{p[0]} n={n:4} pass {r['pass_ci']:13} wrong {r['wrong_ci']:13} "
                  f"P&W {k_pw:3} {r['pw_ci']:13} W|P {k_pw:3}/{k_p:<3} {r['wgp_ci']:13} flip {r['flip_ci']}")
    pd.DataFrame(rows).to_csv(out / "summary_by_class.csv", index=False)

    rows = []
    for name in ("reference", "finer_keep"):
        if name not in cfgs: continue
        for (bed, e, p), g in cfgs[name].groupby(["bed", "error_type", "protocol"]):
            rows.append(dict(config=name, bed=bed, error_type=e, protocol=p[0], n=len(g), passes=int(g.pass_.sum()),
                             pw=int(g.pw.sum()), flips=int(g.flip.sum()), pw_ci=ci(int(g.pw.sum()), len(g)),
                             flip_ci=ci(int(g.flip.sum()), len(g)), wgp_ci=ci(int(g.pw.sum()), int(g.pass_.sum()))))
    pd.DataFrame(rows).to_csv(out / "summary_by_type.csv", index=False)

    print("\npaired against the reference run (exact McNemar): n paired, k reference, k new, +new/-lost, p")
    base = cfgs["reference"].set_index(RK)
    rows = []
    for name, x in cfgs.items():
        if name == "reference": continue
        j = base.join(x.set_index(RK)[["pw", "flip", "pass_", "wrong", "ffr", "outlet_flow_residual"]],
                      how="outer", rsuffix="_n")
        miss = int(j.pw.isna().sum() + j.pw_n.isna().sum())
        j = j.dropna(subset=["pw", "pw_n"]).reset_index()
        print(f" {name}: rows only in one run {miss}; max|ffr diff| {np.abs(j.ffr - j.ffr_n).max():.3g}; "
              f"max|resid diff| {np.abs(j.outlet_flow_residual - j.outlet_flow_residual_n).max():.3g}")
        for (bed, cls, p), g in j.groupby(["bed", "cls", "protocol"]):
            for ep, a, b in (("pw", g.pw, g.pw_n), ("flip", g.flip, g.flip_n)):
                n01, n10, pv = mcn(a.astype(int), b.astype(int))
                rows.append(dict(config=name, bed=bed, cls=cls, protocol=p[0], endpoint=ep, n=len(g),
                                 k_reference=int(a.sum()), k_new=int(b.sum()), new=n01, lost=n10, p=pv))
                print(f"   {bed:9}{cls:5}{p[0]} {ep:5} n={len(g):4} {int(a.sum()):3} -> {int(b.sum()):3} "
                      f"(+{n01}/-{n10}) p={pv:.3g}")
    pd.DataFrame(rows).to_csv(out / "mcnemar_vs_reference.csv", index=False)

    if "finer_keep" in cfgs:
        x = cfgs["finer_keep"]
        print("\nfiner_keep: rows with >= 1 unperfused territory (n_empty > 0)")
        for (bed, e, p), g in x.groupby(["bed", "error_type", "protocol"]):
            ne = int((g.n_empty.fillna(0) > 0).sum())
            print(f"  {bed:9}{e:20}{p[0]} n={len(g):4} with empty {ne:4}  pass among those "
                  f"{int(g[g.n_empty.fillna(0) > 0].pass_.sum())}")

if __name__ == "__main__":
    main()
