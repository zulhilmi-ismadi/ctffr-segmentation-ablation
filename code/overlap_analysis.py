import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
TOPO = ("T1_missed_branch", "T2_truncation")
ETYPES = ("T1_missed_branch", "T2_truncation", "T3_stenosis_length", "T4_taper")
SEED, NBOOT = 20261009, 2000
SCORES = {"net_dsc_tree": -1, "net_dsc_scan": -1, "net_cldice": -1, "flow_lost": +1, "full_dsc_scan": -1}

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return np.nan
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())

def boot_auc(g, ycol, scol, sign, rng):
    inst = g["inst"].to_numpy(); u = np.unique(inst); idx = {k: np.where(inst == k)[0] for k in u}
    y = g[ycol].to_numpy(); s = sign * g[scol].to_numpy()
    point = auc(y, s); bs = []
    for _ in range(NBOOT):
        pick = np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])
        a = auc(y[pick], s[pick])
        if np.isfinite(a): bs.append(a)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return point, lo, hi, len(bs)

def wil(k, n):
    lo, hi = proportion_confint(k, n, method="wilson"); return f"{100*k/n:.0f} ({100*lo:.0f}--{100*hi:.0f})"

def med(x, d=3):
    x = x.dropna()
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}--{x.quantile(.75):.{d}f})" if len(x) else "--"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ov"); ap.add_argument("abl"); ap.add_argument("elig")
    ap.add_argument("out"); a = ap.parse_args(); out = Path(a.out)
    ov = pd.read_csv(a.ov); ab = pd.read_csv(a.abl)
    el = pd.read_csv(a.elig); el = el[el.eligible.astype(bool)][KEY].assign(_e=1)

    cl = ab[ab.error_type == "clean"][KEY + ["bed", "ffr_clean"]].merge(
        ov[ov.error_type == "clean"][KEY + ["bed", "ffr_clean_overlap"]], on=KEY + ["bed"], how="outer", indicator=True)
    print("clean rows joined:", cl._merge.value_counts().to_dict(),
          " max |ffr_clean diff|:", float((cl.ffr_clean - cl.ffr_clean_overlap).abs().max()))
    abA = ab[(ab.status == "ok") & (ab.protocol == "A_fixed")]
    abB = ab[(ab.status == "ok") & (ab.protocol == "B_rederived")]
    ck = ov[ov.status == "ok"].merge(abB, on=KEY + ["bed", "error_type"])
    for c in ("n_outlets", "L_resolved_mm", "w_sum", "r_ref_root_mm"):
        d = (ck[f"chk_{c}"] - ck[c]).abs() / ck[c].abs().clip(lower=1e-30)
        print(f"corrupted-tree check {c}: n={len(ck)} max rel diff {d.max():.2e}")
    okk = ov[ov.status == "ok"]
    st_ab = ab[ab.error_type != "clean"].groupby(KEY + ["bed", "error_type"]).status.apply(
        lambda s: "ok" if (s == "ok").any() else s.iloc[0]).reset_index()
    j = ov[ov.error_type != "clean"].merge(st_ab, on=KEY + ["bed", "error_type"], suffixes=("_overlap", "_ab"))
    print("status agreement (ok vs not):", int(((j.status_overlap == "ok") == (j.status_ab == "ok")).sum()), "/", len(j))

    e = okk[okk.error_type != "clean"]
    print("\nnesting (network set): by error type and bed")
    print(e.groupby(["bed", "error_type"]).agg(
        n=("net_dsc_tree", "size"), unmatched=("net_nest_unmatched", "max"), inactive=("net_nest_inactive", "max"),
        n_r_excess=("net_nest_r_excess_mm", lambda x: int((x > 1e-9).sum())), max_r_excess_mm=("net_nest_r_excess_mm", "max"),
        ds_mm=("net_nest_ds_mismatch_mm", "max"), full_unmatched=("full_nest_unmatched", "max"),
        b0_corr_max=("b0_corr", "max"), b0_corr_min=("b0_corr", "min"), b0_clean=("b0_clean", "max")).to_string())

    x = e.merge(el, on=KEY, how="left"); x = x[(x.bed == "leaky") | (x._e == 1)].drop(columns="_e")
    for p, src in (("A", abA), ("B", abB)):
        s = src[KEY + ["bed", "error_type", "flip", "dFFR"]].rename(columns={"flip": f"flip_{p}", "dFFR": f"dFFR_{p}"})
        x = x.merge(s, on=KEY + ["bed", "error_type"], how="left")
        x[f"wrong_{p}"] = np.where(x[f"dFFR_{p}"].notna(), (x[f"dFFR_{p}"].abs() > 0.05).astype(float), np.nan)
    x["inst"] = x[KEY].astype(str).agg("_".join, axis=1)
    x["host"] = np.where(x.vessel == "RCA", "RCA", "left")
    x.to_csv(out / "overlap_joined.csv", index=False)
    print("\ninstances per bed:", x.groupby("bed").inst.nunique().to_dict())

    rows = []
    print("\n=== medians (IQR) by bed and error type; flip/wrong under A ===")
    for (bed, et), g in x.groupby(["bed", "error_type"]):
        gA = g.dropna(subset=["flip_A"])
        r = dict(bed=bed, error_type=et, n=len(g), dsc_tree=med(g.net_dsc_tree), dsc_scan=med(g.net_dsc_scan),
                 cldice=med(g.net_cldice), flow_lost=med(100 * g.flow_lost, 1), vol_lost=med(100 * g.net_vol_lost, 1),
                 full_dsc_tree=med(g.full_dsc_tree), full_dsc_scan=med(g.full_dsc_scan),
                 dsc_tree_min=g.net_dsc_tree.min(), nA=len(gA), flipA=wil(int(gA.flip_A.sum()), len(gA)),
                 wrongA=wil(int(gA.wrong_A.sum()), len(gA)))
        rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(out / "medians.csv", index=False)

    print("\n=== by host side (T1, T2) ===")
    rows = []
    for (bed, et, h), g in x[x.error_type.isin(TOPO)].groupby(["bed", "error_type", "host"]):
        gA = g.dropna(subset=["flip_A"])
        r = dict(bed=bed, error_type=et, host=h, n=len(g), dsc_tree=med(g.net_dsc_tree), dsc_scan=med(g.net_dsc_scan),
                 dsc_tree_range=f"{g.net_dsc_tree.min():.3f}--{g.net_dsc_tree.max():.3f}",
                 dsc_scan_range=f"{g.net_dsc_scan.min():.3f}--{g.net_dsc_scan.max():.3f}",
                 cldice=med(g.net_cldice), flow_lost=med(100 * g.flow_lost, 1),
                 flipA=wil(int(gA.flip_A.sum()), len(gA)) if len(gA) else "--")
        rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(out / "medians_by_host.csv", index=False)

    print("\n=== T4 ===")
    t4 = x[x.error_type == "T4_taper"]
    for bed, g in t4.groupby("bed"):
        print(bed, "scaled clean-volume share", med(g.t4_scaled_vol_share), " dsc_tree", med(g.net_dsc_tree))

    print("\n=== paired T1 vs T4 DSC within instance ===")
    for bed, g in x.groupby("bed"):
        pv = g.pivot_table(index="inst", columns="error_type", values="net_dsc_tree").dropna(subset=["T1_missed_branch", "T4_taper"])
        d = pv.T1_missed_branch - pv.T4_taper
        print(bed, f"n={len(d)} median T1-T4 {d.median():+.4f} T1<T4 in {(d<0).sum()} p={wilcoxon(d).pvalue:.3g}")

    print("\n=== AUC (instance bootstrap, seed %d, %d reps) ===" % (SEED, NBOOT))
    rng = np.random.default_rng(SEED); rows = []
    for bed in ("leaky", "discrete"):
        for subset, ets in (("pooled", ETYPES), ("topological", TOPO), ("T1", ("T1_missed_branch",)),
                            ("T2", ("T2_truncation",))):
            for yc in ("flip_A", "wrong_A", "flip_B", "wrong_B"):
                g = x[(x.bed == bed) & x.error_type.isin(ets)].dropna(subset=[yc])
                for sc, sign in SCORES.items():
                    p, lo, hi, nb = boot_auc(g, yc, sc, sign, rng)
                    rows.append(dict(bed=bed, subset=subset, outcome=yc, metric=sc, n=len(g), events=int(g[yc].sum()),
                                     auc=p, lo=lo, hi=hi, nboot=nb))
    au = pd.DataFrame(rows); au.to_csv(out / "auc.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(au[au.outcome.isin(["flip_A", "wrong_A"])].round(3).to_string(index=False))
        print(au[au.outcome.isin(["flip_B", "wrong_B"]) & au.subset.isin(["pooled", "topological"])].round(3).to_string(index=False))

if __name__ == "__main__":
    main()
