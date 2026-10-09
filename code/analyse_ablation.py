from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm, wilcoxon
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

THRESHOLD, RESID, MATERIAL, FLOOR_SD = 0.80, 0.10, 0.05, 0.018
PROTOS = ["A_fixed", "B_rederived", "C_flowmatched"]
ETYPES = ["T1_missed_branch", "T2_truncation", "T3_lesion_length", "T4_taper"]
BANDS = np.round(np.arange(0.65, 0.951, 0.05), 2)

def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    return proportion_confint(k, n, alpha=0.05, method="wilson")

def fmt_p(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100*k/n:.1f}%, {100*lo:.1f}-{100*hi:.1f})" if n else "0/0"

def mcnemar(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p

def band_of(f):
    i = np.floor((f - 0.65) / 0.05 + 1e-9)
    return np.where((f >= 0.65) & (f < 0.95), 0.65 + 0.05 * i, np.nan).round(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--negatives", default=None); ap.add_argument("--outdir", default=None)
    ap.add_argument("--eligibility", default=str(Path(__file__).parent.parent / "results" / "discrete_arm_eligibility.csv"))
    a = ap.parse_args()
    src = Path(a.csv); df = pd.read_csv(src)
    out = Path(a.outdir or src.parent / f"analysis-{src.stem}"); out.mkdir(parents=True, exist_ok=True)
    df["inst"] = df.scan.astype(str) + "_" + df.side + "_" + df.vessel + "_" + df["loc"].astype(str) + "_" \
        + df.L_mm.astype(str) + "_" + df.ds_pct.astype(str)
    df["slot"] = df.scan.astype(str) + "_" + df.side + "_" + df.vessel + "_" + df["loc"].astype(str) + "_" + df.L_mm.astype(str)

    el = pd.read_csv(a.eligibility)
    key = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
    el = el[el.eligible.astype(bool)][key].assign(_elig=True)
    df = df.merge(el, on=key, how="left"); df["_elig"] = df._elig.fillna(False).astype(bool)
    n_before = df[df.bed == "discrete"].inst.nunique()
    df = df[(df.bed != "discrete") | df._elig].copy()
    print(f"discrete arm: {n_before} instances run -> {df[df.bed == 'discrete'].inst.nunique()} healthy-network eligible kept")
    df["band_bed"] = band_of(df.ffr_clean.values)
    clean = df[df.error_type == "clean"]
    rows = df[df.error_type != "clean"]
    ok = rows[rows.status == "ok"].copy()
    ok["flip"] = ok.flip.astype(bool).astype(int)
    ok["absd"] = ok.dFFR.abs()
    R = []

    def say(s=""): R.append(s); print(s)

    say(f"# Ablation analysis — {src.name}")
    say(f"instances: {clean.inst.nunique()} | clean rows per bed: {clean.groupby('bed').size().to_dict()}")
    say(f"corrupted rows: {len(rows)}, solved ok: {len(ok)}")
    nonconv = int((ok.converged.astype('boolean') != True).sum())
    say(f"non-converged ok rows: {nonconv} | max mass error {ok.mass_err.max():.1e}")

    st = rows.assign(s=rows.status.str.replace(r":.*", "", regex=True).where(rows.status == "ok", rows.status)) \
        .groupby(["bed", "error_type", "protocol", "s"]).size().unstack(fill_value=0)
    st.to_csv(out / "exclusions.csv")
    say("\n## Row status by bed x error x protocol\n" + st.to_string())

    ok["floor_repeat"] = norm.cdf(-np.abs(ok.ffr_clean - THRESHOLD) / FLOOR_SD)
    t1 = []
    for (bed, e, p), g in ok.groupby(["bed", "error_type", "protocol"]):
        k, n = int(g.flip.sum()), len(g); lo, hi = wilson(k, n)
        t1.append(dict(bed=bed, error_type=e, protocol=p, n=n, flips=k, flip_pct=100*k/n, lo=100*lo, hi=100*hi,
                       mean_dFFR=g.dFFR.mean(), median_dFFR=g.dFFR.median(), mean_abs_dFFR=g.absd.mean(),
                       wrong_pct=100*(g.absd > MATERIAL).mean(), floor_repeat_pct=100*g.floor_repeat.mean(),
                       grey_flips=int((g.flip & g.ffr_clean.between(0.75, 0.85)).sum()),
                       median_resid=g.outlet_flow_residual.median()))
    t1 = pd.DataFrame(t1); t1.to_csv(out / "flips.csv", index=False)
    say("\n## Flips and dFFR (Wilson 95% CI); floor_repeat = expected flip % from repeat-FFR SD 0.018\n"
        + t1.round(4).to_string(index=False))

    pb = ok.groupby(["bed", "error_type", "protocol", "band_bed"]).agg(n=("flip", "size"), flips=("flip", "sum"),
                                                                       floor_repeat=("floor_repeat", "mean")).reset_index()
    pb["flip_pct"] = 100 * pb.flips / pb.n; pb["floor_repeat_pct"] = 100 * pb.floor_repeat
    pb.to_csv(out / "flip_by_band.csv", index=False)

    mc = []
    for (bed, e), g in ok.groupby(["bed", "error_type"]):
        w = g.pivot_table(index="inst", columns="protocol", values="flip")
        fam = []
        for x, y in (("A_fixed", "B_rederived"), ("A_fixed", "C_flowmatched"), ("B_rederived", "C_flowmatched")):
            if x in w and y in w:
                pr = w[[x, y]].dropna()
                n01, n10, p = mcnemar(pr[x], pr[y])
                fam.append(dict(bed=bed, error_type=e, contrast=f"{y[0]}-{x[0]}", n_pairs=len(pr),
                                flip_only_second=n01, flip_only_first=n10, p=p))
        if fam:
            ph = multipletests([f["p"] for f in fam], method="holm")[1]
            for f, q in zip(fam, ph): f["p_holm"] = q
            mc += fam
    mc = pd.DataFrame(mc); mc.to_csv(out / "mcnemar.csv", index=False)
    say("\n## Paired McNemar (exact), Holm within bed x error\n" + mc.round(4).to_string(index=False))

    wc = []
    for (bed, e), g in ok.groupby(["bed", "error_type"]):
        w = g.pivot_table(index="inst", columns="protocol", values="dFFR")
        for x, y in (("A_fixed", "B_rederived"), ("A_fixed", "C_flowmatched"), ("B_rederived", "C_flowmatched")):
            if x in w and y in w:
                pr = w[[x, y]].dropna()
                if len(pr) > 5:
                    d = pr[y].abs() - pr[x].abs()
                    wc.append(dict(bed=bed, error_type=e, contrast=f"|{y[0]}|-|{x[0]}|", n=len(pr),
                                   median_diff=d.median(), p=wilcoxon(pr[y].abs(), pr[x].abs()).pvalue))
    wc = pd.DataFrame(wc); wc.to_csv(out / "abs_dffr_wilcoxon.csv", index=False)

    ok["topo"] = ok.error_type.isin(ETYPES[:2]).astype(int)
    tc = []
    for (bed, p), g in ok.groupby(["bed", "protocol"]):
        tp, cb = g[g.topo == 1], g[g.topo == 0]

        w = g.groupby(["inst", "topo"]).flip.mean().unstack().dropna()
        d = w[1] - w[0]
        tc.append(dict(bed=bed, protocol=p, topo=fmt_p(int(tp.flip.sum()), len(tp)), calibre=fmt_p(int(cb.flip.sum()), len(cb)),
                       topo_abs_dFFR=tp.absd.mean(), calibre_abs_dFFR=cb.absd.mean(), n_paired=len(w),
                       paired_mean_diff=d.mean(),
                       p_sign=binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue if (d != 0).any() else 1.0))
    tc = pd.DataFrame(tc); tc.to_csv(out / "topological_vs_caliber.csv", index=False)
    say("\n## Topological (T1,T2) vs caliber (T3,T4); paired sign test on per-instance flip means\n"
        + tc.round(4).to_string(index=False))

    pwrows = []
    for thr in (RESID, 0.13, 0.16):
        for (bed, p), g in ok.groupby(["bed", "protocol"]):
            for e, h in list(g.groupby("error_type")) + [("ALL", g), ("TOPOLOGICAL", g[g.topo == 1])]:
                hh = h.dropna(subset=["outlet_flow_residual"])
                passes = hh[hh.outlet_flow_residual < thr]
                pw = passes[passes.absd > MATERIAL]
                lo, hi = wilson(len(pw), len(hh))
                pwrows.append(dict(threshold=thr, bed=bed, protocol=p, error_type=e, n=len(hh), passes=len(passes),
                               passes_and_wrong=len(pw), pct=100*len(pw)/len(hh) if len(hh) else np.nan,
                               lo=100*lo, hi=100*hi, passes_and_flip=int(passes.flip.sum()),
                               median_resid=hh.outlet_flow_residual.median()))
    pwt = pd.DataFrame(pwrows); pwt.to_csv(out / "passes_and_wrong.csv", index=False)
    say("\n## Passes and wrong: passes check (resid < thr) AND |dFFR| > 0.05 (Wilson 95% CI over all rows)\n"
        + pwt[pwt.threshold == RESID].round(4).to_string(index=False))
    say("\n### sensitivity thresholds (ALL / TOPOLOGICAL only)\n"
        + pwt[(pwt.threshold != RESID) & pwt.error_type.isin(["ALL", "TOPOLOGICAL"])].round(4).to_string(index=False))

    mb = []
    for bed, g in ok[ok.error_type == "T1_missed_branch"].groupby("bed"):
        w = g.pivot_table(index="inst", columns="protocol", values="dFFR")
        if {"A_fixed", "C_flowmatched"} <= set(w):
            pr = w[["A_fixed", "C_flowmatched"]].dropna()
            mb.append(dict(bed=bed, n_pairs=len(pr), A_median_dFFR=pr.A_fixed.median(), C_median_dFFR=pr.C_flowmatched.median(),
                           A_mean_abs=pr.A_fixed.abs().mean(), C_mean_abs=pr.C_flowmatched.abs().mean(),
                           A_pct_rel_gt13=100*(pr.A_fixed.abs() / g.set_index("inst").ffr_clean.groupby(level=0).first()
                                               .reindex(pr.index) > 0.13).mean(),
                           p_wilcoxon=wilcoxon(pr.A_fixed.abs(), pr.C_flowmatched.abs()).pvalue if len(pr) > 5 else np.nan))
    mb = pd.DataFrame(mb); mb.to_csv(out / "missed_branch_A_vs_C.csv", index=False)
    say("\n## T1 under A vs C (paired)\n" + mb.round(4).to_string(index=False))

    ag = t1.pivot_table(index=["error_type", "protocol"], columns="bed", values=["mean_dFFR", "flip_pct"])
    ag.to_csv(out / "bed_agreement.csv")
    say("\n## Per-bed effect sizes\n" + ag.round(4).to_string())

    say("\n## Flip models")
    for bed, g in ok.groupby("bed"):
        g = g.drop(columns=["C"]).copy(); g["band_c"] = g.band_bed.fillna(-1).astype(str)
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = BinomialBayesMixedGLM.from_formula(
                    "flip ~ C(error_type) * C(protocol) + ds_pct + L_mm + C(band_c)",
                    {"scan": "0 + C(scan)", "slot": "0 + C(slot)"}, g).fit_vb()
            fe = pd.DataFrame(dict(post_mean=m.fe_mean, post_sd=m.fe_sd), index=m.model.exog_names)
            fe["lo95"] = fe.post_mean - 1.96 * fe.post_sd; fe["hi95"] = fe.post_mean + 1.96 * fe.post_sd
            fe.to_csv(out / f"mixed_model_{bed}.csv")
            say(f"\n### Bayesian mixed logistic (VB), bed={bed}; vc sd (log): {np.round(m.vcp_mean, 3).tolist()}\n"
                + fe[~fe.index.str.startswith("C(band_c)")].round(4).to_string())
        except Exception as ex:
            say(f"GLMM {bed} failed: {ex.__class__.__name__}: {ex}")

    sec = ok.groupby(["bed", "error_type", "protocol", "vessel"]).agg(n=("flip", "size"), flips=("flip", "sum"),
                                                                     mean_dFFR=("dFFR", "mean")).reset_index()
    sec.to_csv(out / "by_vessel.csv", index=False)

    if a.negatives and Path(a.negatives).exists():
        ng = pd.read_csv(a.negatives)
        say(f"\n## Simulated floor: physiological noise on clean anatomy ({a.negatives}), columns: {list(ng.columns)[:40]}")
        if "dFFR" in ng:
            ngo = ng[ng.get("status", "ok") == "ok"] if "status" in ng else ng
            fb = ngo.groupby("bed").agg(n=("dFFR", "size"), median_abs_dFFR=("dFFR", lambda s: s.abs().median()),
                                        p95_abs_dFFR=("dFFR", lambda s: s.abs().quantile(.95)),
                                        flip_pct=("flip", lambda s: 100*s.astype(bool).mean()) if "flip" in ngo else ("dFFR", "size"))
            if "outlet_flow_residual" in ngo:
                fb["pass_pct"] = ngo.groupby("bed").outlet_flow_residual.apply(lambda s: 100*(s < RESID).mean())
            fb.to_csv(out / "floor_simulated.csv"); say(fb.round(4).to_string())

    (out / "REPORT.md").write_text("\n".join(R) + "\n")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
