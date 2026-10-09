from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ABL = TERR = ELIG = NEG = OUT = None
KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
TOPO = ["T1_missed_branch", "T2_truncation"]
CHECK = 0.10
MATERIAL = 0.05
NBOOT = 2000
SEED = 20261009

def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)

def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def boot_auc(pos, pos_cl, neg, neg_cl, rng):
    cl = np.unique(np.concatenate([pos_cl, neg_cl]))
    pi = {c: np.where(pos_cl == c)[0] for c in cl}
    ni = {c: np.where(neg_cl == c)[0] for c in cl}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        s = rng.choice(cl, size=len(cl), replace=True)
        ip = np.concatenate([pi[c] for c in s]) if len(s) else np.array([], int)
        iN = np.concatenate([ni[c] for c in s])
        out[b] = auc(pos[ip.astype(int)], neg[iN.astype(int)])
    return np.nanpercentile(out, [2.5, 97.5])

def cluster_id(df):
    return (df.scan.astype(str) + "_" + df.side + "_" + df.vessel + "_" + df["loc"] + "_"
            + df.L_mm.map(lambda x: f"{x:g}") + "_" + df.ds_pct.astype(int).astype(str)).to_numpy()

def evaluate(label, bed, detector, pos_df, pos_col, neg_df, neg_col, rng):
    pos = pos_df[pos_col].to_numpy(float); neg = neg_df[neg_col].to_numpy(float)
    m = ~np.isnan(pos); pos, pcl = pos[m], cluster_id(pos_df)[m]
    m = ~np.isnan(neg); neg, ncl = neg[m], cluster_id(neg_df)[m]
    a = auc(pos, neg); lo, hi = boot_auc(pos, pcl, neg, ncl, rng)
    thr95 = float(np.quantile(neg, 0.95))
    rec = dict(bed=bed, detector=detector, target=label, n_pos=len(pos), n_pos_inst=len(np.unique(pcl)),
               n_neg=len(neg), n_neg_inst=len(np.unique(ncl)), auc=a, auc_lo=lo, auc_hi=hi,
               pos_median=float(np.median(pos)), neg_median=float(np.median(neg)), thr95=thr95)
    for tag, thr in (("10", CHECK), ("95", thr95)):
        tp = int((pos >= thr).sum()) if tag == "10" else int((pos > thr).sum())
        fa = int((neg >= thr).sum()) if tag == "10" else int((neg > thr).sum())
        s, sl, sh = wilson(tp, len(pos)); f, fl, fh = wilson(fa, len(neg))
        rec.update({f"sens{tag}_k": tp, f"sens{tag}": s, f"sens{tag}_lo": sl, f"sens{tag}_hi": sh,
                    f"far{tag}_k": fa, f"far{tag}": f, f"far{tag}_lo": fl, f"far{tag}_hi": fh})
    return rec

def load_positives():
    a = pd.read_csv(ABL)
    e = pd.read_csv(ELIG)
    el = e[e.eligible][KEY].assign(elig=1)
    a = a.merge(el, on=KEY, how="left"); a["elig"] = a.elig.fillna(0)
    a = a[(a.bed == "leaky") | (a.elig == 1)]
    a = a[a.error_type.isin(TOPO + ["T3_stenosis_length", "T4_taper"])]
    b = a[(a.protocol == "B_rederived") & (a.status == "ok")].set_index(KEY + ["bed", "error_type"])
    c = a[(a.protocol == "C_flowmatched") & (a.status == "ok")].set_index(KEY + ["bed", "error_type"])
    p = b[["run_id", "outlet_flow_residual", "flip", "dFFR", "C_ratio"]].rename(columns=lambda x: x + "_B")
    p = p.join(c[["outlet_flow_residual", "flip", "dFFR", "C_ratio"]].rename(columns=lambda x: x + "_C"), how="left")
    p = p.reset_index()
    p["topo"] = p.error_type.isin(TOPO)
    p["tune_factor"] = np.abs(np.log(p.C_ratio_C / p.C_ratio_B))
    p["tune_factor_vs_clean"] = np.abs(np.log(p.C_ratio_C))
    p["pw_C"] = ((p.outlet_flow_residual_C < CHECK) & (p.dFFR_C.abs() > MATERIAL)).astype(float)
    p.loc[p.outlet_flow_residual_C.isna(), "pw_C"] = np.nan
    p["decision_C"] = ((p.flip_C == 1) | (p.pw_C == 1)).astype(float)
    p.loc[p.outlet_flow_residual_C.isna(), "decision_C"] = np.nan
    return p, el

def load_negatives(el):
    n = pd.read_csv(NEG)
    n = n.merge(el, on=KEY, how="left"); n["elig"] = n.elig.fillna(0)
    n = n[(n.bed == "leaky") | (n.elig == 1)].copy()
    n = n[n.pretune_resid.notna()]
    n["tune_factor"] = np.where(n.status == "ok", np.abs(np.log(n.C_ratio)), np.nan)
    return n

def noisy_positive_residuals(p, n):
    terr = pd.read_csv(TERR)
    clean = terr[terr.run_id.str.endswith("_clean_none")]
    clean_idx = {(r.run_id[: -len("_clean_none")], int(r.root_node)): int(r.terr_id) for r in clean.itertuples()}
    tb = terr[terr.run_id.isin(set(p.run_id_B))]
    tgroups = {rid: g for rid, g in tb.groupby("run_id")}
    ncl = cluster_id(n)
    nk = {}
    for i, (cid, bed) in enumerate(zip(ncl, n.bed)):
        nk.setdefault((cid, bed), []).append(i)
    tg = n.terr_target_mls.to_numpy()
    rows, maxdev = [], 0.0
    pcl = cluster_id(p)
    for i, r in enumerate(p.itertuples()):
        g = tgroups.get(r.run_id_B)
        if g is None or np.isnan(r.outlet_flow_residual_B):
            continue
        prefix = r.run_id_B.split(f"_{r.bed}_")[0] + f"_{r.bed}"
        idx = [clean_idx[(prefix, int(rn))] for rn in g.root_node]
        qa = g.Q_achieved_mls.to_numpy()
        chk = np.sqrt(np.mean(((qa - g.Q_target_mls.to_numpy()) / g.Q_target_mls.to_numpy()) ** 2))
        maxdev = max(maxdev, abs(chk - r.outlet_flow_residual_B))
        for j in nk.get((pcl[i], r.bed), []):
            t = np.array([float(x) for x in tg[j].split(";")])[idx]
            rows.append(dict(**{k: getattr(r, k) for k in KEY}, bed=r.bed, error_type=r.error_type, topo=r.topo,
                             flip_B=r.flip_B, draw=int(n.draw.iloc[j]),
                             resid_noisy=float(np.sqrt(np.mean(((qa - t) / t) ** 2)))))
    return pd.DataFrame(rows), maxdev

def main():
    global ABL, TERR, ELIG, NEG, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("abl"); ap.add_argument("territory"); ap.add_argument("elig")
    ap.add_argument("negatives"); ap.add_argument("out_dir"); a = ap.parse_args()
    ABL, TERR, ELIG, NEG = a.abl, a.territory, a.elig, a.negatives
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    p, el = load_positives()
    n = load_negatives(el)
    pn, maxdev = noisy_positive_residuals(p, n)
    lines = [f"territory-row recomputation of the B residual: max |diff| = {maxdev:.2e}"]
    recs = []
    for bed in ("discrete", "leaky"):
        P, N, PN = p[p.bed == bed], n[n.bed == bed], pn[pn.bed == bed]
        Pd = P[P.outlet_flow_residual_B.notna()]
        PC = P[P.outlet_flow_residual_C.notna()]
        recs.append(evaluate("topological", bed, "B_residual", Pd[Pd.topo], "outlet_flow_residual_B", N, "pretune_resid", rng))
        recs.append(evaluate("flip_B_any_type", bed, "B_residual", Pd[Pd.flip_B == 1], "outlet_flow_residual_B", N, "pretune_resid", rng))
        recs.append(evaluate("flip_B_topological", bed, "B_residual", Pd[(Pd.flip_B == 1) & Pd.topo], "outlet_flow_residual_B", N, "pretune_resid", rng))
        recs.append(evaluate("topological_vs_caliber", bed, "B_residual", Pd[Pd.topo], "outlet_flow_residual_B",
                             Pd[~Pd.topo], "outlet_flow_residual_B", rng))
        recs.append(evaluate("topological|noisy_targets", bed, "B_residual", PN[PN.topo], "resid_noisy", N, "pretune_resid", rng))
        recs.append(evaluate("flip_B_any_type|noisy_targets", bed, "B_residual", PN[PN.flip_B == 1], "resid_noisy", N, "pretune_resid", rng))
        recs.append(evaluate("topological", bed, "abs_ln_C", PC[PC.topo], "tune_factor", N, "tune_factor", rng))
        recs.append(evaluate("flip_or_PW_C_any_type", bed, "abs_ln_C", PC[PC.decision_C == 1], "tune_factor", N, "tune_factor", rng))
        recs.append(evaluate("flip_or_PW_C_topological", bed, "abs_ln_C", PC[(PC.decision_C == 1) & PC.topo], "tune_factor", N, "tune_factor", rng))
        recs.append(evaluate("topological_vs_caliber", bed, "abs_ln_C", PC[PC.topo], "tune_factor", PC[~PC.topo], "tune_factor", rng))
        recs.append(evaluate("topological", bed, "abs_ln_C_vs_clean", PC[PC.topo], "tune_factor_vs_clean", N, "tune_factor", rng))
        q = N.terr_pred_start_mls.str.split(";")
        lines.append(f"{bed}: negatives {len(N)} draws on {N[KEY].drop_duplicates().shape[0]} instances; "
                     f"pre-tuning residual median {N.pretune_resid.median():.4f} IQR "
                     f"[{N.pretune_resid.quantile(.25):.4f}, {N.pretune_resid.quantile(.75):.4f}]; "
                     f"|ln C| median {N.tune_factor.median():.4f} (failed fits {int((N.status != 'ok').sum())})")
        lines.append(f"{bed}: error models with defined B residual: topological {int(Pd.topo.sum())}, caliber "
                     f"{int((~Pd.topo).sum())}; B flips {int(Pd.flip_B.sum())} (topological {int(Pd[Pd.topo].flip_B.sum())}); "
                     f"C defined topological {int(PC.topo.sum())}, C flip-or-P&W {int(PC.decision_C.sum())} "
                     f"(topological {int(PC[PC.topo].decision_C.sum())})")
        tt = PC[PC.topo]
        lines.append(f"{bed}: re-derived topological models passing (C defined) {int((tt.outlet_flow_residual_B < CHECK).sum())}/{len(tt)}; "
                     f"all with defined B residual {int((Pd[Pd.topo].outlet_flow_residual_B < CHECK).sum())}/{int(Pd.topo.sum())}")
    m = pd.DataFrame(recs)
    m.to_csv(OUT / "detector_metrics.csv", index=False)
    terr = pd.read_csv(TERR)
    clean = terr[terr.run_id.str.endswith("_clean_none")]
    devs = []
    for r in n.drop_duplicates(KEY + ["bed"]).itertuples():
        rid = f"{r.scan}_{r.side}_{r.vessel}_{r.loc}_{r.L_mm:g}mm_{int(r.ds_pct)}ds_{r.bed}_clean_none"
        g = clean[clean.run_id == rid].sort_values("terr_id")
        if not len(g):
            continue
        pr = np.array([float(x) for x in r.terr_pred_start_mls.split(";")])
        devs.append(np.sqrt(np.mean(((pr - g.Q_target_mls.to_numpy()) / g.Q_target_mls.to_numpy()) ** 2)))
    lines.append(f"noise-free correct anatomy, B residual: {len(devs)} instance-beds, max {max(devs):.2e}, median {np.median(devs):.2e}")
    pd.set_option("display.width", 250)
    txt = "\n".join(lines) + "\n\n" + m.round(4).to_string()
    (OUT / "detector_summary.txt").write_text(txt + "\n")
    print(txt)

if __name__ == "__main__":
    main()
