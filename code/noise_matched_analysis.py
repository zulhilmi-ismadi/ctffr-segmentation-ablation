from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROWS = Path(sys.argv[1])
ELIG = Path(sys.argv[2])
RES = Path(sys.argv[3]); RES.mkdir(parents=True, exist_ok=True)
MAX_DRAW = int(sys.argv[4]) if len(sys.argv) > 4 else None
KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
MODEL = KEY + ["bed", "error_type"]
CHECK, MATERIAL = 0.10, 0.05
CLS = {"T1_missed_branch": "topo", "T2_truncation": "topo", "T3_stenosis_length": "cal", "T4_taper": "cal",
       "T5_vox_narrow": "throat", "T5_vox_wide": "throat", "T0_identity": "floor"}
CLS_NAME = {"topo": "Topological (T1+T2)", "cal": "Caliber (T3+T4)", "throat": "Throat ($\\pm\\tfrac12$ voxel)",
            "floor": "Correct anatomy"}
PROTO = {"A_fixed": "A", "B_rederived": "B", "C_flowmatched": "C", "D_perterritory": "D"}
NBOOT, SEED = 2000, 20261009

def load():
    x = pd.read_csv(ROWS, low_memory=False)
    el = pd.read_csv(ELIG); el = el[el.eligible.astype(bool)][KEY].assign(_e=1)
    x = x[(x.status == "ok") & (x.error_type != "clean")].merge(el, on=KEY, how="left")
    x = x[(x.bed == "leaky") | (x._e == 1)].drop(columns="_e")
    x = x[x.outlet_flow_residual.notna()]
    if MAX_DRAW is not None:
        x = x[x.draw <= MAX_DRAW]
    x = x.copy()
    x["pw"] = ((x.outlet_flow_residual < CHECK) & (x.dFFR.abs() > MATERIAL)).astype(float)
    x["pass"] = (x.outlet_flow_residual < CHECK).astype(float)
    x["wrong"] = (x.dFFR.abs() > MATERIAL).astype(float)
    x["cls"] = x.error_type.map(CLS)
    return x

def pct(v):
    return 100 * v

def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson"); return 100 * lo, 100 * hi

def boot_mean(values, clusters, rng, weights=None):
    df = pd.DataFrame(dict(v=values, c=clusters))
    s = df.groupby("c").v.agg(["sum", "count"])
    cl = s.index.to_numpy(); sums = s["sum"].to_numpy(); cnts = s["count"].to_numpy()
    idx = rng.integers(0, len(cl), size=(NBOOT, len(cl)))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))

def auc(pos, neg):
    allv = np.concatenate([pos, neg]); r = pd.Series(allv).rank().to_numpy()
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def boot_auc(pos, pcl, neg, ncl, rng):
    cl = np.unique(np.concatenate([pcl, ncl]))
    pi = {c: pos[pcl == c] for c in cl}; ni = {c: neg[ncl == c] for c in cl}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        s = rng.choice(cl, size=len(cl), replace=True)
        p = np.concatenate([pi[c] for c in s]); n = np.concatenate([ni[c] for c in s])
        out[b] = auc(p, n) if len(p) and len(n) else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))

def main():
    rng = np.random.default_rng(SEED)
    x = load()
    noisy = x[x.draw >= 0]; nominal = x[x.draw == -1]
    n_draws = noisy.groupby(MODEL + ["protocol"]).draw.nunique()
    lines = [f"rows {len(x)}; draws per model-protocol: min {int(n_draws.min())}, max {int(n_draws.max())}, "
             f"median {n_draws.median():.0f}; instances per bed: "
             + str(x.groupby('bed').apply(lambda g: g[KEY].drop_duplicates().shape[0], include_groups=False).to_dict())]
    floor = noisy[noisy.cls == "floor"][KEY + ["bed", "draw", "protocol", "pw", "flip", "wrong"]]
    floor = floor.rename(columns={"pw": "pw_floor", "flip": "flip_floor", "wrong": "wrong_floor"})
    bres = noisy[noisy.protocol == "B_rederived"][MODEL + ["draw", "pw"]].rename(columns={"pw": "pw_B"})
    d = noisy.merge(floor, on=KEY + ["bed", "draw", "protocol"], how="left").merge(bres, on=MODEL + ["draw"], how="left")
    d["ex_floor"] = d.pw - d.pw_floor
    d["ex_B"] = d.pw - d.pw_B
    pm = d.groupby(MODEL + ["cls", "protocol"]).agg(n_draws=("pw", "size"), pw=("pw", "mean"), flip=("flip", "mean"),
                                                   pass_=("pass", "mean"), wrong=("wrong", "mean"),
                                                   ex_floor=("ex_floor", "mean"), ex_B=("ex_B", "mean")).reset_index()
    recs = []
    for (bed, cls, proto), g in pm.groupby(["bed", "cls", "protocol"]):
        nom = nominal[(nominal.bed == bed) & (nominal.cls == cls) & (nominal.protocol == proto)]
        k_nom = int(nom.pw.sum()); n_nom = len(nom)
        gd = d[(d.bed == bed) & (d.cls == cls) & (d.protocol == proto)]
        k_pool = int(gd.pw.sum()); n_pool = len(gd)
        r = dict(bed=bed, cls=cls, protocol=proto, n_models=len(g), n_model_draws=n_pool,
                 nominal_n=n_nom, nominal_pw=pct(k_nom / n_nom) if n_nom else np.nan,
                 nominal_flip=pct(nom.flip.mean()) if n_nom else np.nan,
                 matched_pw=pct(g.pw.mean()), matched_flip=pct(g.flip.mean()), matched_pass=pct(g.pass_.mean()),
                 matched_wrong=pct(g.wrong.mean()))
        r["matched_pw_lo"], r["matched_pw_hi"] = (pct(v) for v in boot_mean(g.pw.to_numpy(), g.scan.to_numpy(), rng))
        r["matched_pw_wilson_lo"], r["matched_pw_wilson_hi"] = wilson(k_pool, n_pool)
        r["matched_flip_lo"], r["matched_flip_hi"] = (pct(v) for v in boot_mean(g.flip.to_numpy(), g.scan.to_numpy(), rng))
        if cls != "floor":
            ge = g[g.ex_floor.notna()]
            r["excess_floor"] = pct(ge.ex_floor.mean())
            r["excess_floor_lo"], r["excess_floor_hi"] = (pct(v) for v in boot_mean(ge.ex_floor.to_numpy(), ge.scan.to_numpy(), rng))
            gb = g[g.ex_B.notna()]
            r["excess_B"] = pct(gb.ex_B.mean())
            r["excess_B_lo"], r["excess_B_hi"] = (pct(v) for v in boot_mean(gb.ex_B.to_numpy(), gb.scan.to_numpy(), rng))
        recs.append(r)
    rates = pd.DataFrame(recs)
    rates.to_csv(RES / "matched_rates.csv", index=False)
    pm.to_csv(RES / "matched_per_model.csv", index=False)
    arecs = []
    for bed in ("discrete", "leaky"):
        nb = noisy[(noisy.bed == bed) & (noisy.protocol == "B_rederived")]
        neg = nb[nb.cls == "floor"]
        for cls in ("topo", "cal", "throat"):
            pos = nb[nb.cls == cls]
            if not len(pos):
                continue
            a = auc(pos.outlet_flow_residual.to_numpy(), neg.outlet_flow_residual.to_numpy())
            lo, hi = boot_auc(pos.outlet_flow_residual.to_numpy(), pos.scan.to_numpy(),
                              neg.outlet_flow_residual.to_numpy(), neg.scan.to_numpy(), rng)
            flag = pct((pos.outlet_flow_residual >= CHECK).mean()); far = pct((neg.outlet_flow_residual >= CHECK).mean())
            arecs.append(dict(bed=bed, positives=cls, detector="pretuning_residual", n_pos=len(pos),
                              n_pos_models=pos[MODEL].drop_duplicates().shape[0], n_neg=len(neg), auc=a, auc_lo=lo, auc_hi=hi,
                              flagged_at_10pct=flag, floor_flagged_at_10pct=far))
        cb = noisy[(noisy.bed == bed) & (noisy.protocol == "B_rederived")][MODEL + ["draw", "C"]].rename(columns={"C": "C_B"})
        nc = noisy[(noisy.bed == bed) & (noisy.protocol == "C_flowmatched")].merge(cb, on=MODEL + ["draw"], how="left")
        nc["tf"] = np.abs(np.log(nc.C / nc.C_B))
        neg = nc[(nc.cls == "floor") & nc.tf.notna()]
        for cls in ("topo",):
            pos = nc[(nc.cls == cls) & nc.tf.notna()]
            if len(pos) and len(neg):
                p = pos.tf.to_numpy(); n = neg.tf.to_numpy()
                a = auc(p, n); lo, hi = boot_auc(p, pos.scan.to_numpy(), n, neg.scan.to_numpy(), rng)
                arecs.append(dict(bed=bed, positives=cls, detector="abs_ln_C_vs_start", n_pos=len(pos),
                                  n_pos_models=pos[MODEL].drop_duplicates().shape[0], n_neg=len(neg), auc=a, auc_lo=lo, auc_hi=hi))
    aucs = pd.DataFrame(arecs); aucs.to_csv(RES / "matched_auc.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["bed", "cls", "protocol", "n_models", "nominal_pw", "matched_pw", "matched_pw_lo", "matched_pw_hi",
            "excess_floor", "excess_floor_lo", "excess_floor_hi", "excess_B", "excess_B_lo", "excess_B_hi",
            "matched_flip", "matched_pass", "matched_wrong"]
    lines.append("\npasses-and-wrong, % of models (nominal = noise-free targets; matched = averaged over the recorded draws; "
                 "intervals = patient-level cluster bootstrap)")
    lines.append(rates[cols].round(1).to_string(index=False))
    lines.append("\ndetectors against correct anatomy on the same draws")
    lines.append(aucs.round(3).to_string(index=False))
    txt = "\n".join(lines); print(txt); (RES / "summary.txt").write_text(txt + "\n")
    table(rates)

def num(v):
    r = int(round(v))
    return f"$-${abs(r)}" if r < 0 else f"{r}"

def fmt(v, lo, hi):
    return f"{num(v)} ({num(lo)}--{num(hi)})"

def table(rates):
    out = []
    for cls in ("topo", "cal", "throat", "floor"):
        for i, proto in enumerate(("C_flowmatched", "D_perterritory")):
            cells = [CLS_NAME[cls] if i == 0 else "", PROTO[proto]]
            for bed in ("discrete", "leaky"):
                r = rates[(rates.bed == bed) & (rates.cls == cls) & (rates.protocol == proto)]
                if not len(r):
                    cells += ["--"] * 4; continue
                r = r.iloc[0]
                cells += [f"{int(r.n_models)}", f"{r.nominal_pw:.0f}", fmt(r.matched_pw, r.matched_pw_lo, r.matched_pw_hi)]
                cells += ["--" if cls == "floor" else fmt(r.excess_floor, r.excess_floor_lo, r.excess_floor_hi)]
            out.append(" & ".join(cells) + " \\\\")
        if cls != "floor": out.append("\\addlinespace[2pt]")
    (RES / "tab_noise_matched.tex").write_text("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
