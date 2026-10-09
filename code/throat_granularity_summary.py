from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_endpoints import load, ci

PL = {"A_fixed": "A", "B_rederived": "B", "C_flowmatched": "C", "D_perterritory": "D"}
VARIANTS = ("T5_ds_plus10", "T5_ds_minus10", "T5_vox_narrow", "T5_vox_wide")
POOLS = {"T5_ds_pooled": ("T5_ds_plus10", "T5_ds_minus10"), "T5_vox_pooled": ("T5_vox_narrow", "T5_vox_wide")}
V = ["pw", "pass_", "flip", "wrong", "outlet_flow_residual", "ffr", "fit_at_bound", "n_empty"]

def level(out, cfg, elig):
    x = load(out / f"ablation_throat_{cfg}.csv", out / f"perterritory_throat_{cfg}.csv", elig)
    x["pass_"] = (x.outlet_flow_residual < 0.10).astype(int)
    for c in ("fit_at_bound", "n_empty"):
        x[c] = x[c] if c in x else np.nan
    x["fit_at_bound"] = x.fit_at_bound.astype(object).map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    return x

def mcn(a, b):
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    return n01, n10, float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("outdir"); ap.add_argument("elig")
    a = ap.parse_args(); out = Path(a.outdir); ELIG = a.elig
    lm, lf = level(out, "main_drop", ELIG), level(out, "finer_keep", ELIG)
    j = lm.set_index("run_id")[["bed", "error_type", "protocol"] + V].join(
        lf.set_index("run_id")[V], how="inner", rsuffix="_2").reset_index()
    only = len(set(lm.run_id) ^ set(lf.run_id))
    rows = []
    for bed in ("discrete", "leaky"):
        for e in VARIANTS + tuple(POOLS):
            for p in PL:
                g = j[(j.bed == bed) & j.error_type.isin(POOLS.get(e, (e,))) & (j.protocol == p)]
                n = len(g); r = dict(bed=bed, error_type=e, protocol=PL[p], n=n)
                for lv, s in (("main", ""), ("finer", "_2")):
                    k_p, k_pw, k_f = int(g["pass_" + s].sum()), int(g["pw" + s].sum()), int(g["flip" + s].sum())
                    nb = int(g["fit_at_bound" + s].sum()); h = g[g["fit_at_bound" + s] == 0]
                    r.update({f"{lv}_pass": k_p, f"{lv}_pw": k_pw, f"{lv}_flip": k_f,
                              f"{lv}_pw_ci": ci(k_pw, n), f"{lv}_wgp_ci": ci(k_pw, k_p), f"{lv}_flip_ci": ci(k_f, n),
                              f"{lv}_pass_ci": ci(k_p, n), f"{lv}_bound": nb,
                              f"{lv}_pw_exbound_ci": ci(int(h["pw" + s].sum()), len(h)),
                              f"{lv}_med_resid": float(g["outlet_flow_residual" + s].median()),
                              f"{lv}_n_empty": int((g["n_empty" + s].fillna(0) > 0).sum())})
                for ep in ("pw", "flip", "pass_"):
                    n01, n10, pv = mcn(g[ep], g[ep + "_2"])
                    r.update({f"{ep}_new": n01, f"{ep}_lost": n10, f"{ep}_p": pv})
                r["max_abs_ffr_diff"] = float((g.ffr - g.ffr_2).abs().max()) if n else np.nan
                rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "throat_granularity_cells.csv", index=False)

    n1 = {b: len(lm[(lm.bed == b)]) for b in ("discrete", "leaky")}
    lines = [f"paired rows {len(j)}; run_ids at only one level {only}; main-branch ok rows {n1}",
             "",
             "bed       error           P  n   | pass main -> finer | P&W main -> finer (+new/-lost, p)       | "
             "P(wrong|pass) main -> finer      | flip main -> finer (+/-, p)           | D bound main/finer"]
    for _, r in t.iterrows():
        lines.append(
            f"{r.bed:9} {r.error_type:15} {r.protocol} {r.n:3} | {r.main_pass:3} -> {r.finer_pass:3} (p {r.pass__p:.2g}) | "
            f"{r.main_pw_ci:13} -> {r.finer_pw_ci:13} (+{r.pw_new}/-{r.pw_lost}, p {r.pw_p:.2g}) | "
            f"{r.main_pw}/{r.main_pass} {r.main_wgp_ci:13} -> {r.finer_pw}/{r.finer_pass} {r.finer_wgp_ci:13} | "
            f"{r.main_flip_ci:13} -> {r.finer_flip_ci:13} (+{r.flip_new}/-{r.flip_lost}, p {r.flip_p:.2g}) | "
            f"{r.main_bound}/{r.finer_bound}  max|dffr main-finer| {r.max_abs_ffr_diff:.2g}  empty {r.main_n_empty}/{r.finer_n_empty}")
    lines += ["", "D passes-and-wrong excluding fits on the parameter bound (main -> finer):"]
    for _, r in t[t.protocol == "D"].iterrows():
        lines.append(f"  {r.bed:9} {r.error_type:15} bound {r.main_bound:3}/{r.finer_bound:3}  "
                     f"{r.main_pw_exbound_ci} -> {r.finer_pw_exbound_ci}")

    lines += ["", "status counts by level (all rows, both beds, before eligibility):"]
    for cfg in ("main_drop", "finer_keep"):
        a_ = pd.read_csv(out / f"ablation_throat_{cfg}.csv", low_memory=False)
        d = pd.read_csv(out / f"perterritory_throat_{cfg}.csv")
        lines.append(f"  {cfg}: A-C {a_.status.value_counts().to_dict()}; D {d.status.value_counts().to_dict()}; "
                     f"D fit_at_bound {int(d.fit_at_bound.astype(str).eq('True').sum())} of {int(d.status.eq('ok').sum())}")
    tc = pd.read_csv(out / "territory_counts_finer_keep.csv")
    el = pd.read_csv(ELIG); el = el[el.eligible.astype(bool)][["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]].assign(_e=1)
    tc = tc.merge(el, on=["scan", "side", "vessel", "loc", "L_mm", "ds_pct"], how="left")
    tc = tc[(tc.bed == "leaky") | (tc._e == 1)]
    for bed, g in tc.groupby("bed"):
        lines.append(f"  territories per tree, {bed}: main {g.n_terr_main.value_counts().sort_index().to_dict()}, "
                     f"finer {g.n_terr_pos.value_counts().sort_index().to_dict()} (median {g.n_terr_pos.median():.0f})")
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
