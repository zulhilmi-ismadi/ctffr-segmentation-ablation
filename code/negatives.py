from __future__ import annotations
import argparse, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Tree, P_AORTA, P_VEN, MU
from severity_sweep import load, plan, insert, HOSTS
from ablation import (THRESHOLD, VALIDATED_RESIDUAL, bed_flow, territories, subtree,
                      run_id_of, sc_covariates, pullback_rows)

SD_CO = 0.05

SD_MAP_REL = 0.056

SD_MU = 0.02

SD_TERRITORY_SHARE = 0.10

SHARE_SENSITIVITY = (0.05, 0.10, 0.15, 0.20)

WSCV_TARGET = 0.083

def perturbed_targets(t: Tree, C_clean: float, r_clean: np.ndarray, terr, rng: np.random.Generator):
    co = float(rng.normal(1.0, SD_CO))
    map_pa = P_AORTA * float(rng.normal(1.0, SD_MAP_REL))
    mu = float(rng.normal(MU, SD_MU * MU))
    co, mu, map_pa = max(co, 0.2), max(mu, 0.2 * MU), max(map_pa, 40 * 133.322)

    mu_saved, C_saved = t.mu, dict(t._C)
    t.mu = mu; t._C.clear()
    try:
        C_true = t.calibrate(t.demand("murray", co), map_pa, P_VEN)
        ffr_t, _, info_t, _, _ = t.evaluate(C_true, r_clean, map_pa, P_VEN)
        q_true = bed_flow(t, C_true, ffr_t) * (map_pa / P_AORTA)
        tgt = np.array([float(q_true[sub].sum()) for sub in terr])
    finally:
        t.mu = mu_saved; t._C.clear(); t._C.update(C_saved)

    share = rng.normal(1.0, SD_TERRITORY_SHARE, size=len(tgt))
    tgt_shared = tgt * share
    if tgt_shared.sum() > 0:
        tgt_shared *= tgt.sum() / tgt_shared.sum()
    measured = tgt_shared * rng.normal(1.0, WSCV_TARGET, size=len(tgt))
    return np.maximum(measured, 1e-12), dict(draw_co=co, draw_map_mmhg=map_pa / 133.322, draw_mu=mu,
                                             draw_share_sd=SD_TERRITORY_SHARE, draw_wscv=WSCV_TARGET)

def fit_global_scaling(t: Tree, r: np.ndarray, terr, targets):
    t._C.clear(); C_start = t.calibrate(t.demand("murray", 1.0))
    lo_b, hi_b = np.log10(C_start) - 1.5, np.log10(C_start) + 1.5
    def loss(lc):
        f, _, _, _, _ = t.evaluate(10 ** lc, r)
        q = bed_flow(t, 10 ** lc, f)
        pred = np.array([q[sub].sum() for sub in terr])
        return float(np.mean(((pred - targets) / targets) ** 2))
    grid = np.linspace(lo_b, hi_b, 61); lv = np.array([loss(x) for x in grid]); k = int(np.argmin(lv))
    res = minimize_scalar(loss, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                          method="bounded", options=dict(xatol=1e-9))
    lc = float(res.x) if float(res.fun) <= lv[k] else float(grid[k])
    n_basins = int(sum(1 for i in range(1, len(lv) - 1) if lv[i] < lv[i - 1] and lv[i] < lv[i + 1]))
    at_bound = bool(min(abs(lc - lo_b), abs(lc - hi_b)) < 1e-6)
    return 10 ** lc, C_start, n_basins, at_bound

def run_instance(root: Path, row, bed: str, draws: int, rng):
    out, pull = [], []
    t = load(root, int(row.scan), row.side, bed)
    o = t.ffr("murray", 1.0); C_clean = o["C"]
    sl = next((s for s in plan(t, row.side, t.last["ffr"].copy())[0]
               if s["vessel"] == row.vessel and s["loc"] == row["loc"] and abs(s["L"] * 1e3 - row.L_mm) < 1e-6), None)
    if sl is None: return out, pull, "slot not eligible under this bed"
    path, s_arc, c, L, mi = sl["path"], sl["s"], sl["c"], sl["L"], sl["mi"]
    meas = int(path[mi])
    r_clean, _ = insert(t, path, s_arc, c, L, row.ds_pct / 100)
    ffr0, _, info0, _, _ = t.evaluate(C_clean, r_clean)
    f0 = float(ffr0[meas])
    terr = [sub for sub in territories(t)]
    if len(terr) < 2: return out, pull, "fewer than 2 territories — Protocol C not applicable"

    base = dict(scan=int(row.scan), side=row.side, vessel=row.vessel, loc=row["loc"], L_mm=row.L_mm,
                ds_pct=int(row.ds_pct), bed=bed, band_cohort=row.band, ffr_clean=f0,
                error_type="negative", protocol="C_flowmatched", n_territories=len(terr), **sc_covariates(t))
    for d in range(draws):
        rid = run_id_of(row, bed, "negative", f"draw{d}")
        rec = {**base, "run_id": rid, "draw": d, "status": "ok"}
        try:
            targets, params = perturbed_targets(t, C_clean, r_clean, terr, rng)
            rec.update(params)
            C2, C_start, nb, at_bound = fit_global_scaling(t, r_clean, terr, targets)
            ffr2, Q2, info2, _, _ = t.evaluate(C2, r_clean)
            qb = bed_flow(t, C2, ffr2)
            pred = np.array([qb[sub].sum() for sub in terr])
            resid = float(np.sqrt(np.mean(((pred - targets) / targets) ** 2)))
            f2 = float(ffr2[meas])
            rec.update(C_clean=C_clean, C=C2, C_abs=C2, C_ratio=C2 / C_clean, ffr=f2, dFFR=f2 - f0,
                       flip=int((f2 <= THRESHOLD) != (f0 <= THRESHOLD)),
                       outlet_flow_residual=resid, passes_check=int(resid < VALIDATED_RESIDUAL),
                       fit_n_basins=nb, fit_at_bound=at_bound,
                       inflow_mls=info2["inflow"] * 1e6, converged=bool(info2["converged"]))
            if at_bound: rec["status"] = "failed fit: optimum at search bound"
            else: pull += pullback_rows(t, path, s_arc, c, L, meas, ffr2, Q2, info2["inflow"], rid)
        except Exception as e:
            rec["status"] = f"error: {e.__class__.__name__}: {e}"
        out.append(rec)
    return out, pull, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--cohort", default=None)
    ap.add_argument("--beds", default="leaky,discrete"); ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260919); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); root = Path(a.root); here = Path(__file__).parent.parent
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    out_path = Path(a.out or here / "results" / "negatives.csv")
    rng = np.random.default_rng(a.seed)
    rows, pulls, t0, fails = [], [], time.time(), 0

    print(f"{len(coh)} instances x {len(a.beds.split(','))} beds x {a.draws} draw(s), seed {a.seed}\n")
    for n, (_, r) in enumerate(coh.iterrows(), 1):
        for bed in a.beds.split(","):
            try:
                res, pl, why = run_instance(root, r, bed, a.draws, rng)
                if why: fails += 1; print(f"  SKIP {r.scan}_{r.side} {bed}: {why}", file=sys.stderr)
                rows += res; pulls += pl
            except Exception as e:
                fails += 1; print(f"  FAIL {r.scan}_{r.side} {bed}: {e.__class__.__name__}: {e}", file=sys.stderr)
                if fails <= 3: traceback.print_exc()
        if n % 10 == 0: print(f"  {n}/{len(coh)}  {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); out_path.parent.mkdir(exist_ok=True); df.to_csv(out_path, index=False)
    pd.DataFrame(pulls).to_csv(out_path.with_name(out_path.stem + "_pullback.csv"), index=False)
    print(f"\nwrote {out_path}  ({len(df)} rows, {fails} skips, {time.time()-t0:.0f}s)")
    ok = df[df.status == "ok"]
    if len(ok):
        print(f"\nCorrect anatomy tuned to noisy perfusion")
        print(f"  residual: median {ok.outlet_flow_residual.median():.4f}  "
              f"p10 {ok.outlet_flow_residual.quantile(.1):.4f}  p90 {ok.outlet_flow_residual.quantile(.9):.4f}")
        print(f"  passes the {VALIDATED_RESIDUAL:.0%} check: {int(ok.passes_check.sum())}/{len(ok)} "
              f"({100*ok.passes_check.mean():.0f}%)")
        print(f"  |dFFR| median {ok.dFFR.abs().median():.4f}  flips {int(ok.flip.sum())}")
        print(f"  C_ratio: median {ok.C_ratio.median():.4f}  IQR "
              f"[{ok.C_ratio.quantile(.25):.4f}, {ok.C_ratio.quantile(.75):.4f}]")

if __name__ == "__main__":
    main()
