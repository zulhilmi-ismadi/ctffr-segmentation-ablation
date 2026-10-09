from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import zerod_ffr

def run_instance(A, root: Path, row, bed: str):
    from zerod_ffr import Tree
    from severity_sweep import load, plan, insert, HOSTS, RUNOFF
    from error_types import ERROR_TYPES, T3_LENGTH_DELTA
    out = []
    t = load(root, int(row.scan), row.side, bed)
    o = t.ffr("murray", 1.0); C_clean = o["C"]
    slots, _ = plan(t, row.side, t.last["ffr"].copy())
    sl = next((s for s in slots if s["vessel"] == row.vessel and s["loc"] == row["loc"]
               and abs(s["L"] * 1e3 - row.L_mm) < 1e-6), None)
    if sl is None: return out, "slot not eligible under this bed"
    path, s_arc, c, L, mi = sl["path"], sl["s"], sl["c"], sl["L"], sl["mi"]
    meas_clean = int(path[mi]); ds = row.ds_pct / 100
    r_clean, _ = insert(t, path, s_arc, c, L, ds)
    ffr0, _, info0, _, _ = t.evaluate(C_clean, r_clean)
    if not info0["converged"]: return out, "clean solve did not converge"
    f0 = float(ffr0[meas_clean]); q0_all = A.bed_flow(t, C_clean, ffr0)
    base = dict(scan=int(row.scan), side=row.side, vessel=row.vessel, loc=row["loc"], L_mm=row.L_mm,
                ds_pct=int(row.ds_pct), bed=bed, band_cohort=row.band, ffr_clean=f0, flip_clean=int(f0 <= A.THRESHOLD))
    for etype, fn in ERROR_TYPES.items():
        rec = {**base, "run_id": A.run_id_of(row, bed, etype, "D_perterritory"), "error_type": etype,
               "protocol": "D_perterritory", "status": "ok"}
        segs2, info = fn(list(t.segments), t, path, s_arc, c, L)
        if segs2 is None: rec["status"] = f"skipped: {info}"; out.append(rec); continue
        try:
            t2 = Tree(segs2, f"{t.name}_{etype}", bed=bed, r_trunc=A.trunc_for(bed, etype),
                      trunc_ref=t if etype in A.CALIBRE_ONLY else None)
        except ValueError as e:
            rec["status"] = f"skipped: {e}"; out.append(rec); continue
        m = A.node_map(t, t2)
        p2, _ = t2.vessel_path(HOSTS[row.side][row.vessel])
        if p2 is None or len(p2) < 3: rec["status"] = "skipped: host vessel lost"; out.append(rec); continue
        s2 = t2.arc[p2] - t2.arc[p2[0]]
        L2 = L + T3_LENGTH_DELTA if etype == "T3_stenosis_length" else L
        if c + L2 / 2 >= s2[-1]: rec["status"] = "skipped: lesion outside vessel"; out.append(rec); continue
        r2, _ = insert(t2, p2, s2, c, L2, ds)
        cand = np.where(m == meas_clean)[0]
        meas2 = int(cand[0]) if len(cand) else int(p2[min(int(np.searchsorted(s2, c + L / 2 + RUNOFF)), len(p2) - 1)])
        t_pairs = A.protocol_c_targets(t, t2, m, q0_all)
        if len(t_pairs) < 2:
            rec["status"] = "skipped: fewer than 2 shared territories to match"; out.append(rec); continue
        q_target = np.array([q for _, q, _ in t_pairs])
        t2._C.clear(); C2 = t2.calibrate(t2.demand("murray", 1.0))
        w0 = t2.w.copy()

        def apply(x):
            w = w0.copy()
            for (mem, _, _), xj in zip(t_pairs, x): w[mem] = w0[mem] * 10 ** xj
            t2.w = w

        def resid(x):
            apply(x)
            f, _, _, _, _ = t2.evaluate(C2, r2)
            q = A.bed_flow(t2, C2, f)
            return np.array([q[mem].sum() for mem, _, _ in t_pairs]) / q_target - 1.0

        try:
            fit = least_squares(resid, np.zeros(len(t_pairs)), bounds=(-3.0, 3.0), xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, diff_step=1e-6, max_nfev=400)
            apply(fit.x)
            ffr2, _, info2, _, _ = t2.evaluate(C2, r2); qb = A.bed_flow(t2, C2, ffr2)
            t2.w = w0
            pred = np.array([qb[mem].sum() for mem, _, _ in t_pairs])
            res = float(np.sqrt(np.mean(((pred - q_target) / q_target) ** 2)))
            f2 = float(ffr2[meas2])
            rec.update(n_territories=len(t_pairs), meas_same_point=bool(len(cand)), C=C2,
                       scale_min=float(10 ** fit.x.min()), scale_max=float(10 ** fit.x.max()),
                       fit_at_bound=bool(np.any(np.abs(np.abs(fit.x) - 3.0) < 1e-6)), fit_nfev=int(fit.nfev),
                       ffr=f2, dFFR=f2 - f0, flip=int((f2 <= A.THRESHOLD) != (f0 <= A.THRESHOLD)),
                       outlet_flow_residual=res, inflow_mls=info2["inflow"] * 1e6,
                       inflow_clean_mls=info0["inflow"] * 1e6, converged=bool(info2["converged"]),
                       mass_err=info2["mass_err"])
        except Exception as e:
            t2.w = w0; rec["status"] = f"error: {e.__class__.__name__}: {e}"
        out.append(rec)
    return out, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--cohort", default=None); ap.add_argument("--beds", default="leaky,discrete")
    ap.add_argument("--kscale", type=float, default=1.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); here = Path(__file__).parent.parent
    zerod_ffr.K_MURRAY = 562.0 * a.kscale
    import ablation as A
    A.K_MURRAY = zerod_ffr.K_MURRAY
    print(f"K_MURRAY = {zerod_ffr.K_MURRAY:.1f} s^-1 (scale {a.kscale})", flush=True)
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    rows, t0 = [], time.time()
    for n, (_, r) in enumerate(coh.iterrows(), 1):
        for bed in a.beds.split(","):
            try:
                res, why = run_instance(A, Path(a.root), r, bed)
                if why: print(f"  SKIP {r.scan}_{r.side} {bed}: {why}", file=sys.stderr)
                rows += res
            except Exception as e:
                print(f"  FAIL {r.scan}_{r.side} {bed}: {e.__class__.__name__}: {e}", file=sys.stderr)
        if n % 10 == 0: print(f"  {n}/{len(coh)}  {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out or here / "results" / "ablation-perterritory.csv"); df.to_csv(out, index=False)
    ok = df[df.status == "ok"]
    print(f"\nwrote {out} ({len(df)} rows, {len(ok)} solved, {time.time()-t0:.0f}s)")
    for (bed, e), h in ok.groupby(["bed", "error_type"]):
        pw = ((h.outlet_flow_residual < A.VALIDATED_RESIDUAL) & (h.dFFR.abs() > A.MATERIAL_DFFR)).sum()
        print(f"{bed:<9}{e:<20}n={len(h):>4}  flips {int(h.flip.sum()):>3} ({100*h.flip.mean():4.1f}%)  "
              f"median |dFFR| {h.dFFR.abs().median():.4f}  median resid {h.outlet_flow_residual.median():.4f}  "
              f"pass&wrong {pw} ({100*pw/len(h):.1f}%)  at_bound {int(h.fit_at_bound.sum())}")

if __name__ == "__main__":
    main()
