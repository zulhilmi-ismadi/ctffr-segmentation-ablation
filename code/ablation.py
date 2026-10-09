from __future__ import annotations
import argparse, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Tree, P_AORTA, P_VEN, R_TRUNC, R_TRUNC_DISCRETE, K_MURRAY
from severity_sweep import load, plan, insert, HOSTS, RUNOFF
from error_types import ERROR_TYPES, TOPOLOGICAL, T3_LENGTH_DELTA, T4_RADIUS_SCALE, T2_KEEP_BEYOND
import severity_sweep as ss

assert T2_KEEP_BEYOND > RUNOFF, (
    f"T2_KEEP_BEYOND ({T2_KEEP_BEYOND*1e3:.1f} mm) must EXCEED RUNOFF ({RUNOFF*1e3:.1f} mm), so that the measurement node survives T2")

THRESHOLD = 0.80

VALIDATED_RESIDUAL = 0.10
MATERIAL_DFFR = 0.05

def node_map(clean: Tree, corrupt: Tree):
    key = {tuple(np.round(x, 9)): i for i, x in enumerate(clean.xyz)}
    return np.array([key.get(tuple(np.round(x, 9)), -1) for x in corrupt.xyz])

def bed_flow(tree: Tree, C: float, ffr: np.ndarray) -> np.ndarray:
    return tree.w / C * (ffr * P_AORTA - P_VEN)

CALIBRE_ONLY = ("T3_stenosis_length", "T4_taper")

def protocol_c_targets(t: Tree, t2: Tree, m: np.ndarray, q0_all: np.ndarray):
    terr_clean = territories(t)
    owner = {int(v): j for j, sub in enumerate(terr_clean) for v in sub}
    pairs = []
    for j, sub_c in enumerate(terr_clean):
        q_clean = float(q0_all[sub_c].sum())
        members = np.array([v for v in range(len(t2.parent))
                            if t2.active[v] and m[v] >= 0 and owner.get(int(m[v]), -1) == j], dtype=int)
        if q_clean > 0 and len(members): pairs.append((members, q_clean, sub_c))

    return pairs

def trunc_for(bed: str, etype: str) -> float:
    base = R_TRUNC if bed == "leaky" else R_TRUNC_DISCRETE
    return base * T4_RADIUS_SCALE if etype == "T4_taper" else base

def subtree(tree: Tree, v: int) -> np.ndarray:
    out, stack = [], [int(v)]
    while stack:
        u = stack.pop(); out.append(u); stack += [int(x) for x in tree.children[u]]
    return np.array(out, dtype=int)

def territories(tree: Tree):
    br = next((v for v in np.where(tree.active)[0] if len(tree.children[v]) >= 2), None)
    if br is None: return []
    out = []
    for c in tree.children[br]:
        sub, stack = [], [int(c)]
        while stack:
            v = stack.pop(); sub.append(v); stack += [int(x) for x in tree.children[v]]
        out.append(np.array(sub, dtype=int))
    return out

PULLBACK_GRID_MM, PULLBACK_CAP, PULLBACK_MERGE_MM = 5.0, 40, 1.0

def run_id_of(row, bed: str, etype: str, proto: str) -> str:
    return (f"{int(row.scan)}_{row.side}_{row.vessel}_{row['loc']}_{int(row.L_mm)}mm_"
            f"{int(row.ds_pct)}ds_{bed}_{etype}_{proto or 'none'}")

def sc_covariates(tree: Tree) -> dict:
    return dict(w_sum=float(tree.w.sum()), r_ref_root_mm=float(tree.r_ref[0] * 1e3),
                L_resolved_mm=float(tree.ds[tree.resolved].sum() * 1e3), n_outlets=int(len(tree.leaves)))

def pullback_rows(tree: Tree, path, s_arc, c, L, meas, ffr, Q, inflow, rid: str):
    res = tree.resolved[path]
    s_end = float(s_arc[res][-1]) if res.any() else float(s_arc[-1])
    anchors = [(float(c - L / 2), "lesion_prox"), (float(c), "throat"), (float(c + L / 2), "lesion_dist")]
    if meas in path: anchors.append((float(s_arc[int(np.argmin(np.abs(path - meas)))]), "measurement"))
    anchors = [(s, k) for s, k in anchors if 0.0 <= s <= s_end]
    for step in (PULLBACK_GRID_MM * 1e-3, 2 * PULLBACK_GRID_MM * 1e-3):
        merged = []
        for s, kind in sorted(anchors, key=lambda z: z[0]):
            if merged and abs(s - merged[-1][0]) < PULLBACK_MERGE_MM * 1e-3: continue
            merged.append((s, kind))
        for s in np.arange(0.0, s_end + 1e-12, step):
            if all(abs(float(s) - m0) >= PULLBACK_MERGE_MM * 1e-3 for m0, _ in merged):
                merged.append((float(s), "grid"))
        merged.sort(key=lambda z: z[0])
        if len(merged) <= PULLBACK_CAP: break
    rows = []
    for s, kind in merged:
        k = int(np.argmin(np.abs(s_arc - s))); v = int(path[k])

        q = float(inflow) if v == 0 else float(Q[v])
        rref = float(tree.r_ref[v])

        kids = tree.children[v]
        orphan = (float(rref ** 3 - sum(tree.r_ref[ch] ** 3 for ch in kids)) / rref ** 3) if (kids and rref > 0) else np.nan
        rows.append(dict(run_id=rid, station_mm=s * 1e3, s_from_lesion_mm=(s - c) * 1e3, kind=kind, node=v,
                         ffr=float(ffr[v]), Q_mls=q * 1e6, r_mm=float(tree.r[v] * 1e3), r_ref_mm=rref * 1e3,
                         r_fit_mm=float(tree.r_fit[v] * 1e3),
                         q_norm=q / (K_MURRAY * rref ** 3) if rref > 0 else np.nan,
                         n_children=len(kids), orphan_frac=orphan, resolved=int(tree.resolved[v])))
    return rows

def run_instance(root: Path, row, bed: str):
    out, terr_out, pull_out = [], [], []
    t = load(root, int(row.scan), row.side, bed)
    o = t.ffr("murray", 1.0); C_clean = o["C"]
    slots, _ = plan(t, row.side, t.last["ffr"].copy())
    sl = next((s for s in slots if s["vessel"] == row.vessel and s["loc"] == row["loc"]
               and abs(s["L"] * 1e3 - row.L_mm) < 1e-6), None)
    if sl is None: return out, terr_out, pull_out, "slot not eligible under this bed"
    path, s_arc, c, L, mi = sl["path"], sl["s"], sl["c"], sl["L"], sl["mi"]
    meas_clean = int(path[mi]); ds = row.ds_pct / 100

    r_clean, _ = insert(t, path, s_arc, c, L, ds)
    ffr0, Q0, info0, _, _ = t.evaluate(C_clean, r_clean)
    if not info0["converged"]: return out, terr_out, pull_out, "clean solve did not converge"
    f0 = float(ffr0[meas_clean]); q0_all = bed_flow(t, C_clean, ffr0)
    base = dict(scan=int(row.scan), side=row.side, vessel=row.vessel, loc=row["loc"], L_mm=row.L_mm,
                ds_pct=int(row.ds_pct), bed=bed, band_cohort=row.band, ffr_clean=f0,
                flip_clean=int(f0 <= THRESHOLD), n_outlets_clean=len(t.leaves))

    terr_clean = territories(t)
    rid0 = run_id_of(row, bed, "clean", "")
    out.append({**base, "run_id": rid0, "error_type": "clean", "protocol": "", "status": "ok",
                "meas_same_point": True, "n_outlets_shared": len(t.leaves), "n_territories": len(terr_clean),
                "C_clean": C_clean, "C": C_clean, "C_abs": C_clean, "C_ratio": 1.0,
                "ffr": f0, "dFFR": 0.0, "flip": 0, "flip_dir": "",
                "outlet_flow_residual": 0.0, "inflow_mls": info0["inflow"] * 1e6,
                "inflow_clean_mls": info0["inflow"] * 1e6, "converged": True,
                "iters": info0["iters"], "mass_err": info0["mass_err"], **sc_covariates(t)})
    pull_out += pullback_rows(t, path, s_arc, c, L, meas_clean, ffr0, Q0, info0["inflow"], rid0)
    for j, sub_c in enumerate(terr_clean):
        terr_out.append(dict(run_id=rid0, terr_id=j, root_node=int(sub_c[0]),
                             root_xyz_mm=";".join(f"{v:.4f}" for v in t.xyz[int(sub_c[0])] * 1e3),
                             Q_target_mls=float(q0_all[sub_c].sum()) * 1e6,
                             Q_achieved_mls=float(q0_all[sub_c].sum()) * 1e6, residual=0.0,
                             sum_w=float(t.w[sub_c].sum()), n_nodes=int(len(sub_c)),
                             n_outlets=int(sum(1 for v in sub_c if v in set(int(x) for x in t.leaves))),
                             contains_lesion=bool(meas_clean in sub_c), contains_error=False))

    for etype, fn in ERROR_TYPES.items():
        segs2, info = fn(list(t.segments), t, path, s_arc, c, L)
        if segs2 is None:
            out.append({**base, "run_id": run_id_of(row, bed, etype, ""), "error_type": etype, "protocol": "",
                        "status": f"skipped: {info}"}); continue
        try:
            t2 = Tree(segs2, f"{t.name}_{etype}", bed=bed, r_trunc=trunc_for(bed, etype),
                      trunc_ref=t if etype in CALIBRE_ONLY else None)
        except ValueError as e:
            out.append({**base, "run_id": run_id_of(row, bed, etype, ""), "error_type": etype, "protocol": "",
                        "status": f"skipped: {e}"}); continue
        m = node_map(t, t2)
        p2, _ = t2.vessel_path(HOSTS[row.side][row.vessel])
        if p2 is None or len(p2) < 3:
            out.append({**base, "run_id": run_id_of(row, bed, etype, ""), "error_type": etype, "protocol": "",
                        "status": "skipped: host vessel lost"}); continue
        s2 = t2.arc[p2] - t2.arc[p2[0]]
        L2 = L + T3_LENGTH_DELTA if etype == "T3_stenosis_length" else L
        if c + L2 / 2 >= s2[-1]:
            out.append({**base, "run_id": run_id_of(row, bed, etype, ""), "error_type": etype, "protocol": "",
                        "status": "skipped: lesion outside vessel"}); continue
        r2, _ = insert(t2, p2, s2, c, L2, ds)

        cand = np.where(m == meas_clean)[0]
        meas2 = int(cand[0]) if len(cand) else int(p2[min(int(np.searchsorted(s2, c + L / 2 + RUNOFF)), len(p2) - 1)])
        meas_same_point = bool(len(cand))

        t_pairs = protocol_c_targets(t, t2, m, q0_all)
        q_target = np.array([q for _, q, _ in t_pairs])
        surv = [v for v in t2.leaves if m[v] >= 0]

        for proto in ("A_fixed", "B_rederived", "C_flowmatched"):
            rec = {**base, "run_id": run_id_of(row, bed, etype, proto),
                   "error_type": etype, "protocol": proto, "status": "ok",
                   "meas_same_point": meas_same_point, "n_outlets_shared": len(surv),
                   "n_territories": len(t_pairs), **sc_covariates(t2),
                   **{f"info_{k}": v for k, v in (info.items() if isinstance(info, dict) else [])}}
            try:
                if proto == "A_fixed":

                    w_saved = t2.w.copy()
                    t2.w = np.where(m >= 0, t.w[np.maximum(m, 0)], 0.0) * (t2.w > 0)
                    if t2.w.sum() <= 0: t2.w = w_saved; rec["status"] = "skipped: no bed left"; out.append(rec); continue
                    C2 = C_clean
                    ffr2, Q2, info2, _, _ = t2.evaluate(C2, r2); qb = bed_flow(t2, C2, ffr2)
                    t2.w = w_saved
                elif proto == "B_rederived":
                    t2._C.clear(); C2 = t2.calibrate(t2.demand("murray", 1.0))
                    ffr2, Q2, info2, _, _ = t2.evaluate(C2, r2); qb = bed_flow(t2, C2, ffr2)
                else:

                    if len(t_pairs) < 2:
                        rec["status"] = "skipped: fewer than 2 shared territories to match"; out.append(rec); continue
                    t2._C.clear(); C_start = t2.calibrate(t2.demand("murray", 1.0))
                    lo_b, hi_b = np.log10(C_start) - 1.5, np.log10(C_start) + 1.5
                    def loss(lc):
                        f, _, _, _, _ = t2.evaluate(10 ** lc, r2)
                        q = bed_flow(t2, 10 ** lc, f)
                        pred = np.array([q[sub].sum() for sub, _, _ in t_pairs])
                        return float(np.mean(((pred - q_target) / q_target) ** 2))

                    grid = np.linspace(lo_b, hi_b, 61)
                    lv = np.array([loss(x) for x in grid])
                    k = int(np.argmin(lv))
                    n_basins = int(sum(1 for i in range(1, len(lv) - 1) if lv[i] < lv[i - 1] and lv[i] < lv[i + 1]))
                    res = minimize_scalar(loss, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                                          method="bounded", options=dict(xatol=1e-9))
                    lc_opt = float(res.x) if float(res.fun) <= lv[k] else float(grid[k])
                    at_bound = bool(min(abs(lc_opt - lo_b), abs(lc_opt - hi_b)) < 1e-6)
                    rec.update(fit_n_basins=n_basins, fit_loss_at_Cstart=float(lv[30]),
                               fit_loss=float(min(float(res.fun), lv[k])), fit_at_bound=at_bound)

                    if at_bound:
                        rec["status"] = "failed fit: Protocol C optimum at search bound"; out.append(rec); continue
                    C2 = 10 ** lc_opt
                    ffr2, Q2, info2, _, _ = t2.evaluate(C2, r2); qb = bed_flow(t2, C2, ffr2)
                f2 = float(ffr2[meas2])
                if len(t_pairs):
                    pred = np.array([qb[sub].sum() for sub, _, _ in t_pairs])
                    resid = float(np.sqrt(np.mean(((pred - q_target) / q_target) ** 2)))
                else:
                    resid = np.nan
                rec.update(C_clean=C_clean, C=C2, C_abs=C2, C_ratio=C2 / C_clean, ffr=f2, dFFR=f2 - f0,
                           flip=int((f2 <= THRESHOLD) != (f0 <= THRESHOLD)),
                           flip_dir=("to_positive" if f2 <= THRESHOLD < f0 else
                                     "to_negative" if f0 <= THRESHOLD < f2 else ""),
                           outlet_flow_residual=resid, inflow_mls=info2["inflow"] * 1e6,
                           inflow_clean_mls=info0["inflow"] * 1e6, converged=bool(info2["converged"]),
                           iters=info2["iters"], mass_err=info2["mass_err"])

                rid = rec["run_id"]
                pull_out += pullback_rows(t2, p2, s2, c, L2, meas2, ffr2, Q2, info2["inflow"], rid)
                covered = set(int(x) for x in m[m >= 0])
                for j, (members, q_tgt_j, sub_c) in enumerate(t_pairs):
                    terr_out.append(dict(
                        run_id=rid, terr_id=j, root_node=int(sub_c[0]),
                        root_xyz_mm=";".join(f"{v:.4f}" for v in t.xyz[int(sub_c[0])] * 1e3),
                        Q_target_mls=q_tgt_j * 1e6, Q_achieved_mls=float(qb[members].sum()) * 1e6,
                        residual=float((qb[members].sum() - q_tgt_j) / q_tgt_j) if q_tgt_j else np.nan,
                        sum_w=float(t2.w[members].sum()), n_nodes=int(len(members)),
                        n_outlets=int(sum(1 for v in members if v in set(int(x) for x in t2.leaves))),
                        contains_lesion=bool(meas_clean in sub_c),

                        contains_error=bool(any(int(v) not in covered for v in sub_c))))
            except Exception as e:
                rec["status"] = f"error: {e.__class__.__name__}: {e}"
            out.append(rec)
    return out, terr_out, pull_out, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--cohort", default=None); ap.add_argument("--beds", default="leaky,discrete")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args(); root = Path(a.root); here = Path(__file__).parent.parent
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    beds = a.beds.split(",")
    out_path = Path(a.out or here / "results" / "ablation.csv")
    rows, terrs, pulls, t0, fails = [], [], [], time.time(), 0
    print(f"cohort {len(coh)} instances x {len(beds)} beds x 4 error types x 3 protocols "
          f"= up to {len(coh)*len(beds)*12} rows\n")
    for n, (_, r) in enumerate(coh.iterrows(), 1):
        for bed in beds:
            try:
                res, tr, pl, why = run_instance(root, r, bed)
                if why: fails += 1; print(f"  SKIP {r.scan}_{r.side} {bed}: {why}", file=sys.stderr)
                rows += res; terrs += tr; pulls += pl
            except Exception as e:
                fails += 1; print(f"  FAIL {r.scan}_{r.side} {bed}: {e.__class__.__name__}: {e}", file=sys.stderr)
                if fails <= 3: traceback.print_exc()
        if n % 10 == 0:
            print(f"  {n}/{len(coh)} instances  {len(rows)} rows  {fails} failures  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); out_path.parent.mkdir(exist_ok=True); df.to_csv(out_path, index=False)

    terr_path = out_path.with_name(out_path.stem + "_territory.csv")
    pull_path = out_path.with_name(out_path.stem + "_pullback.csv")
    pd.DataFrame(terrs).to_csv(terr_path, index=False)
    pd.DataFrame(pulls).to_csv(pull_path, index=False)
    print(f"\nwrote {out_path}  ({len(df)} rows, {fails} instance failures, {time.time()-t0:.0f}s)")
    print(f"      {terr_path.name}  ({len(terrs)} territory rows)")
    print(f"      {pull_path.name}  ({len(pulls)} pullback stations)")
    summarise(df)

def summarise(df):
    ok = df[df.status == "ok"].copy()
    print(f"\n{'='*94}\nABLATION — {len(ok)}/{len(df)} rows solved"
          f"{'' if len(ok)==len(df) else '  (skips: ' + '; '.join(f'{k}: {v}' for k,v in df[df.status!='ok'].status.value_counts().head(4).items()) + ')'}"
          f"\n{'='*94}")
    if ok.empty: return

    nonconv = int((ok.converged.astype("boolean") != True).sum())
    print(f"non-converged {nonconv}  |  max mass error {ok.mass_err.max():.1e}  |  territories per row: "
          f"median {ok.n_territories.median():.0f}")
    for bed, g in ok.groupby("bed"):
        print(f"\n--- bed = {bed} ---")
        print(f"{'error type':<20}{'protocol':<16}{'n':>4}{'mean dFFR':>11}{'mean |dFFR|':>13}{'flips':>8}{'flip %':>8}{'flow resid':>12}")
        for (e, p), h in g.groupby(["error_type", "protocol"]):
            print(f"{e:<20}{p:<16}{len(h):>4}{h.dFFR.mean():>+11.4f}{h.dFFR.abs().mean():>13.4f}"
                  f"{int(h.flip.sum()):>8}{100*h.flip.mean():>7.1f}%{h.outlet_flow_residual.median():>12.4f}")
    print(f"\n--- Models that pass the perfusion check (residual < {VALIDATED_RESIDUAL:.0%}) and are wrong by > {MATERIAL_DFFR:.2f} ---")
    print(f"{'bed':<10}{'protocol':<16}{'n':>4}{'median resid':>14}{'passes check':>14}{'…and wrong':>12}{'…and flips':>12}")
    for bed, g in ok.groupby("bed"):
        for p in ("A_fixed", "B_rederived", "C_flowmatched"):
            h = g[g.protocol == p]
            if h.empty: continue
            passes = h[h.outlet_flow_residual < VALIDATED_RESIDUAL]
            absorbed = passes[passes.dFFR.abs() > MATERIAL_DFFR]
            print(f"{bed:<10}{p:<16}{len(h):>4}{h.outlet_flow_residual.median():>14.4f}"
                  f"{len(passes):>8} ({100*len(passes)/len(h):>3.0f}%){len(absorbed):>8}"
                  f"{int(passes.flip.sum()):>12}")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarise":
        summarise(pd.read_csv(sys.argv[2]))
    else:
        main()
