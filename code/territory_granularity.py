from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

_CFG = None

def territories_finer(tree, base_territories):
    out = []
    for sub in base_territories(tree):
        v = int(sub[0]); trunk = [v]
        while len(tree.children[v]) == 1:
            v = int(tree.children[v][0]); trunk.append(v)
        if len(tree.children[v]) < 2:
            out.append(sub); continue
        out.append(np.array(trunk, dtype=int))
        for ch in tree.children[v]:
            out.append(np.asarray(_subtree(tree, int(ch)), dtype=int))
    return out

def _subtree(tree, v):
    res, stack = [], [v]
    while stack:
        u = stack.pop(); res.append(u); stack += [int(x) for x in tree.children[u]]
    return res

def make_targets(A, level: int, keep_empty: bool):
    base_targets = A.protocol_c_targets
    base_terr = A.territories

    def partition(t):
        return base_terr(t) if level == 1 else territories_finer(t, base_terr)

    def targets(t, t2, m, q0_all):
        base = base_targets(t, t2, m, q0_all)
        if len(base) < 2:
            return base
        terr = partition(t)
        owner = {int(v): j for j, sub in enumerate(terr) for v in sub}
        mm = np.array([owner.get(int(m[v]), -1) if (t2.active[v] and m[v] >= 0) else -1
                       for v in range(len(t2.parent))])
        pairs = []
        for j, sub_c in enumerate(terr):
            q_clean = float(q0_all[sub_c].sum())
            members = np.where(mm == j)[0].astype(int)
            if q_clean > 0 and (len(members) or keep_empty):
                pairs.append((members, q_clean, sub_c))
        return pairs

    return targets, partition

def setup(config: str):
    global _CFG
    import ablation as A
    level = 1 if config.startswith("main") else 2
    keep = config.endswith("keep")
    targets, partition = make_targets(A, level, keep)
    A.protocol_c_targets = targets
    _CFG = dict(config=config, level=level, keep=keep, partition=partition)

def run_d(A, root: Path, row, bed: str):
    from zerod_ffr import Tree
    from severity_sweep import load, plan, insert, HOSTS, RUNOFF
    from error_types import ERROR_TYPES, T3_LENGTH_DELTA
    out = []
    t = load(root, int(row.scan), row.side, bed)
    o = t.ffr("murray", 1.0); C_clean = o["C"]
    slots, _ = plan(t, row.side, t.last["ffr"].copy())
    sl = next((s for s in slots if s["vessel"] == row.vessel and s["loc"] == row["loc"]
               and abs(s["L"] * 1e3 - row.L_mm) < 1e-6), None)
    if sl is None: return out, None, "slot not eligible under this bed"
    path, s_arc, c, L, mi = sl["path"], sl["s"], sl["c"], sl["L"], sl["mi"]
    meas_clean = int(path[mi]); ds = row.ds_pct / 100
    r_clean, _ = insert(t, path, s_arc, c, L, ds)
    ffr0, _, info0, _, _ = t.evaluate(C_clean, r_clean)
    if not info0["converged"]: return out, None, "clean solve did not converge"
    f0 = float(ffr0[meas_clean]); q0_all = A.bed_flow(t, C_clean, ffr0)
    terr = _CFG["partition"](t)
    tinfo = dict(scan=int(row.scan), side=row.side, vessel=row.vessel, loc=row["loc"], L_mm=row.L_mm,
                 ds_pct=int(row.ds_pct), bed=bed, n_terr_main=len(A.territories(t)),
                 n_terr_all=len(terr), n_terr_pos=int(sum(1 for s in terr if q0_all[s].sum() > 0)))
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
        free = [k for k, (mem, _, _) in enumerate(t_pairs) if len(mem)]
        t2._C.clear(); C2 = t2.calibrate(t2.demand("murray", 1.0))
        w0 = t2.w.copy()

        def apply(x):
            w = w0.copy()
            for k, xj in zip(free, x):
                mem = t_pairs[k][0]; w[mem] = w0[mem] * 10 ** xj
            t2.w = w

        def resid(x):
            apply(x)
            f, _, _, _, _ = t2.evaluate(C2, r2)
            q = A.bed_flow(t2, C2, f)
            return np.array([q[mem].sum() for mem, _, _ in t_pairs]) / q_target - 1.0

        try:
            fit = least_squares(resid, np.zeros(len(free)), bounds=(-3.0, 3.0), xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, diff_step=1e-6, max_nfev=400)
            apply(fit.x)
            ffr2, _, info2, _, _ = t2.evaluate(C2, r2); qb = A.bed_flow(t2, C2, ffr2)
            t2.w = w0
            pred = np.array([qb[mem].sum() for mem, _, _ in t_pairs])
            res = float(np.sqrt(np.mean(((pred - q_target) / q_target) ** 2)))
            f2 = float(ffr2[meas2])
            rec.update(n_territories=len(t_pairs), n_empty=len(t_pairs) - len(free),
                       meas_same_point=bool(len(cand)), C=C2,
                       scale_min=float(10 ** fit.x.min()), scale_max=float(10 ** fit.x.max()),
                       fit_at_bound=bool(np.any(np.abs(np.abs(fit.x) - 3.0) < 1e-6)), fit_nfev=int(fit.nfev),
                       ffr=f2, dFFR=f2 - f0, flip=int((f2 <= A.THRESHOLD) != (f0 <= A.THRESHOLD)),
                       outlet_flow_residual=res, inflow_mls=info2["inflow"] * 1e6,
                       inflow_clean_mls=info0["inflow"] * 1e6, converged=bool(info2["converged"]),
                       mass_err=info2["mass_err"])
        except Exception as e:
            t2.w = w0; rec["status"] = f"error: {e.__class__.__name__}: {e}"
        out.append(rec)
    return out, tinfo, ""

def job(args):
    root, row, bed = args
    import ablation as A
    abc, terr, _, why1 = A.run_instance(root, row, bed)
    d, tinfo, why2 = run_d(A, root, row, bed)
    for r in abc:
        if r.get("protocol") == "C_flowmatched" and r.get("status") == "ok":
            r["n_empty"] = int(sum(1 for x in terr if x["run_id"] == r["run_id"] and x["n_nodes"] == 0))
    return abc, d, tinfo, (why1 or why2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--config", required=True, choices=["main_drop", "main_keep", "finer_keep", "finer_drop"])
    ap.add_argument("--cohort", default=None); ap.add_argument("--beds", default="leaky,discrete")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(); here = HERE.parent
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    jobs = [(Path(a.root), r, bed) for _, r in coh.iterrows() for bed in a.beds.split(",")]
    t0 = time.time(); abc_all, d_all, tinfo_all = [], [], []
    with Pool(a.workers, initializer=setup, initargs=(a.config,)) as pool:
        for n, (abc, d, tinfo, why) in enumerate(pool.imap(job, jobs), 1):
            abc_all += abc; d_all += d
            if tinfo: tinfo_all.append(tinfo)
            if why: print(f"  SKIP job {n}: {why}", file=sys.stderr)
            if n % 20 == 0: print(f"  {n}/{len(jobs)} jobs  {time.time()-t0:.0f}s", flush=True)
    out = Path(a.outdir or here / "results" / "territory_granularity"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(abc_all).to_csv(out / f"ablation_{a.config}.csv", index=False)
    pd.DataFrame(d_all).to_csv(out / f"perterritory_{a.config}.csv", index=False)
    pd.DataFrame(tinfo_all).to_csv(out / f"territory_counts_{a.config}.csv", index=False)
    print(f"wrote {out} config {a.config}: {len(abc_all)} A-C rows, {len(d_all)} D rows, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
