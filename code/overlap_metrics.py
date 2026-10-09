from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Tree, R_FLOOR
from severity_sweep import load, plan, insert, HOSTS, RUNOFF
from error_types import ERROR_TYPES, T3_LENGTH_DELTA
from ablation import node_map, bed_flow, trunc_for, CALIBRE_ONLY, sc_covariates

TOL = 1e-12

def key(x):
    return tuple(np.round(x, 9))

def full_nodes(segments):
    segs = {s.sid: s for s in segments}; kids = {sid: [] for sid in segs}; roots = []
    for s in segments:
        (kids[s.parent].append(s.sid) if s.parent is not None else roots.append(s.sid))
    order, q = [], [roots[0]]
    while q:
        u = q.pop(0); order.append(u); q += kids[u]
    xyz, r, ds = [], [], []
    for sid in order:
        sg = segs[sid]; d = np.linalg.norm(np.diff(sg.pts, axis=0), axis=1); rr = np.maximum(sg.r, R_FLOOR)
        if sg.parent is None:
            xyz.append(sg.pts[0]); r.append(rr[0]); ds.append(0.0)
        for j in range(1, len(rr)):
            xyz.append(sg.pts[j]); r.append(rr[j]); ds.append(max(d[j - 1], 1e-6))
    return np.array(xyz), np.array(r), np.array(ds)

def with_lesion(xyz, r, tree: Tree, r_les):
    r = r.copy(); changed = np.where(np.abs(r_les - tree.r) > 0)[0]
    pos = {key(x): i for i, x in enumerate(xyz)}
    for v in changed:
        i = pos.get(key(tree.xyz[v]))
        if i is None: raise RuntimeError("lesion node absent from full node list")
        r[i] = r_les[v]
    return r

def pair_metrics(x0, r0, d0, x1, r1, d1):
    pos = {key(x): i for i, x in enumerate(x0)}
    m = np.array([pos.get(key(x), -1) for x in x1])
    hit = m >= 0; mm = m[hit]
    V0 = float(np.sum(np.pi * r0 ** 2 * d0)); V1 = float(np.sum(np.pi * r1 ** 2 * d1))
    Vi = float(np.sum(np.pi * np.minimum(r0[mm], r1[hit]) ** 2 * d1[hit]))
    covered = np.zeros(len(x0), bool); covered[mm] = True
    Ls0 = float(d0.sum()); Ls1 = float(d1.sum())
    tsens = float(d0[covered].sum() / Ls0); tprec = float(d1[hit].sum() / Ls1)
    nest = dict(nest_unmatched=int((~hit).sum()),
                nest_r_excess_mm=float(max(0.0, np.max(r1[hit] - r0[mm])) * 1e3) if hit.any() else 0.0,
                nest_ds_mismatch_mm=float(np.max(np.abs(d1[hit] - d0[mm])) * 1e3) if hit.any() else 0.0)
    return dict(V0=V0, V1=V1, Vi=Vi, tsens=tsens, tprec=tprec, covered=covered, **nest)

def betti0(tree: Tree):
    G = nx.Graph(); act = np.where(tree.active)[0]; G.add_nodes_from(act.tolist())
    for v in act:
        p = tree.parent[v]
        if p >= 0 and tree.active[p]: G.add_edge(int(v), int(p))
    return nx.number_connected_components(G)

def summarise_pair(pm, Vo, pre):
    dsc_tree = 2 * pm["Vi"] / (pm["V0"] + pm["V1"])
    dsc_scan = 2 * (pm["Vi"] + Vo) / (pm["V0"] + pm["V1"] + 2 * Vo)
    cld = 2 * pm["tprec"] * pm["tsens"] / (pm["tprec"] + pm["tsens"])
    return {f"{pre}dsc_tree": dsc_tree, f"{pre}dsc_scan": dsc_scan, f"{pre}cl_tprec": pm["tprec"],
            f"{pre}cl_tsens": pm["tsens"], f"{pre}cldice": cld, f"{pre}vol_lost": 1 - pm["V1"] / pm["V0"],
            f"{pre}V_clean_mm3": pm["V0"] * 1e9, f"{pre}V_corr_mm3": pm["V1"] * 1e9, f"{pre}V_other_mm3": Vo * 1e9,
            f"{pre}nest_unmatched": pm["nest_unmatched"], f"{pre}nest_r_excess_mm": pm["nest_r_excess_mm"],
            f"{pre}nest_ds_mismatch_mm": pm["nest_ds_mismatch_mm"]}

def run_instance(args):
    root, row, bed = args
    out = []
    t = load(root, int(row["scan"]), row["side"], bed)
    o = t.ffr("murray", 1.0); C_clean = o["C"]
    slots, _ = plan(t, row["side"], t.last["ffr"].copy())
    sl = next((s for s in slots if s["vessel"] == row["vessel"] and s["loc"] == row["loc"]
               and abs(s["L"] * 1e3 - row["L_mm"]) < 1e-6), None)
    base = dict(scan=int(row["scan"]), side=row["side"], vessel=row["vessel"], loc=row["loc"], L_mm=row["L_mm"],
                ds_pct=int(row["ds_pct"]), bed=bed)
    if sl is None: return [{**base, "error_type": "clean", "status": "slot not eligible under this bed"}]
    path, s_arc, c, L, mi = sl["path"], sl["s"], sl["c"], sl["L"], sl["mi"]
    ds = row["ds_pct"] / 100
    r_clean, _ = insert(t, path, s_arc, c, L, ds)
    ffr0, Q0, info0, _, _ = t.evaluate(C_clean, r_clean)
    f0 = float(ffr0[int(path[mi])]); q0 = bed_flow(t, C_clean, ffr0)
    act0 = t.active
    x0n, r0n, d0n = t.xyz[act0], r_clean[act0], t.ds[act0]
    x0f, r0f, d0f = full_nodes(t.segments); r0f = with_lesion(x0f, r0f, t, r_clean)
    other = "right" if row["side"] == "left" else "left"
    try:
        to = load(root, int(row["scan"]), other, bed)
        Vo_net = float(np.sum(np.pi * to.r[to.active] ** 2 * to.ds[to.active]))
        _, rof, dof = full_nodes(to.segments); Vo_full = float(np.sum(np.pi * rof ** 2 * dof))
        other_ok = True
    except Exception:
        Vo_net = Vo_full = 0.0; other_ok = False
    out.append({**base, "error_type": "clean", "status": "ok", "ffr_clean_overlap": f0, "other_tree_present": other_ok,
                "b0_clean": betti0(t)})
    for etype, fn in ERROR_TYPES.items():
        rec = {**base, "error_type": etype, "ffr_clean_overlap": f0, "other_tree_present": other_ok}
        segs2, info = fn(list(t.segments), t, path, s_arc, c, L)
        if segs2 is None: out.append({**rec, "status": f"skipped: {info}"}); continue
        try:
            t2 = Tree(segs2, f"{t.name}_{etype}", bed=bed, r_trunc=trunc_for(bed, etype),
                      trunc_ref=t if etype in CALIBRE_ONLY else None)
        except ValueError as e:
            out.append({**rec, "status": f"skipped: {e}"}); continue
        p2, _ = t2.vessel_path(HOSTS[row["side"]][row["vessel"]])
        if p2 is None or len(p2) < 3: out.append({**rec, "status": "skipped: host vessel lost"}); continue
        s2 = t2.arc[p2] - t2.arc[p2[0]]
        L2 = L + T3_LENGTH_DELTA if etype == "T3_stenosis_length" else L
        if c + L2 / 2 >= s2[-1]: out.append({**rec, "status": "skipped: lesion outside vessel"}); continue
        r2, _ = insert(t2, p2, s2, c, L2, ds)
        m = node_map(t, t2)
        act2 = t2.active
        pmn = pair_metrics(x0n, r0n, d0n, t2.xyz[act2], r2[act2], t2.ds[act2])
        nest_inactive = int(np.sum((m[act2] >= 0) & ~t.active[np.maximum(m[act2], 0)]))
        covered_clean = np.zeros(len(t.r), bool); mm = m[act2]; covered_clean[mm[mm >= 0]] = True
        deleted = act0 & ~covered_clean
        flow_lost = float(q0[deleted].sum() / q0[act0].sum())
        w_lost = float(t.w[deleted].sum() / t.w[act0].sum())
        x1f, r1f, d1f = full_nodes(segs2); r1f = with_lesion(x1f, r1f, t2, r2)
        pmf = pair_metrics(x0f, r0f, d0f, x1f, r1f, d1f)
        rec.update(status="ok", **summarise_pair(pmn, Vo_net, "net_"), **summarise_pair(pmf, Vo_full, "full_"),
                   net_nest_inactive=nest_inactive, flow_lost=flow_lost, w_lost=w_lost,
                   n_deleted_nodes=int(deleted.sum()), b0_corr=betti0(t2), b0_clean=betti0(t),
                   **{f"chk_{k}": v for k, v in sc_covariates(t2).items()})
        if etype == "T4_taper":
            scaled = act2 & (np.abs(t2.r - np.where(m >= 0, t.r[np.maximum(m, 0)], np.nan)) > 0)
            rec["t4_scaled_vol_share"] = float(np.sum(np.pi * r_clean[m[scaled]] ** 2 * t.ds[m[scaled]]) / pmn["V0"])
        out.append(rec)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("cohort"); ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--beds", default="leaky,discrete")
    a = ap.parse_args(); root = Path(a.root); outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    coh = pd.read_csv(a.cohort)
    if a.limit: coh = coh.head(a.limit)
    jobs = [(root, r.to_dict(), bed) for _, r in coh.iterrows() for bed in a.beds.split(",")]
    t0 = time.time(); rows = []
    with Pool(a.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(run_instance, jobs, chunksize=1), 1):
            rows += res
            if n % 20 == 0: print(f"{n}/{len(jobs)} jobs  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["scan", "side", "vessel", "loc", "L_mm", "ds_pct", "bed", "error_type"])
    df.to_csv(outd / "overlap_metrics.csv", index=False)
    print(f"wrote {outd/'overlap_metrics.csv'} ({len(df)} rows) in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
