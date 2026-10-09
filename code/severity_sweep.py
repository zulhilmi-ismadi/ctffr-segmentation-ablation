from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Tree, Segment, DS_LESION
from imagecasx_loader import load_tree

HOSTS = {"left": {"LAD": ("LAD",), "LCX": ("LCX", "LCx")}, "right": {"RCA": ("RCA",)}}
LOCS = {"prox": 20e-3, "mid": 45e-3}
LENGTHS = (10e-3, 20e-3)
SEVERITIES = (40, 50, 55, 60, 65, 70, 75, 80)
MIN_HOST_RFIT, RUNOFF, NATIVE_MAX, BASE_MIN, QUALITY_MIN = 1.0e-3, 20e-3, 0.40, 0.90, 2
SEED = 20260918

def insert(tree: Tree, path, s, c, L, ds_frac):
    inw = np.abs(s - c) < L / 2
    w = 0.5 * (1 + np.cos(np.pi * (s[inw] - c) / (L / 2)))
    nodes = path[inw]; r_new = tree.r.copy()
    r_new[nodes] = np.minimum(tree.r[nodes], (1 - w) * tree.r[nodes] + w * tree.r_fit[nodes] * (1 - ds_frac))
    return r_new, nodes

def plan(tree: Tree, side: str, base_ffr):
    slots, rejects = [], []
    for vessel, labels in HOSTS[side].items():
        path, n_same = tree.vessel_path(labels)
        if path is None: rejects.append((vessel, "-", "-", "vessel absent")); continue
        s = tree.arc[path] - tree.arc[path[0]]; s_same = s[n_same - 1]
        bif = np.array([s[k] for k, v in enumerate(path) if len(tree.children[v]) >= 2])
        for loc, c0 in LOCS.items():
            for L in LENGTHS:
                ci = int(np.argmin(np.abs(s - c0))); c = s[ci]; why = None
                if c - L / 2 < 0 or c + L / 2 > s_same: why = "window outside same-label vessel"
                elif tree.r_fit[path[ci]] < MIN_HOST_RFIT: why = f"host r_fit {tree.r_fit[path[ci]]*1e3:.2f} mm < 1.0"
                else:
                    s_meas = c + L / 2 + RUNOFF
                    if s_meas > s[-1]: why = "run-off < 20 mm"
                    else:
                        mi = int(np.searchsorted(s, s_meas))
                        span = (s >= c - L / 2 - 5e-3) & (s <= s[mi])
                        if not tree.resolved[path[mi]]: why = "measurement node unresolved"
                        elif tree.stenosis[path[span]].max() >= NATIVE_MAX: why = f"native {100*tree.stenosis[path[span]].max():.0f}%DS >= 40"
                        elif base_ffr[path[mi]] < BASE_MIN: why = f"baseline FFR {base_ffr[path[mi]]:.3f} < 0.90"
                if why: rejects.append((vessel, loc, int(L * 1e3), why)); continue
                up = bif[bif <= c - L / 2]; dn = bif[bif >= c + L / 2]
                slots.append(dict(vessel=vessel, loc=loc, L=L, c=c, ci=ci, mi=mi, path=path, s=s,
                                  r_fit_c_mm=tree.r_fit[path[ci]] * 1e3,
                                  d_up_bif_mm=float((c - L / 2 - up.max()) * 1e3) if len(up) else np.nan,
                                  d_dn_bif_mm=float((dn.min() - c - L / 2) * 1e3) if len(dn) else np.nan,
                                  bif_in_window=bool(((bif > c - L / 2) & (bif < c + L / 2)).any()),
                                  base_ffr_meas=float(base_ffr[path[mi]])))
    return slots, rejects

def scans_of(root: Path, split: str):
    d = pd.read_excel(root / "Descriptors.xlsx"); d.columns = [c.strip() for c in d.columns]
    d["Scan ID"] = d["Scan ID"].astype(int); d = d[d["Image Quality"] >= QUALITY_MIN]
    keep = {int(l) for l in (root / "filelist" / f"{split}.txt").read_text().split()}
    return d[d["Scan ID"].isin(keep)].set_index("Scan ID")

def load(root, sid, side, bed="leaky"):
    return load_tree(str(root / "centerlines" / f"{sid}.coronary_{side}_centerline.vtk"),
                     str(root / "segmentations" / f"{sid}.coronary.nii.gz"), name=f"{sid}_{side}", bed=bed)

def run(root: Path, split: str):
    meta = scans_of(root, split); rows, rej, t0, fails = [], [], time.time(), 0
    for n, (sid, m) in enumerate(meta.iterrows(), 1):
        for side in ("left", "right"):
            try:
                t = load(root, sid, side); o = t.ffr("murray", 1.0); base = t.last["ffr"].copy()
                slots, rj = plan(t, side, base); rej += [(sid, side) + r for r in rj]
                for sl in slots:
                    for ds in SEVERITIES:
                        r_new, _ = insert(t, sl["path"], sl["s"], sl["c"], sl["L"], ds / 100)
                        ffr, Q, info, _, _ = t.evaluate(o["C"], r_new)
                        main = t.resolved & np.isin(t.label, HOSTS[side][sl["vessel"]])
                        rows.append(dict(scan=sid, side=side, quality=int(m["Image Quality"]), dominance=m["Dominance"],
                                         disease=m["Disease"], vessel=sl["vessel"], loc=sl["loc"], c_mm=sl["c"] * 1e3,
                                         L_mm=sl["L"] * 1e3, ds_pct=ds, r_fit_c_mm=sl["r_fit_c_mm"],
                                         d_up_bif_mm=sl["d_up_bif_mm"], d_dn_bif_mm=sl["d_dn_bif_mm"],
                                         bif_in_window=sl["bif_in_window"], r_in_mm=o["r_in_mm"],
                                         base_ffr_meas=sl["base_ffr_meas"], ffr_meas=float(ffr[sl["path"][sl["mi"]]]),
                                         ffr_min_vessel=float(np.nanmin(ffr[main])), Q_in_pre=o["Q_in_mls"],
                                         Q_in_post=info["inflow"] * 1e6, iters=info["iters"], converged=info["converged"],
                                         mass_err=info["mass_err"]))
            except Exception as e:
                fails += 1; print(f"  FAIL {sid} {side}: {e.__class__.__name__}: {e}", file=sys.stderr)
        if n % 20 == 0: print(f"  {n}/{len(meta)} scans  {len(rows)} instances  {fails} failures  {time.time()-t0:.0f}s", flush=True)
    res = Path(__file__).parent.parent / "results"; res.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(res / f"sweep_{split}.csv", index=False)
    pd.DataFrame(rej, columns=["scan", "side", "vessel", "loc", "L_mm", "why"]).to_csv(res / f"sweep_{split}_rejections.csv", index=False)
    print(f"\nwrote results/sweep_{split}.csv: {len(df)} instances, {df.groupby(['scan','side','vessel']).ngroups} hosts, "
          f"{df.scan.nunique()} scans, {fails} failures, {time.time()-t0:.0f}s")
    summarise(df)

def summarise(df):
    bands = np.arange(0.40, 1.0001, 0.05)
    print(f"\nnon-converged: {int((~df.converged).sum())};  max mass error {df.mass_err.max():.1e};  baseline FFR_meas median {df.base_ffr_meas.median():.3f}")
    print("\nFFR_meas by inserted severity (median [p10, p90]):")
    for ds, g in df.groupby("ds_pct"): print(f"   {ds:>3}%DS  {g.ffr_meas.median():.3f}  [{g.ffr_meas.quantile(.1):.3f}, {g.ffr_meas.quantile(.9):.3f}]   <=0.80: {100*(g.ffr_meas<=0.80).mean():5.1f}%")
    print("\ninstances per FFR_meas band (supply for selection):")
    h, _ = np.histogram(df.ffr_meas, bins=bands)
    for lo, c in zip(bands[:-1], h): print(f"   [{lo:.2f},{lo+0.05:.2f})  {c:6d}  {'#' * int(60 * c / max(h.max(), 1))}")
    print("\nhosts by vessel:"); print(df.groupby("vessel").apply(lambda g: g.groupby(["scan", "side"]).ngroups, include_groups=False).to_string())

def select(csv_path, per_band=25, max_per_tree=2):
    df = pd.read_csv(csv_path); df = df[df.converged]; rng = np.random.default_rng(SEED)
    edges = np.arange(0.65, 0.9501, 0.05); df["band"] = pd.cut(df.ffr_meas, edges, right=False)
    df = df.dropna(subset=["band"]).sample(frac=1.0, random_state=SEED)
    per_tree, used_host_band, chosen = {}, set(), []
    for band in sorted(df.band.unique(), key=lambda b: (df.band == b).sum()):
        pool = df[df.band == band]; got = 0; vessels = list(rng.permutation(pool.vessel.unique()))
        while got < per_band:
            progressed = False
            for v in vessels:
                for idx, row in pool[pool.vessel == v].iterrows():
                    tree, host = (row.scan, row.side), (row.scan, row.side, row.vessel, str(band))
                    if idx in chosen or per_tree.get(tree, 0) >= max_per_tree or host in used_host_band: continue
                    chosen.append(idx); per_tree[tree] = per_tree.get(tree, 0) + 1; used_host_band.add(host); got += 1; progressed = True; break
                if got >= per_band: break
            if not progressed: break
    sel = df.loc[chosen].sort_values(["band", "vessel", "scan"]); out = Path(csv_path).with_name(Path(csv_path).stem + "_selected.csv")
    sel.to_csv(out, index=False)
    print(f"selected {len(sel)} instances from {sel.groupby(['scan','side']).ngroups} trees (seed {SEED}) -> {out.name}")
    print(pd.crosstab(sel.band, sel.vessel, margins=True).to_string())
    print("\nshortfall vs target of", per_band, "per band:", {str(b): per_band - int(n) for b, n in sel.band.value_counts().items() if n < per_band} or "none")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", nargs=1); g.add_argument("--select", nargs=1)
    ap.add_argument("--split", default="test"); ap.add_argument("--bed", default="leaky", choices=("leaky", "discrete"))
    a = ap.parse_args()
    if a.run: run(Path(a.run[0]), a.split)
    if a.select: select(a.select[0])
