from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Tree, R_TRUNC_DISCRETE, MURRAY_EXP
from severity_sweep import load, plan, insert, HOSTS, RUNOFF, SEVERITIES
import severity_sweep as ss

GATE_HEALTHY, GATE_OUTLETS, GATE_BASELINE = 0.90, 2, 0.90
BANDS = np.arange(0.65, 0.9501, 0.05)

def band_idx(f):
    if not np.isfinite(f) or f < BANDS[0] or f >= BANDS[-1]: return -1
    return int(np.searchsorted(BANDS, f, "right") - 1)

def band_of(f):
    i = band_idx(f)
    return "out" if i < 0 else f"[{BANDS[i]:.2f}, {BANDS[i+1]:.2f})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root); here = Path(__file__).parent.parent
    cohort = Path(a.cohort or here / "cohort" / "cohort.csv")
    out = Path(a.out or here / "results" / "discrete_arm_eligibility.csv")
    coh = pd.read_csv(cohort)
    print(f"cohort: {cohort.name} — {len(coh)} instances, {coh.groupby(['scan','side']).ngroups} trees, "
          f"{coh.scan.nunique()} patients")
    print(f"discrete bed: leaf r_ref^{MURRAY_EXP}, truncation r_ref >= {R_TRUNC_DISCRETE*1e3:.2f} mm, "
          f"gates: healthy-network FFR >= {GATE_HEALTHY}, outlets >= {GATE_OUTLETS}, baseline >= {GATE_BASELINE}\n")

    rows, tcache, t0 = [], {}, time.time()
    for n, r in coh.iterrows():
        rec = dict(scan=int(r.scan), side=r.side, vessel=r.vessel, loc=r["loc"], c_mm=r.c_mm, L_mm=r.L_mm,
                   ds_pct=int(r.ds_pct), band_leaky=r.band, ffr_leaky=r.ffr_meas, base_leaky=r.base_ffr_meas,
                   eligible=False, reason="", n_outlets=np.nan, healthy_main=np.nan, base_discrete=np.nan,
                   ffr_discrete=np.nan, band_discrete="", n_branch_ge_cut=np.nan, deletable_branch=False)
        key = (int(r.scan), r.side)
        if key not in tcache:
            try:
                t = load(root, key[0], key[1], "discrete"); o = t.ffr("murray", 1.0)
                tcache[key] = (t, o, t.last["ffr"].copy(), plan(t, key[1], t.last["ffr"].copy())[0], "")
            except ValueError as e:
                tcache[key] = (None, None, None, None, ("E2 calibration infeasible" if "passes at most" in str(e)
                                                        else "E1 no outlet survives truncation"))
        t, o, base, slots, err = tcache[key]
        if t is None:
            rec["reason"] = err; rows.append(rec); continue
        rec["n_outlets"] = len(t.leaves)
        rec["healthy_main"] = hm = t.healthy_main_ffr()
        if len(t.leaves) < GATE_OUTLETS: rec["reason"] = "E4 fewer than 2 outlets"
        elif not (hm >= GATE_HEALTHY): rec["reason"] = f"E3 healthy-network FFR {hm:.3f} < {GATE_HEALTHY}"
        if rec["reason"]: rows.append(rec); continue
        sl = next((s for s in slots if s["vessel"] == r.vessel and s["loc"] == r["loc"]
                   and abs(s["L"] * 1e3 - r.L_mm) < 1e-6), None)
        if sl is None:
            rec["reason"] = "E5 slot not eligible under discrete truncation"; rows.append(rec); continue
        if not (sl["base_ffr_meas"] >= GATE_BASELINE):
            rec["reason"] = f"E6 discrete baseline {sl['base_ffr_meas']:.3f} < {GATE_BASELINE}"; rows.append(rec); continue

        pset = set(sl["path"].tolist())
        br = [c for k, v in enumerate(sl["path"]) for c in t.children[v]
              if c not in pset and sl["s"][k] >= sl["c"] + sl["L"] / 2 and t.r_ref[c] >= R_TRUNC_DISCRETE]
        rec["n_branch_ge_cut"] = len(br); rec["deletable_branch"] = len(br) > 0
        rr, _ = insert(t, sl["path"], sl["s"], sl["c"], sl["L"], r.ds_pct / 100)
        ffr, Q, info, _, _ = t.evaluate(o["C"], rr)
        if not info["converged"] or not np.isfinite(ffr[sl["path"][sl["mi"]]]):
            rec["reason"] = "E2 solve did not converge"; rows.append(rec); continue
        rec.update(eligible=True, reason="", base_discrete=float(sl["base_ffr_meas"]),
                   ffr_discrete=float(ffr[sl["path"][sl["mi"]]]))
        rec["band_discrete"] = band_of(rec["ffr_discrete"])
        rows.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(coh)}  eligible so far {sum(x['eligible'] for x in rows)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows); out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False)
    report(df); print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")

def report(df):
    n = len(df); el = df[df.eligible]
    print(f"\n{'='*78}\nDISCRETE ARM ELIGIBILITY — CONSORT-style flow\n{'='*78}")
    print(f"  cohort                                {n:>4}")
    print(f"  eligible for the discrete arm         {len(el):>4}   ({100*len(el)/n:.0f}%)")
    print(f"    of which a deletable branch exists  {int(el.deletable_branch.sum()):>4}   (missed-branch / truncation error types)")
    same = el.ffr_discrete.map(band_idx) == el.ffr_leaky.map(band_idx)
    both = el[same]
    print(f"    of which the FFR band is unchanged  {len(both):>4}")
    print(f"\n  excluded                              {n-len(el):>4}")
    for why, k in df[~df.eligible].reason.value_counts().items():
        print(f"      {k:>3}  {why}")
    if len(el):
        print(f"\n  discrete vs leaky on the eligible set:")
        d = el.ffr_discrete - el.ffr_leaky
        print(f"      FFR offset      mean {d.mean():+.4f}  sd {d.std():.4f}  min {d.min():+.4f}  max {d.max():+.4f}")
        dis = int(((el.ffr_discrete <= 0.80) != (el.ffr_leaky <= 0.80)).sum())
        print(f"      decision disagreement  {dis}/{len(el)}  ({100*dis/len(el):.0f}%)")
        print(f"      outlets per tree: median {el.n_outlets.median():.0f} (min {el.n_outlets.min():.0f}, max {el.n_outlets.max():.0f})")
        print(f"\n  eligible by vessel: {dict(el.vessel.value_counts())}")
        print(f"  eligible by leaky band:    {dict(el.band_leaky.value_counts().sort_index())}")
        print(f"  eligible by DISCRETE band: {dict(el.band_discrete.value_counts().sort_index())}")


if __name__ == "__main__":
    main()
