from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import throat_error as TE
import ablation as A
import ablation_per_territory as P

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]

def build_types(spec):
    if spec == "throat": return TE.THROAT_TYPES
    return {k: TE.constant_ds_type(v) for k, v in spec.items()}

def one(args):
    root, row, bed, spec = args
    TE.install(build_types(spec), root)
    TE.LOG.clear()
    TE.reset(); abc, _, _, why = A.run_instance(root, row, bed)
    TE.reset(); d, why_d = P.run_instance(A, root, row, bed)
    TE.reset()
    log = [dict(m, bed=bed) for m in TE.LOG]
    return abc, d, log, why or why_d

def run_all(root, coh, beds, types, workers):
    tasks = [(root, r, bed, types) for _, r in coh.iterrows() for bed in beds]
    abc, d, log, t0 = [], [], [], time.time()
    with Pool(workers) as pool:
        for n, (a_, d_, l_, why) in enumerate(pool.imap(one, tasks, chunksize=1), 1):
            abc += a_; d += d_; log += l_
            if n % 20 == 0: print(f"  {n}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(abc), pd.DataFrame(d), pd.DataFrame(log), time.time() - t0

def reference_clean(ref_abl):
    f = pd.read_csv(ref_abl, low_memory=False, float_precision="round_trip")
    return f[f.error_type == "clean"][KEY + ["bed", "ffr"]].rename(columns={"ffr": "ffr_ref"})

def check_clean(abc, d, ref_abl, ref_per):
    fc = reference_clean(ref_abl)
    c = abc[abc.error_type == "clean"][KEY + ["bed", "ffr"]].merge(fc, on=KEY + ["bed"], how="outer", indicator=True)
    both = c[c._merge == "both"]
    dmax = float((both.ffr - both.ffr_ref).abs().max())
    fp = pd.read_csv(ref_per, float_precision="round_trip")[KEY + ["bed", "ffr_clean"]].drop_duplicates()
    dd = d[KEY + ["bed", "ffr_clean"]].drop_duplicates().merge(fp, on=KEY + ["bed"], suffixes=("", "_ref"))
    dmax_d = float((dd.ffr_clean - dd.ffr_clean_ref).abs().max())
    return dict(n_clean_rows=len(both), only_new=int((c._merge == "left_only").sum()),
                only_reference=int((c._merge == "right_only").sum()), max_abs_diff_clean_ffr=dmax,
                identical=bool(dmax == 0.0), n_D_instances=len(dd), max_abs_diff_D_ffr_clean=dmax_d)

def validate(root, coh, out, n, workers, ref_abl, ref_per):
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.concat([coh[coh.side == "left"].head(n // 2 + 1), coh[coh.side == "right"].head(n // 2)]).head(n)
    lines = [f"validation on {len(sub)} instances: " + ", ".join(f"{r.scan}_{r.side}_{r.vessel}_{r['loc']}_{int(r.L_mm)}mm_{int(r.ds_pct)}"
                                                         for _, r in sub.iterrows())]
    zero = {"T5_zero": 0.0}
    abc, d, log, _ = run_all(root, sub, ["leaky", "discrete"], zero, workers)
    ok = pd.concat([abc[(abc.error_type == "T5_zero") & (abc.status == "ok")], d[d.status == "ok"]])
    lines.append("(1) zero-magnitude throat path, |dFFR| by protocol (max):")
    for p, g in ok.groupby("protocol"):
        lines.append(f"    {p:16} n={len(g):3}  max|dFFR|={g.dFFR.abs().max():.3e}  flips={int(g.flip.sum())}")
    lines.append(f"    insertions logged {len(log)}, all ds_applied == ds_orig: {bool((log.ds_applied == log.ds_orig).all())}")
    if ref_abl and ref_per:
        lines.append(f"(2) clean FFR vs reference run: {check_clean(abc, d, ref_abl, ref_per)}")
    grid = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    types = {f"T5_d{int(round(g*100)):+d}": g for g in grid}
    abc3, _, log3, _ = run_all(root, sub, ["leaky", "discrete"], types, workers)
    x = abc3[(abc3.status == "ok") & abc3.protocol.isin(["A_fixed", "B_rederived"])].copy()
    x["dds"] = x.error_type.str.extract(r"d([+-]\d+)").astype(float)
    x["ds_new"] = (x.ds_pct + x.dds).clip(5, 95)
    viol, steps = 0, 0
    mono_rows = []
    for k, g in x.groupby(KEY + ["bed", "protocol"]):
        g = g.sort_values("ds_new"); f = g.ffr.values; dif = np.diff(f)
        steps += len(dif); viol += int((dif >= 0).sum())
        mono_rows.append(dict(zip(KEY + ["bed", "protocol"], k), **{f"ffr_ds{int(a)}": b for a, b in zip(g.ds_new, f)}))
    lines.append(f"(3) FFR strictly decreasing in DS (Protocols A and B, DS change -20..+15 pp): "
                 f"{viol} violations in {steps} steps over {x.groupby(KEY + ['bed']).ngroups} instance-beds")
    pd.DataFrame(mono_rows).to_csv(out / "validation_monotonic.csv", index=False)
    txt = "\n".join(lines); print(txt); (out / "validation.txt").write_text(txt + "\n")

def run(root, coh, out, workers, ref_abl, ref_per):
    out.mkdir(parents=True, exist_ok=True)
    abc, d, log, dt = run_all(root, coh, ["leaky", "discrete"], "throat", workers)
    abc.to_csv(out / "ablation_throat.csv", index=False)
    d.to_csv(out / "perterritory_throat.csv", index=False)
    log.to_csv(out / "throat_insert_log.csv", index=False)
    txt = f"rows A-C {len(abc)}, D {len(d)}, insertions {len(log)}, {dt:.0f}s\n"
    if ref_abl and ref_per:
        txt += f"clean FFR vs reference run: {check_clean(abc, d, ref_abl, ref_per)}\n"
    txt += f"status A-C: {abc.status.value_counts().to_dict()}\nstatus D: {d.status.value_counts().to_dict()}\n"
    print(txt); (out / "run.txt").write_text(txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("validate", "run")); ap.add_argument("root")
    ap.add_argument("--cohort", default=None); ap.add_argument("--outdir", default=None)
    ap.add_argument("--reference-ablation", default=None); ap.add_argument("--reference-perterritory", default=None)
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); here = HERE.parent
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    out = Path(a.outdir or here / "results" / "throat")
    if a.mode == "validate":
        validate(Path(a.root), coh, out, a.n, a.workers, a.reference_ablation, a.reference_perterritory)
    else:
        run(Path(a.root), coh, out, a.workers, a.reference_ablation, a.reference_perterritory)
