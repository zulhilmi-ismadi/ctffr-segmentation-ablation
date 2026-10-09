from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_endpoints import load, ci, KEY

PROTOS = ("A_fixed", "B_rederived", "C_flowmatched", "D_perterritory")
VARIANTS = ("T5_ds_plus10", "T5_ds_minus10", "T5_vox_narrow", "T5_vox_wide")
POOLS = {"T5_ds_pooled": ("T5_ds_plus10", "T5_ds_minus10"), "T5_vox_pooled": ("T5_vox_narrow", "T5_vox_wide")}
LO, HI = 0.75, 0.85

def grey(x):
    f0, f1 = x.ffr_clean, x.ffr
    x["beyond"] = ((x.flip == 1) & (((f0 > 0.80) & (f1 < LO)) | ((f0 <= 0.80) & (f1 > HI)))).astype(int)
    x["to_pos"] = ((f1 <= 0.80) & (f0 > 0.80)).astype(int)
    x["to_neg"] = ((f0 <= 0.80) & (f1 > 0.80)).astype(int)
    return x

def cell(g):
    n = len(g); fl = int(g.flip.sum()); d = g[g.outlet_flow_residual.notna()]; nr = len(d); pw = int(d.pw.sum())
    return dict(n=n, flips=fl, flip_ci=ci(fl, n), n_resid=nr, pw=pw, pw_ci=ci(pw, nr),
                med_abs_dffr=float(g.dFFR.abs().median()), mean_dffr=float(g.dFFR.mean()),
                beyond=int(g.beyond.sum()), beyond_ci=ci(int(g.beyond.sum()), n),
                to_pos=int(g.to_pos.sum()), to_neg=int(g.to_neg.sum()))

def sign_test(x, a_types, b_types, bed, proto):
    g = x[(x.bed == bed) & (x.protocol == proto)]
    pa = g[g.error_type.isin(a_types)].groupby(KEY).flip.mean()
    pb = g[g.error_type.isin(b_types)].groupby(KEY).flip.mean()
    j = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
    d = j.a - j.b; k = int((d > 0).sum()); m = int((d < 0).sum()); n = k + m
    return dict(n_inst=len(j), rate_a=float(j.a.mean()), rate_b=float(j.b.mean()), a_gt_b=k, b_gt_a=m,
                p=float(binomtest(k, n).pvalue) if n else 1.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("throat_dir"); ap.add_argument("abl"); ap.add_argument("per")
    ap.add_argument("elig"); a = ap.parse_args(); out = Path(a.throat_dir)
    th = grey(load(out / "ablation_throat.csv", out / "perterritory_throat.csv", a.elig))
    fr = grey(load(a.abl, a.per, a.elig))
    x = pd.concat([fr, th], ignore_index=True)
    rows = []
    for bed in ("discrete", "leaky"):
        for e in VARIANTS + tuple(POOLS):
            members = POOLS.get(e, (e,))
            for p in PROTOS:
                g = th[(th.bed == bed) & th.error_type.isin(members) & (th.protocol == p)]
                rows.append(dict(bed=bed, error_type=e, protocol=p, **cell(g)))
        for e, members in (("T1T2_topo", ("T1_missed_branch", "T2_truncation")),
                           ("T3T4_cal", ("T3_stenosis_length", "T4_taper"))):
            for p in PROTOS:
                g = fr[(fr.bed == bed) & fr.error_type.isin(members) & (fr.protocol == p)]
                rows.append(dict(bed=bed, error_type=e, protocol=p, **cell(g)))
    tab = pd.DataFrame(rows); tab.to_csv(out / "throat_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab[["bed", "error_type", "protocol", "n", "flips", "flip_ci", "n_resid", "pw", "pw_ci", "med_abs_dffr",
               "mean_dffr", "beyond", "beyond_ci", "to_pos", "to_neg"]].to_string(index=False))

    topo = ("T1_missed_branch", "T2_truncation"); cal = ("T3_stenosis_length", "T4_taper")
    comps = [("T1T2 vs T3T4", topo, cal)]
    comps += [(f"T1T2 vs {e}", topo, (e,)) for e in VARIANTS]
    comps += [(f"T1T2 vs {e}", topo, m) for e, m in POOLS.items()]
    comps += [(f"{e} vs T3T4", (e,), cal) for e in VARIANTS]
    st = []
    for bed in ("discrete", "leaky"):
        for p in PROTOS:
            for name, a_, b_ in comps:
                st.append(dict(bed=bed, protocol=p, comparison=name, **sign_test(x, a_, b_, bed, p)))
    st = pd.DataFrame(st); st.to_csv(out / "throat_sign_tests.csv", index=False)
    print("\npaired sign test on per-instance flip proportions (a > b counts, b > a counts, exact two-sided p)")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    log = pd.read_csv(out / "throat_insert_log.csv")
    log["scan"] = log.tree.str.extract(r"^(\d+)_").astype(int)
    ab = pd.read_csv(out / "ablation_throat.csv", low_memory=False)
    v = ab[ab.error_type.str.startswith("T5_vox") & (ab.protocol == "A_fixed") & (ab.status == "ok")]
    lines = [f"insertions {len(log)}, clipped {int(log.clipped.sum())}",
             f"realised throat DS differs from applied DS by > 1e-9 in {int(((log.ds_realised - log.ds_applied).abs() > 1e-9).sum())} insertions",
             "in-plane spacing (mm) per scan: " + v.drop_duplicates('scan').info_spacing_mm.describe().round(4).to_dict().__repr__(),
             "voxel-variant |dDS| (pp) per instance-bed: " + v[v.error_type == 'T5_vox_narrow'].info_dDS_requested_pp.abs().describe().round(2).to_dict().__repr__(),
             "throat radius change (mm) = spacing/4: median " + f"{v.drop_duplicates('scan').info_spacing_mm.median()/4:.4f}"]
    txt = "\n".join(lines); print("\n" + txt); (out / "throat_magnitude.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
