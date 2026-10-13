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
import error_types as ET

KEY = ["scan", "side", "vessel", "loc", "L_mm", "ds_pct"]
IDENTITY = "T0_identity"
BASE_TYPES = {k: v for k, v in ET.ERROR_TYPES.items() if k.startswith(("T1", "T2", "T3", "T4"))}
THROAT = {k: TE.THROAT_TYPES[k] for k in ("T5_vox_narrow", "T5_vox_wide")}
CALIBRE = ("T3_stenosis_length", "T4_taper", IDENTITY, "T5_vox_narrow", "T5_vox_wide")
RATIO = {"v": None}

def noisy_targets(t, t2, m, q0_all):
    terr_clean = A.territories(t)
    owner = {int(v): j for j, sub in enumerate(terr_clean) for v in sub}
    ratio = RATIO["v"]
    pairs = []
    for j, sub_c in enumerate(terr_clean):
        q_clean = float(q0_all[sub_c].sum())
        if ratio is not None:
            q_clean *= float(ratio[j])
        members = np.array([v for v in range(len(t2.parent))
                            if t2.active[v] and m[v] >= 0 and owner.get(int(m[v]), -1) == j], dtype=int)
        if q_clean > 0 and len(members):
            pairs.append((members, q_clean, sub_c))
    return pairs

def identity(segs, tree, path, s_arc, c, L):
    return list(segs), {}

def install(root):
    TE.install(THROAT, root)
    types = {IDENTITY: identity, **BASE_TYPES, **THROAT}
    ET.ERROR_TYPES = types
    A.ERROR_TYPES = types
    A.CALIBRE_ONLY = CALIBRE
    A.protocol_c_targets = noisy_targets

def one(args):
    root, row, bed, draw, ratio = args
    install(root)
    RATIO["v"] = ratio
    TE.reset(); abc, _, _, why = A.run_instance(root, row, bed)
    TE.reset(); d, why_d = P.run_instance(A, root, row, bed)
    TE.reset(); RATIO["v"] = None
    return [dict(r, draw=draw) for r in abc] + [dict(r, draw=draw) for r in d], why or why_d

def build_tasks(root, coh, el, neg, n_draws):
    elig = set(map(tuple, el[KEY].astype(str).values))
    tasks = []
    for _, r in coh.iterrows():
        k = tuple(str(r[c]) for c in KEY)
        for bed in ("leaky", "discrete"):
            if bed == "discrete" and k not in elig:
                continue
            g = neg[(neg.scan == r.scan) & (neg.side == r.side) & (neg.vessel == r.vessel) & (neg["loc"] == r["loc"])
                    & np.isclose(neg.L_mm, r.L_mm) & (neg.ds_pct == r.ds_pct) & (neg.bed == bed)].sort_values("draw")
            tasks.append((root, r, bed, -1, None))
            for _, n in g.head(n_draws).iterrows():
                tgt = np.array([float(x) for x in str(n.terr_target_mls).split(";")])
                nom = np.array([float(x) for x in str(n.terr_pred_start_mls).split(";")])
                tasks.append((root, r, bed, int(n.draw), tgt / nom))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("cohort"); ap.add_argument("eligibility"); ap.add_argument("negatives_pretune")
    ap.add_argument("out_dir"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-draw", type=int, default=19); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    coh = pd.read_csv(a.cohort)
    if a.limit: coh = coh.head(a.limit)
    el = pd.read_csv(a.eligibility); el = el[el.eligible.astype(bool)][KEY]
    neg = pd.read_csv(a.negatives_pretune, low_memory=False)
    tasks = build_tasks(Path(a.root), coh, el, neg, a.max_draw + 1)
    rows, fails, t0 = [], [], time.time()
    with Pool(a.workers) as pool:
        for n, (rr, why) in enumerate(pool.imap_unordered(one, tasks, chunksize=1), 1):
            rows += rr
            if why: fails.append(why)
            if n % 50 == 0 or n == len(tasks): print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "matched_rows.csv", index=False)
    print(f"rows {len(df)}, failures: {fails}")
    print("status:", df.status.value_counts().to_dict())

if __name__ == "__main__":
    main()
