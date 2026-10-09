from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def setup(config: str, root):
    import throat_error as TE
    import territory_granularity as TG
    TE.install(TE.THROAT_TYPES, root)
    TG.setup(config)

def job(args):
    root, row, bed = args
    import ablation as A
    import throat_error as TE
    import territory_granularity as TG
    TE.LOG.clear()
    TE.reset(); abc, terr, _, why1 = A.run_instance(root, row, bed)
    TE.reset(); d, tinfo, why2 = TG.run_d(A, root, row, bed)
    TE.reset()
    empty = {}
    for x in terr:
        if x["n_nodes"] == 0: empty[x["run_id"]] = empty.get(x["run_id"], 0) + 1
    for r in abc:
        if r.get("status") == "ok" and r.get("protocol"):
            r["n_empty"] = empty.get(r["run_id"], 0)
    log = [dict(m, bed=bed) for m in TE.LOG]
    return abc, d, tinfo, log, (why1 or why2)

def run_all(root, coh, config, workers, beds=("leaky", "discrete")):
    jobs = [(root, r, bed) for _, r in coh.iterrows() for bed in beds]
    abc_all, d_all, ti_all, log_all, t0 = [], [], [], [], time.time()
    with Pool(workers, initializer=setup, initargs=(config, root)) as pool:
        for n, (abc, d, ti, log, why) in enumerate(pool.imap(job, jobs, chunksize=1), 1):
            abc_all += abc; d_all += d; log_all += log
            if ti: ti_all.append(ti)
            if why: print(f"  skip job {n}: {why}", file=sys.stderr)
            if n % 20 == 0: print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(abc_all), pd.DataFrame(d_all), pd.DataFrame(ti_all), pd.DataFrame(log_all), time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--config", default="finer_keep", choices=("main_drop", "finer_keep"))
    ap.add_argument("--cohort", default=None); ap.add_argument("--outdir", default=None)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); here = HERE.parent
    coh = pd.read_csv(a.cohort or here / "cohort" / "cohort.csv")
    if a.limit: coh = coh.head(a.limit)
    out = Path(a.outdir or here / "results" / "throat_granularity"); out.mkdir(parents=True, exist_ok=True)
    abc, d, ti, log, dt = run_all(Path(a.root), coh, a.config, a.workers)
    abc.to_csv(out / f"ablation_throat_{a.config}.csv", index=False)
    d.to_csv(out / f"perterritory_throat_{a.config}.csv", index=False)
    ti.to_csv(out / f"territory_counts_{a.config}.csv", index=False)
    log.to_csv(out / f"throat_insert_log_{a.config}.csv", index=False)
    txt = (f"{a.config}: rows A-C {len(abc)}, D {len(d)}, insertions {len(log)}, {dt:.0f}s\n"
           f"status A-C: {abc.status.value_counts().to_dict()}\nstatus D: {d.status.value_counts().to_dict()}\n")
    print(txt); (out / f"run_{a.config}.txt").write_text(txt)

if __name__ == "__main__":
    main()
