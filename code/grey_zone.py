import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).parent))
from summarise_endpoints import load, ci, KEY
x = load(sys.argv[1], sys.argv[2], sys.argv[3])
LO, HI = 0.75, 0.85
f0, f1 = x.ffr_clean, x.ffr
x["within"] = (x.flip == 1) & f0.between(LO, HI) & f1.between(LO, HI)
x["beyond"] = (x.flip == 1) & (((f0 > 0.80) & (f1 < LO)) | ((f0 <= 0.80) & (f1 > HI)))
x["clean_in_gz"] = f0.between(LO, HI)
rows = []
for (bed, cls, p), g in x.groupby(["bed", "cls", "protocol"]):
    n, fl, w, b = len(g), int(g.flip.sum()), int(g.within.sum()), int(g.beyond.sum())
    rows.append(dict(bed=bed, cls=cls, protocol=p, n=n, flips=fl, within=w, beyond=b,
                     flip_ci=ci(fl, n), beyond_ci=ci(b, n), within_share=(100 * w / fl if fl else np.nan)))
t = pd.DataFrame(rows); out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "grey_zone.csv", index=False); print(t.to_string(index=False))
print("\nclean FFR inside grey zone:", x.drop_duplicates(KEY + ["bed"]).groupby("bed").clean_in_gz.mean().round(3).to_dict())
for bed in ("discrete", "leaky"):
    for p in ("A_fixed", "B_rederived", "C_flowmatched", "D_perterritory"):
        g = x[(x.bed == bed) & (x.protocol == p)].groupby(KEY + ["cls"]).beyond.mean().unstack().dropna()
        d = g.topo - g.cal; k, m = int((d > 0).sum()), int((d != 0).sum())
        print(f"{bed:9}{p:16} beyond-grey-zone topo>cal in {k}/{m}, sign test p={binomtest(k, m).pvalue if m else 1:.3g}")
