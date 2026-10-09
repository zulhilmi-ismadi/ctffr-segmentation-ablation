from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MU, RHO, KT, MMHG = 0.004, 1060.0, 1.52, 133.322
R_FLOOR, R_TRUNC, R_RESOLVED = 0.15e-3, 0.50e-3, 0.75e-3

R_TRUNC_DISCRETE = 0.60e-3

MURRAY_EXP = 2.66
DS_LESION = 0.30
K_MURRAY = 562.0
MAIN = ("LM", "LAD", "LCX", "LCx", "RCA")
P_AORTA, P_VEN = 90 * MMHG, 5 * MMHG

@dataclass
class Segment:
    sid: int
    parent: int | None
    pts: np.ndarray
    r: np.ndarray
    label: str = ""

def robust_taper(s, r, n_iter=4):
    if len(s) < 4 or s[-1] - s[0] < 1e-9:
        return np.full_like(r, np.median(r))
    A = np.vstack([s, np.ones_like(s)]).T; w = np.ones_like(r)
    for _ in range(n_iter):
        coef, *_ = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)
        fit = A @ coef
        w = np.where(r - fit >= -0.12 * np.maximum(fit, 1e-6), 1.0, 0.15)
    return np.maximum(fit, 0.5 * np.median(r))

class Tree:
    def __init__(self, segments: list[Segment], name: str = "", bed: str = "leaky", r_trunc: float | None = None,
                 reference: "Tree | None" = None, trunc_ref: "Tree | None" = None):
        assert bed in ("leaky", "discrete")
        self.name = name; self.bed = bed
        self.r_trunc = r_trunc if r_trunc is not None else (R_TRUNC if bed == "leaky" else R_TRUNC_DISCRETE)
        segs = {s.sid: s for s in segments}; kids = {sid: [] for sid in segs}; roots = []
        for s in segments:
            (kids[s.parent].append(s.sid) if s.parent is not None else roots.append(s.sid))
        assert len(roots) == 1, f"need exactly one root segment, got {roots}"
        order, q = [], [roots[0]]
        while q:
            u = q.pop(0); order.append(u); q += kids[u]
        P, R, RF, RFIT, DSD, LAB, SEG, XYZ = [-1], [], [], [], [0.0], [], [], []
        end_node, end_ref = {}, {}
        self._seg_s, self._seg_fit, self._seg_ref = {}, {}, {}
        for sid in order:
            sg = segs[sid]
            d = np.linalg.norm(np.diff(sg.pts, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(d)])
            r = np.maximum(sg.r, R_FLOOR)
            if reference is not None:

                fit = np.interp(s, reference._seg_s[sid], reference._seg_fit[sid])
                ref = np.interp(s, reference._seg_s[sid], reference._seg_ref[sid])
            else:
                fit = robust_taper(s, r)
                cap = 1.15 * end_ref.get(sg.parent, np.inf)
                ref = np.minimum.accumulate(np.minimum(fit, cap))
            self._seg_s[sid], self._seg_fit[sid], self._seg_ref[sid] = s, fit, ref
            if sg.parent is None:
                R.append(r[0]); RF.append(ref[0]); RFIT.append(fit[0]); LAB.append(sg.label); SEG.append(sid)
                XYZ.append(sg.pts[0]); prev = 0
            else:
                prev = end_node[sg.parent]
            for j in range(1, len(r)):
                P.append(prev); DSD.append(max(d[j - 1], 1e-6)); R.append(r[j]); RF.append(ref[j]); RFIT.append(fit[j])
                LAB.append(sg.label); SEG.append(sid); XYZ.append(sg.pts[j]); prev = len(P) - 1
            end_node[sid] = prev; end_ref[sid] = ref[-1]

        self.parent = np.array(P); self.r = np.asarray(R, dtype=np.float64)
        self.r_ref = np.asarray(RF, dtype=np.float64); self.r_fit = np.asarray(RFIT, dtype=np.float64)
        self.ds = np.asarray(DSD, dtype=np.float64); self.label = np.array(LAB); self.seg = np.array(SEG)

        self.xyz = np.asarray(XYZ, dtype=np.float64)
        n = len(self.parent)
        self.active = self.r_ref >= self.r_trunc; self.active[0] = True
        if trunc_ref is not None:

            key = {tuple(np.round(x, 9)) for x in trunc_ref.xyz[trunc_ref.active]}
            self.active = np.array([tuple(np.round(x, 9)) in key for x in self.xyz])
            self.active[0] = True
        for v in range(1, n):
            self.active[v] &= self.active[self.parent[v]]
        self.children = [[] for _ in range(n)]
        for v in range(1, n):
            if self.active[v]: self.children[self.parent[v]].append(v)
        self.arc = np.zeros(n)
        for v in range(1, n): self.arc[v] = self.arc[self.parent[v]] + self.ds[v]
        self.resolved = self.active & (self.r_fit >= R_RESOLVED)
        self._set_radius(self.r)
        self.leaves = np.array([v for v in np.where(self.active)[0] if not self.children[v]])

        if len(self.leaves) == 0 or (len(self.leaves) == 1 and self.leaves[0] == 0):
            raise ValueError(f"{name}: no outlet survives truncation at r_ref >= {self.r_trunc*1e3:.2f} mm")
        if bed == "leaky":
            w = self.r_ref ** 3
            for v in range(n):
                if self.children[v]:
                    w[v] = max(self.r_ref[v] ** 3 - sum(self.r_ref[c] ** 3 for c in self.children[v]), 0.0)
        else:

            w = np.zeros(n); w[self.leaves] = self.r_ref[self.leaves] ** MURRAY_EXP
        self.w = np.where(self.active, w, 0.0)

        act = np.where(self.active)[0]; self.nr = act[act != 0]
        loc = -np.ones(n, int); loc[self.nr] = np.arange(len(self.nr))
        self._i = loc[self.nr]; self._j = loc[self.parent[self.nr]]

        self.mu = MU
        self._C = {}

    def _set_radius(self, r):
        self.r = r
        self.stenosis = np.clip(1.0 - self.r / np.maximum(self.r_fit, 1e-9), 0.0, 0.99)
        n = len(r); K = np.zeros(n)
        cand = self.resolved & (self.stenosis >= DS_LESION)

        s = self.stenosis
        is_max = cand.copy()
        for v in range(n):
            if not cand[v]: continue
            p = self.parent[v]
            if p >= 0 and cand[p] and s[p] > s[v]: is_max[v] = False
            for c in self.children[v]:
                if cand[c] and s[c] > s[v]: is_max[v] = False; break

        def near(a, b):
            lo, hi = (a, b) if self.arc[a] <= self.arc[b] else (b, a)
            v = hi
            while v >= 0 and self.arc[v] >= self.arc[lo] - 1e-12:
                if v == lo: return self.arc[hi] - self.arc[lo] < 5e-3
                if self.arc[hi] - self.arc[v] >= 5e-3: return False
                v = self.parent[v]
            return False
        throats = list(np.where(is_max)[0])
        throats.sort(key=lambda v: -s[v])
        kept = []
        for v in throats:
            if not any(near(v, u) for u in kept): kept.append(v)
        for v in kept:
            A0 = np.pi * self.r_fit[v] ** 2; As = np.pi * self.r[v] ** 2
            K[v] = RHO * KT / (2 * A0 ** 2) * (A0 / As - 1.0) ** 2
        self.K = K

    def _solve(self, C, P_in, P_v, healthy, max_iter=200, tol=1e-8):
        el, par, i, j = self.nr, self.parent[self.nr], self._i, self._j
        m = len(el); nrp = j >= 0
        rr = self.r_ref if healthy else self.r
        r_mid = np.maximum(0.5 * (rr[el] + rr[par]), R_FLOOR)
        R_lin = 8 * self.mu * self.ds[el] / (np.pi * r_mid ** 4)
        Kel = np.zeros(m) if healthy else self.K[el]
        g_bed = self.w[el] / C; g_root = self.w[0] / C
        Q = np.zeros(m); conv = False; it = 0
        for it in range(1, max_iter + 1):
            g = 1.0 / (R_lin + Kel * np.abs(Q))
            rows = np.concatenate([i, j[nrp], i[nrp], j[nrp]])
            cols = np.concatenate([i, j[nrp], j[nrp], i[nrp]])
            vals = np.concatenate([g + g_bed, g[nrp], -g[nrp], -g[nrp]])
            G = coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
            b = g_bed * P_v; b[~nrp] += g[~nrp] * P_in
            Pn = spsolve(G, b)
            Pp = np.where(nrp, Pn[np.maximum(j, 0)], P_in)
            Qn = (Pp - Pn) * g
            if healthy or not Kel.any():
                Q = Qn; conv = True; break
            if np.max(np.abs(Qn - Q)) <= tol * (np.max(np.abs(Qn)) + 1e-18):
                Q = Qn; conv = True; break
            Q = 0.5 * (Q + Qn)
        inflow = float(np.sum(Q[~nrp]) + g_root * (P_in - P_v))
        bed_out = float(np.sum(g_bed * (Pn - P_v)) + g_root * (P_in - P_v))
        n = len(self.r); Pf = np.full(n, np.nan); Pf[0] = P_in; Pf[el] = Pn; Qf = np.zeros(n); Qf[el] = Q
        self.info = dict(iters=it, converged=conv, inflow=inflow, bed_out=bed_out,
                         mass_err=abs(inflow - bed_out) / max(abs(inflow), 1e-18))
        return Pf, Qf, inflow

    def demand(self, mode="murray", scale=1.0, Q_territory=None):
        return (K_MURRAY * self.r_ref[0] ** 3 if mode == "murray" else Q_territory) * scale

    def calibrate(self, Q_demand, P_in=P_AORTA, P_v=P_VEN):
        key = round(Q_demand * 1e12)
        if key in self._C: return self._C[key]
        f = lambda lc: self._solve(10 ** lc, P_in, P_v, healthy=True)[2] - Q_demand
        q_max = self._solve(1e-30, P_in, P_v, healthy=True)[2]
        if not np.isfinite(q_max) or q_max <= Q_demand:
            raise ValueError(f"{self.name}: healthy epicardial network passes at most {q_max*1e6:.3f} mL/s, "
                             f"below the demand {Q_demand*1e6:.3f} mL/s — tree ineligible for this bed/truncation")
        lo, hi = 0.0, 8.0
        while f(lo) < 0: lo -= 2
        while f(hi) > 0: hi += 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0: lo = mid
            else: hi = mid
        C = 10 ** (0.5 * (lo + hi))
        if not (np.isfinite(C) and C > 1e-12):
            raise ValueError(f"{self.name}: calibration failed, C = {C:.3e}")
        self._C[key] = C
        return C

    def healthy_main_ffr(self, mode="murray", scale=1.0, Q_territory=None, P_in=P_AORTA, P_v=P_VEN) -> float:
        C = self.calibrate(self.demand(mode, scale, Q_territory), P_in, P_v)
        P, _, _ = self._solve(C, P_in, P_v, healthy=True)
        main = self.resolved & np.isin(self.label, MAIN)
        return float(np.nanmin(P[main] / P_in)) if main.any() else np.nan

    def evaluate(self, C, r_override=None, P_in=P_AORTA, P_v=P_VEN):
        r0 = self.r
        if r_override is not None: self._set_radius(np.maximum(r_override, R_FLOOR))
        P, Q, inflow = self._solve(C, P_in, P_v, healthy=False); info = dict(self.info)
        sten, K = self.stenosis.copy(), self.K.copy()
        if r_override is not None: self._set_radius(r0)
        return P / P_in, Q, info, sten, K

    def vessel_path(self, labels):
        cand = np.where(self.active & np.isin(self.label, labels))[0]
        if not len(cand): return None, 0
        v = cand[np.argmin(self.arc[cand])]; path = [v]; n_same = 1; in_label = True
        while self.children[v]:
            ch = self.children[v]
            same = [c for c in ch if self.label[c] in labels] if in_label else []
            if same: v = max(same, key=lambda c: self.r_ref[c]); n_same += 1
            else: in_label = False; v = max(ch, key=lambda c: self.r_ref[c])
            path.append(v)
        return np.array(path), n_same

    def ffr(self, mode="murray", scale=1.0, Q_territory=None, P_in=P_AORTA, P_v=P_VEN) -> dict:
        Qd = self.demand(mode, scale, Q_territory); C = self.calibrate(Qd, P_in, P_v)
        ffr, Q, info, _, _ = self.evaluate(C, None, P_in, P_v)
        main = self.resolved & np.isin(self.label, MAIN)
        out = dict(name=self.name, mode=mode, scale=scale, Q_demand_mls=Qd * 1e6, Q_in_mls=info["inflow"] * 1e6,
                   r_in_mm=self.r_ref[0] * 1e3, n_nodes=int(self.active.sum()),
                   resolved_len_mm=float(self.ds[self.resolved].sum() * 1e3),
                   min_ffr_main=float(np.nanmin(ffr[main])) if main.any() else np.nan,
                   lesion_ds_pct=np.nan, lesion_label="", lesion_ffr20=np.nan, n_lesions=int((self.K > 0).sum()),
                   C=C, iters=info["iters"], converged=info["converged"])
        les = np.where(main & (self.K > 0))[0]
        if len(les):
            v = les[np.argmax(self.stenosis[les])]; d = 0.0; u = v
            while d < 20e-3 and self.children[u]:
                u = max(self.children[u], key=lambda c: self.r_ref[c]); d += self.ds[u]
            out.update(lesion_ds_pct=float(100 * self.stenosis[v]), lesion_label=str(self.label[v]),
                       lesion_ffr20=float(ffr[u]))
        self.last = dict(ffr=ffr, Q=Q)
        return out

    def segment_table(self):
        f, Q, rows = self.last["ffr"], self.last["Q"], []
        for sid in np.unique(self.seg):
            nd = np.where((self.seg == sid) & self.active)[0]
            if len(nd) == 0: continue
            res = nd[self.resolved[nd]]; q0 = nd[1] if (nd[0] == 0 and len(nd) > 1) else nd[0]
            rows.append(dict(seg=int(sid), label=str(self.label[nd[0]]), len_mm=float(self.ds[nd].sum() * 1e3),
                             r_prox=float(self.r[nd[0]] * 1e3), r_dist=float(self.r[nd[-1]] * 1e3),
                             maxDS_resolved=float(100 * self.stenosis[res].max()) if len(res) else np.nan,
                             Q_prox=float(Q[q0] * 1e6), Q_dist=float(Q[nd[-1]] * 1e6),
                             ffr_dist=float(f[nd[-1]]), lesions=int((self.K[nd] > 0).sum())))
        return rows

def _ideal_vessel(ds_pct, r0_mm=1.8, L_mm=90.0, lesion_len_mm=10.0, r_end_mm=0.9, n=181):
    s = np.linspace(0, L_mm, n); r = r0_mm + (r_end_mm - r0_mm) * s / L_mm; c = L_mm * 0.35
    inl = np.abs(s - c) < lesion_len_mm / 2
    r = np.where(inl, r * (1 - ds_pct / 100 * 0.5 * (1 + np.cos(np.pi * (s - c) / (lesion_len_mm / 2)))), r)
    return Segment(0, None, np.stack([s, 0 * s, 0 * s], 1) * 1e-3, r * 1e-3, "LAD")

if __name__ == "__main__":
    print("self-test: tapering LAD-like vessel 1.8 -> 0.9 mm over 90 mm, Murray demand, distributed leakage")
    print(f"{'%DS':>4} {'DS_est':>7} {'Q_in':>6} {'minFFR':>7} {'FFR20':>7} {'iters':>6} {'conv':>5} {'mass_err':>9}")
    for ds in (0, 30, 50, 60, 70, 80, 90):
        t = Tree([_ideal_vessel(ds)]); o = t.ffr()
        print(f"{ds:>4} {o['lesion_ds_pct'] if o['n_lesions'] else 0:>7.1f} {o['Q_in_mls']:>6.2f} {o['min_ffr_main']:>7.3f} "
              f"{o['lesion_ffr20'] if o['n_lesions'] else float('nan'):>7.3f} {o['iters']:>6} {str(o['converged']):>5} {t.info['mass_err']:>9.1e}")
