from __future__ import annotations
import numpy as np
from zerod_ffr import Segment

T4_RADIUS_SCALE = 0.930

T3_LENGTH_DELTA = 2.46e-3

T2_KEEP_BEYOND = 25e-3

from severity_sweep import RUNOFF as _RUNOFF
if not T2_KEEP_BEYOND > _RUNOFF:
    raise ValueError(
        f"T2_KEEP_BEYOND ({T2_KEEP_BEYOND*1e3:.1f} mm) must EXCEED RUNOFF ({_RUNOFF*1e3:.1f} mm) so that the measurement node survives T2")

def _descendants(segs, sid):
    dead = {sid}; grew = True
    while grew:
        grew = False
        for s in segs:
            if s.parent in dead and s.sid not in dead: dead.add(s.sid); grew = True
    return dead

def t1_missed_branch(segs, tree, path, s_arc, c, L, min_r=None):
    pset = set(int(v) for v in path)
    cand = [(tree.r_ref[c_], int(c_), float(s_arc[k])) for k, v in enumerate(path) for c_ in tree.children[v]
            if int(c_) not in pset and s_arc[k] >= c + L / 2 and (min_r is None or tree.r_ref[c_] >= min_r)]
    if not cand: return None, "no deletable downstream branch"
    r_br, node, s_br = max(cand)
    dead = _descendants(segs, int(tree.seg[node]))
    out = [s for s in segs if s.sid not in dead]
    if not out: return None, "deletion would empty the tree"
    return out, dict(branch_r_mm=r_br * 1e3, branch_arc_mm=s_br * 1e3, segments_deleted=len(dead),
                     branch_to_host_ratio=float(r_br / tree.r_fit[path[int(np.argmin(np.abs(s_arc - c)))]]))

def t2_truncation(segs, tree, path, s_arc, c, L):
    s_cut = c + L / 2 + T2_KEEP_BEYOND
    if s_cut >= s_arc[-1]: return None, "vessel already ends before the truncation point"
    k_cut = int(np.searchsorted(s_arc, s_cut))
    node = int(path[k_cut]); sid = int(tree.seg[node])
    host = next(s for s in segs if s.sid == sid)
    keep_pts = [i for i in range(len(host.pts)) if np.linalg.norm(host.pts[i] - tree.xyz[node]) > 1e-12]

    d = np.linalg.norm(host.pts - tree.xyz[node], axis=1); i_cut = int(np.argmin(d))
    if i_cut < 2: return None, "truncation point falls at the start of its segment"
    dead = set()
    for s in segs:
        if s.parent == sid: dead |= _descendants(segs, s.sid)
    out = []
    for s in segs:
        if s.sid in dead: continue
        out.append(Segment(s.sid, s.parent, s.pts[:i_cut + 1].copy(), s.r[:i_cut + 1].copy(), s.label)
                   if s.sid == sid else s)
    return out, dict(cut_arc_mm=float(s_arc[k_cut] * 1e3), length_lost_mm=float((s_arc[-1] - s_arc[k_cut]) * 1e3),
                     segments_deleted=len(dead))

def t3_stenosis_length(segs, tree, path, s_arc, c, L):
    return list(segs), dict(length_delta_mm=T3_LENGTH_DELTA * 1e3)

def t4_taper(segs, tree, path, s_arc, c, L, scale=T4_RADIUS_SCALE):
    s0 = c - L / 2
    k0 = int(np.searchsorted(s_arc, s0)); node = int(path[max(k0, 0)]); sid = int(tree.seg[node])
    fam = _descendants(segs, sid)
    out = []
    for s in segs:
        if s.sid not in fam: out.append(s); continue
        r = s.r.copy()
        if s.sid == sid:
            d = np.linalg.norm(s.pts - tree.xyz[node], axis=1); i0 = int(np.argmin(d))
            r[i0:] *= scale
        else:
            r *= scale
        out.append(Segment(s.sid, s.parent, s.pts, r, s.label))
    return out, dict(radius_scale=scale, segments_scaled=len(fam), from_arc_mm=float(s_arc[max(k0, 0)] * 1e3))

ERROR_TYPES = {"T1_missed_branch": t1_missed_branch, "T2_truncation": t2_truncation,
               "T3_stenosis_length": t3_stenosis_length, "T4_taper": t4_taper}
TOPOLOGICAL = ("T1_missed_branch", "T2_truncation")
