from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import networkx as nx
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from zerod_ffr import Segment, Tree

def _polylines(m: pv.PolyData) -> list[np.ndarray]:
    L = np.asarray(m.lines); out = []; i = 0
    while i < len(L):
        n = int(L[i]); out.append(L[i + 1:i + 1 + n].astype(int)); i += n + 1
    return out

@lru_cache(maxsize=2)
def _edt(mask_path: str):
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt
    img = nib.load(mask_path)
    lab = np.asarray(img.dataobj) > 0
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    nz = np.argwhere(lab)
    lo = np.maximum(nz.min(0) - 4, 0); hi = np.minimum(nz.max(0) + 5, lab.shape)
    crop = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    edt = distance_transform_edt(crop, sampling=sp).astype(np.float32)
    return edt, lo, np.linalg.inv(img.affine), min(sp)

def radius_from_mask(mask_path: str, pts_lps_mm: np.ndarray):
    from scipy.ndimage import map_coordinates
    edt, lo, inv, hmin = _edt(mask_path)
    ras = pts_lps_mm * np.array([-1.0, -1.0, 1.0])
    ijk = (inv @ np.c_[ras, np.ones(len(ras))].T)[:3] - lo[:, None]

    r = map_coordinates(edt, ijk, order=1, mode="nearest").astype(np.float64)
    inside = r > 0

    return np.maximum(r, 0.15), inside

def _smooth(r: np.ndarray, sigma_pts: float = 2.0) -> np.ndarray:
    if len(r) < 5:
        return r
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(r, sigma_pts, mode="nearest")

def load_tree(vtk_path: str, mask_path: str, name: str = "", verbose: bool = False, bed: str = "leaky") -> Tree:
    m = pv.read(vtk_path)
    pts = np.asarray(m.points, dtype=float)
    seg_name = np.asarray(m.point_data["segment_name"]).astype(str)
    start = np.asarray(m.point_data["start_points"]).astype(bool)
    r_mm, inside = radius_from_mask(mask_path, pts)

    keys = {}; canon = np.empty(len(pts), int)
    for i, p in enumerate(pts):
        canon[i] = keys.setdefault(tuple(np.round(p, 2)), i)
    G = nx.Graph()
    for pl in _polylines(m):
        c = canon[pl]
        for a, b in zip(c[:-1], c[1:]):
            if a != b:
                G.add_edge(int(a), int(b))
    if G.number_of_nodes() == 0:
        raise ValueError("empty centreline")
    comp = max(nx.connected_components(G), key=len)
    if len(comp) < G.number_of_nodes() and verbose:
        print(f"  note: kept largest component {len(comp)}/{G.number_of_nodes()} points")
    G = G.subgraph(comp).copy()

    cand = [int(canon[i]) for i in np.where(start)[0] if int(canon[i]) in G]
    if not cand:
        cand = [n for n in G if G.degree(n) == 1]
    root = max(cand, key=lambda n: r_mm[n])
    T = nx.bfs_tree(G, root)

    segs = []; seg_of_endnode = {}
    stack = [(root, None)]
    while stack:
        node, parent_sid = stack.pop()
        for child in T.successors(node):
            chain = [node, child]
            while T.out_degree(chain[-1]) == 1:
                chain.append(next(T.successors(chain[-1])))
            idx = np.array(chain)
            names, counts = np.unique(seg_name[idx[1:]], return_counts=True)
            sid = len(segs)
            rs = _smooth(r_mm[idx])
            if parent_sid is None:

                arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[idx], axis=0), axis=1))])
                j = int(np.searchsorted(arc, 4.0))
                if 0 < j < len(rs):
                    rs[:j] = np.maximum(rs[:j], rs[j])
            segs.append(Segment(sid, parent_sid, pts[idx] * 1e-3, rs * 1e-3, str(names[np.argmax(counts)])))
            stack.append((chain[-1], sid))
    if verbose:
        tot_len = sum(np.linalg.norm(np.diff(s.pts, axis=0), axis=1).sum() for s in segs) * 1e3
        print(f"  {name}: {len(segs)} segments, {tot_len:.0f} mm, points-inside-mask={inside.mean():.1%}, "
              f"radius mm min/med/max = {r_mm.min():.2f}/{np.median(r_mm):.2f}/{r_mm.max():.2f}")
    tree = Tree(segs, name, bed=bed)
    tree.segments = segs
    return tree
