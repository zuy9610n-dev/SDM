"""D 5-2:レベル0(幾何学的評価、ボクセルなし)。

仮想の断面に測定点を置き、各元素について「その元素を含む粒子の領域」までの距離を求め、
半径 L 以内に届く元素の数を数える。粒子の領域はラゲール分割の近似として、
各粒子を体積が充填率の逆数倍になるよう膨らませた球で表す(RSA の隙間を埋めるため)。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .packing import Packing, laguerre_labels


def section_points(box_um: float, spacing_um: float, z_um: float | None = None) -> np.ndarray:
    g = np.arange(spacing_um / 2, box_um, spacing_um)
    X, Y = np.meshgrid(g, g, indexing="ij")
    z = box_um / 2 if z_um is None else z_um
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)


def element_reach_counts(pk: Packing, points: np.ndarray, L_um: float, present_at: float = 2.0,
                         k: int = 32) -> np.ndarray:
    """各点で半径 L 以内に届く元素の数。"""
    comps = pk.element_compositions()               # (n_powder, 5)
    has = comps[pk.powder_idx] > present_at         # (n_particle, 5)
    inflate = (1.0 / max(pk.rsa_fraction, 1e-6)) ** (1 / 3)
    r_eff = pk.radii * inflate
    own = laguerre_labels(pk, points)
    reach = has[own].copy()                         # 自分の粒子の元素は距離0
    for e in range(comps.shape[1]):
        sel = np.where(has[:, e])[0]
        if len(sel) == 0:
            continue
        tree = cKDTree(np.mod(pk.centers[sel], pk.box_um), boxsize=pk.box_um)
        kk = min(k, len(sel))
        d, nb = tree.query(np.mod(points, pk.box_um), k=kk)
        d = np.atleast_2d(d.T).T if kk == 1 else d
        nb = np.atleast_2d(nb.T).T if kk == 1 else nb
        surf = np.min(d - r_eff[sel][nb], axis=1)
        reach[:, e] |= surf <= L_um
    return reach.sum(1)


def f_k(pk: Packing, L_um: float, spacing_um: float = 1.0, n_sections: int = 3,
        present_at: float = 2.0, seed: int = 0) -> dict[int, float]:
    """k元素に届く測定点の割合 f_k(L)(k=1..5)。"""
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_sections):
        pts = section_points(pk.box_um, spacing_um, rng.uniform(0, pk.box_um))
        counts.append(element_reach_counts(pk, pts, L_um, present_at))
    c = np.concatenate(counts)
    return {k: float((c == k).mean()) for k in range(1, 6)}
