"""D 5-1・5-2:粒子の充填(v0.2.1 修正メモ 8、R13)。

ランダム逐次配置(RSA)は3次元で充填率が約 38% 止まりで、圧粉体(≥60%)とかけ離れる。
そこで RSA で置いた粒子の中心と半径から、ラゲール分割(パワー図、周期境界)で空間を埋め、
緻密な圧粉体の近似とする。粉末ごとの体積分率は分割後に実測して報告する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..common.composition import normalize


@dataclass(frozen=True)
class Powder:
    name: str
    composition: tuple[float, ...]     # at%、固定の元素順
    d50_um: float                      # 体積基準ではなく個数基準の中央径(対数正規)
    sigma_ln: float = 0.25             # 対数正規の幅
    volume_fraction: float = 0.2

    @property
    def comp(self) -> np.ndarray:
        return normalize(np.asarray(self.composition, float))


@dataclass
class Packing:
    box_um: float
    centers: np.ndarray                # (n, 3)
    radii: np.ndarray                  # (n,)
    powder_idx: np.ndarray             # (n,)
    powders: list[Powder]
    rsa_fraction: float
    meta: dict = field(default_factory=dict)

    def element_compositions(self) -> np.ndarray:
        return np.stack([p.comp for p in self.powders])


def pure(el_index: int, d50_um: float, vf: float, sigma_ln: float = 0.25, name: str | None = None) -> Powder:
    from ..common.composition import ELEMENTS
    c = [0.0] * 5
    c[el_index] = 100.0
    return Powder(name or ELEMENTS[el_index], tuple(c), d50_um, sigma_ln, vf)


class TooManyParticlesError(ValueError):
    pass


def pack(powders: Sequence[Powder], box_um: float | None = None, target_fraction: float = 0.30,
         seed: int = 0, box_factor: float = 8.0, max_particles: int = 50_000,
         tries_per_particle: int = 300) -> Packing:
    """RSA で重なりなく球を置く。box_um を省略すると最大粒径の box_factor 倍(D 5-1)。

    大きい粒子から順に置き、重なりの判定は格子のハッシュで近傍だけを調べる。
    粒子数が max_particles を超える条件(大小の粒径差が極端な場合など)は TooManyParticlesError。
    その場合は D 5-1 のとおり、粗い計算か2次元の断面計算で扱う。
    """
    rng = np.random.default_rng(seed)
    powders = list(powders)
    vf = np.array([p.volume_fraction for p in powders], float)
    vf = vf / vf.sum()
    if box_um is None:
        dmax = max(p.d50_um * np.exp(2 * p.sigma_ln) for p in powders)
        box_um = box_factor * dmax
    V = box_um ** 3
    radii, idx = [], []
    for i, p in enumerate(powders):
        need = target_fraction * vf[i] * V
        mean_vol = 4 / 3 * np.pi * (0.5 * p.d50_um) ** 3 * np.exp(4.5 * p.sigma_ln ** 2)
        if need / mean_vol + len(radii) > max_particles:
            raise TooManyParticlesError(
                f"粒子数が約 {int(need / mean_vol + len(radii))} 個になります(上限 {max_particles})。"
                "box_um を小さくするか、粒径の範囲を狭めてください")
        vol = 0.0
        while vol < need:
            r = min(0.5 * p.d50_um * np.exp(rng.normal(0, p.sigma_ln)), 0.45 * box_um)
            radii.append(r)
            idx.append(i)
            vol += 4 / 3 * np.pi * r ** 3
    order = np.argsort(-np.asarray(radii), kind="stable")
    radii = np.asarray(radii)[order]
    idx = np.asarray(idx)[order]
    rmax = float(radii[0]) if len(radii) else 1.0
    g = max(2 * rmax, box_um / 64)                       # 格子の大きさ
    ng = max(1, int(box_um // g))
    g = box_um / ng
    grid: dict[tuple[int, int, int], list[int]] = {}
    C = np.empty((len(radii), 3))
    Rr = np.empty(len(radii))
    Pi = np.empty(len(radii), int)
    n = 0
    for r, i in zip(radii, idx):
        cand = rng.uniform(0, box_um, (tries_per_particle, 3))
        reach = int(np.ceil((r + rmax) / g))
        for c in cand:
            gi = (c // g).astype(int) % ng
            nb = []
            rng_ = range(-reach, reach + 1) if 2 * reach + 1 < ng else range(ng)
            full = not (2 * reach + 1 < ng)
            for dx in rng_:
                for dy in rng_:
                    for dz in rng_:
                        key = (dx % ng, dy % ng, dz % ng) if full else \
                              ((gi[0] + dx) % ng, (gi[1] + dy) % ng, (gi[2] + dz) % ng)
                        lst = grid.get(key)
                        if lst:
                            nb.extend(lst)
            if nb:
                nb_a = np.asarray(nb)
                d = np.abs(C[nb_a] - c)
                d = np.minimum(d, box_um - d)
                if np.any((d ** 2).sum(1) < (Rr[nb_a] + r) ** 2):
                    continue
            C[n], Rr[n], Pi[n] = c, r, i
            grid.setdefault(tuple(gi), []).append(n)
            n += 1
            break
    centers, rr = C[:n], Rr[:n]
    frac = float((4 / 3 * np.pi * rr ** 3).sum() / V)
    return Packing(box_um, centers, rr, Pi[:n], powders, frac,
                   {"seed": seed, "target_fraction": target_fraction, "rmax": rmax, "n_requested": len(radii)})


def laguerre_labels(pk: Packing, points: np.ndarray, k: int = 24) -> np.ndarray:
    """点ごとに、パワー距離 |x−c|² − r² が最小の粒子番号(周期境界)。"""
    tree = cKDTree(np.mod(pk.centers, pk.box_um), boxsize=pk.box_um)
    k = min(k, len(pk.centers))
    P = np.mod(np.asarray(points, float), pk.box_um)
    dist, nb = tree.query(P, k=k)
    if k == 1:
        return nb.astype(int)
    power = dist ** 2 - pk.radii[nb] ** 2
    return nb[np.arange(len(P)), np.argmin(power, axis=1)]


def voxel_labels(pk: Packing, n: int) -> np.ndarray:
    """n³ ボクセルのラゲール分割(粒子番号)。"""
    h = pk.box_um / n
    g = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lab = np.empty(len(pts), int)
    step = 200_000
    for s in range(0, len(pts), step):
        lab[s:s + step] = laguerre_labels(pk, pts[s:s + step])
    return lab.reshape(n, n, n)


def measured_volume_fractions(pk: Packing, labels: np.ndarray) -> np.ndarray:
    pidx = pk.powder_idx[labels.ravel()]
    return np.bincount(pidx, minlength=len(pk.powders)) / pidx.size
