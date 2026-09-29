"""D 3章・6章:カバー率、期待点数 λ、包囲判定、キャリブレーション指標、GD2。"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ..common.cells import all_cells, cell_center, cell_ids
from ..common.composition import normalize


# ---------------------------------------------------------------- 区画への割り当て
def assign_cells(points_at: np.ndarray, delta: float = 5.0) -> list[str]:
    return cell_ids(np.atleast_2d(points_at), delta)


def expected_points(points_at: np.ndarray, n_points_planned: int, delta: float = 5.0) -> pd.Series:
    """λ(区画)= 模擬で区画に入った割合 × 予定の測定点数 N(修正メモ8、R14)。"""
    c = Counter(assign_cells(points_at, delta))
    n = len(points_at)
    return pd.Series({k: v / n * n_points_planned for k, v in c.items()}, name="d_expected_points")


def cell_design_table(points_at: np.ndarray, design_id: str, n_points_planned: int,
                      model_version: str, delta: float = 5.0) -> pd.DataFrame:
    lam = expected_points(points_at, n_points_planned, delta)
    return pd.DataFrame({"cell_id": lam.index, "design_id": design_id, "d_expected_points": lam.values,
                         "n_points_planned": n_points_planned, "model_version": model_version})


def _n_el(cid: str) -> int:
    return int((cell_center(cid) > 0).sum())


def coverage(counts: Mapping[str, float], n_min: float = 1, delta: float = 5.0,
             single_phase: Mapping[str, bool] | None = None,
             target_cells: Iterable[str] | None = None) -> dict[str, float]:
    """C_k(N_min) を k=2..5 で返す。single_phase を与えると単相区画だけで数え直した値も返す(R15)。

    counts:区画ごとの点数(観測値)または λ(予測値)。
    """
    cells = all_cells(delta)
    by_k: dict[int, list[str]] = {}
    for c in cells:
        by_k.setdefault(_n_el(c), []).append(c)
    out: dict[str, float] = {}
    for k in range(2, 6):
        cs = by_k.get(k, [])
        out[f"coverage_{k}"] = float(np.mean([counts.get(c, 0) >= n_min for c in cs])) if cs else np.nan
        if single_phase is not None:
            sp = [c for c in cs if single_phase.get(c, False)]
            out[f"coverage_{k}_mask"] = float(np.mean([counts.get(c, 0) >= n_min for c in sp])) if sp else np.nan
    if target_cells is not None:
        tc = list(target_cells)
        out["coverage_target"] = float(np.mean([counts.get(c, 0) >= n_min for c in tc])) if tc else np.nan
    return out


def required_points(points_at: np.ndarray, target_cells: Sequence[str], lambda_min: float = 5.0,
                    quantile: float = 1.0, delta: float = 5.0) -> float:
    """狙い区画の(quantile の割合)で λ ≥ lambda_min となる最小の測定点数 N。届かない区画があれば inf。"""
    c = Counter(assign_cells(points_at, delta))
    n = len(points_at)
    p = np.array([c.get(t, 0) / n for t in target_cells])
    if len(p) == 0:
        return np.nan
    p = np.sort(p)[::-1]
    m = max(1, int(np.ceil(quantile * len(p))))
    pk = p[m - 1]
    return float(np.inf if pk == 0 else np.ceil(lambda_min / pk))


# ---------------------------------------------------------------- 3-3 包囲判定
def in_hull(p: np.ndarray, pts: np.ndarray) -> bool:
    """p が pts の凸包に含まれるか(線形計画法、退化した配置でも動く)。

    修正メモ5:点を合計100に規格化し、最後の成分を落として4成分で判定する(冗長な合計の行を除く)。
    """
    pts = normalize(pts)[:, :-1]
    p = normalize(p)[:-1]
    n = len(pts)
    A_eq = np.vstack([pts.T, np.ones(n)])
    b_eq = np.append(p, 1.0)
    res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.status == 0


def surrounded(candidate, points, R: float = 10.0, M: int = 50) -> tuple[bool, int]:
    """(判定, 近傍の点数)。組成はすべて at%。"""
    points = np.atleast_2d(np.asarray(points, float))
    candidate = np.asarray(candidate, float)
    d = np.linalg.norm(points - candidate, axis=1)
    near = points[d <= R]
    if len(near) < M:
        return False, len(near)
    return in_hull(candidate, near), len(near)


def cell_sample_table(points_at: np.ndarray, sample_id: str, cells: Iterable[str], R=10.0, M=50) -> pd.DataFrame:
    rows = []
    for c in cells:
        ok, nn = surrounded(cell_center(c), points_at, R, M)
        rows.append({"cell_id": c, "sample_id": sample_id, "surrounded": ok, "n_near": nn})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6-1 キャリブレーション指標
def element_count_hist(points_at: np.ndarray, present_at: float = 2.0) -> np.ndarray:
    """測定点あたりの元素数(present_at 以上)のヒストグラム(割合、k=1..5)。"""
    k = (np.asarray(points_at) >= present_at).sum(1)
    return np.array([(k == i).mean() for i in range(1, 6)])


def interface_width(profile_at: np.ndarray, x_um: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """1本の濃度分布(単調に近い)の 10–90% 幅(µm)。"""
    c = np.asarray(profile_at, float)
    c = (c - c.min()) / max(c.max() - c.min(), 1e-12)
    if c[0] > c[-1]:
        c = 1 - c
    x_lo = np.interp(lo, np.maximum.accumulate(c), x_um)
    x_hi = np.interp(hi, np.maximum.accumulate(c), x_um)
    return float(abs(x_hi - x_lo))


def gd2(sim_points: np.ndarray, exp_points: np.ndarray, tol: float = 0.20, present_at: float = 2.0) -> dict:
    """GD2:元素数ヒストグラムの誤差。誤差 = Σ|h_sim − h_exp| / 2(全変動距離、0〜1)。"""
    hs = element_count_hist(sim_points, present_at)
    he = element_count_hist(exp_points, present_at)
    err = float(np.abs(hs - he).sum() / 2)
    return {"hist_sim": hs.tolist(), "hist_exp": he.tolist(), "error": err, "pass": err <= tol}
