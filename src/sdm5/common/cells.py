"""組成区画(上位文書 3-1、D 3-1、v0.2.1 修正メモ 1・5)。

cell_id の書式: "<解像度コード>:<i_Al>-<i_Si>-<i_Ti>-<i_Fe>-<i_Cu>"
  5 at% 区画  -> "05:04-04-04-04-04"(添字の合計 20)
  2.5 at% 区画 -> "025:08-08-08-08-08"(添字の合計 40)
細分区画の親は、細分区画の中心組成を 5 at% 格子で丸めた区画(同じ丸め規則)。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .composition import ELEMENTS, VEC_COLS, chemsys, distance, normalize


def res_code(delta: float) -> str:
    return "0" + f"{delta:g}".replace(".", "")


def code_to_delta(code: str) -> float:
    s = code.lstrip("0")
    return float(s) if len(s) == 1 else float(s[0] + "." + s[1:])


def to_cell(x_at: Sequence[float] | np.ndarray, delta: float = 5.0) -> tuple[int, ...]:
    """組成を刻み delta の格子点に割り当てる(合計制約つき丸め)。

    修正メモ5: 先に合計100へ規格化し、剰余が同じときは元素順で前の元素を切り上げる。
    """
    n = int(round(100.0 / delta))
    y = normalize(x_at) / delta
    base = np.floor(y + 1e-9).astype(int)
    frac = y - base
    rem = n - int(base.sum())
    # 剰余の大きい順、同値は元素順(安定ソート)。丸め誤差で同値が崩れないよう量子化する
    order = np.argsort(-np.round(frac, 9), kind="stable")
    base[order[:rem]] += 1
    return tuple(int(v) for v in base)


def to_cells(X_at: np.ndarray, delta: float = 5.0) -> np.ndarray:
    """to_cell の一括版(行ごと、同じ丸め規則)。(n, 5) の整数配列を返す。"""
    n = int(round(100.0 / delta))
    y = normalize(np.atleast_2d(X_at)) / delta
    base = np.floor(y + 1e-9).astype(int)
    frac = np.round(y - base, 9)
    rem = n - base.sum(1)
    order = np.argsort(-frac, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(y.shape[1])[None, :].repeat(len(y), 0), axis=1)
    return base + (rank < rem[:, None]).astype(int)


def cell_ids(X_at: np.ndarray, delta: float = 5.0) -> list[str]:
    code = res_code(delta)
    idx = to_cells(X_at, delta)
    return [f"{code}:" + "-".join(f"{i:02d}" for i in row) for row in idx]


def cell_id(idx: Sequence[int], delta: float = 5.0) -> str:
    return f"{res_code(delta)}:" + "-".join(f"{i:02d}" for i in idx)


def parse_cell_id(cid: str) -> tuple[float, tuple[int, ...]]:
    code, rest = cid.split(":")
    return code_to_delta(code), tuple(int(v) for v in rest.split("-"))


def cell_center(cid: str) -> np.ndarray:
    delta, idx = parse_cell_id(cid)
    return np.asarray(idx, float) * delta


def composition_to_cell_id(x_at, delta: float = 5.0) -> str:
    return cell_id(to_cell(x_at, delta), delta)


def parent_cell_id(cid: str, parent_delta: float = 5.0) -> str | None:
    delta, _ = parse_cell_id(cid)
    if delta >= parent_delta:
        return None
    return composition_to_cell_id(cell_center(cid), parent_delta)


def iter_lattice(n: int, k: int = 5) -> Iterator[tuple[int, ...]]:
    """合計 n の非負整数 k 組をすべて列挙する(stars and bars)。"""
    for bars in combinations(range(n + k - 1), k - 1):
        prev = -1
        parts = []
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(n + k - 1 - prev - 1)
        yield tuple(parts)


def all_cells(delta: float = 5.0) -> list[str]:
    n = int(round(100.0 / delta))
    return [cell_id(idx, delta) for idx in iter_lattice(n)]


@dataclass(frozen=True)
class TargetRegion:
    """狙い組成領域(中心と半径、at%)。細分区画を作る範囲に使う。"""
    name: str
    center: tuple[float, ...]
    radius: float


def fine_cells(delta_fine: float = 2.5, al_min: float = 60.0,
               targets: Iterable[TargetRegion] = ()) -> list[str]:
    """Al の多い角(x_Al >= al_min)と狙い組成領域の細分区画。"""
    n = int(round(100.0 / delta_fine))
    targets = list(targets)
    out = []
    for idx in iter_lattice(n):
        c = np.asarray(idx, float) * delta_fine
        keep = c[0] >= al_min - 1e-9
        if not keep:
            for t in targets:
                if distance(c, np.asarray(t.center, float)) <= t.radius + 1e-9:
                    keep = True
                    break
        if keep:
            out.append(cell_id(idx, delta_fine))
    return out


def build_cells_table(delta_base: float = 5.0, delta_fine: float = 2.5, al_min: float = 60.0,
                      targets: Iterable[TargetRegion] = ()) -> pd.DataFrame:
    """cells(区画マスタ)表を作る(修正メモ1)。E-1 の列は空で用意する。"""
    rows = []
    for cid in all_cells(delta_base) + fine_cells(delta_fine, al_min, targets):
        c = cell_center(cid)
        delta, _ = parse_cell_id(cid)
        rows.append({
            "cell_id": cid,
            "parent_cell_id": parent_cell_id(cid, delta_base),
            "resolution": delta,
            **{col: v for col, v in zip(VEC_COLS, c)},
            "n_elements": int((c > 0).sum()),
            "chemsys": chemsys(c),
        })
    df = pd.DataFrame(rows)
    for col, dt in [("e1_solidus", float), ("e1_solidus_flag", object), ("e1_confidence", object)]:
        df[col] = pd.Series([None] * len(df), dtype=dt if dt is object else "float64")
    return df


def count_by_n_elements(delta: float = 5.0) -> dict[int, int]:
    """元素数ごとの区画数(D 3-1 の検算用)。"""
    out: dict[int, int] = {}
    for cid in all_cells(delta):
        k = int((cell_center(cid) > 0).sum())
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


__all__ = [
    "ELEMENTS", "to_cell", "cell_id", "parse_cell_id", "cell_center", "composition_to_cell_id",
    "parent_cell_id", "all_cells", "fine_cells", "build_cells_table", "TargetRegion", "iter_lattice",
    "count_by_n_elements",
]
