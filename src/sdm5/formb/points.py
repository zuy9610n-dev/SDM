"""形B 2章・B1:組成点の列挙、学習データの整形。"""
from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..common.cells import composition_to_cell_id, iter_lattice
from ..common.composition import ELEMENTS, VEC_COLS, reduced_formula
from .features import chemsys_any, parse_any


def _reduce(counts: Sequence[int]) -> tuple[int, ...]:
    g = reduce(gcd, counts)
    return tuple(c // g for c in counts)


def _formula(els: Sequence[str], counts: Sequence[int]) -> str:
    """任意元素の約分した化学式(対象5元素は pymatgen 準拠の並び、それ以外は記号順)。"""
    if set(els) <= set(ELEMENTS):
        return reduced_formula(dict(zip(els, counts)))
    return "".join(f"{e}{c if c != 1 else ''}" for e, c in sorted(zip(els, counts)))


def enumerate_points(elements: Sequence[str] = ELEMENTS, max_atoms: int = 20, min_elements: int = 2,
                     max_elements: int | None = None, with_cells: bool = True) -> pd.DataFrame:
    """原子数 ≤ max_atoms、全元素 > 0 の整数比の組成点(約分して重複を除く)。"""
    rows, seen = [], set()
    kmax = max_elements or len(elements)
    for k in range(min_elements, kmax + 1):
        for els in combinations(sorted(elements), k):
            for total in range(k, max_atoms + 1):
                for parts in iter_lattice(total - k, k):
                    counts = _reduce([p + 1 for p in parts])
                    key = (els, counts)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"chemsys": "-".join(els), "reduced_formula": _formula(els, counts),
                                 "n_elements": k, "n_atoms": sum(counts)})
    df = pd.DataFrame(rows)
    if set(elements) <= set(ELEMENTS) and with_cells:
        X = np.zeros((len(df), 5))
        for i, f in enumerate(df["reduced_formula"]):
            p = parse_any(f)
            t = sum(p.values())
            for e, a in p.items():
                X[i, ELEMENTS.index(e)] = a / t * 100
        for j, c in enumerate(VEC_COLS):
            df[c] = X[:, j]
        df["cell_id"] = [composition_to_cell_id(x) for x in X]
    return df


def canonical_key(formula: str, max_atoms: int = 20) -> tuple[str, str, float, bool]:
    """既知化合物を組成点に丸める(B1-3)。(chemsys, 組成点の化学式, 丸め距離 at%, 不規則フラグ)。

    非整数や原子数 > max_atoms の組成は、同じ元素で原子数 ≤ max_atoms の最も近い整数比に丸める。
    """
    p = parse_any(formula)
    els = sorted(p)
    a = np.array([p[e] for e in els], float)
    disordered = bool(np.any(np.abs(a - np.round(a)) > 1e-6))
    x = a / a.sum()
    if not disordered and a.sum() <= max_atoms:
        return "-".join(els), _formula(els, _reduce([int(round(v)) for v in a])), 0.0, False
    best, bestc = np.inf, None
    k = len(els)
    for total in range(k, max_atoms + 1):
        c = np.maximum(1, np.round(x * total)).astype(int)
        d = np.linalg.norm(c / c.sum() - x) * 100
        if d < best - 1e-12:
            best, bestc = d, c
    return "-".join(els), _formula(els, _reduce(list(bestc))), float(best), disordered


def build_training_set(records: pd.DataFrame, max_atoms: int = 20, exclude_status=("計算予測のみ", "薄膜のみ"),
                       metals_only: bool = True, allowed_elements: Iterable[str] | None = None) -> pd.DataFrame:
    """B1:正例の組成リスト。records の列:formula, [year], [report_status], [high_pressure]。

    - 報告の状況が「計算予測のみ」「薄膜のみ」は主の版から除く(ルール1・B1)。
    - 高圧相は除く。組成の重複をまとめ、最初の報告年とエントリ数を記録する。
    - 対象5元系の部分系に属するものに target_system の印をつける。
    """
    r = records.copy()
    if "report_status" in r:
        r = r[~r["report_status"].isin(exclude_status)]
    if "high_pressure" in r:
        r = r[~r["high_pressure"].fillna(False).astype(bool)]
    if allowed_elements is not None:
        allowed = set(allowed_elements)
        r = r[[set(parse_any(f)) <= allowed for f in r["formula"]]]
    keys = [canonical_key(f, max_atoms) for f in r["formula"]]
    r["chemsys"] = [k[0] for k in keys]
    r["reduced_formula"] = [k[1] for k in keys]
    r["snap_d"] = [k[2] for k in keys]
    r["disordered"] = [k[3] for k in keys]
    r = r[r["chemsys"].str.count("-") >= 1]
    agg = {"formula": "first", "snap_d": "min", "disordered": "max"}
    if "year" in r:
        agg["year"] = "min"
    g = r.groupby(["chemsys", "reduced_formula"], as_index=False).agg(agg)
    g["n_entries"] = r.groupby(["chemsys", "reduced_formula"]).size().values
    g["n_elements"] = g["chemsys"].str.count("-") + 1
    g["target_system"] = [set(c.split("-")) <= set(ELEMENTS) for c in g["chemsys"]]
    return g
