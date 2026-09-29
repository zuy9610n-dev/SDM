"""E-2:凸包(v0.2.1 修正メモ 2)。

一つの凸包には同じ計算手法(method タグ)のエントリだけを入れる。混在した場合と、
系の元素について同じ手法の単体の参照がない場合はエラーにする。
pymatgen があればそれを使い、なければ内蔵の線形計画法による実装を使う(結果は一致する)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ..common.composition import ELEMENTS, VEC_COLS, chemsys, parse_formula, reduced_formula, to_vector


class MixedMethodError(ValueError):
    pass


class MissingReferenceError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    struct_id: str
    formula: str
    energy_per_atom: float     # eV/atom(全エネルギー、または手法内で一貫した基準)
    method: str                # 例: "uMLIP:CHGNet", "DFT:PBE-MP2020", "DB:MP-GGA_GGA+U"

    @property
    def x(self) -> np.ndarray:
        return to_vector(self.formula) / 100.0


class Hull:
    def __init__(self, entries: Sequence[Entry]):
        entries = list(entries)
        if not entries:
            raise ValueError("エントリがありません")
        methods = {e.method for e in entries}
        if len(methods) != 1:
            raise MixedMethodError(f"計算手法が混在しています: {sorted(methods)}")
        self.method = methods.pop()
        self.entries = entries
        X = np.stack([e.x for e in entries])
        self.active = np.where(X.sum(0) > 0)[0]
        self.elements = [ELEMENTS[i] for i in self.active]
        # 単体の参照(最も低いもの)
        self.mu = np.zeros(len(ELEMENTS))
        for i in self.active:
            el_e = [e.energy_per_atom for e, x in zip(entries, X) if x[i] > 1 - 1e-9]
            if not el_e:
                raise MissingReferenceError(f"{self.method}: 単体 {ELEMENTS[i]} の参照がありません")
            self.mu[i] = min(el_e)
        self.X = X
        self.E = np.array([e.energy_per_atom for e in entries])
        self.Ef = self.E - X @ self.mu        # 生成エネルギー(eV/atom)

    def hull_energy(self, x: np.ndarray, exclude: int | None = None) -> float:
        """組成 x(モル分率)での凸包の生成エネルギー(LP:min Σ w Ef, Σ w x = x, w ≥ 0)。"""
        idx = np.arange(len(self.entries)) if exclude is None else np.delete(np.arange(len(self.entries)), exclude)
        A = self.X[idx][:, self.active].T
        res = linprog(self.Ef[idx], A_eq=A, b_eq=x[self.active], bounds=(0, None), method="highs")
        if res.status != 0:
            raise MissingReferenceError(f"組成 {x} を凸包で表せません")
        return float(res.fun)

    def e_above_hull(self, i: int) -> float:
        return max(0.0, float(self.Ef[i] - self.hull_energy(self.X[i])))

    def e_hull_of(self, formula: str, energy_per_atom: float) -> float:
        """凸包に入れていない候補の E_hull(負なら凸包を下回る=新たな安定相)。"""
        x = to_vector(formula) / 100.0
        if np.any(x[np.setdiff1d(np.arange(5), self.active)] > 0):
            raise MissingReferenceError(f"{formula} は凸包の元素 {self.elements} の外です")
        ef = energy_per_atom - x @ self.mu
        return float(ef - self.hull_energy(x))

    def table(self) -> pd.DataFrame:
        rows = []
        for i, e in enumerate(self.entries):
            rows.append({"struct_id": e.struct_id, "formula": e.formula,
                         "reduced_formula": reduced_formula(e.formula), "chemsys": chemsys(e.x),
                         "method": self.method, "e_form": float(self.Ef[i]),
                         "e_hull": self.e_above_hull(i)})
        return pd.DataFrame(rows)


def hulls_by_method(entries: Iterable[Entry]) -> dict[str, Hull]:
    by: dict[str, list[Entry]] = {}
    for e in entries:
        by.setdefault(e.method, []).append(e)
    return {m: Hull(es) for m, es in by.items()}


def ehull_columns(structures: pd.DataFrame, entries: Iterable[Entry]) -> pd.DataFrame:
    """構造表に ehull__<method> 列を加える(手法ごとに別の凸包)。"""
    out = structures.copy()
    for m, h in hulls_by_method(entries).items():
        t = h.table()[["struct_id", "e_hull"]].rename(columns={"e_hull": f"ehull__{m}"})
        out = out.merge(t, on="struct_id", how="left")
    return out


def entries_from_frame(df: pd.DataFrame) -> list[Entry]:
    return [Entry(str(r.struct_id), str(r.formula), float(r.energy_per_atom), str(r.method))
            for r in df.itertuples()]
