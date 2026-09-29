"""組成と距離の表し方(上位文書 3-1、v0.2.1 修正メモ 5)。

- 組成ベクトルは x_Al, x_Si, x_Ti, x_Fe, x_Cu の順(at%、合計100)。
- chemsys は元素をアルファベット順にハイフンでつないだもの。
- reduced_formula は pymatgen と同じ規則(電気陰性度順、同値は記号順)で約分した化学式。
"""
from __future__ import annotations

import math
import re
from functools import reduce
from typing import Iterable, Mapping, Sequence

import numpy as np

ELEMENTS: tuple[str, ...] = ("Al", "Si", "Ti", "Fe", "Cu")
EL_INDEX = {e: i for i, e in enumerate(ELEMENTS)}
# Pauling 電気陰性度(pymatgen の Element.X と同じ値)
_PAULING = {"Al": 1.61, "Si": 1.90, "Ti": 1.54, "Fe": 1.83, "Cu": 1.90}
VEC_COLS = [f"x_{e}" for e in ELEMENTS]

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """'Al2Fe3Si3' -> {'Al':2,'Fe':3,'Si':3}。対象外の元素は ValueError。"""
    out: dict[str, float] = {}
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"化学式を解釈できません: {formula!r}")
        el, num = m.group(1), m.group(2)
        if el not in EL_INDEX:
            raise ValueError(f"対象外の元素 {el} を含みます: {formula!r}")
        out[el] = out.get(el, 0.0) + (float(num) if num else 1.0)
        pos = m.end()
    if pos != len(formula) or not out:
        raise ValueError(f"化学式を解釈できません: {formula!r}")
    return out


def normalize(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """負の値を0にし、合計100(at%)に規格化する(修正メモ5)。2次元配列は行ごと。"""
    a = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = a.sum(axis=-1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("組成の合計が0です")
    return a / s * 100.0


def to_vector(comp: Mapping[str, float] | str) -> np.ndarray:
    """化学式または {元素: 量} を at% の組成ベクトル(固定順)にする。"""
    if isinstance(comp, str):
        comp = parse_formula(comp)
    v = np.zeros(len(ELEMENTS))
    for el, amt in comp.items():
        if el not in EL_INDEX:
            raise ValueError(f"対象外の元素 {el}")
        v[EL_INDEX[el]] += amt
    return normalize(v)


def round_vector(x: np.ndarray, ndigits: int = 1) -> np.ndarray:
    """小数第1位に丸めつつ合計100を保つ(最大剰余法)。"""
    q = 10 ** ndigits
    y = normalize(x) * q
    base = np.floor(y + 1e-9)
    rem = int(round(100 * q - base.sum()))
    order = np.argsort(-(y - base), kind="stable")
    base[order[:rem]] += 1
    return base / q


def chemsys(x_or_elements: np.ndarray | Iterable[str], tol: float = 1e-9) -> str:
    """組成ベクトル(0 より大きい元素)または元素の集まりから chemsys を作る。"""
    if isinstance(x_or_elements, np.ndarray) or (
        isinstance(x_or_elements, (list, tuple)) and x_or_elements and not isinstance(x_or_elements[0], str)
    ):
        a = np.asarray(x_or_elements, dtype=float)
        els = [ELEMENTS[i] for i in range(len(ELEMENTS)) if a[i] > tol]
    else:
        els = list(x_or_elements)
    return "-".join(sorted(els))


def n_elements(x: np.ndarray, tol: float = 1e-9) -> int:
    return int((np.asarray(x) > tol).sum())


def reduced_formula(comp: Mapping[str, int | float] | str) -> str:
    """整数比を約分し、pymatgen と同じ並び(電気陰性度、同値は記号)で化学式にする。"""
    if isinstance(comp, str):
        comp = parse_formula(comp)
    items = {k: v for k, v in comp.items() if v > 0}
    if all(abs(v - round(v)) < 1e-8 for v in items.values()):
        ints = {k: int(round(v)) for k, v in items.items()}
        g = reduce(math.gcd, ints.values())
        amounts: dict[str, float] = {k: v // g for k, v in ints.items()}
    else:  # 非整数は最小値で割るだけ
        m = min(items.values())
        amounts = {k: v / m for k, v in items.items()}
    order = sorted(amounts, key=lambda e: (_PAULING[e], e))
    parts = []
    for e in order:
        n = amounts[e]
        if abs(n - 1) < 1e-8:
            parts.append(e)
        elif abs(n - round(n)) < 1e-8:
            parts.append(f"{e}{int(round(n))}")
        else:
            parts.append(f"{e}{n:.4g}")
    return "".join(parts)


def formula_from_counts(counts: Sequence[int]) -> str:
    """固定順の整数ベクトルから約分した化学式を作る。"""
    return reduced_formula({ELEMENTS[i]: c for i, c in enumerate(counts) if c > 0})


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """組成距離 d = sqrt(Σ Δx²)(at%)。ブロードキャスト可。"""
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)


def vector_to_row(x: np.ndarray) -> dict[str, float]:
    return {c: float(v) for c, v in zip(VEC_COLS, round_vector(x))}
