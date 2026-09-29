"""形B B2:特徴量(組成だけから計算)。

学習データは金属・半金属の化合物全体(B1)なので、任意の元素を扱う。
- 元素の性質の統計量(組成で重みづけした平均・偏差・最大・最小・範囲)。Magpie の簡易版。
  元素の性質は pymatgen があればそこから取り、なければ内蔵表(対象5元素)を使う。
- 金属間化合物向けの指標:VEC、電気陰性度差、原子サイズの不一致度 δ。
- 化学量論の特徴:元素数、Lp ノルム、原子数、最大の整数。
- 部分系の情報:低次の部分系にある既知化合物の数、最も近い既知化合物までの距離。
  リーク対策として、exclude_systems に含まれる系と、それを含む上位の系の既知化合物を数えない。
"""
from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

# 内蔵表:Pauling 電気陰性度、金属結合半径 pm、価電子数、融点 K、族、周期、原子番号
_BUILTIN = {
    "Al": (1.61, 143., 3, 933.5, 13, 3, 13), "Si": (1.90, 117., 4, 1687., 14, 3, 14),
    "Ti": (1.54, 147., 4, 1941., 4, 4, 22), "Fe": (1.83, 126., 8, 1811., 8, 4, 26),
    "Cu": (1.90, 128., 11, 1357.8, 11, 4, 29),
}
PROPS = ("X", "radius", "valence", "Tm", "group", "row", "Z")


def parse_any(formula: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for el, num in _TOKEN.findall(formula):
        out[el] = out.get(el, 0.0) + (float(num) if num else 1.0)
    if not out:
        raise ValueError(formula)
    return out


def chemsys_any(formula: str) -> str:
    return "-".join(sorted(parse_any(formula)))


@lru_cache(maxsize=None)
def element_props(el: str) -> tuple[float, ...]:
    if el in _BUILTIN:
        return _BUILTIN[el]
    try:
        from pymatgen.core import Element
    except ImportError as e:  # pragma: no cover
        raise KeyError(f"元素 {el} の性質がありません(pymatgen を入れてください)") from e
    E = Element(el)
    r = E.metallic_radius or E.atomic_radius or np.nan
    r = float(r) * 100 if r and r < 10 else float(r)   # Å → pm
    try:
        val = float(E.valence[1]) if E.valence else float(E.group)
    except Exception:
        val = float(E.group if E.group <= 12 else E.group - 10)
    return (float(E.X), r, val, float(E.melting_point or np.nan), float(E.group), float(E.row), float(E.Z))


def composition_matrix(formulas: Iterable[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(原子分率の行列 n×E, 元素の一覧, 原子数)。"""
    parsed = [parse_any(f) for f in formulas]
    els = sorted({e for p in parsed for e in p})
    idx = {e: i for i, e in enumerate(els)}
    X = np.zeros((len(parsed), len(els)))
    n_atoms = np.zeros(len(parsed))
    for r, p in enumerate(parsed):
        for e, a in p.items():
            X[r, idx[e]] = a
        n_atoms[r] = sum(p.values())
    return X / X.sum(1, keepdims=True), els, n_atoms


def elemental_features(formulas: list[str]) -> pd.DataFrame:
    W, els, n_atoms = composition_matrix(formulas)
    P = np.array([element_props(e) for e in els], float)        # E × p
    present = W > 0
    feats: dict[str, np.ndarray] = {}
    for j, name in enumerate(PROPS):
        prop = np.nan_to_num(P[:, j], nan=np.nanmean(P[:, j]))
        mean = W @ prop
        feats[f"{name}_mean"] = mean
        feats[f"{name}_dev"] = np.sqrt(np.clip(W @ prop ** 2 - mean ** 2, 0, None))
        big = np.where(present, prop, -np.inf).max(1)
        small = np.where(present, prop, np.inf).min(1)
        feats[f"{name}_max"], feats[f"{name}_min"], feats[f"{name}_range"] = big, small, big - small
    r = np.nan_to_num(P[:, 1], nan=np.nanmean(P[:, 1]))
    rbar = W @ r
    feats["delta_size"] = np.sqrt((W * (1 - r[None, :] / rbar[:, None]) ** 2).sum(1))
    feats["vec"] = W @ np.nan_to_num(P[:, 2])
    feats["n_el"] = present.sum(1)
    for p in (2, 3, 5, 10):
        feats[f"L{p}"] = (W ** p).sum(1) ** (1 / p)
    feats["n_atoms"] = n_atoms
    feats["max_int"] = np.round(W * n_atoms[:, None]).max(1)
    return pd.DataFrame(feats)


def _lower_subsystems(cs: str) -> list[str]:
    els = cs.split("-")
    return ["-".join(sorted(c)) for k in range(2, len(els)) for c in combinations(els, k)]


def _blocked(cs: str, exclude: set[frozenset]) -> bool:
    s = set(cs.split("-"))
    return any(e <= s for e in exclude)


def subsystem_features(formulas: list[str], known_formulas: list[str],
                       exclude_systems: Iterable[str] = ()) -> pd.DataFrame:
    """低次の部分系の既知化合物の数と、最も近い既知化合物までの距離(at%、同じ部分系・低次の部分系)。"""
    exclude = {frozenset(e.split("-")) for e in exclude_systems}
    kf = [f for f in known_formulas if not _blocked(chemsys_any(f), exclude)] if exclude else list(known_formulas)
    kcs = [chemsys_any(f) for f in kf]
    cnt: dict[str, int] = {}
    for c in kcs:
        cnt[c] = cnt.get(c, 0) + 1
    by_sys: dict[str, list[dict]] = {}
    for f, c in zip(kf, kcs):
        p = parse_any(f)
        t = sum(p.values())
        by_sys.setdefault(c, []).append({e: a / t * 100 for e, a in p.items()})
    rows = []
    for f in formulas:
        cs = chemsys_any(f)
        subs = _lower_subsystems(cs)
        c = [cnt.get(s, 0) for s in subs]
        p = parse_any(f)
        t = sum(p.values())
        x = {e: a / t * 100 for e, a in p.items()}
        best = 100.0
        for s in [cs] + subs:
            for k in by_sys.get(s, []):
                d = np.sqrt(sum((x.get(e, 0) - k.get(e, 0)) ** 2 for e in x))
                if d > 1e-6 and d < best:            # 自分自身は除く
                    best = d
        rows.append({"sub_known": sum(c), "sub_known_max": max(c) if c else 0,
                     "sub_known_min": min(c) if c else 0, "d_near_known": best})
    return pd.DataFrame(rows)


def build_features(formulas: list[str], known_formulas: list[str], exclude_systems: Iterable[str] = (),
                   use_subsystem: bool = True) -> pd.DataFrame:
    parts = [elemental_features(formulas)]
    if use_subsystem:
        parts.append(subsystem_features(formulas, known_formulas, exclude_systems))
    return pd.concat(parts, axis=1)
