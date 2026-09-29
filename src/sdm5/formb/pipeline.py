"""形B:B4 の検証と B5 の適用をまとめる。"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..common.composition import ELEMENTS
from .features import build_features, chemsys_any
from .models import PUBagging, ensemble, heuristic_score, m1_scores, percentile_within
from .points import enumerate_points
from .validate import bootstrap_diff, g2_decide, rank_metrics


def _universe(train: pd.DataFrame, max_atoms: int, elements: Sequence[str]) -> pd.DataFrame:
    """学習・評価の母集団:elements から作れる組成点すべて(正例の印つき)。"""
    pts = enumerate_points(elements, max_atoms, with_cells=False)
    pos = set(zip(train["chemsys"], train["reduced_formula"]))
    pts["y"] = [int((c, f) in pos) for c, f in zip(pts["chemsys"], pts["reduced_formula"])]
    meta = train.set_index(["chemsys", "reduced_formula"])
    if "year" in train:
        pts["year"] = [meta["year"].get((c, f), np.nan) for c, f in zip(pts["chemsys"], pts["reduced_formula"])]
    return pts


def _score(pts: pd.DataFrame, known_mask: np.ndarray, exclude: Iterable[str] = (), n_bags=30, seed=0,
           test_mask: np.ndarray | None = None):
    """known_mask の正例だけを使って学習し、M2・経験則・無作為の元素数別パーセンタイルを返す。"""
    formulas = pts["reduced_formula"].tolist()
    known = [f for f, k in zip(formulas, known_mask) if k]
    F = build_features(formulas, known, exclude_systems=exclude)
    X = F.to_numpy(float)
    y = known_mask.astype(int)
    mean, std, _, _ = PUBagging(n_bags=n_bags, seed=seed).fit_score(X, y)
    ne = pts["n_elements"].to_numpy()
    rng = np.random.default_rng(seed)
    return {"m2": percentile_within(mean, ne), "heur": percentile_within(heuristic_score(F), ne),
            "rand": percentile_within(rng.random(len(pts)), ne), "m2_std": std, "features": F}


def _pack(s: dict, test: np.ndarray, k: float, n_boot: int, seed: int) -> dict:
    return {"m2": rank_metrics(s["m2"], test), "heur": rank_metrics(s["heur"], test),
            "rand": rank_metrics(s["rand"], test),
            "boot": bootstrap_diff(s["m2"], s["heur"], test, k=k, n_boot=n_boot, seed=seed)}


def run_validation(train: pd.DataFrame, elements: Sequence[str], max_atoms: int = 20, split_year: int | None = None,
                   exclude_system: str | None = "Al-Fe-Si", n_bags: int = 30, k: float = 10.0,
                   n_boot: int = 1000, seed: int = 0):
    """B4。train は build_training_set の出力。elements は検証に使う元素の範囲。"""
    pts = _universe(train, max_atoms, elements)
    y = pts["y"].to_numpy().astype(bool)
    results = {}
    # 1) 時系列分割
    if split_year is not None and "year" in pts and pts.loc[y, "year"].notna().any():
        before = y & (pts["year"].fillna(np.inf).to_numpy() < split_year)
        test = y & ~before
        if test.any():
            s = _score(pts, before, n_bags=n_bags, seed=seed)
            # 評価は「学習時に未ラベルだった点」の中での順位
            results["time_split"] = _pack(s, test, k, n_boot, seed)
    # 2) 系の除外
    if exclude_system:
        in_sys = (pts["chemsys"] == exclude_system).to_numpy()
        test = y & in_sys
        if test.any():
            s = _score(pts, y & ~in_sys, exclude=[exclude_system], n_bags=n_bags, seed=seed)
            results["system_exclusion"] = _pack(s, test, k, n_boot, seed)
    # 3) 元素数の外挿(3元系以下で学習し、4元系の既知化合物を評価)
    ne = pts["n_elements"].to_numpy()
    test = y & (ne == 4)
    if test.any():
        s = _score(pts, y & (ne <= 3), n_bags=n_bags, seed=seed)
        results["element_extrapolation"] = _pack(s, test, k, n_boot, seed)
    return g2_decide(results), results


def apply_b5(train: pd.DataFrame, max_atoms: int = 20, n_bags: int = 50, seed: int = 0,
             model_version: str = "formB-v0", use_m1: bool = True) -> pd.DataFrame:
    """B5:最終モデルで対象5元系の全組成点をスコアづけし、組成点表の b_ 列を返す。

    学習の母集団は学習データに現れる元素全体ではなく、計算量を抑えるため対象5元系の組成点に、
    対象外の系の正例(特徴量だけ計算)を加えたものとする。
    """
    pts = enumerate_points(ELEMENTS, max_atoms)
    pos = set(zip(train["chemsys"], train["reduced_formula"]))
    pts["y"] = [int((c, f) in pos) for c, f in zip(pts["chemsys"], pts["reduced_formula"])]
    other = train[~train["target_system"]]
    known_all = train["reduced_formula"].tolist()
    f_t = pts["reduced_formula"].tolist()
    f_o = other["reduced_formula"].tolist()
    F = build_features(f_t + f_o, known_all)
    y = np.r_[pts["y"].to_numpy(), np.ones(len(f_o), int)]
    mean, std, _, _ = PUBagging(n_bags=n_bags, seed=seed).fit_score(F.to_numpy(float), y)
    n = len(pts)
    pts["score_M2"] = mean[:n]
    pts["m2_std"] = std[:n]
    cols = ["score_M2"]
    if use_m1:
        pts["score_M1"] = m1_scores(f_t, pts["y"].to_numpy().astype(bool))
        # M1 は2〜3元系のベースライン:4元系以上は M2 のみでアンサンブル
        cols.append("score_M1")
    out = ensemble(pts, cols)
    if use_m1:
        hi = out["n_elements"] >= 4
        out.loc[hi, "b_score"] = out.loc[hi, "pct_score_M2"]
        out.loc[hi, "b_rank_spread"] = 0.0
        out["b_percentile"] = percentile_within(out["b_score"], out["n_elements"])
        out["b_percentile_chemsys"] = percentile_within(out["b_score"], out["chemsys"])
    out["b_uncertainty"] = out["b_rank_spread"] + 100 * out["m2_std"]
    out["b_model_version"] = model_version
    out["label_status"] = np.where(out["y"] == 1, "正例", "未ラベル")
    return out
