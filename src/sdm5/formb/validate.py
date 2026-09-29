"""形B B4:検証(G2)。時系列分割・系の除外・元素数の外挿、recall@k%、順位の中央値、ブートストラップ(R23)。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


def rank_metrics(pct: np.ndarray, is_test: np.ndarray, ks=(1, 5, 10)) -> dict:
    """pct:元素数別パーセンタイル(大きいほど上位)。is_test:評価対象(後で報告された既知化合物)。

    上位 k% = パーセンタイル ≥ 100 − k。
    """
    t = np.asarray(pct)[np.asarray(is_test, bool)]
    if len(t) == 0:
        return {**{f"recall@{k}%": np.nan for k in ks}, "median_top_pct": np.nan, "n_test": 0}
    top = 100 - t
    return {**{f"recall@{k}%": float((top <= k).mean()) for k in ks},
            "median_top_pct": float(np.median(top)), "n_test": int(len(t))}


def bootstrap_diff(pct_a: np.ndarray, pct_b: np.ndarray, is_test: np.ndarray, k: float = 10.0,
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """recall@k% の差(a − b)の 95% 信頼区間(評価対象を復元抽出)。"""
    rng = np.random.default_rng(seed)
    a = (100 - np.asarray(pct_a)[is_test]) <= k
    b = (100 - np.asarray(pct_b)[is_test]) <= k
    n = len(a)
    if n == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    idx = rng.integers(0, n, (n_boot, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "n": int(n)}


@dataclass
class G2Result:
    passed: bool
    table: pd.DataFrame
    note: str


def g2_decide(results: dict[str, dict]) -> G2Result:
    """G2:時系列分割と元素数の外挿の両方で、M2 − 経験則 の recall@10% 差の 95% CI 下限 > 0。

    results:{検証名: {"m2": metrics, "heur": metrics, "rand": metrics, "boot": bootstrap_diff}}。
    時系列分割ができない場合(報告年なし)は、元素数の外挿と系の除外で判定し、その旨を記録する。
    """
    rows = []
    for name, r in results.items():
        rows.append({"validation": name, **{f"m2_{k}": v for k, v in r["m2"].items()},
                     **{f"heur_{k}": v for k, v in r["heur"].items()},
                     **{f"rand_{k}": v for k, v in r.get("rand", {}).items()},
                     "diff_recall10": r["boot"]["diff"], "ci_lo": r["boot"]["lo"], "ci_hi": r["boot"]["hi"]})
    t = pd.DataFrame(rows)
    required = [n for n in ("time_split", "element_extrapolation") if n in results]
    note = ""
    if "time_split" not in results:
        required = [n for n in ("element_extrapolation", "system_exclusion") if n in results]
        note = "報告年がないため、時系列分割の代わりに系の除外で判定"
    ok = bool(required) and all(results[n]["boot"]["lo"] > 0 for n in required)
    return G2Result(ok, t, note)
