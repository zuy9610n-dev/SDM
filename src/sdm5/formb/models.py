"""形B B3:モデル(M1 推薦・M2 PU バギング・経験則)と順位の付け方。

- パーセンタイルは元素数ごとの集団の中で計算する(v0.2.1 修正メモ 3)。部分系ごとは補助の列。
- アンサンブル:各モデルの元素数別パーセンタイルを平均し、モデル間の差(最大−最小)を不確かさとする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import element_props, parse_any


# ---------------------------------------------------------------- 順位
def percentile_within(scores: np.ndarray, groups: Sequence) -> np.ndarray:
    """グループ内のパーセンタイル順位(0〜100、大きいほど上位、同点は平均順位)。"""
    s = pd.Series(np.asarray(scores, float))
    g = pd.Series(list(groups))
    return (s.groupby(g).rank(pct=True, method="average") * 100).to_numpy()


def add_percentiles(df: pd.DataFrame, score_col: str, prefix: str = "b") -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_percentile"] = percentile_within(out[score_col], out["n_elements"])
    out[f"{prefix}_percentile_chemsys"] = percentile_within(out[score_col], out["chemsys"])
    return out


# ---------------------------------------------------------------- 経験則(比較の基準)
def heuristic_score(feats: pd.DataFrame) -> np.ndarray:
    """「部分系に既知化合物が多い組成ほど上位」。同数は近い既知化合物がある方を上に。"""
    return feats["sub_known"].to_numpy(float) - 1e-3 * feats["d_near_known"].to_numpy(float)


# ---------------------------------------------------------------- M2
@dataclass
class PUBagging:
    """PU バギング(Mordelet & Vert 2014)。未ラベルから正例と同数を仮の負例として抜き、多数回学習して平均する。

    各回で選ばれなかった未ラベル(out-of-bag)のスコアだけを平均する(未ラベルの過学習を避ける)。
    LightGBM があれば使い、なければ scikit-learn の HistGradientBoosting を使う。
    """
    n_bags: int = 50
    neg_ratio: float = 1.0
    seed: int = 0
    params: dict = field(default_factory=lambda: {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 15})

    def _model(self, rs):
        try:
            import lightgbm as lgb  # noqa: F401
            return lgb.LGBMClassifier(n_estimators=self.params["max_iter"], learning_rate=self.params["learning_rate"],
                                      num_leaves=self.params["max_leaf_nodes"], random_state=rs, verbose=-1)
        except ImportError:
            return HistGradientBoostingClassifier(random_state=rs, **self.params)

    def fit_score(self, X: np.ndarray, y: np.ndarray, X_new: np.ndarray | None = None):
        """(学習集合のスコア, そのばらつき, X_new のスコア, そのばらつき) を返す。"""
        rng = np.random.default_rng(self.seed)
        P, U = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(U), max(1, int(round(self.neg_ratio * len(P)))))
        s_sum = np.zeros(len(X)); s_sq = np.zeros(len(X)); cnt = np.zeros(len(X))
        new = [] if X_new is not None else None
        for b in range(self.n_bags):
            neg = rng.choice(U, n_neg, replace=False)
            idx = np.concatenate([P, neg])
            m = self._model(int(rng.integers(1 << 31)))
            m.fit(X[idx], np.r_[np.ones(len(P)), np.zeros(len(neg))])
            oob = np.ones(len(X), bool)
            oob[neg] = False
            p = m.predict_proba(X)[:, 1]
            s_sum[oob] += p[oob]; s_sq[oob] += p[oob] ** 2; cnt[oob] += 1
            if new is not None:
                new.append(m.predict_proba(X_new)[:, 1])
        cnt[cnt == 0] = 1
        mean = s_sum / cnt
        std = np.sqrt(np.clip(s_sq / cnt - mean ** 2, 0, None))
        if new is None:
            return mean, std, None, None
        a = np.stack(new)
        return mean, std, a.mean(0), a.std(0)


# ---------------------------------------------------------------- M1
def _pattern(formula: str) -> tuple[str, tuple[int, ...]]:
    """元素を原子番号順に並べたときの整数比の型(元素を問わない列の軸)。"""
    p = parse_any(formula)
    els = sorted(p, key=lambda e: element_props(e)[6])
    return "-".join(sorted(p)), tuple(int(round(p[e])) for e in els)


def m1_scores(formulas: Sequence[str], positive: np.ndarray, rank: int = 4, n_iter: int = 30) -> np.ndarray:
    """推薦モデル(Seko ら 2018 の考え方の簡易版):系 × 整数比の型 の行列を、元素数ごとに低ランク近似する。

    既知の位置を 1 にし、欠測を反復 SVD で埋める(soft-impute)。5元系では非常に疎なので参考値。
    """
    keys = [_pattern(f) for f in formulas]
    k_el = np.array([len(k[1]) for k in keys])
    out = np.zeros(len(formulas))
    for k in np.unique(k_el):
        ids = np.where(k_el == k)[0]
        systems = sorted({keys[i][0] for i in ids})
        pats = sorted({keys[i][1] for i in ids})
        si = {s: j for j, s in enumerate(systems)}
        pi = {p: j for j, p in enumerate(pats)}
        Y = np.zeros((len(systems), len(pats)))
        for i in ids:
            if positive[i]:
                Y[si[keys[i][0]], pi[keys[i][1]]] = 1.0
        r = max(1, min(rank, min(Y.shape) - 1)) if min(Y.shape) > 1 else 1
        Z = Y.copy()
        mask = Y > 0
        for _ in range(n_iter):
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            L = (U[:, :r] * s[:r]) @ Vt[:r]
            Z = np.where(mask, 1.0, L)
        for i in ids:
            out[i] = L[si[keys[i][0]], pi[keys[i][1]]]
    return out


# ---------------------------------------------------------------- アンサンブル
def ensemble(df: pd.DataFrame, score_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    pcs = []
    for c in score_cols:
        pc = percentile_within(out[c], out["n_elements"])
        out[f"pct_{c}"] = pc
        pcs.append(pc)
    P = np.stack(pcs)
    out["b_score"] = P.mean(0)
    out["b_rank_spread"] = P.max(0) - P.min(0)
    return add_percentiles(out, "b_score")
