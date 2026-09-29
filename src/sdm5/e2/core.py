"""E-2:P2 信頼性評価、P4 暫定階層と P5 選別、P6 配置エントロピー、P7 正式な階層。

v0.2.1 修正メモ 4・9(R4, R16, R17, R26)。
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

K_B_meV = 8.617333262e-2   # meV/K
CONF_RANK = {"高": 2, "中": 1, "低": 0}
RANK_CONF = {v: k for k, v in CONF_RANK.items()}


# ---------------------------------------------------------------- P2
def p2_reliability(known: pd.DataFrame, hull_table: pd.DataFrame, recall_meV: float = 25.0,
                   high: float = 0.9, mid: float = 0.7) -> pd.DataFrame:
    """部分系ごとの再現率・誤安定率・欠落相数と信頼度(R16)。

    known:既知相の参照表(chemsys, formula, report_status)。「計算予測のみ」は正解に含めない(ルール5)。
    hull_table:Hull.table() の出力(chemsys, reduced_formula, e_hull[eV/atom])。
    """
    from ..common.composition import reduced_formula

    k = known[known["report_status"].isin(["バルクの平衡相", "準安定相"])].copy()
    k["rf"] = [reduced_formula(f) for f in k["formula"]]
    h = hull_table.copy()
    best = h.groupby(["chemsys", "reduced_formula"])["e_hull"].min().reset_index()
    rows = []
    systems = sorted(set(best["chemsys"]) | set(k["chemsys"]))
    for cs in systems:
        if "-" not in cs:
            continue
        kk = set(k.loc[k["chemsys"] == cs, "rf"])
        bb = best[best["chemsys"] == cs]
        emap = dict(zip(bb["reduced_formula"], bb["e_hull"] * 1000))
        present = [f for f in kk if f in emap]
        recalled = [f for f in present if emap[f] <= recall_meV]
        on_hull = [f for f, e in emap.items() if e <= 1e-6]
        false_stable = [f for f in on_hull if f not in kk]
        recall = len(recalled) / len(kk) if kk else np.nan
        fs_rate = len(false_stable) / len(on_hull) if on_hull else 0.0
        if kk:
            conf = "高" if recall >= high else ("中" if recall >= mid else "低")
        else:  # 既知化合物なし(Al–Si、Cu–Fe など):誤安定の有無で判定(R16c)
            conf = "高" if not false_stable else "中"
        rows.append({"chemsys": cs, "n_known": len(kk), "recall": recall, "false_stable_rate": fs_rate,
                     "n_missing": len(kk) - len(present), "missing": ";".join(sorted(kk - set(present))),
                     "false_stable": ";".join(sorted(false_stable)), "confidence": conf})
    return pd.DataFrame(rows)


def system_confidence(chem: str, sub_conf: Mapping[str, str], default: str = "低") -> str:
    """多元系の信頼度 = 構成する2元・3元部分系の信頼度の最小値(R16d)。"""
    els = chem.split("-")
    subs = ["-".join(sorted(c)) for k in (2, 3) for c in combinations(els, k) if k <= len(els)]
    if not subs:
        return "高"
    return RANK_CONF[min(CONF_RANK[sub_conf.get(s, default)] for s in subs)]


# ---------------------------------------------------------------- P4
def p4_screen(df: pd.DataFrame, col_a: str, col_b: str, in_target: str | None = "in_target",
              S_max=25.0, A_max=70.0, p5_both=70.0, p5_either=25.0, disagree=50.0,
              conf_col: str | None = "sys_confidence") -> pd.DataFrame:
    """暫定階層と P5 に進めるかの判定(値は meV/atom)。

    暫定S:両モデル ≤ S_max、暫定A:両モデル ≤ A_max、保留:2モデルの差 > disagree または部分系の信頼度が低。
    P5:両モデル ≤ p5_both(= A_max、R4)、またはどちらか ≤ p5_either かつ狙い組成領域。
    """
    out = df.copy()
    a, b = out[col_a].astype(float), out[col_b].astype(float)
    mx = np.maximum(a, b)
    tier = np.where(mx <= S_max, "暫定S", np.where(mx <= A_max, "暫定A", "不安定"))
    hold = (np.abs(a - b) > disagree)
    if conf_col and conf_col in out:
        hold |= (out[conf_col] == "低").to_numpy()
    out["e2_tier_prov"] = np.where(hold, "保留", tier)
    out["model_diff_meV"] = np.abs(a - b)
    tgt = out[in_target].astype(bool) if in_target and in_target in out else False
    out["to_p5"] = (mx <= p5_both) | ((np.minimum(a, b) <= p5_either) & tgt)
    return out


# ---------------------------------------------------------------- P6
def config_entropy_meV(T_K: float, site_occupancies: Sequence[Sequence[float]],
                       site_multiplicities: Sequence[float]) -> float:
    """原子あたりの配置エントロピーによる安定化 T·S_conf(meV/atom、正の値)。

    site_occupancies:乱雑サイトごとの占有率の組(合計1)。site_multiplicities:そのサイトの原子数の割合。
    例:2種が半分ずつ占めるサイトが全体の100% → 790 K で約 47 meV/atom。
    """
    tot = 0.0
    for occ, m in zip(site_occupancies, site_multiplicities):
        o = np.asarray(occ, float)
        o = o[o > 0]
        tot += m * float(-(o * np.log(o)).sum())
    return K_B_meV * T_K * tot


def corrected_ehull(e_hull_meV: float, T_K: float | Sequence[float], ts_conf_meV_fn=None,
                    dF_vib_meV: float = 0.0, occupancies=None, multiplicities=None) -> float:
    """E_hull − T·S_conf − ΔF_vib。T が複数(T_anneal の暫定集合、R22)なら最小値を返す。"""
    Ts = np.atleast_1d(T_K).astype(float)
    vals = []
    for T in Ts:
        ts = config_entropy_meV(T, occupancies, multiplicities) if occupancies is not None else 0.0
        vals.append(e_hull_meV - ts - dF_vib_meV)
    return float(min(vals))


# ---------------------------------------------------------------- P7
def formal_tier(e_hull_dft_meV: float | None, reported: bool, sys_conf: str, model_diff_meV: float | None = None,
                corrected_meV: float | None = None, large_solubility: bool = False,
                S_max=25.0, A_max=70.0, disagree=50.0) -> str:
    """正式な階層(S / A / B / 保留 / 既知 / 不安定)。R16a・R26 を反映。

    - 保留:部分系の信頼度が低、または2モデルの差 > disagree
    - S  :DFT の E_hull ≤ S_max、実験で未報告、信頼度が高
    - A  :DFT の E_hull ≤ S_max で信頼度が中(R16a)、または S_max < E_hull ≤ A_max かつ補正後 ≤ S_max(R26)
    - B  :既知相への大きな固溶(報告済みの相の固溶の評価を含む)
    """
    if sys_conf == "低" or (model_diff_meV is not None and model_diff_meV > disagree):
        return "保留"
    if large_solubility:
        return "B"
    if reported:
        return "既知"
    if e_hull_dft_meV is None or np.isnan(e_hull_dft_meV):
        return "不安定"
    if e_hull_dft_meV <= S_max:
        return "S" if sys_conf == "高" else "A"
    if e_hull_dft_meV <= A_max and corrected_meV is not None and corrected_meV <= S_max:
        return "A"
    return "不安定"
