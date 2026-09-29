"""E-1:到達可能集合 S、T_max、信頼度、e1_ok、GE1、C3 の単相マスク、C5 の T_ad。

上位文書 4章 段階2、E-1 3章・5章・8章、v0.2.1 修正メモ 6・7・11。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..common.composition import ELEMENTS, VEC_COLS, chemsys, normalize

R_GAS = 8.314462618  # J/mol/K
CONF_RANK = {"高": 2, "中": 1, "低": 0}


# ---------------------------------------------------------------- C1: ΔT_safe
def dT_safe_from_errors(errors_K: Iterable[float], minimum: float = 20.0) -> float:
    """最優先の部分系での誤差の最大値の2倍、ただし minimum 以上(E-1 C1)。"""
    errs = [abs(float(e)) for e in errors_K]
    return max(2.0 * max(errs), minimum) if errs else float("nan")


# ---------------------------------------------------------------- 3-1: 到達可能集合 S
def distance_to_hull(points: np.ndarray, vertices: np.ndarray, w_eq: float = 1e3) -> np.ndarray:
    """各点から、頂点(原料組成)の凸包までのユークリッド距離(at%)。

    min ||p - V^T w|| s.t. w >= 0, Σw = 1 を、Σw=1 を重みつきの行として NNLS で解く。
    """
    V = np.asarray(vertices, float)
    P = np.atleast_2d(np.asarray(points, float))
    A = np.vstack([V.T, w_eq * np.ones(len(V))])
    out = np.empty(len(P))
    for i, p in enumerate(P):
        w, _ = nnls(A, np.append(p, w_eq))
        out[i] = np.linalg.norm(p - V.T @ w)
    return out


def reachable_mask(cells: pd.DataFrame, powder_compositions: Sequence[Sequence[float]],
                   delta_S: float = 5.0) -> np.ndarray:
    """区画の中心が S(凸包を δ 広げたもの)に入るか。"""
    V = normalize(np.asarray(powder_compositions, float))
    return distance_to_hull(cells[VEC_COLS].to_numpy(float), V) <= delta_S + 1e-9


# ---------------------------------------------------------------- 3-3: 信頼度
def subsystems_of(chem: str, orders=(2, 3)) -> list[str]:
    els = chem.split("-")
    return ["-".join(sorted(c)) for k in orders for c in combinations(els, k) if k <= len(els)]


def assign_confidence(cells: pd.DataFrame, db_eval: pd.DataFrame, default: str = "低") -> pd.Series:
    """区画の信頼度 = 区画に含まれる2元・3元部分系の信頼度の最小値。

    データベース評価表にない部分系は default(低)とみなす。単体の区画は「高」。
    """
    conf = dict(zip(db_eval["subsystem"], db_eval["confidence"]))
    out = []
    for chem in cells["chemsys"]:
        subs = subsystems_of(chem)
        if not subs:
            out.append("高")
            continue
        ranks = [CONF_RANK[conf.get(s, default)] for s in subs]
        out.append({v: k for k, v in CONF_RANK.items()}[min(ranks)])
    return pd.Series(out, index=cells.index, name="e1_confidence")


def apply_ge2(db_eval: pd.DataFrame, ternary_ok: Mapping[str, bool]) -> pd.DataFrame:
    """GE2:再現が不十分な3元系の信頼度を低にする。"""
    out = db_eval.copy()
    for sub, ok in ternary_ok.items():
        if not ok:
            out.loc[out["subsystem"] == sub, "confidence"] = "低"
    return out


# ---------------------------------------------------------------- 3-1: T_max
@dataclass
class TmaxResult:
    T_max: float
    status: str               # 確定 / 要DSC確認
    limiting_cell: str
    min_solidus: float
    n_cells_in_S: int
    reasons: list[str]


def t_max_for_design(cells: pd.DataFrame, powder_compositions, dT_safe: float, delta_S: float = 5.0,
                     dT_low_conf: float = 50.0) -> TmaxResult:
    """T_max = min{S の中の e1_solidus} − ΔT_safe と、その状態(R11)。"""
    if cells["e1_solidus"].isna().any():
        raise ValueError("e1_solidus が未計算の区画があります")
    m = reachable_mask(cells, powder_compositions, delta_S)
    sub = cells[m]
    if sub.empty:
        raise ValueError("S に入る区画がありません")
    i = sub["e1_solidus"].idxmin()
    smin = float(sub.at[i, "e1_solidus"])
    reasons = []
    if sub.at[i, "e1_confidence"] == "低":
        reasons.append("最小値をとる区画の信頼度が低")
    near_low = sub[(sub["e1_confidence"] == "低") & (sub["e1_solidus"] <= smin + dT_low_conf)]
    if len(near_low) and not reasons:
        reasons.append(f"信頼度が低い区画が最小値 +{dT_low_conf:g} K 以内に {len(near_low)} 個")
    if sub.at[i, "e1_solidus_flag"] in ("非単調", "下限で液相あり"):
        reasons.append(f"最小値をとる区画の固相線が「{sub.at[i, 'e1_solidus_flag']}」")
    return TmaxResult(T_max=smin - dT_safe, status="要DSC確認" if reasons else "確定",
                      limiting_cell=str(sub.at[i, "cell_id"]), min_solidus=smin,
                      n_cells_in_S=int(m.sum()), reasons=reasons)


# ---------------------------------------------------------------- 4章 段階2: e1_ok
def e1_ok(cell_row: Mapping, T_anneal: float, e1_T_max: float, e1_T_max_status: str,
          dT_safe: float) -> tuple[str, str]:
    """(値, 理由) を返す。値は 可 / 不可 / 要DSC確認(上位文書 4章、修正メモ 6)。"""
    if T_anneal > e1_T_max + 1e-9:
        return "不可", "T_anneal が設計の e1_T_max を超える"
    conf = cell_row.get("e1_confidence")
    if conf == "低" or conf is None or (isinstance(conf, float) and np.isnan(conf)):
        return "要DSC確認", "区画の信頼度が低"
    if e1_T_max_status == "要DSC確認":
        return "要DSC確認", "設計の e1_T_max_status が要DSC確認"
    flag = cell_row.get("e1_solidus_flag")
    if flag == "下限で液相あり":
        return "不可", "区画が計算下限で液相あり"
    if float(cell_row["e1_solidus"]) < T_anneal + dT_safe:
        return "不可", "e1_solidus < T_anneal + ΔT_safe"
    if flag == "非単調":
        return "要DSC確認", "区画の固相線が非単調"
    return "可", ""


# ---------------------------------------------------------------- C4a / GE1
def diffusion_length(D_m2s: float, t_h: float) -> float:
    """L = 2√(Dt)(µm)。修正メモ7。"""
    return 2.0 * np.sqrt(D_m2s * t_h * 3600.0) * 1e6


def time_required_h(L_um: float, D_m2s: float) -> float:
    return (L_um * 1e-6) ** 2 / (4.0 * D_m2s) / 3600.0


def arrhenius(D0: float, Q_kJ: float, T: float) -> float:
    return D0 * np.exp(-Q_kJ * 1e3 / (R_GAS * T))


@dataclass
class GE1Result:
    verdict: str              # 可 / 不確定 / 不可
    limiting: str
    t_req_h_fast: float
    t_req_h_slow: float
    detail: pd.DataFrame


def ge1(L_required_um: float, D_ranges: pd.DataFrame, t_limit_h: float = 300.0) -> GE1Result:
    """GE1:必要な拡散距離に、t_limit_h 以内で届くか。

    D_ranges:列 species, D_lo, D_hi(m²/s、T_max での値)。最も遅い種で判定する。
    可   … 遅い側の値(D_lo)でも t_req ≤ t_limit
    不可 … 速い側の値(D_hi)でも t_req > t_limit
    不確定 … その間
    """
    d = D_ranges.copy()
    d["t_req_h_fast"] = [time_required_h(L_required_um, v) for v in d["D_hi"]]
    d["t_req_h_slow"] = [time_required_h(L_required_um, v) for v in d["D_lo"]]
    worst = d.loc[d["t_req_h_fast"].idxmax()]
    worst_slow = d["t_req_h_slow"].max()
    if worst_slow <= t_limit_h:
        verdict = "可"
    elif worst["t_req_h_fast"] > t_limit_h:
        verdict = "不可"
    else:
        verdict = "不確定"
    return GE1Result(verdict, str(worst["species"]), float(worst["t_req_h_fast"]), float(worst_slow), d)


# ---------------------------------------------------------------- C3: 平衡相と単相マスク
def phase_table(backend, cells: pd.DataFrame, temps: Iterable[float]) -> pd.DataFrame:
    """cell_phases 表(区画×温度)を作る(修正メモ1)。"""
    rows = []
    for T in temps:
        for _, r in cells.iterrows():
            x = {e.upper(): float(r[c]) for e, c in zip(ELEMENTS, VEC_COLS) if r[c] > 0}
            ph = backend.stable_phases(x, float(T))
            rows.append({"cell_id": r["cell_id"], "T_K": float(T),
                         "e1_phases": ";".join(f"{p}:{f:.3f}" for p, f in ph),
                         "e1_single_phase": len(ph) == 1})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C5: 断熱到達温度
def adiabatic_temperature(dH_kJ_per_mol_atom: float, T0: float = 298.15,
                          cp_J_per_mol_atom_K: float = 3 * R_GAS) -> float:
    """T_ad = T0 − ΔH / Cp。Cp は既定で Dulong–Petit(3R)。融解熱を無視するため高めに出る(安全側、R28)。"""
    if dH_kJ_per_mol_atom >= 0:
        return T0
    return T0 - dH_kJ_per_mol_atom * 1e3 / cp_J_per_mol_atom_K


def t_ad_for_powders(powders: Sequence[str], dH_table: pd.DataFrame, T0: float = 298.15) -> pd.DataFrame:
    """原料の2つの組み合わせごとに、両者の元素だけでできる生成物のうち最も発熱の大きいものの T_ad。

    powders:原料の化学式(純元素は 'Al' など)。dH_table:列 formula, dH_kJ_per_mol_atom(生成物の生成エンタルピー)。
    反応熱は安全側に ΔH_rxn ≈ ΔH_f(生成物) − max(ΔH_f(a), ΔH_f(b)) とする(純元素は 0)。
    """
    from ..common.composition import parse_formula

    dh = dict(zip(dH_table["formula"], dH_table["dH_kJ_per_mol_atom"]))
    rows = []
    for a, b in combinations(powders, 2):
        els = set(parse_formula(a)) | set(parse_formula(b))
        h_src = max(dh.get(a, 0.0), dh.get(b, 0.0))
        best, best_f = 0.0, None
        for f, h in dh.items():
            if f in (a, b) or not set(parse_formula(f)) <= els or len(set(parse_formula(f))) < 2:
                continue
            gain = h - h_src
            if gain < best:
                best, best_f = gain, f
        rows.append({"pair": f"{a}+{b}", "product": best_f, "dH_kJ_per_mol_atom": best,
                     "T_ad_K": adiabatic_temperature(best, T0)})
    return pd.DataFrame(rows).sort_values("T_ad_K", ascending=False, ignore_index=True)
