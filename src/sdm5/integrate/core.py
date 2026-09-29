"""統合(上位文書 4章)と実験結果の戻し方(6章)。v0.2.1 修正メモ 10(R18〜R21、R29)。"""
from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from ..common.composition import VEC_COLS
from ..d.metrics import surrounded as _surrounded
from ..e1.core import e1_ok as _e1_ok

Round = Literal["暫定", "本"]
STAGE2_ORDER = ("不可", "要DSC確認", "可")


# ---------------------------------------------------------------- 照合(E-2 → 組成点)
def attach_e2(points: pd.DataFrame, e2: pd.DataFrame, threshold_at: float = 3.0) -> pd.DataFrame:
    """E-2 の候補を組成点に結合する。

    e2 の列:struct_id, chemsys, reduced_formula, x_*, e_hull_meV, e2_tier, e2_tier_prov, [mag_checked], [source]
    1) (chemsys, reduced_formula) が一致すれば結合(距離 0)
    2) 一致しない候補は、同じ chemsys の組成点のうち d ≤ threshold の最も近いものに結合
    組成点ごとに E_hull が最小の構造を代表にする。
    """
    pts = points.copy()
    pts["match_threshold_at"] = threshold_at
    key = {(c, f): i for i, (c, f) in enumerate(zip(pts["chemsys"], pts["reduced_formula"]))}
    by_sys = {cs: g for cs, g in pts.groupby("chemsys")}
    rows = []
    for r in e2.itertuples(index=False):
        i = key.get((r.chemsys, r.reduced_formula))
        d = 0.0
        if i is None:
            g = by_sys.get(r.chemsys)
            if g is None:
                continue
            x = np.array([getattr(r, c) for c in VEC_COLS], float)
            dd = np.linalg.norm(g[VEC_COLS].to_numpy(float) - x, axis=1)
            j = int(np.argmin(dd))
            if dd[j] > threshold_at:
                continue
            i, d = pts.index.get_loc(g.index[j]), float(dd[j])
        rows.append({"_row": i, "e2_best_struct_id": r.struct_id, "e2_min_ehull": float(r.e_hull_meV),
                     "e2_tier": getattr(r, "e2_tier", None), "e2_tier_prov": getattr(r, "e2_tier_prov", None),
                     "e2_match_d": d, "e2_mag_checked": getattr(r, "mag_checked", None),
                     "e2_source": getattr(r, "source", None)})
    for c in ("e2_best_struct_id", "e2_min_ehull", "e2_tier", "e2_tier_prov", "e2_match_d", "e2_mag_checked",
              "e2_source"):
        pts[c] = pd.Series([None] * len(pts), index=pts.index, dtype=object)
    if rows:
        m = pd.DataFrame(rows).sort_values("e2_min_ehull", kind="stable").drop_duplicates("_row")
        for rec in m.to_dict("records"):
            idx = pts.index[rec.pop("_row")]
            for c, v in rec.items():
                pts.at[idx, c] = v
    return pts


# ---------------------------------------------------------------- 段階1
def quadrant(row: Mapping, rnd: Round, g2_passed: bool, high_pct: float = 90.0) -> str:
    """①〜④、または E-2 で保留(R18)、E2単独(R21)。"""
    tier = row.get("e2_tier_prov") if rnd == "暫定" else row.get("e2_tier")
    stable_set = ("暫定S", "暫定A") if rnd == "暫定" else ("S", "A")
    if tier == "保留":
        return "E2保留"
    stable = tier in stable_set
    if not g2_passed:
        return "E2単独" if stable else "④"
    pct = row.get("b_percentile")
    high = pct is not None and not pd.isna(pct) and float(pct) >= high_pct
    if stable and high:
        return "①"
    if stable:
        return "②"
    if high:
        return "③"
    return "④"


# ---------------------------------------------------------------- 段階2
def combine_stage2(*vals: str | None) -> str:
    """不可 > 要DSC確認 > 可(R20)。None(未評価)は無視し、すべて None なら「未評価」。"""
    vs = [v for v in vals if v is not None]
    if not vs:
        return "未評価"
    for v in STAGE2_ORDER:
        if v in vs:
            return v
    return "未評価"


def d_coverage_ok(lam: float | None, lambda_min: float = 5.0) -> str | None:
    if lam is None or pd.isna(lam):
        return "不可"
    return "可" if float(lam) >= lambda_min else "不可"


def stage2_for_points(pts: pd.DataFrame, cells: pd.DataFrame, designs: pd.DataFrame, cell_design: pd.DataFrame,
                      dT_safe: float, lambda_min: float = 5.0) -> pd.DataFrame:
    """target_design_id のある組成点について e1_ok・d_coverage_ok・stage2 を計算する(5 at% 区画の λ、R14)。"""
    out = pts.copy()
    cmap = cells.set_index("cell_id")
    dmap = designs.set_index("design_id")
    lam = cell_design.set_index(["cell_id", "design_id"])["d_expected_points"]
    e1s, dcs, st, why = [], [], [], []
    for r in out.itertuples(index=False):
        did = getattr(r, "target_design_id", None)
        if did is None or pd.isna(did) or did not in dmap.index or r.cell_id not in cmap.index:
            e1s.append(None); dcs.append(None); st.append("未評価"); why.append("狙う試料設計なし")
            continue
        dz = dmap.loc[did]
        v, reason = _e1_ok(cmap.loc[r.cell_id].to_dict(), float(dz["T_anneal_K"]), float(dz["e1_T_max"]),
                           str(dz["e1_T_max_status"]), dT_safe)
        dc = d_coverage_ok(lam.get((r.cell_id, did)), lambda_min)
        e1s.append(v); dcs.append(dc); st.append(combine_stage2(v, dc)); why.append(reason)
    out["e1_ok"], out["d_coverage_ok"], out["stage2"], out["stage2_reason"] = e1s, dcs, st, why
    return out


# ---------------------------------------------------------------- 段階3
def final_priority(row: Mapping, rnd: Round, sys_conf_low: bool = False) -> str:
    q = row.get("quadrant")
    s2 = row.get("stage2", "未評価")
    if q == "E2保留":
        return "保留"
    if q == "③":
        return "構造探索"
    if q == "④":
        return "低"
    # ① ② E2単独
    if s2 == "不可" or (rnd == "本" and sys_conf_low):
        return "保留"
    if s2 in ("要DSC確認", "未評価"):
        return "条件付き" if s2 == "要DSC確認" else "保留"
    # s2 == 可
    if rnd == "暫定":
        return "優先" if q in ("①", "E2単独") else "保留"      # ②は本統合で判定
    if q in ("①", "E2単独"):
        return "最優先"
    # ②:Fe を含まないものは磁気確認不要(R19)
    has_fe = "Fe" in str(row.get("chemsys", "")).split("-")
    mag = row.get("e2_mag_checked")
    return "優先" if (not has_fe or mag is True) else "保留"


def integrate(pts: pd.DataFrame, rnd: Round, g2_passed: bool, high_pct: float = 90.0,
              sys_conf: Mapping[str, str] | None = None) -> pd.DataFrame:
    out = pts.copy()
    out["integration_round"] = rnd
    out["quadrant"] = [quadrant(r, rnd, g2_passed, high_pct) for r in out.to_dict("records")]
    lows = [bool(sys_conf and sys_conf.get(c) == "低") for c in out["chemsys"]]
    out["final_priority"] = [final_priority(r, rnd, lo) for r, lo in zip(out.to_dict("records"), lows)]
    return out


def to_p3b(integrated: pd.DataFrame) -> pd.DataFrame:
    """③ を E-2 の P3b へ(生成方法は「形B主導」と記録する、ルール2)。"""
    t = integrated[integrated["quadrant"] == "③"][["chemsys", "reduced_formula", *VEC_COLS, "b_percentile"]].copy()
    t["source"] = "形B主導"
    return t.sort_values("b_percentile", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- 6章 実験結果の戻し方
def classify_observation(obs: Mapping, candidates: pd.DataFrame, known: pd.DataFrame | None, points_at: np.ndarray,
                         threshold_at: float = 3.0, R: float = 10.0, M: int = 50) -> dict:
    """観測された点の集まり(または、予測した候補の位置)を 一致/新相候補/未出現/判定不能 に分類する。

    obs:{"kind": "cluster" | "candidate", x_*, "xrd_ebsd_consistent": bool, "xrd_ebsd_confirmed": bool,
         "time_varied_checked": bool}
    candidates:予測した候補(x_* 列)。known:既知相(x_* 列)。points_at:その試料の測定点(at%)。
    """
    x = np.array([obs[c] for c in VEC_COLS], float)

    def nearest(df):
        if df is None or df.empty:
            return None, np.inf
        d = np.linalg.norm(df[VEC_COLS].to_numpy(float) - x, axis=1)
        j = int(np.argmin(d))
        return df.index[j], float(d[j])

    if obs.get("kind", "cluster") == "cluster":
        ci, cd = nearest(candidates)
        ki, kd = nearest(known)
        if cd <= threshold_at and obs.get("xrd_ebsd_consistent", False):
            return {"feedback_class": "一致", "matched": ci, "d": cd}
        if cd > threshold_at and kd > threshold_at:
            if obs.get("xrd_ebsd_confirmed", False):
                return {"feedback_class": "新相候補", "matched": None, "d": min(cd, kd)}
            return {"feedback_class": "判定不能", "matched": None, "d": min(cd, kd),
                    "reason": "構造確認待ち(R29)"}
        if kd <= threshold_at:
            return {"feedback_class": "既知相", "matched": ki, "d": kd}
        return {"feedback_class": "判定不能", "matched": ci, "d": cd, "reason": "XRD/EBSD と矛盾または未確認"}
    # 予測した候補の位置に相が現れなかった
    ok, nn = _surrounded(x, points_at, R, M)
    if ok and obs.get("time_varied_checked", False):
        return {"feedback_class": "未出現", "n_near": nn}
    return {"feedback_class": "判定不能", "n_near": nn,
            "reason": "包囲判定を満たさない" if not ok else "焼鈍時間を変えた確認がまだ"}
