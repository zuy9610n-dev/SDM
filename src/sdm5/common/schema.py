"""共通のデータ形式(上位文書 3-2、v0.2.1 修正メモ 1)。

各表の列と、書き込む計画(owner)を定義する。validate() で列の欠落と値域を確認する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .composition import VEC_COLS

E1_OK_VALUES = ("可", "不可", "要DSC確認")
SOLIDUS_FLAGS = ("正常", "非単調", "下限で液相あり", "上限まで液相なし")
CONFIDENCE = ("高", "中", "低")
T_MAX_STATUS = ("確定", "要DSC確認")
LABEL_STATUS = ("正例", "未ラベル", "負例(実験)", "新相候補")
E2_TIERS = ("S", "A", "B", "保留", "既知", "不安定")
E2_TIERS_PROV = ("暫定S", "暫定A", "保留", "不安定")
FINAL_PRIORITY = ("最優先", "優先", "条件付き", "構造探索", "保留", "低")
STRUCT_SOURCE = ("DB", "P3a", "P3b", "形B主導")
FEEDBACK_CLASS = ("一致", "新相候補", "未出現", "判定不能", "既知相")  # 既知相は6章の外の記録用


@dataclass(frozen=True)
class TableSpec:
    name: str
    key: tuple[str, ...]
    columns: dict[str, str]              # 列 -> 書き込む計画
    enums: dict[str, tuple] = field(default_factory=dict)


def _cols(owner: str, *names: str) -> dict[str, str]:
    return {n: owner for n in names}


COMPOSITION_POINTS = TableSpec(
    name="composition_points",
    key=("chemsys", "reduced_formula"),
    columns={
        **_cols("共通", "chemsys", "reduced_formula", *VEC_COLS, "n_elements", "cell_id",
                "nearest_known", "d_nearest"),
        **_cols("形B", "b_score", "b_percentile", "b_percentile_chemsys", "b_uncertainty", "b_model_version"),
        **_cols("E-2", "e2_best_struct_id", "e2_min_ehull", "e2_tier", "e2_tier_prov", "e2_match_d",
                "e2_mag_checked", "e2_source"),
        **_cols("D", "target_design_id"),
        **_cols("統合", "e1_ok", "d_coverage_ok", "stage2", "quadrant", "final_priority",
                "match_threshold_at", "integration_round"),
        **_cols("実験", "label_status", "label_T_K"),
    },
    enums={"e1_ok": E1_OK_VALUES, "d_coverage_ok": E1_OK_VALUES, "label_status": LABEL_STATUS,
           "final_priority": FINAL_PRIORITY},
)

CELLS = TableSpec(
    name="cells",
    key=("cell_id",),
    columns={
        **_cols("共通", "cell_id", "parent_cell_id", "resolution", *VEC_COLS, "n_elements", "chemsys"),
        **_cols("E-1", "e1_solidus", "e1_solidus_flag", "e1_confidence"),
    },
    enums={"e1_solidus_flag": SOLIDUS_FLAGS, "e1_confidence": CONFIDENCE},
)

CELL_PHASES = TableSpec(
    name="cell_phases",
    key=("cell_id", "T_K"),
    columns=_cols("E-1", "cell_id", "T_K", "e1_phases", "e1_single_phase"),
)

CELL_DESIGN = TableSpec(
    name="cell_design",
    key=("cell_id", "design_id"),
    columns=_cols("D", "cell_id", "design_id", "d_expected_points", "n_points_planned", "model_version"),
)

CELL_SAMPLE = TableSpec(
    name="cell_sample",
    key=("cell_id", "sample_id"),
    columns=_cols("D・実験", "cell_id", "sample_id", "surrounded", "n_near"),
)

STRUCTURES = TableSpec(
    name="structures",
    key=("struct_id",),
    columns={
        **_cols("E-2", "struct_id", "chemsys", "reduced_formula", *VEC_COLS, "source", "stage", "method",
                "energy_per_atom", "magnetic_state", "calc_hash", "cif_path"),
        # E_hull は ehull__<method> 列を動的に追加する(修正メモ2)
    },
    enums={"source": STRUCT_SOURCE},
)

DESIGNS = TableSpec(
    name="designs",
    key=("design_id",),
    columns={
        **_cols("D", "design_id", "powders", "particle_size_um", "volume_fractions", "T_anneal_K",
                "t_anneal_h", "relative_density", "model_level", "model_version",
                "coverage_2", "coverage_3", "coverage_4", "coverage_5", "coverage_target",
                "required_points", "n_points_planned", "sample_id"),
        **_cols("E-1", "e1_T_max", "e1_T_max_status", "e1_T_max_cell", "e1_T_ad_max"),
    },
    enums={"e1_T_max_status": T_MAX_STATUS},
)

OBSERVATIONS = TableSpec(
    name="observations",
    key=("sample_id", "obs_id"),
    columns=_cols("実験", "sample_id", "obs_id", "T_anneal_K", "t_anneal_h", *VEC_COLS, "spread_at",
                  "n_points", "phase_id", "id_basis", "matched_point", "cell_id", "surrounded",
                  "time_varied_checked", "xrd_ebsd_confirmed", "feedback_class"),
    enums={"feedback_class": FEEDBACK_CLASS},
)

DB_EVAL = TableSpec(
    name="db_evaluation",
    key=("subsystem",),
    columns=_cols("E-1", "subsystem", "assessed", "source", "ternary_compounds", "missing_known_phases",
                  "validation_error_K", "confidence"),
    enums={"confidence": CONFIDENCE},
)

KNOWN_PHASES = TableSpec(
    name="known_phases",
    key=("phase_name", "chemsys"),
    columns=_cols("E-2(P0)", "phase_name", "chemsys", "formula", "comp_range", "space_group", "prototype",
                  "T_range", "source", "report_status"),
    enums={"report_status": ("バルクの平衡相", "準安定相", "薄膜のみ", "計算予測のみ", "要確認")},
)

ALL_TABLES = {t.name: t for t in (COMPOSITION_POINTS, CELLS, CELL_PHASES, CELL_DESIGN, CELL_SAMPLE,
                                  STRUCTURES, DESIGNS, OBSERVATIONS, DB_EVAL, KNOWN_PHASES)}


class SchemaError(ValueError):
    pass


def empty(spec: TableSpec) -> pd.DataFrame:
    return pd.DataFrame(columns=list(spec.columns))


def conform(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """不足列を NA で補い、定義順に並べる(ehull__* などの追加列は後ろに残す)。"""
    out = df.copy()
    for c in spec.columns:
        if c not in out.columns:
            out[c] = pd.NA
    extra = [c for c in out.columns if c not in spec.columns]
    return out[list(spec.columns) + extra]


def validate(df: pd.DataFrame, spec: TableSpec, require_all: bool = False) -> None:
    missing = [c for c in spec.key if c not in df.columns]
    if require_all:
        missing += [c for c in spec.columns if c not in df.columns and c not in missing]
    if missing:
        raise SchemaError(f"{spec.name}: 列がありません: {missing}")
    if df.duplicated(list(spec.key)).any():
        dup = df[df.duplicated(list(spec.key), keep=False)][list(spec.key)].head()
        raise SchemaError(f"{spec.name}: キーが重複しています:\n{dup}")
    for col, allowed in spec.enums.items():
        if col in df.columns:
            bad = df[col].dropna()
            bad = bad[~bad.isin(allowed)]
            if len(bad):
                raise SchemaError(f"{spec.name}.{col}: 許されない値 {sorted(set(bad))[:5]}(許可: {allowed})")
