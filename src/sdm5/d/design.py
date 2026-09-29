"""D 7章:試料設計の評価と探索(ラテン超方格でふるい分け → レベル1a で評価)。

ベイズ最適化は任意(scikit-optimize などを後から差し替えられるよう、評価関数を分離している)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..common.composition import ELEMENTS
from .level0 import f_k
from .level1 import diffuse, eds_multi, initial_field
from .metrics import cell_design_table, coverage, expected_points, required_points
from .packing import Powder, TooManyParticlesError, pack


@dataclass
class Design:
    design_id: str
    powders: list[Powder]
    T_anneal_K: float
    t_anneal_h: float
    relative_density: float = 1.0
    n_points_planned: int = 10000

    def to_row(self) -> dict:
        return {
            "design_id": self.design_id,
            "powders": json.dumps([{"name": p.name, "comp": p.composition} for p in self.powders], ensure_ascii=False),
            "particle_size_um": json.dumps({p.name: p.d50_um for p in self.powders}, ensure_ascii=False),
            "volume_fractions": json.dumps({p.name: p.volume_fraction for p in self.powders}, ensure_ascii=False),
            "T_anneal_K": self.T_anneal_K, "t_anneal_h": self.t_anneal_h,
            "relative_density": self.relative_density, "n_points_planned": self.n_points_planned,
        }


DFn = Callable[[float], Sequence[float]]   # T -> 元素ごとの実効拡散係数(m²/s)


def evaluate_level1a(design: Design, D_of_T: DFn, n_vox: int = 64, n_sections: int = 3, seed: int = 0,
                     target_cells: Sequence[str] | None = None, lambda_min: float = 5.0,
                     eds: dict | None = None, model_version: str = "L1a-uncal") -> tuple[dict, pd.DataFrame, np.ndarray]:
    """1条件を評価し、(試料設計表の行, cell_design 表, 模擬 EDS 点) を返す。"""
    pk = pack(design.powders, seed=seed)
    f0 = initial_field(pk, n_vox)
    f1 = diffuse(f0, D_of_T(design.T_anneal_K), design.t_anneal_h)
    pts = eds_multi(f1, n_sections=n_sections, seed=seed, **(eds or {}))
    lam = expected_points(pts, design.n_points_planned)
    row = design.to_row()
    row.update(coverage(lam.to_dict(), n_min=lambda_min, target_cells=target_cells))
    row.update({"model_level": "1a", "model_version": model_version,
                "required_points": required_points(pts, target_cells, lambda_min) if target_cells else np.nan,
                "voxel_um": f0.h_um, "box_um": pk.box_um})
    cd = cell_design_table(pts, design.design_id, design.n_points_planned, model_version)
    return row, cd, pts


def lhs_screen_level0(base_powders: Sequence[Powder], n: int, L_bounds_um=(2.0, 30.0),
                      d_bounds_um=(5.0, 150.0), seed: int = 0, max_particles: int = 20_000) -> pd.DataFrame:
    """レベル0のふるい分け:粒径(粉末ごと)と拡散距離 L をラテン超方格で振り、f_5 を返す。

    粒子数が多すぎる条件は計算せず skipped=True とする(D 5-1:粗い計算か断面計算で扱う)。
    """
    k = len(base_powders)
    sampler = qmc.LatinHypercube(d=k + 1, seed=seed)
    u = sampler.random(n)
    lo = np.log([d_bounds_um[0]] * k + [L_bounds_um[0]])
    hi = np.log([d_bounds_um[1]] * k + [L_bounds_um[1]])
    x = np.exp(qmc.scale(u, lo, hi))
    rows = []
    for i, xi in enumerate(x):
        pw = [Powder(p.name, p.composition, float(xi[j]), p.sigma_ln, p.volume_fraction)
              for j, p in enumerate(base_powders)]
        try:
            pk = pack(pw, seed=seed + i, max_particles=max_particles)
        except TooManyParticlesError:
            rows.append({**{f"d_{p.name}": float(xi[j]) for j, p in enumerate(base_powders)},
                         "L_um": float(xi[-1]), **{f"f_{kk}": np.nan for kk in range(1, 6)}, "skipped": True})
            continue
        fk = f_k(pk, float(xi[-1]), spacing_um=max(1.0, pk.box_um / 60), n_sections=1, seed=i)
        rows.append({**{f"d_{p.name}": float(xi[j]) for j, p in enumerate(base_powders)},
                     "L_um": float(xi[-1]), **{f"f_{kk}": v for kk, v in fk.items()}, "skipped": False})
    return pd.DataFrame(rows).sort_values("f_5", ascending=False, ignore_index=True)
