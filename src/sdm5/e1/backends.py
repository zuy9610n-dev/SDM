"""熱力学計算のバックエンド。

- PycalphadBackend:C1 で検証済みの TDB を使う(pycalphad は任意の依存)。
- TableBackend:Thermo-Calc などで外部計算した固相線・相の表を取り込む。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

ORDER = ["AL", "SI", "TI", "FE", "CU"]   # 上位文書 3-1 の元素順


class PycalphadBackend:
    """E-1 C2 のコード(v0.2)を関数化したもの。pycalphad の版により引数が変わるので要確認。"""

    def __init__(self, tdb_path: str, phases: list[str] | None = None):
        try:
            from pycalphad import Database  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pycalphad が必要です: pip install 'sdm5[calphad]'") from e
        import pycalphad
        from pycalphad import Database

        self.db = Database(tdb_path)
        self.phases = phases or list(self.db.phases.keys())
        self.name = f"pycalphad:{tdb_path}"
        self.version = pycalphad.__version__

    def _eq(self, x_at: Mapping[str, float], temps):
        from pycalphad import equilibrium, variables as v

        els = [e for e in ORDER if x_at.get(e, 0) > 0]
        cond = {v.T: np.atleast_1d(temps).astype(float), v.P: 101325, v.N: 1}
        for e in els[1:]:                       # 先頭の元素を従属変数にする
            cond[v.X(e)] = x_at[e] / 100.0
        return equilibrium(self.db, els + ["VA"], self.phases, cond), len(cond[v.T])

    def liquid_fraction(self, x_at, temps):
        eq, nt = self._eq(x_at, temps)
        ph = eq.Phase.values.reshape(nt, -1)
        npf = eq.NP.values.reshape(nt, -1)
        return np.nansum(np.where(ph == "LIQUID", npf, 0.0), axis=1)

    def stable_phases(self, x_at, T):
        eq, _ = self._eq(x_at, [T])
        ph = eq.Phase.values.ravel()
        npf = eq.NP.values.ravel()
        out: dict[str, float] = {}
        for p, f in zip(ph, npf):
            if p and isinstance(p, str) and np.isfinite(f) and f > 1e-6:
                out[p] = out.get(p, 0.0) + float(f)
        return sorted(out.items(), key=lambda t: -t[1])


class TableBackend:
    """外部計算の結果(cell_id, e1_solidus, e1_solidus_flag [, T_K, e1_phases])を取り込む。

    liquid_fraction は提供しない。cells 表に直接結合して使う。
    """

    def __init__(self, solidus_csv: str, name: str = "external", version: str = "unknown"):
        self.df = pd.read_csv(solidus_csv)
        self.name, self.version = name, version

    def apply(self, cells: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("e1_solidus", "e1_solidus_flag") if c in self.df.columns]
        out = cells.drop(columns=[c for c in cols if c in cells.columns]).merge(
            self.df[["cell_id", *cols]], on="cell_id", how="left")
        return out
