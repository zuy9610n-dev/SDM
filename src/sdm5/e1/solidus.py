"""E-1 C2:固相線の探索(v0.2 のコードを R9 に従って修正)。

熱力学の計算そのものは「液相分率を返す関数」として外から与える
(pycalphad の場合は backends.PycalphadBackend、Thermo-Calc の結果は CSV で取り込む)。
"""
from __future__ import annotations

from typing import Callable, Mapping, Protocol

import numpy as np

LiquidFn = Callable[[Mapping[str, float], np.ndarray], np.ndarray]


class ThermoBackend(Protocol):
    name: str
    version: str

    def liquid_fraction(self, x_at: Mapping[str, float], temps: np.ndarray) -> np.ndarray: ...

    def stable_phases(self, x_at: Mapping[str, float], T: float) -> list[tuple[str, float]]: ...


def solidus(liquid_fraction: LiquidFn, x_at: Mapping[str, float], t_lo: float = 600.0,
            t_hi: float = 1800.0, step: float = 10.0, eps: float = 1e-4,
            refine: int = 11) -> tuple[float, str]:
    """固相線(K)と状態を返す。

    状態:
      正常            … 粗い格子で初めて液相が出た区間を細かく探索した
      非単調          … 一度出た液相が高温側で消える点がある(再固化・偏晶の可能性)
      下限で液相あり  … t_lo ですでに液相がある。値は t_lo(上限値)
      上限まで液相なし … t_hi まで液相なし。値は t_hi(下限値)。v0.2 では区別できなかった(R9a)
    """
    temps = np.arange(t_lo, t_hi + step * 0.5, step)
    fl = np.asarray(liquid_fraction(x_at, temps), float)
    above = fl > eps
    if not above.any():
        return float(temps[-1]), "上限まで液相なし"
    i = int(np.argmax(above))
    if i == 0:
        return float(t_lo), "下限で液相あり"
    flag = "非単調" if (~above[i:]).any() else "正常"
    fine = np.linspace(temps[i - 1], temps[i], refine)
    ff = np.asarray(liquid_fraction(x_at, fine), float) > eps
    if not ff.any():            # R9b: 細かい格子で液相が再現しない(数値の揺らぎ)
        return float(temps[i - 1]), "非単調"
    j = int(np.argmax(ff))
    if j == 0:                  # 粗い格子の下端で液相がないはずなのに出た → 揺らぎ
        return float(fine[0]), "非単調"
    return float(fine[j - 1]), flag


def solidus_table(backend: ThermoBackend, cells, t_lo=600.0, t_hi=1800.0, step=10.0, eps=1e-4,
                  progress: Callable[[int, int], None] | None = None):
    """cells 表(pandas)の各区画について固相線を計算して列を埋めた複製を返す。

    区画の中心組成のうち 0 at% の元素は系から除く(E-1 C2)。
    """
    from ..common.composition import ELEMENTS, VEC_COLS

    out = cells.copy()
    sol, flags = [], []
    n = len(out)
    for k, (_, row) in enumerate(out.iterrows()):
        x = {e.upper(): float(row[c]) for e, c in zip(ELEMENTS, VEC_COLS) if row[c] > 0}
        s, f = solidus(backend.liquid_fraction, x, t_lo, t_hi, step, eps)
        sol.append(s)
        flags.append(f)
        if progress:
            progress(k + 1, n)
    out["e1_solidus"] = sol
    out["e1_solidus_flag"] = flags
    return out
