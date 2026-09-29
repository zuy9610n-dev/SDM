"""テストと動作確認用の合成バックエンド(物理的に正しい値ではない)。

固相線を「純元素の融点の組成加重平均から、Al と Cu・Si が共存するほど下げる」形で作る。
実際の計算には PycalphadBackend か外部計算の取り込みを使うこと。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

TM = {"AL": 933.5, "SI": 1687.0, "TI": 1941.0, "FE": 1811.0, "CU": 1357.8}


class ToyBackend:
    name = "toy"
    version = "0"

    def __init__(self, width_K: float = 40.0, nonmonotonic_cells: set | None = None):
        self.width = width_K

    def solidus_of(self, x_at: Mapping[str, float]) -> float:
        tot = sum(x_at.values())
        w = {k: v / tot for k, v in x_at.items()}
        base = sum(w.get(e, 0) * TM[e] for e in TM)
        al, cu, si = w.get("AL", 0), w.get("CU", 0), w.get("SI", 0)
        eut = 700.0 * al * (cu + si) * 2 + 250.0 * al * w.get("FE", 0) + 200 * cu * si
        return max(700.0, base - eut)

    def liquid_fraction(self, x_at, temps):
        ts = self.solidus_of(x_at)
        t = np.asarray(temps, float)
        return np.clip((t - ts) / self.width, 0, 1)

    def stable_phases(self, x_at, T):
        n = sum(1 for v in x_at.values() if v > 0)
        # 単体と、Al が 50 at% 前後の2元系だけ単相とする(形だけ)
        tot = sum(x_at.values())
        if n == 1 or (n == 2 and abs(x_at.get("AL", 0) / tot - 0.5) < 0.06):
            return [("PHASE_" + "_".join(sorted(k for k, v in x_at.items() if v > 0)), 1.0)]
        return [("A", 0.6), ("B", 0.4)]
