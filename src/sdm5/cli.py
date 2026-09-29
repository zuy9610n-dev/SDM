"""sdm5 コマンド(第1段階の作業を回すための入口)。

  sdm5 cells     --out out/          区画表(5 at% + 2.5 at% 細分区画)を作る
  sdm5 points    --out out/          形B の組成点表を作る(cell_id つき)
  sdm5 ge1       --L 10 --D c4a.csv  GE1 の判定(C4a の拡散係数の範囲から)
  sdm5 tmax      --cells cells.csv --powders "100,0,0,0,0;0,100,0,0,0;..."
  sdm5 level0    --d50 20 --L 1,5,10,20   レベル0の f_k(L)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common.params import load_params
from .common.store import Store


def _store(a) -> Store:
    return Store(a.out)


def cmd_cells(a, P):
    from .common.cells import build_cells_table
    c = P["composition"]
    t = build_cells_table(c["delta_base"], c["delta_fine"], c["fine_al_min"])
    v = _store(a).save("cells", t, tag=a.tag, meta={"params_version": P["params_version"]})
    print(f"cells: {len(t)} 行 → {a.out}/cells/{v}.csv")
    print(t.groupby(["resolution", "n_elements"]).size().to_string())


def cmd_points(a, P):
    from .formb.points import enumerate_points
    t = enumerate_points(max_atoms=P["formb"]["max_atoms"])
    v = _store(a).save("composition_points", t, tag=a.tag, meta={"params_version": P["params_version"]})
    print(f"composition_points: {len(t)} 行 → {a.out}/composition_points/{v}.csv")
    print(t.groupby("n_elements").size().to_string())


def cmd_ge1(a, P):
    from .e1.core import ge1
    D = pd.read_csv(a.D).rename(columns={"D_lo_m2s": "D_lo", "D_hi_m2s": "D_hi"}).dropna(subset=["D_lo", "D_hi"])
    if D.empty:
        sys.exit("拡散係数の範囲(D_lo_m2s, D_hi_m2s)が空です。C4a の表を記入してください")
    r = ge1(a.L, D, a.t_limit or P["e1"]["t_anneal_max_h"])
    print(f"GE1 = {r.verdict}(律速: {r.limiting}、必要時間 {r.t_req_h_fast:.3g}〜{r.t_req_h_slow:.3g} h)")
    print(r.detail.to_string(index=False))


def cmd_tmax(a, P):
    from .e1.core import t_max_for_design
    cells = pd.read_csv(a.cells)
    V = np.array([[float(v) for v in s.split(",")] for s in a.powders.split(";")])
    r = t_max_for_design(cells, V, a.dT_safe or P["e1"]["dT_safe_K"], P["e1"]["delta_S_at"],
                         P["e1"]["dT_low_conf_K"])
    print(json.dumps(r.__dict__, ensure_ascii=False, indent=2, default=float))


def cmd_level0(a, P):
    from .d.level0 import f_k
    from .d.packing import pack, pure
    pk = pack([pure(i, a.d50, 0.2) for i in range(5)], seed=a.seed)
    print(f"粒子 {len(pk.radii)} 個、箱 {pk.box_um:.0f} µm、RSA 充填率 {pk.rsa_fraction:.2f}")
    rows = [{"L_um": L, **{f"f_{k}": v for k, v in f_k(pk, L, spacing_um=max(1, pk.box_um / 80)).items()}}
            for L in (float(x) for x in a.L.split(","))]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sdm5")
    ap.add_argument("--params", default=None, help="上書き用の YAML")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cells", "points"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="out")
        s.add_argument("--tag", default="v")
    s = sub.add_parser("ge1")
    s.add_argument("--L", type=float, required=True, help="必要な拡散距離 µm(D のレベル0から)")
    s.add_argument("--D", required=True, help="C4a の表(species, D_lo_m2s, D_hi_m2s)")
    s.add_argument("--t-limit", dest="t_limit", type=float, default=None)
    s = sub.add_parser("tmax")
    s.add_argument("--cells", required=True)
    s.add_argument("--powders", required=True, help="原料組成 at%(固定順)を ; 区切り")
    s.add_argument("--dT-safe", dest="dT_safe", type=float, default=None)
    s = sub.add_parser("level0")
    s.add_argument("--d50", type=float, default=20.0)
    s.add_argument("--L", default="1,5,10,20")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    P = load_params(a.params)
    {"cells": cmd_cells, "points": cmd_points, "ge1": cmd_ge1, "tmax": cmd_tmax, "level0": cmd_level0}[a.cmd](a, P)


if __name__ == "__main__":
    main()
