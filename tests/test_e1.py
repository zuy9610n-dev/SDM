import numpy as np
import pandas as pd
import pytest

from sdm5.common.cells import build_cells_table
from sdm5.e1 import core as E
from sdm5.e1.solidus import solidus, solidus_table
from sdm5.e1.toy import ToyBackend


def step_liquid(ts):
    return lambda x, t: (np.asarray(t, float) > ts).astype(float)


def test_solidus_normal_and_refined():
    s, f = solidus(step_liquid(853.4), {}, 600, 1800, 10)
    assert f == "正常"
    assert 852.0 <= s <= 853.4 + 1e-9


def test_solidus_flags_R9():
    assert solidus(step_liquid(10.0), {}, 600, 1800)[1] == "下限で液相あり"
    s, f = solidus(step_liquid(5000.0), {}, 600, 1800)
    assert f == "上限まで液相なし" and s == 1800.0
    # 液相が一度出て消える(非単調)
    nonmono = lambda x, t: ((np.asarray(t) > 900) & (np.asarray(t) < 950)).astype(float)
    assert solidus(nonmono, {}, 600, 1800)[1] == "非単調"


def test_dT_safe():
    assert E.dT_safe_from_errors([3, 8, 5]) == 20.0
    assert E.dT_safe_from_errors([3, 15]) == 30.0


def test_reachable_set_and_hull_distance():
    V = np.eye(5) * 100
    assert np.allclose(E.distance_to_hull(np.array([[20, 20, 20, 20, 20]]), V), 0)
    # Al と Fe だけの原料 → Al-Fe 線からの距離
    V2 = np.array([[100, 0, 0, 0, 0], [0, 0, 0, 100, 0]])
    d = E.distance_to_hull(np.array([[50, 0, 0, 50, 0], [50, 10, 0, 40, 0]]), V2)
    assert d[0] == pytest.approx(0, abs=1e-6)
    assert d[1] > 5


@pytest.fixture(scope="module")
def cells_with_solidus():
    t = build_cells_table()
    t = t[t.resolution == 5.0].reset_index(drop=True)
    tb = solidus_table(ToyBackend(), t, step=20)
    tb["e1_confidence"] = "高"
    return tb


def test_t_max_pure_vs_compound(cells_with_solidus):
    t = cells_with_solidus
    pure = np.eye(5) * 100
    r_pure = E.t_max_for_design(t, pure, dT_safe=30)
    assert r_pure.n_cells_in_S == len(t)
    assert r_pure.status == "確定"
    # Al を FeAl・Al3Ti で入れ、Cu を使わない → S が狭くなり T_max が上がる
    comp = np.array([[50, 0, 0, 50, 0], [75, 0, 25, 0, 0], [0, 100, 0, 0, 0], [0, 0, 0, 100, 0]])
    r_c = E.t_max_for_design(t, comp, dT_safe=30)
    assert r_c.n_cells_in_S < r_pure.n_cells_in_S
    assert r_c.T_max >= r_pure.T_max


def test_t_max_status_R11(cells_with_solidus):
    t = cells_with_solidus.copy()
    r = E.t_max_for_design(t, np.eye(5) * 100, 30)
    # 最小値 +20 K の区画を低信頼に → 要DSC確認
    j = t.index[(t.e1_solidus > r.min_solidus) & (t.e1_solidus <= r.min_solidus + 40)][0]
    t.loc[j, "e1_confidence"] = "低"
    r2 = E.t_max_for_design(t, np.eye(5) * 100, 30, dT_low_conf=50)
    assert r2.status == "要DSC確認"


def test_e1_ok_three_values_R10():
    row = {"e1_solidus": 900.0, "e1_confidence": "高", "e1_solidus_flag": "正常"}
    assert E.e1_ok(row, 800, 850, "確定", 30)[0] == "可"
    assert E.e1_ok(row, 880, 850, "確定", 30)[0] == "不可"          # T_anneal > T_max
    assert E.e1_ok({**row, "e1_solidus": 820.0}, 800, 850, "確定", 30)[0] == "不可"
    assert E.e1_ok({**row, "e1_confidence": "低"}, 800, 850, "確定", 30)[0] == "要DSC確認"
    assert E.e1_ok(row, 800, 850, "要DSC確認", 30)[0] == "要DSC確認"
    assert E.e1_ok({**row, "e1_solidus_flag": "非単調"}, 800, 850, "確定", 30)[0] == "要DSC確認"


def test_confidence_assignment():
    cells = pd.DataFrame({"chemsys": ["Al", "Al-Fe", "Al-Fe-Si", "Al-Cu-Fe-Si-Ti"]})
    ev = pd.DataFrame({"subsystem": ["Al-Fe", "Al-Si", "Fe-Si", "Al-Fe-Si"], "confidence": ["高", "高", "中", "高"]})
    c = E.assign_confidence(cells, ev)
    assert list(c) == ["高", "高", "中", "低"]
    ev2 = E.apply_ge2(ev, {"Al-Fe-Si": False})
    assert ev2.set_index("subsystem").at["Al-Fe-Si", "confidence"] == "低"


def test_diffusion_and_ge1():
    L = E.diffusion_length(1e-16, 300)
    assert L == pytest.approx(2 * np.sqrt(1e-16 * 300 * 3600) * 1e6)
    assert E.time_required_h(L, 1e-16) == pytest.approx(300)
    D = pd.DataFrame({"species": ["Cu", "Ti"], "D_lo": [1e-14, 1e-19], "D_hi": [1e-13, 1e-17]})
    r = E.ge1(10.0, D, 300)            # Ti: t_req(1e-17)=694 h > 300 → 不可
    assert r.verdict == "不可" and r.limiting == "Ti"
    D2 = D.assign(D_hi=[1e-13, 1e-15])
    assert E.ge1(10.0, D2, 300).verdict == "不確定"
    assert E.ge1(1.0, D.assign(D_lo=[1e-14, 1e-15]), 300).verdict == "可"


def test_t_ad():
    assert E.adiabatic_temperature(-40.0) == pytest.approx(298.15 + 40000 / (3 * 8.314462618))
    dh = pd.DataFrame({"formula": ["TiSi", "Al3Ti", "FeAl"], "dH_kJ_per_mol_atom": [-65.0, -40.0, -25.0]})
    t = E.t_ad_for_powders(["Ti", "Si", "Al", "Fe"], dh)
    assert t.iloc[0]["pair"] == "Ti+Si"
    assert t.set_index("pair").at["Si+Al", "product"] is None


def test_phase_table_toy():
    t = build_cells_table()
    t = t[(t.resolution == 5.0) & (t.n_elements <= 2)].head(30)
    ph = E.phase_table(ToyBackend(), t, [780, 830])
    assert set(ph.columns) == {"cell_id", "T_K", "e1_phases", "e1_single_phase"}
    assert len(ph) == 60
