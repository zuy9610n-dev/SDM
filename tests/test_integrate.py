import numpy as np
import pandas as pd
import pytest

from sdm5.integrate import core as I


def _pts():
    return pd.DataFrame({
        "chemsys": ["Al-Fe-Si", "Al-Fe-Si", "Al-Cu-Ti", "Al-Si-Ti"],
        "reduced_formula": ["Al2Fe3Si3", "Al3FeSi", "AlCuTi", "Al5Si12Ti7"],
        "x_Al": [25.0, 60.0, 33.3, 20.8], "x_Si": [37.5, 20.0, 0, 50.0], "x_Ti": [0, 0, 33.3, 29.2],
        "x_Fe": [37.5, 20.0, 0, 0], "x_Cu": [0, 0, 33.4, 0],
        "n_elements": [3, 3, 3, 3], "b_percentile": [95, 20, 99, 50],
    })


def test_attach_e2_exact_and_near():
    e2 = pd.DataFrame({"struct_id": ["s1", "s2", "s3"], "chemsys": ["Al-Fe-Si", "Al-Fe-Si", "Al-Fe-Si"],
                       "reduced_formula": ["Al2Fe3Si3", "Al2Fe3Si3", "Al61Fe20Si19"],
                       "x_Al": [25, 25, 61], "x_Si": [37.5, 37.5, 19], "x_Ti": [0, 0, 0], "x_Fe": [37.5, 37.5, 20],
                       "x_Cu": [0, 0, 0], "e_hull_meV": [30, 10, 5], "e2_tier": ["A", "S", "S"],
                       "e2_tier_prov": ["暫定A", "暫定S", "暫定S"]})
    p = I.attach_e2(_pts(), e2, 3.0)
    assert p.loc[0, "e2_best_struct_id"] == "s2"          # 最小 E_hull を代表に
    assert p.loc[1, "e2_best_struct_id"] == "s3" and 0 < p.loc[1, "e2_match_d"] <= 3
    assert pd.isna(p.loc[2, "e2_best_struct_id"])
    assert (p["match_threshold_at"] == 3.0).all()


def test_quadrants_and_R18_R21():
    q = lambda **kw: I.quadrant(kw, "本", True)
    assert q(e2_tier="S", b_percentile=95) == "①"
    assert q(e2_tier="A", b_percentile=10) == "②"
    assert q(e2_tier=None, b_percentile=95) == "③"
    assert q(e2_tier="不安定", b_percentile=10) == "④"
    assert q(e2_tier="保留", b_percentile=99) == "E2保留"       # R18:P3b に送らない
    assert I.quadrant({"e2_tier_prov": "暫定A", "b_percentile": 95}, "暫定", True) == "①"
    assert I.quadrant({"e2_tier": "S", "b_percentile": 1}, "本", False) == "E2単独"   # R21
    assert I.quadrant({"e2_tier": None, "b_percentile": 99}, "本", False) == "④"


def test_stage2_combination_R20():
    assert I.combine_stage2("可", "可") == "可"
    assert I.combine_stage2("要DSC確認", "可") == "要DSC確認"
    assert I.combine_stage2("要DSC確認", "不可") == "不可"
    assert I.combine_stage2(None, None) == "未評価"
    assert I.d_coverage_ok(5.0) == "可" and I.d_coverage_ok(4.9) == "不可" and I.d_coverage_ok(None) == "不可"


@pytest.mark.parametrize("rnd,q,s2,extra,expect", [
    ("本", "①", "可", {}, "最優先"),
    ("暫定", "①", "可", {}, "優先"),
    ("暫定", "②", "可", {}, "保留"),
    ("本", "②", "可", {"chemsys": "Al-Fe-Si", "e2_mag_checked": True}, "優先"),
    ("本", "②", "可", {"chemsys": "Al-Fe-Si", "e2_mag_checked": None}, "保留"),       # R19
    ("本", "②", "可", {"chemsys": "Al-Cu-Ti"}, "優先"),                               # Fe なし
    ("本", "①", "要DSC確認", {}, "条件付き"),
    ("暫定", "②", "要DSC確認", {}, "条件付き"),
    ("本", "①", "不可", {}, "保留"),
    ("本", "③", "可", {}, "構造探索"),
    ("本", "④", "可", {}, "低"),
    ("本", "E2保留", "可", {}, "保留"),
])
def test_final_priority_table(rnd, q, s2, extra, expect):
    assert I.final_priority({"quadrant": q, "stage2": s2, **extra}, rnd) == expect


def test_final_priority_low_confidence_system():
    assert I.final_priority({"quadrant": "①", "stage2": "可"}, "本", sys_conf_low=True) == "保留"


def test_stage2_for_points_end_to_end():
    pts = pd.DataFrame({"chemsys": ["Al-Fe"], "reduced_formula": ["AlFe"], "cell_id": ["05:10-00-00-10-00"],
                        "target_design_id": ["D1"]})
    cells = pd.DataFrame({"cell_id": ["05:10-00-00-10-00"], "e1_solidus": [1500.0], "e1_confidence": ["高"],
                          "e1_solidus_flag": ["正常"]})
    designs = pd.DataFrame({"design_id": ["D1"], "T_anneal_K": [800.0], "e1_T_max": [820.0],
                            "e1_T_max_status": ["確定"]})
    cd = pd.DataFrame({"cell_id": ["05:10-00-00-10-00"], "design_id": ["D1"], "d_expected_points": [12.0]})
    out = I.stage2_for_points(pts, cells, designs, cd, dT_safe=30)
    assert out.loc[0, "e1_ok"] == "可" and out.loc[0, "d_coverage_ok"] == "可" and out.loc[0, "stage2"] == "可"
    out2 = I.stage2_for_points(pts, cells, designs, cd.assign(d_expected_points=2.0), dT_safe=30)
    assert out2.loc[0, "stage2"] == "不可"


def test_integrate_and_p3b():
    p = _pts().assign(e2_tier=["S", None, None, "保留"], stage2=["可", "未評価", "未評価", "可"])
    out = I.integrate(p, "本", g2_passed=True)
    assert list(out["quadrant"]) == ["①", "④", "③", "E2保留"]
    assert list(out["final_priority"]) == ["最優先", "低", "構造探索", "保留"]
    p3b = I.to_p3b(out)
    assert list(p3b["reduced_formula"]) == ["AlCuTi"] and (p3b["source"] == "形B主導").all()


def test_feedback_classification():
    cand = pd.DataFrame({"x_Al": [25.0], "x_Si": [37.5], "x_Ti": [0.0], "x_Fe": [37.5], "x_Cu": [0.0]})
    known = pd.DataFrame({"x_Al": [50.0], "x_Si": [0.0], "x_Ti": [0.0], "x_Fe": [50.0], "x_Cu": [0.0]})
    base = dict(x_Al=26, x_Si=37, x_Ti=0, x_Fe=37, x_Cu=0)
    r = I.classify_observation({**base, "kind": "cluster", "xrd_ebsd_consistent": True}, cand, known, None)
    assert r["feedback_class"] == "一致"
    far = dict(x_Al=10, x_Si=10, x_Ti=40, x_Fe=10, x_Cu=30)
    assert I.classify_observation({**far, "kind": "cluster"}, cand, known, None)["feedback_class"] == "判定不能"
    assert I.classify_observation({**far, "kind": "cluster", "xrd_ebsd_confirmed": True}, cand, known,
                                  None)["feedback_class"] == "新相候補"
    rng = np.random.default_rng(0)
    c = np.array([25, 37.5, 0.0001, 37.5, 0.0])
    pts = np.clip(c + rng.normal(0, 3, (300, 5)), 0, None)
    pts[:, 2] = np.abs(rng.normal(0, 1, 300)); pts[:, 4] = np.abs(rng.normal(0, 1, 300))
    ob = {"x_Al": 25, "x_Si": 36, "x_Ti": 1, "x_Fe": 36, "x_Cu": 2, "kind": "candidate"}
    assert I.classify_observation({**ob, "time_varied_checked": False}, cand, known, pts)["feedback_class"] == "判定不能"
    assert I.classify_observation({**ob, "time_varied_checked": True}, cand, known, pts)["feedback_class"] == "未出現"
