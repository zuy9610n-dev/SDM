import numpy as np
import pandas as pd
import pytest

from sdm5.formb.features import build_features, subsystem_features
from sdm5.formb.models import PUBagging, ensemble, m1_scores, percentile_within
from sdm5.formb.points import build_training_set, canonical_key, enumerate_points
from sdm5.formb.validate import bootstrap_diff, g2_decide, rank_metrics


def test_candidate_counts_match_design():
    # 形B 2章:約分前の候補数 2元 190・3元 1140・4元 4845・5元 15504(1系あたり)。約分後は少なくなる
    from math import comb
    assert [comb(20, k) for k in (2, 3, 4, 5)] == [190, 1140, 4845, 15504]
    p = enumerate_points(max_atoms=20)
    assert set(p["n_elements"]) == {2, 3, 4, 5}
    assert not p.duplicated(["chemsys", "reduced_formula"]).any()
    assert p.groupby("chemsys").size().loc["Al-Fe"] < 190
    assert p["cell_id"].str.startswith("05:").all()


def test_canonical_key_rounding():
    assert canonical_key("Al4Fe6Si6") == ("Al-Fe-Si", "Al2Fe3Si3", 0.0, False)
    cs, f, d, dis = canonical_key("Al62Cu25Fe13")          # 原子数 > 20 → 丸める
    assert cs == "Al-Cu-Fe" and d < 3.0 and not dis
    cs, f, d, dis = canonical_key("Al0.5Si0.5Fe")
    assert dis and cs == "Al-Fe-Si"


def test_training_set_rules():
    rec = pd.DataFrame({"formula": ["AlFe", "FeAl", "Fe2TiSi", "Al3Ti", "NiAl", "AlCu"],
                        "year": [1950, 1930, 2000, 1960, 1940, 1970],
                        "report_status": ["バルクの平衡相", "バルクの平衡相", "計算予測のみ", "バルクの平衡相",
                                          "バルクの平衡相", "薄膜のみ"]})
    t = build_training_set(rec)
    assert "Fe2SiTi" not in set(t["reduced_formula"]) and "TiFe2Si" not in set(t["reduced_formula"])
    alfe = t[(t.chemsys == "Al-Fe")].iloc[0]
    assert alfe["year"] == 1930 and alfe["n_entries"] == 2
    assert not t.set_index("chemsys").at["Al-Ni", "target_system"]
    assert "Al-Cu" not in set(t["chemsys"])                   # 薄膜のみは主の版から除く


def test_percentile_by_element_count_R3():
    s = np.array([1, 2, 3, 4, 10, 20, 30, 40])
    g = [2, 2, 2, 2, 3, 3, 3, 3]
    p = percentile_within(s, g)
    assert list(p[:4]) == list(p[4:]) == [25, 50, 75, 100]


def test_subsystem_features_leak_guard():
    known = ["AlFe", "Al2Fe3Si3", "FeSi"]
    f = subsystem_features(["Al2FeSi", "AlFe2Si"], known)
    assert (f["sub_known"] == 2).all()                         # Al-Fe と Fe-Si
    f2 = subsystem_features(["Al2FeSi"], known, exclude_systems=["Al-Fe"])
    assert f2["sub_known"].iloc[0] == 1                        # Al-Fe を数えない
    assert f2["d_near_known"].iloc[0] > 0


def test_pu_bagging_ranks_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y_true = (X[:, 0] + 0.5 * X[:, 1] > 1).astype(int)
    y = y_true * (rng.random(600) < 0.5)                      # 正例の半分だけラベルあり
    mean, std, _, _ = PUBagging(n_bags=10, seed=0).fit_score(X, y)
    hidden = (y_true == 1) & (y == 0)
    assert mean[hidden].mean() > mean[y_true == 0].mean() + 0.2


def test_m1_and_ensemble():
    p = enumerate_points(["Al", "Fe", "Ti"], max_atoms=6, with_cells=False)
    pos = p["reduced_formula"].isin(["AlFe", "AlTi", "Al3Ti", "Al3Fe"]).to_numpy()
    s = m1_scores(p["reduced_formula"].tolist(), pos, rank=2)
    assert s.shape == (len(p),)
    df = p.assign(a=np.arange(len(p)), b=np.arange(len(p))[::-1])
    e = ensemble(df, ["a", "b"])
    assert {"b_score", "b_percentile", "b_percentile_chemsys", "b_rank_spread"} <= set(e.columns)


def test_rank_metrics_and_bootstrap_and_g2():
    pct = np.array([99.5, 96, 91, 50, 10, 99, 80])
    test = np.array([1, 1, 1, 1, 0, 0, 0], bool)
    m = rank_metrics(pct, test)
    assert m["recall@1%"] == 0.25 and m["recall@5%"] == 0.5 and m["recall@10%"] == 0.75
    good = np.full(200, 99.0); bad = np.full(200, 50.0)
    t = np.ones(200, bool)
    b = bootstrap_diff(good, bad, t, n_boot=200)
    assert b["lo"] > 0
    res = {"element_extrapolation": {"m2": {}, "heur": {}, "boot": b},
           "system_exclusion": {"m2": {}, "heur": {}, "boot": b}}
    g = g2_decide(res)
    assert g.passed and "時系列" in g.note
    res["element_extrapolation"]["boot"] = bootstrap_diff(bad, good, t, n_boot=200)
    assert not g2_decide(res).passed


def test_validation_pipeline_small():
    from sdm5.formb.pipeline import run_validation
    rec = pd.DataFrame({"formula": ["AlFe", "Al3Fe", "Al5Fe2", "FeSi", "Fe3Si", "FeSi2", "AlSi", "Al2Fe3Si3",
                                    "Al3FeSi", "AlFeSi", "Al4Fe"], "year": [1930, 1940, 1950, 1935, 1945, 1955,
                                                                            1960, 1990, 1995, 2000, 2010]})
    t = build_training_set(rec, max_atoms=10)
    g, raw = run_validation(t, ["Al", "Fe", "Si"], max_atoms=10, split_year=1980, exclude_system="Al-Fe-Si",
                            n_bags=5, n_boot=100)
    assert "time_split" in raw and "system_exclusion" in raw
    assert isinstance(g.passed, bool)
