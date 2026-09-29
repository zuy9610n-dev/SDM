import numpy as np
import pandas as pd
import pytest

from sdm5.common import cells as C
from sdm5.common.composition import (chemsys, distance, normalize, parse_formula, reduced_formula, round_vector,
                                     to_vector)
from sdm5.common.schema import CELLS, COMPOSITION_POINTS, SchemaError, validate
from sdm5.common.store import Store, VersionExistsError


def test_cell_counts_match_design():
    # D 3-1 の区画数
    assert C.count_by_n_elements(5.0) == {1: 5, 2: 190, 3: 1710, 4: 4845, 5: 3876}
    assert len(C.all_cells(5.0)) == 10626


def test_cell_id_format_and_parent():
    assert C.composition_to_cell_id([20, 20, 20, 20, 20]) == "05:04-04-04-04-04"
    fine = C.composition_to_cell_id([20, 20, 20, 20, 20], 2.5)
    assert fine == "025:08-08-08-08-08"
    assert C.parent_cell_id(fine) == "05:04-04-04-04-04"
    assert C.parent_cell_id("05:04-04-04-04-04") is None
    assert C.parse_cell_id(fine) == (2.5, (8, 8, 8, 8, 8))


def test_to_cell_normalizes_and_is_deterministic_on_ties():
    # 合計が100でない EDS 値(R7)
    assert sum(C.to_cell([40.1, 19.9, 20.0, 10.0, 9.5])) == 20
    # タイ:62.5/12.5/12.5/7.5/5 → 剰余 0.5 が3つ。元素順で前を切り上げる(R6)
    a = C.to_cell([62.5, 12.5, 12.5, 7.5, 5.0])
    assert a == (13, 3, 2, 1, 1)
    for _ in range(5):
        assert C.to_cell([62.5, 12.5, 12.5, 7.5, 5.0]) == a


def test_every_fine_cell_has_parent():
    t = C.build_cells_table()
    fine = t[t.resolution == 2.5]
    assert fine["parent_cell_id"].notna().all()
    assert set(fine["parent_cell_id"]) <= set(t.loc[t.resolution == 5.0, "cell_id"])
    assert (fine["x_Al"] >= 60).all()
    validate(t, CELLS)


def test_fine_cells_for_target_region():
    tr = C.TargetRegion("tau1", (25.0, 37.5, 0, 37.5, 0), 5.0)
    f = C.fine_cells(2.5, 101.0, [tr])
    assert len(f) > 0
    for cid in f:
        assert distance(C.cell_center(cid), np.array(tr.center)) <= 5.0 + 1e-9


def test_formula_helpers():
    assert reduced_formula("Al4Fe6Si6") == "Al2Fe3Si3"
    assert reduced_formula("FeAl") == "AlFe"
    assert reduced_formula("Ti5Si3") == "Ti5Si3"
    assert chemsys(to_vector("Al2Fe3Si3")) == "Al-Fe-Si"
    assert np.isclose(to_vector("AlFe").sum(), 100)
    with pytest.raises(ValueError):
        parse_formula("NiAl")
    assert round_vector(np.array([33.333, 33.333, 33.334, 0, 0])).sum() == pytest.approx(100.0)


def test_reduced_formula_matches_pymatgen_order():
    # 上位文書 3-1 は括弧なしの形(Al2Fe3Si3)を使う。新しい pymatgen は Al2(FeSi)3 とまとめるので、
    # 括弧なしの reduced_composition.formula と比べる。
    pmg = pytest.importorskip("pymatgen.core")
    for f in ["Al2Fe3Si3", "Ti7Al5Si12", "Al7Cu2Fe", "TiFeSi2", "Cu15Si4", "AlCuFeSiTi", "Al3Ti", "Fe2TiSi"]:
        flat = pmg.Composition(f).reduced_composition.formula.replace(" ", "")
        flat = "".join(t[:-1] if t.endswith("1") and not t[-2].isdigit() else t
                       for t in __import__("re").findall(r"[A-Z][a-z]?\d*", flat))
        assert reduced_formula(f) == flat


def test_schema_validation_and_store(tmp_path):
    df = pd.DataFrame({"chemsys": ["Al-Fe"], "reduced_formula": ["AlFe"], "e1_ok": ["可"]})
    validate(df, COMPOSITION_POINTS)
    with pytest.raises(SchemaError):
        validate(df.assign(e1_ok="OK"), COMPOSITION_POINTS)
    with pytest.raises(SchemaError):
        validate(pd.concat([df, df]), COMPOSITION_POINTS)
    s = Store(tmp_path)
    v = s.save("composition_points", df, tag="t", meta={"params_version": "x"}, date="2026-09-29")
    assert v == "2026-09-29_t"
    with pytest.raises(VersionExistsError):
        s.save("composition_points", df, tag="t", date="2026-09-29")
    assert s.load("composition_points").equals(df)
    assert s.meta("composition_points", v)["params_version"] == "x"


def test_vectorized_to_cells_matches_scalar():
    rng = np.random.default_rng(1)
    X = rng.dirichlet(np.ones(5) * 0.7, 3000) * 100
    X = np.vstack([X, [[62.5, 12.5, 12.5, 7.5, 5.0]], np.eye(5) * 100])
    for d in (5.0, 2.5):
        v = C.to_cells(X, d)
        for x, row in zip(X, v):
            assert tuple(row) == C.to_cell(x, d)
