import numpy as np
import pandas as pd
import pytest

from sdm5.e2 import core as E
from sdm5.e2.hull import Entry, Hull, MissingReferenceError, MixedMethodError, ehull_columns


def _entries(method="uMLIP:A"):
    return [Entry("al", "Al", -3.7, method), Entry("fe", "Fe", -8.3, method), Entry("si", "Si", -5.4, method),
            Entry("alfe", "AlFe", -6.3, method),          # Ef = -6.3 - (-6.0) = -0.30
            Entry("al3fe", "Al3Fe", -4.6, method),        # Ef = -4.6 - (-4.85) = +0.25 → 凸包の上
            Entry("fesi", "FeSi", -7.2, method)]


def test_hull_basic():
    h = Hull(_entries())
    t = h.table().set_index("struct_id")
    assert t.at["alfe", "e_hull"] == pytest.approx(0)
    # Al3Fe:凸包は Al と AlFe の間 0.5·(−0.30) = −0.15、Ef = +0.25 → 0.40
    assert t.at["al3fe", "e_hull"] == pytest.approx(0.40)
    assert h.e_hull_of("Al3Fe", -5.1) == pytest.approx(-0.10)


def test_hull_rejects_mixed_methods_R2():
    es = _entries() + [Entry("x", "AlFe", -6.5, "DFT:PBE")]
    with pytest.raises(MixedMethodError):
        Hull(es)
    with pytest.raises(MissingReferenceError):
        Hull([Entry("a", "Al", -3.7, "m"), Entry("b", "AlFe", -6.0, "m")])


def test_ehull_columns_per_method():
    es = _entries("uMLIP:A") + _entries("uMLIP:B")
    es = [Entry(e.struct_id, e.formula, e.energy_per_atom + (0.05 if e.method.endswith("B") and e.struct_id == "alfe" else 0),
                e.method) for e in es]
    s = pd.DataFrame({"struct_id": ["alfe", "al3fe"]})
    out = ehull_columns(s, es)
    assert {"ehull__uMLIP:A", "ehull__uMLIP:B"} <= set(out.columns)


def test_hull_matches_pymatgen():
    pd_mod = pytest.importorskip("pymatgen.analysis.phase_diagram")
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.core import Composition
    es = _entries()
    ce = [ComputedEntry(Composition(e.formula), e.energy_per_atom * Composition(e.formula).num_atoms, entry_id=e.struct_id)
          for e in es]
    pdg = pd_mod.PhaseDiagram(ce)
    t = Hull(es).table().set_index("struct_id")
    for c in ce:
        assert t.at[c.entry_id, "e_hull"] == pytest.approx(pdg.get_e_above_hull(c), abs=1e-6)


def test_p2_reliability():
    known = pd.DataFrame({"chemsys": ["Al-Fe", "Al-Fe", "Al-Si"], "formula": ["AlFe", "Al5Fe2", "AlSi"],
                          "report_status": ["バルクの平衡相", "バルクの平衡相", "計算予測のみ"]})
    ht = pd.DataFrame({"chemsys": ["Al-Fe", "Al-Fe", "Al-Fe", "Al-Si"],
                       "reduced_formula": ["AlFe", "Al3Fe", "Al5Fe2", "Al3Si"], "e_hull": [0.0, 0.0, 0.1, 0.0]})
    r = E.p2_reliability(known, ht).set_index("chemsys")
    assert r.at["Al-Fe", "recall"] == 0.5 and r.at["Al-Fe", "confidence"] == "低"
    assert r.at["Al-Fe", "false_stable"] == "Al3Fe"
    # 計算予測のみは正解に入れない(ルール5)→ Al-Si は既知化合物なし+誤安定あり → 中
    assert r.at["Al-Si", "n_known"] == 0 and r.at["Al-Si", "confidence"] == "中"
    assert E.system_confidence("Al-Fe-Si", {"Al-Fe": "高", "Al-Si": "高", "Fe-Si": "中", "Al-Fe-Si": "高"}) == "中"


def test_p4_screen_R4():
    df = pd.DataFrame({"a": [10, 20, 60, 60, 5, 100], "b": [15, 30, 65, 40, 80, 120],
                       "in_target": [False, False, False, False, True, False]})
    r = E.p4_screen(df, "a", "b", conf_col=None)
    assert list(r["e2_tier_prov"]) == ["暫定S", "暫定A", "暫定A", "暫定A", "保留", "不安定"]
    # 60/65 meV の暫定A も P5 に進む(v0.2 の 50 meV では落ちていた)
    assert list(r["to_p5"]) == [True, True, True, True, True, False]


def test_config_entropy():
    assert E.config_entropy_meV(790, [[0.5, 0.5]], [1.0]) == pytest.approx(47.2, abs=0.1)
    assert E.config_entropy_meV(1000, [[0.5, 0.5]], [1.0]) == pytest.approx(59.7, abs=0.1)
    assert E.corrected_ehull(40, [780, 830], occupancies=[[0.5, 0.5]], multiplicities=[0.5]) == pytest.approx(
        40 - E.config_entropy_meV(830, [[0.5, 0.5]], [0.5]))


def test_formal_tier_R16():
    assert E.formal_tier(10, False, "高") == "S"
    assert E.formal_tier(10, False, "中") == "A"                   # R16a
    assert E.formal_tier(10, False, "低") == "保留"
    assert E.formal_tier(40, False, "高", corrected_meV=20) == "A"
    assert E.formal_tier(40, False, "高", corrected_meV=30) == "不安定"
    assert E.formal_tier(10, False, "高", model_diff_meV=60) == "保留"
    assert E.formal_tier(0, True, "高") == "既知"
    assert E.formal_tier(0, True, "高", large_solubility=True) == "B"
