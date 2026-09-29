import numpy as np
import pytest

from sdm5.d import metrics as M
from sdm5.d.design import Design, evaluate_level1a, lhs_screen_level0
from sdm5.d.level0 import f_k
from sdm5.d.level1 import Field, diffuse, eds_sample, initial_field
from sdm5.d.packing import pack, pure, voxel_labels, measured_volume_fractions


@pytest.fixture(scope="module")
def pk():
    return pack([pure(i, 20, 0.2) for i in range(5)], box_um=120, seed=3)


def test_packing_no_overlap_and_laguerre_fractions(pk):
    c, r, L = pk.centers, pk.radii, pk.box_um
    d = np.abs(c[:, None] - c[None])
    d = np.sqrt((np.minimum(d, L - d) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert np.all(d >= (r[:, None] + r[None]) - 1e-9)
    assert 0.15 < pk.rsa_fraction < 0.45
    vf = measured_volume_fractions(pk, voxel_labels(pk, 32))
    assert np.allclose(vf, 0.2, atol=0.08)          # 緻密化後もおおむね等体積(R13)


def test_level0_monotone_in_L(pk):
    f_small = f_k(pk, 1.0, spacing_um=3, n_sections=1)
    f_big = f_k(pk, 30.0, spacing_um=3, n_sections=1)
    assert sum(f_small.values()) == pytest.approx(1)
    mean_k = lambda f: sum(k * v for k, v in f.items())
    assert mean_k(f_big) > mean_k(f_small)


def test_fft_diffusion_conserves_and_matches_gaussian():
    n, h = 64, 3.0          # 界面の間隔 96 µm ≫ 2√(Dt) = 12 µm
    conc = np.zeros((5, n, n, n), np.float32)
    conc[0] = 100.0
    conc[0, :, :, : n // 2] = 0
    conc[1, :, :, : n // 2] = 100.0
    f = Field(conc, h, n * h)
    D, t_h = 1e-15, 10.0
    g = diffuse(f, [D, D, 0, 0, 0], t_h)
    assert np.allclose(g.conc.sum(0), 100, atol=1e-3)
    assert g.conc[0].mean() == pytest.approx(50, abs=0.1)
    # 界面の濃度分布 = 誤差関数。10–90% 幅 ≈ 2·1.8126·√(Dt)
    prof = g.conc[0, 0, 0, :]
    x = np.arange(n) * h
    w = M.interface_width(prof[n // 4: 3 * n // 4], x[n // 4: 3 * n // 4])
    expect = 2 * 1.8124 * np.sqrt(D * t_h * 3600) * 1e6
    assert w == pytest.approx(expect, rel=0.15)


def test_in_hull_and_surrounded():
    rng = np.random.default_rng(0)
    center = np.array([20, 20, 20, 20, 20.0])
    pts = center + rng.normal(0, 4, (400, 5))
    pts = pts / pts.sum(1, keepdims=True) * 100
    ok, n = M.surrounded(center, pts, R=10, M=50)
    assert ok and n >= 50
    # 片側だけにある点 → 包囲されない
    one_side = pts[pts[:, 0] > 22]
    assert not M.surrounded(center, one_side, R=10, M=10)[0]
    # 点が少なければ偽
    assert not M.surrounded(center, pts[:10], R=10, M=50)[0]
    # 誤差のある点(合計が100でない)でも動く(R8)
    noisy = pts * rng.uniform(0.98, 1.02, (len(pts), 1))
    assert M.in_hull(center, noisy)


def test_lambda_coverage_required_points():
    pts = np.array([[20, 20, 20, 20, 20.0]] * 50 + [[50, 50, 0, 0, 0.0]] * 50)
    lam = M.expected_points(pts, 1000)
    assert lam["05:04-04-04-04-04"] == pytest.approx(500)
    cov = M.coverage(lam.to_dict(), n_min=5, target_cells=["05:04-04-04-04-04", "05:10-10-00-00-00"])
    assert cov["coverage_5"] == pytest.approx(1 / 3876)
    assert cov["coverage_2"] == pytest.approx(1 / 190)
    assert cov["coverage_target"] == 1.0
    assert M.required_points(pts, ["05:04-04-04-04-04"], 5) == 10
    assert np.isinf(M.required_points(pts, ["05:00-00-00-00-20"], 5))
    cm = M.coverage(lam.to_dict(), 5, single_phase={"05:10-10-00-00-00": True})
    assert cm["coverage_2_mask"] == 1.0


def test_gd2_histogram():
    a = np.array([[50, 50, 0, 0, 0]] * 10 + [[20] * 5] * 10, float)
    r = M.gd2(a, a)
    assert r["error"] == 0 and r["pass"]
    b = np.array([[20] * 5] * 20, float)
    assert M.gd2(a, b)["error"] == pytest.approx(0.5)


def test_eds_sample_normalized(pk):
    f0 = initial_field(pk, 32)
    pts = eds_sample(f0, spacing_um=4, blur_um=1.0, quant_err_at=1.0)
    assert np.allclose(pts.sum(1), 100)
    assert (pts >= 0).all()


def test_evaluate_level1a_end_to_end():
    d = Design("t", [pure(i, 15, 0.2) for i in range(5)], 780, 50, n_points_planned=5000)
    row, cd, pts = evaluate_level1a(d, lambda T: [1e-15] * 5, n_vox=32, target_cells=["05:04-04-04-04-04"])
    assert set(["coverage_2", "coverage_5", "coverage_target", "required_points"]) <= set(row)
    assert cd["d_expected_points"].sum() == pytest.approx(5000)


def test_lhs_screen_small():
    df = lhs_screen_level0([pure(i, 20, 0.2) for i in range(5)], n=3, seed=1, d_bounds_um=(10, 40), max_particles=4000)
    assert len(df) == 3 and "f_5" in df


def test_too_many_particles_guard():
    from sdm5.d.packing import TooManyParticlesError
    with pytest.raises(TooManyParticlesError):
        pack([pure(0, 5, 0.5), pure(3, 150, 0.5)], max_particles=10_000)
