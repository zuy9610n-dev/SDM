"""D 5-3:レベル1a(FFT による連続拡散、周期境界)と EDS 測定の模擬。

各元素 e について c_e(k, t) = c_e(k, 0)·exp(−D_e k² t)。計算後、各ボクセルで合計を 100 at% に規格化する。
元素間の干渉とカーケンダル効果は無視する(近似)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from ..common.composition import normalize
from .packing import Packing, voxel_labels


@dataclass
class Field:
    conc: np.ndarray        # (5, n, n, n) at%
    h_um: float
    box_um: float


def initial_field(pk: Packing, n: int) -> Field:
    lab = voxel_labels(pk, n)
    comps = pk.element_compositions()[pk.powder_idx]     # (n_particle, 5)
    conc = np.moveaxis(comps[lab], -1, 0).astype(np.float32)
    return Field(conc, pk.box_um / n, pk.box_um)


def diffuse(f0: Field, D_m2s: Sequence[float], t_h: float) -> Field:
    """D_m2s:元素ごとの実効拡散係数(固定の元素順)。"""
    n = f0.conc.shape[1]
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=f0.h_um * 1e-6)          # rad/m
    k2 = (k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + k1[None, None, :] ** 2)
    t = t_h * 3600.0
    out = np.empty_like(f0.conc)
    for e in range(f0.conc.shape[0]):
        if not np.any(f0.conc[e]):
            out[e] = 0
            continue
        ck = np.fft.fftn(f0.conc[e])
        ck *= np.exp(-float(D_m2s[e]) * k2 * t)
        out[e] = np.real(np.fft.ifftn(ck))
    out = np.clip(out, 0, None)
    s = out.sum(0, keepdims=True)
    s[s == 0] = 1
    return Field(out / s * 100.0, f0.h_um, f0.box_um)


def eds_sample(field: Field, spacing_um: float = 1.0, blur_um: float = 1.0, quant_err_at: float = 1.5,
               z_index: int | None = None, axis: int = 2, seed: int = 0) -> np.ndarray:
    """断面を切り出し、ぼかし(ガウス、FWHM ではなく σ=blur_um/2.355)、格子状の測定点、定量誤差を加える。"""
    rng = np.random.default_rng(seed)
    n = field.conc.shape[1]
    zi = n // 2 if z_index is None else z_index
    sl = np.take(field.conc, zi, axis=axis + 1)                    # (5, n, n)
    sigma_vox = blur_um / 2.355 / field.h_um
    if sigma_vox > 0:
        sl = np.stack([gaussian_filter(s, sigma_vox, mode="wrap") for s in sl])
    g = np.arange(spacing_um / 2, field.box_um, spacing_um) / field.h_um - 0.5
    gi = np.clip(np.round(g).astype(int), 0, n - 1)
    pts = sl[:, gi[:, None], gi[None, :]].reshape(5, -1).T
    if quant_err_at > 0:
        pts = pts + rng.normal(0, quant_err_at, pts.shape) * (pts > 0.5)
    return normalize(np.clip(pts, 0, None))


def eds_multi(field: Field, n_sections: int = 3, seed: int = 0, **kw) -> np.ndarray:
    n = field.conc.shape[1]
    zs = np.linspace(0, n - 1, n_sections + 2)[1:-1].astype(int)
    return np.vstack([eds_sample(field, z_index=int(z), seed=seed + i, **kw) for i, z in enumerate(zs)])
