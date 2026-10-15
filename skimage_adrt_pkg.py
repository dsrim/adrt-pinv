"""ADRT-shaped data from scikit-image's radon by the route of the adrt package's tomography example: radon at the ADRT
angles (coord_adrt), RectBivariateSpline interpolation at the ADRT coordinates, scaled from arc length to the ADRT's
one-sample-per-column sums, then mapped from the package's layout into this repository's (exact permutation)."""

from collections import defaultdict

import adrt
import numpy as np
from scipy import interpolate
from skimage.transform import radon

import adrtcore as ac

_maps = {}


def layout_maps(n):
    """index arrays (core_idx, pkg_idx) with core.ravel()[core_idx] == pkg.ravel()[pkg_idx] for every non-structural entry."""
    if n in _maps:
        return _maps[n]
    N = 1 << n
    F = np.random.default_rng(7).standard_normal((3, N, N))
    key = lambda v: tuple(np.round(v, 8))
    fc_ = np.stack([ac.adrt(n, f).ravel() for f in F], 1)
    fp_ = np.stack([adrt.adrt(f).ravel() for f in F], 1)
    gp, gc = defaultdict(list), defaultdict(list)
    for i, v in enumerate(fp_):
        if np.abs(v).max() > 1e-12:
            gp[key(v)].append(i)
    for i, v in enumerate(fc_):
        if np.abs(v).max() > 1e-12:
            gc[key(v)].append(i)
    assert gp.keys() == gc.keys() and all(len(gp[k]) == len(gc[k]) for k in gp)
    _maps[n] = (np.array([i for k in gc for i in gc[k]]), np.array([i for k in gc for i in gp[k]]))
    return _maps[n]


def pkg_to_core(bp, n):
    N = 1 << n
    ci, pi = layout_maps(n)
    out = np.zeros(4 * (2 * N - 1) * N)
    out[ci] = bp.ravel()[pi]
    return out.reshape(4, 2 * N - 1, N)


def core_to_pkg(bc, n):
    N = 1 << n
    ci, pi = layout_maps(n)
    out = np.zeros(4 * (2 * N - 1) * N)
    out[pi] = bc.ravel()[ci]
    return out.reshape(4, 2 * N - 1, N)


def skimage_adrt_pkg(x, n, scale="cos", circle=False):
    N = 1 << n
    coord = adrt.utils.coord_adrt(N)
    th1 = np.unique(coord.angle)
    sino = radon(x, theta=90.0 + np.rad2deg(th1), circle=circle)
    t_array = np.linspace(-0.5, 0.5, N) if circle else (np.arange(sino.shape[0]) - sino.shape[0] // 2) / N
    spline = interpolate.RectBivariateSpline(t_array, th1, sino)
    data = spline(coord.offset, np.broadcast_to(coord.angle, coord.offset.shape), grid=False)
    # outside the detector range the spline extrapolates; those lines miss the image
    data[np.abs(coord.offset) > t_array.max() + 0.5 / N] = 0.0
    a = np.abs(np.broadcast_to(coord.angle, coord.offset.shape))
    phi = np.minimum(a, np.abs(np.pi / 2 - a))  # angle from the summation axis
    f = {"none": 1.0, "cos": np.cos(phi), "sec": 1.0 / np.cos(phi)}[scale]
    return pkg_to_core(data * f, n)
