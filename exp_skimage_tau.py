"""Model-mismatch error versus the Tikhonov parameter tau: data from skimage's radon on the ADRT lines (exp_skimage_data.py),
no added noise and 1% noise; L_p and hybrid (compiled) for tau = 0.2N, N, 5N, 25N against CGLS (least squares)."""

import numpy as np

import fastcpp as fc
import pinv
from exp_shepp import shepp_logan
from exp_skimage_data import skimage_adrt

n = 7
N = 1 << n
conv = (-1, -1, 0.5)
i, j = np.mgrid[0:N, 0:N]
phantoms = {
    "smooth": (
        np.exp(-(((i - 0.45 * N) ** 2 + (j - 0.42 * N) ** 2) / (0.15 * N) ** 2))
        + 0.7 * np.exp(-(((i - 0.7 * N) ** 2 + (j - 0.65 * N) ** 2) / (0.07 * N) ** 2))
        + 0.5 * np.exp(-(((i - 0.25 * N) ** 2 + (j - 0.72 * N) ** 2) / (0.1 * N) ** 2))
    ),
    "shepp": -shepp_logan(N),
}
for name, x in phantoms.items():
    b_sk = skimage_adrt(x, n, conv)
    err = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    nz = np.random.default_rng(0).standard_normal(b_sk.shape)
    nz *= np.linalg.norm(b_sk) / np.linalg.norm(nz)
    for lev in (0.0, 0.01):
        b = b_sk + lev * nz
        xc, _ = pinv.cgls(n, b, tol=1e-10, maxit=50000)
        ec = err(xc)
        line = f"{name:6s} noise {lev:4.0%}: CGLS {ec:.3e} |"
        for tf in (0.2, 1.0, 5.0, 25.0):
            el = err(fc.FastLpCpp(n, tau_fac=tf, eps_in=0.1, threads=1).apply(b))
            eh = err(fc.FastHybridCpp(n, tau_fac=tf, eps_in=0.1, threads=1).apply(b))
            line += f" tau={tf:g}N: L_p {el / ec:.2f}x hyb {eh / ec:.2f}x |"
        print(line, flush=True)
print("TAU DONE", flush=True)
