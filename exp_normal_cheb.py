"""Chebyshev semi-iteration on the normal equations R^T R x = R^T b: kappa(R^T R) from lambda_max (power iteration) and the
measured lambda_min (exp_ghost.py), and the degree D needed for 1e-10; compares with the frame-operator degrees of L_p."""

import time

import numpy as np

import adrtcore as ac

lam_min = {8: 15.47, 16: 24.50, 32: 35.67, 64: 50.3, 128: 68.0, 256: 88.69}
for n in range(3, 9):
    N = 1 << n
    rng = np.random.default_rng(0)
    v = rng.standard_normal((N, N))
    v /= np.linalg.norm(v)
    for _ in range(60):
        w = ac.adrt_T(n, ac.adrt(n, v))
        lmax = np.vdot(v, w).real
        v = w / np.linalg.norm(w)
    kap = lmax / lam_min[N]
    rate = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    D = int(np.ceil(np.log(1e-10 / 2) / np.log(rate)))
    print(
        f"N={N:4d}: lambda_max(R^T R)={lmax:9.1f} ({lmax / N**2:.3f} N^2) | lambda_min={lam_min[N]:6.2f} | kappa={kap:8.1f} ({kap / N**1.5:.3f} N^1.5) | Chebyshev degree for 1e-10: D={D:5d} ({D / N**0.75:.2f} N^0.75)",
        flush=True,
    )

# ---- run the fixed-degree semi-iteration x_D = c_D(R^T R) R^T b with the interval [lambda_min, lambda_max] (data independent)
import pinv

for n in (6, 7, 8):
    N = 1 << n
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, N))
    b0 = ac.adrt(n, x)
    noise = rng.standard_normal(b0.shape)
    noise *= 0.01 * np.linalg.norm(b0) / np.linalg.norm(noise)
    b1 = b0 + noise
    lo, hi = lam_min[N], 3.404 * N * N
    kap = hi / lo
    rate = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    D = int(np.ceil(np.log(1e-10 / 2) / np.log(rate)))

    def cheb_normal(b, n=n, hi=hi, lo=lo, D=D):
        y = ac.adrt_T(n, b)
        c = (hi - lo) / (hi + lo)
        a = 2.0 / (hi + lo)
        xp = np.zeros_like(y)
        xc = a * y
        om = 1.0
        for k in range(1, D):
            om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
            r = y - ac.adrt_T(n, ac.adrt(n, xc))
            xn = xc + om * (a * r) + (om - 1.0) * (xc - xp)
            xp, xc = xc, xn
        return xc

    t0 = time.time()
    x0 = cheb_normal(b0)
    ta = time.time() - t0
    x1 = cheb_normal(b1)
    xmp, h = pinv.cgls(n, b1, tol=1e-8, maxit=20000)
    e = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    print(
        f"N={N}: D={D} | apply {ta:.1f}s ({D} ADRT+transpose pairs; CGLS to 1e-8: {len(h) - 1} its) | c_D(R^TR)R^TR x - x: {e(x0):.1e} | 1% noise err {e(x1):.3e} = {e(x1) / e(xmp):.3f} MP",
        flush=True,
    )
