"""Shifted outer solve: x = (E^T F_p E + mu)^{-1} y, explicit interval [mu, 4(1+eps)+mu]. Bias and noisy errors on Shepp-Logan."""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import adrtcore as ac
import pinv
from exp_shepp import shepp_logan
from fastspec import FastSpectralInverse


def apply_shift(P, b, mu, eps=1e-10):
    y = P.backproject(b)
    lo, hi = mu, 4.0 * (1 + 1e-9) + mu
    kap = hi / lo
    D = int(np.ceil(np.log(eps / 2) / np.log((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)))) + 2
    c = (hi - lo) / (hi + lo)
    a = 2.0 / (hi + lo)
    x_prev = np.zeros_like(y)
    x = a * y
    om = 1.0
    for k in range(1, D):
        om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
        r = y - (P.frame_window(x) + mu * x)
        x_new = x + om * (a * r) + (om - 1.0) * (x - x_prev)
        x_prev, x = x, x_new
    return x, D


for n in (6, 7):
    N = 1 << n
    x = -shepp_logan(N)
    rng = np.random.default_rng(0)
    b0 = ac.adrt(n, x)
    noise = rng.standard_normal(b0.shape)
    noise *= np.linalg.norm(b0) / np.linalg.norm(noise)
    rnd = rng.standard_normal((N, N))
    rnd /= np.linalg.norm(rnd)
    brnd = ac.adrt(n, rnd)  # white-noise image: worst case for the bias
    t0 = time.time()
    P = FastSpectralInverse(n, tau_fac=0.2, eps=1e-10, lam_bound="explicit")
    print(f"N={N}: setup {time.time() - t0:.0f}s, Weinstein lo={P.lo:.3f}, D_exact={P.outer_deg}", flush=True)
    for lev in (0.0, 0.01, 0.05):
        b = b0 + lev * noise
        xe = P.apply(b)
        ee = np.linalg.norm(xe - x) / np.linalg.norm(x)
        xc, h = pinv.cgls(n, b, tol=1e-8, maxit=5000)
        ec = np.linalg.norm(xc - x) / np.linalg.norm(x)
        line = f"  noise {lev:4.0%}: exact L_p err {ee:.3e} (D={P.outer_deg}), CGLS {ec:.3e};"
        for mu in (0.1, 0.3, 0.5, 1.0):
            xm, D = apply_shift(P, b, mu)
            em = np.linalg.norm(xm - x) / np.linalg.norm(x)
            line += f" mu={mu}: err {em:.3e} (D={D})"
        print(line, flush=True)
    line = "  bias on a white-noise image ||L_mu R r - r||:"
    for mu in (0.1, 0.3, 0.5, 1.0):
        xm, D = apply_shift(P, brnd, mu)
        line += f" mu={mu}: {np.linalg.norm(xm - rnd):.3f}"
    print(line, flush=True)
