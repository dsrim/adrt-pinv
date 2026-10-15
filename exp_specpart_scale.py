"""Spectral-partition left inverse at larger N: kappa(E^T F E) by Lanczos, exact left-inverse check,
error on 1% noise vs MP (CGLS), and the non-iterative fixed-degree Chebyshev inverse of E^T F E."""

import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import adrtcore as ac
import pinv
import specpart

rng = np.random.default_rng(0)


def cheb_solve(op, y, lo, hi, deg):
    """Chebyshev semi-iteration (fixed degree = non-iterative polynomial in op) for spectrum in [lo,hi]."""
    c = (hi - lo) / (hi + lo)
    a = 2 / (hi + lo)
    x_prev = np.zeros_like(y)
    x = a * y
    om = 1.0
    for k in range(1, deg):
        om = 1 / (1 - c * c / 2) if k == 1 else 1 / (1 - c * c * om / 4)
        r = y - op(x)
        x_new = x + om * (a * r) + (om - 1) * (x - x_prev)
        x_prev, x = x, x_new
    return x


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]]:
        N = 1 << n
        t0 = time.time()
        SP = specpart.SpectralPartition(n, 0.05)
        ts = time.time() - t0
        Fop = spla.LinearOperator(
            (N * N, N * N), matvec=lambda v, SP=SP, N=N: np.real(SP.frame_window(v.reshape(N, N))).ravel()
        )
        lo = spla.eigsh(Fop, k=1, which="SA", tol=1e-4)[0][0]
        hi = spla.eigsh(Fop, k=1, which="LA", tol=1e-4)[0][0]
        x = rng.standard_normal((N, N))
        b = ac.adrt(n, x)
        bn = b + 1e-2 * rng.standard_normal((4, 2 * N - 1, N))
        y = np.real(SP.quadrant_inverse(b)[:N, :N])
        yn = np.real(SP.quadrant_inverse(bn)[:N, :N])
        op = lambda v, SP=SP: np.real(SP.frame_window(v))
        out = [f"N={N}: setup {ts:.0f}s, kappa(E^T F E)={hi / lo:.2f} [{lo:.3f},{hi:.3f}]"]
        xmp, h = pinv.cgls(n, bn, tol=1e-10, maxit=20000)
        emp = np.linalg.norm(xmp - x) / np.linalg.norm(x)
        for deg in (10, 20, 40):
            xl = cheb_solve(op, y, lo, hi, deg)
            xln = cheb_solve(op, yn, lo, hi, deg)
            out.append(
                f"cheb deg {deg}: left-inv err {np.linalg.norm(xl - x) / np.linalg.norm(x):.1e}, noisy err {np.linalg.norm(xln - x) / np.linalg.norm(x):.2e} (MP {emp:.2e})"
            )
        print(" | ".join(out), f"({time.time() - t0:.0f}s)", flush=True)
