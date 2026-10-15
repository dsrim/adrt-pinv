"""Cost of L_p as a function of the INNER filter accuracy eps_in (outer polynomial always to 1e-10): the left inverse is exact
for every inner degree, so eps_in only changes the oblique projection (error ratio to MP) and the outer interval."""

import sys
import time

import numpy as np

import adrtcore as ac
import fastspec as fs
import pinv

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = 1 << n
rng = np.random.default_rng(0)
x = rng.standard_normal((N, N))
b0 = ac.adrt(n, x)
noise = rng.standard_normal(b0.shape)
noise *= 0.01 * np.linalg.norm(b0) / np.linalg.norm(noise)
b1 = b0 + noise
xmp, _ = pinv.cgls(n, b1, tol=1e-8, maxit=20000)
e = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)
emp = e(xmp)
for eps_in in (1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
    t0 = time.time()
    P = fs.FastSpectralInverse(n, tau_fac=0.2, eps=eps_in)  # inner degrees from eps_in
    kap = P.hi / P.lo
    P.outer_deg = (
        int(np.ceil(np.log(1e-10 / 2) / np.log((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)))) + 2
    )  # outer to 1e-10
    ts = time.time() - t0
    P.count = [0]
    t0 = time.time()
    x0 = P.apply(b0)
    ta = time.time() - t0
    x1 = P.apply(b1)
    print(
        f"N={N} eps_in={eps_in:.0e}: sum d_k/N {P.deg.sum() / N:6.1f} | outer [{P.lo:.3f},{P.hi:.3f}] k={kap:.2f} D={P.outer_deg} | apply {ta:5.1f}s | LR=I {e(x0):.1e} | 1% err {e(x1) / emp:.3f} MP",
        flush=True,
    )
