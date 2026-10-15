"""Inner filter accuracy eps_in for L_p (fastspec) and the hybrid (fasthybrid), outer always to 1e-10: cost, exactness, error/MP."""

import sys
import time

import numpy as np

import adrtcore as ac
import fasthybrid as fh
import fastspec as fs
import pinv

n = int(sys.argv[1])
N = 1 << n
eps_list = [float(a) for a in sys.argv[2:]] or [1e-10, 1e-2, 1e-1]
rng = np.random.default_rng(0)
x = rng.standard_normal((N, N))
b0 = ac.adrt(n, x)
noise = rng.standard_normal(b0.shape)
noise *= 0.01 * np.linalg.norm(b0) / np.linalg.norm(noise)
b1 = b0 + noise
xmp, _ = pinv.cgls(n, b1, tol=1e-8, maxit=20000)
e = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)
emp = e(xmp)
for eps_in in eps_list:
    for name, ctor in (
        ("L_p   ", lambda eps_in=eps_in: fs.FastSpectralInverse(n, tau_fac=0.2, eps=eps_in)),
        ("hybrid", lambda eps_in=eps_in: fh.FastHybridInverse(n, tau_fac=0.2, eps=eps_in, precond="g1")),
    ):
        t0 = time.time()
        P = ctor()
        kap = P.hi / P.lo
        P.outer_deg = int(np.ceil(np.log(1e-10 / 2) / np.log((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)))) + 2
        ts = time.time() - t0
        P.count = [0]
        t0 = time.time()
        x0 = P.apply(b0)
        ta = time.time() - t0
        x1 = P.apply(b1)
        print(
            f"N={N} eps_in={eps_in:.0e} {name}: sum d_k/N {P.deg.sum() / N:6.1f} | outer [{P.lo:.3f},{P.hi:.3f}] k={kap:.2f} D={P.outer_deg} | setup {ts:4.0f}s apply {ta:6.1f}s | LR=I {e(x0):.1e} | 1% err {e(x1) / emp:.3f} MP",
            flush=True,
        )
