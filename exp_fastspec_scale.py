import sys
import time

import numpy as np

import adrtcore as ac
import fastspec as fs
import pinv

rng = np.random.default_rng(0)
tau_fac = float(sys.argv[1])
eps = float(sys.argv[2])
for n in [int(a) for a in sys.argv[3:]]:
    N = 1 << n
    t0 = time.time()
    P = fs.FastSpectralInverse(n, tau_fac=tau_fac, eps=eps)
    ts = time.time() - t0
    x = rng.standard_normal((N, N))
    b = ac.adrt(n, x)
    bn = b + 1e-2 * rng.standard_normal((4, 2 * N - 1, N))
    P.count = [0]
    t0 = time.time()
    xl = P.apply(b)
    ta = time.time() - t0
    cnt = P.count[0]
    xln = P.apply(bn)
    t0 = time.time()
    xmp, h = pinv.cgls(n, bn, tol=1e-10, maxit=20000)
    tc = time.time() - t0
    print(
        f"N={N} tau={tau_fac}N eps={eps:g}: setup {ts:.0f}s; sum deg {P.deg.sum() / N:.0f}N, max deg {P.deg.max()}, outer deg {P.outer_deg}, kappa_out {P.hi / P.lo:.2f} | left-inv err {np.linalg.norm(xl - x) / np.linalg.norm(x):.1e} | noisy err {np.linalg.norm(xln - x) / np.linalg.norm(x):.2e} vs MP {np.linalg.norm(xmp - x) / np.linalg.norm(x):.2e} | freq-steps/apply {cnt / N:.0f} N, time {ta:.0f}s (CGLS {len(h) - 1} its, {tc:.0f}s)",
        flush=True,
    )
