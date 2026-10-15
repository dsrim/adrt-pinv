"""Scaling of the fast hybrid inverse (fasthybrid.FastHybridInverse) against the original fast oblique inverse
(fastspec.FastSpectralInverse): setup, degrees, outer interval/degree, cost in level-row units, accuracy on a random
image at 0 and 1% noise relative to the Moore-Penrose solution (CGLS to 1e-8), wall time."""

import sys
import time

import numpy as np

import adrtcore as ac
import fasthybrid as fh
import fastspec as fs
import pinv

ns = [int(a) for a in sys.argv[1:]] or [6, 7, 8]
tf = 0.2
for n in ns:
    N = 1 << n
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, N))
    b0 = ac.adrt(n, x)
    noise = rng.standard_normal(b0.shape)
    noise *= 0.01 * np.linalg.norm(b0) / np.linalg.norm(noise)
    b1 = b0 + noise
    t0 = time.time()
    xmp, h = pinv.cgls(n, b1, tol=1e-8, maxit=20000)
    tmp = time.time() - t0
    emp = np.linalg.norm(xmp - x) / np.linalg.norm(x)
    print(f"N={N}: MP (CGLS {len(h) - 1} its, {tmp:.0f}s) err at 1% noise {emp:.3e}", flush=True)
    t0 = time.time()
    O = fs.FastSpectralInverse(n, tau_fac=tf, eps=1e-10)
    ts = time.time() - t0
    O.count = [0]
    t0 = time.time()
    xo0 = O.apply(b0)
    ta = time.time() - t0
    co = O.count[0] * 2 * n
    xo1 = O.apply(b1)
    e = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    print(
        f"  orig   : setup {ts:5.0f}s | sum deg/N {O.deg.sum() / N:6.1f} | outer [{O.lo:.3f},{O.hi:.3f}] k={O.hi / O.lo:.2f} deg {O.outer_deg:3d} | "
        f"cost {co / (N * N * n):7.1f} N^2 log N | apply {ta:5.0f}s | LR=I {e(xo0):.1e} | 1% err {e(xo1):.3e} = {e(xo1) / emp:.2f} MP",
        flush=True,
    )
    for pc in ("g1", "lorr"):
        t0 = time.time()
        H = fh.FastHybridInverse(n, tau_fac=tf, eps=1e-10, precond=pc)
        ts = time.time() - t0
        H.count = [0]
        t0 = time.time()
        xh0 = H.apply(b0)
        ta = time.time() - t0
        ch = H.count[0]
        xh1 = H.apply(b1)
        print(
            f"  hyb/{pc:4s}: setup {ts:5.0f}s | sum deg/N {H.deg.sum() / N:6.1f} | outer [{H.lo:.3f},{H.hi:.3f}] k={H.hi / H.lo:.2f} deg {H.outer_deg:3d} | "
            f"cost {ch / (N * N * n):7.1f} N^2 log N | apply {ta:5.0f}s | LR=I {e(xh0):.1e} | 1% err {e(xh1):.3e} = {e(xh1) / emp:.2f} MP",
            flush=True,
        )
