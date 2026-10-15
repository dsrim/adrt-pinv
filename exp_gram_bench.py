"""Normal-equation Chebyshev polynomial with the patched adrt package: R^T R by the transform pair (adrt + bdrt) vs row by row in
frequency (adrt.core.gram), same interval and degree; and the C++ oblique inverse L_p (fastcpp) for reference.  Single process;
thread count from OMP_NUM_THREADS (both package paths and fastcpp use OpenMP).  lam_max by power iteration, lam_min from the table."""

import os
import subprocess
import sys
import time

import adrt
import numpy as np

lam_min = {64: 50.3, 128: 68.0, 256: 88.69, 512: 100.0}
T = int(os.environ.get("OMP_NUM_THREADS", "1"))
PAIR_ONLY = bool(os.environ.get("PAIR_ONLY"))  # unpatched package (e.g. the PyPI wheel): time the transform pair only
for n in [int(a) for a in sys.argv[1:]] or [6, 7, 8, 9]:
    N = 1 << n
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, N))
    b = adrt.adrt(x)
    pair = lambda v: adrt.utils.truncate(adrt.bdrt(adrt.adrt(v))).sum(axis=-3)
    v = rng.standard_normal((N, N))
    t0 = time.time()
    [pair(v) for _ in range(5)]
    tp = (time.time() - t0) / 5
    if PAIR_ONLY:
        lo, hi = lam_min[N], 3.4042 * N * N * 1.001
        kap = hi / lo
        rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
        D = int(np.ceil(np.log(1e-10 / 2) / np.log(rho)))
        y = adrt.utils.truncate(adrt.bdrt(b)).sum(axis=-3)
        c, a = (hi - lo) / (hi + lo), 2 / (hi + lo)
        xp, xc, om = np.zeros_like(y), a * y, 1.0
        t0 = time.time()
        for k in range(1, D):
            om = 1 / (1 - c * c / 2) if k == 1 else 1 / (1 - c * c * om / 4)
            r = y - pair(xc)
            xp, xc = xc, xc + om * a * r + (om - 1) * (xc - xp)
        t1 = time.time() - t0
        print(
            f"N={N} T={T} [{adrt.__version__} unpatched]: one R^T R pair {tp * 1e3:8.2f} ms | Chebyshev D={D}: pair {t1:7.2f}s (LR=I {np.linalg.norm(xc - x) / np.linalg.norm(x):.1e})",
            flush=True,
        )
        continue
    t0 = time.time()
    [adrt.core.gram(v) for _ in range(5)]
    tg = (time.time() - t0) / 5
    rel = np.abs(adrt.core.gram(v) - pair(v)).max() / np.abs(pair(v)).max()
    # lam_max once (shared)
    w = rng.standard_normal((N, N))
    w /= np.linalg.norm(w)
    for _ in range(50):
        u = adrt.core.gram(w)
        w = u / np.linalg.norm(u)
    u = adrt.core.gram(w)
    th = float(np.vdot(w, u))
    lmax = th + float(np.linalg.norm(u - th * w))
    lo, hi = lam_min[N], lmax
    kap = hi / lo
    rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    D = int(np.ceil(np.log(1e-10 / 2) / np.log(rho)))

    def cheb(op, b=b, hi=hi, lo=lo, D=D):
        y = adrt.utils.truncate(adrt.bdrt(b)).sum(axis=-3)
        c, a = (hi - lo) / (hi + lo), 2 / (hi + lo)
        xp, xc, om = np.zeros_like(y), a * y, 1.0
        for k in range(1, D):
            om = 1 / (1 - c * c / 2) if k == 1 else 1 / (1 - c * c * om / 4)
            r = y - op(xc)
            xp, xc = xc, xc + om * a * r + (om - 1) * (xc - xp)
        return xc

    t0 = time.time()
    x1 = cheb(pair)
    t1 = time.time() - t0
    t0 = time.time()
    x2 = cheb(adrt.core.gram)
    t2 = time.time() - t0
    t0 = time.time()
    x3 = adrt.core.lstsq_cheb(b, lam_min=lo, lam_max=hi)
    t3 = time.time() - t0
    e = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    print(
        f"N={N} T={T}: one R^T R: pair {tp * 1e3:8.2f} ms, gram {tg * 1e3:8.2f} ms (ratio {tp / tg:.2f}, |diff| {rel:.1e}) | lam_max {lmax / N**2:.4f} N^2, D={D} | "
        f"Chebyshev: pair {t1:7.2f}s (LR=I {e(x1):.1e}), gram {t2:7.2f}s (LR=I {e(x2):.1e}), lstsq_cheb {t3:7.2f}s (LR=I {e(x3):.1e})",
        flush=True,
    )
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import adrtcore as ac
        import fastcpp as fc

        bc = ac.adrt(n, x)  # L_p uses this repository's sinogram layout, not the package's
        F = fc.FastLpCpp(n, threads=T)
        F.apply(bc)
        t0 = time.time()
        xl = F.apply(bc)
        tl = time.time() - t0
        print(
            f"N={N} T={T}: L_p cpp {tl:7.2f}s (LR=I {e(xl):.1e}, D={F.outer_deg}) -> ratio gram-Chebyshev / L_p = {t2 / tl:.2f}",
            flush=True,
        )
    except (ImportError, OSError, subprocess.SubprocessError) as ex:  # no compiler or no OpenMP
        print("fastcpp skipped:", ex, flush=True)
