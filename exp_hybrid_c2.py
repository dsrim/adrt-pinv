"""Single-thread comparison on compute2: hybrid (C++ engine, G_1^{-1} degree 6, dense table none/16/8) vs L_p (C++ engine)
vs the normal-equation Chebyshev polynomial with the PyPI adrt wheel (transform pair).  Methods interleaved, min over reps.
Accuracy: left-inverse error on noiseless data; error on 1% noise relative to the least-squares solution, taken as the
normal-equation polynomial's result (accurate to 1e-10)."""

import os
import sys
import time

import adrt
import numpy as np

import adrtcore as ac
import fastcpp as fc

lam_min = {64: 50.3, 128: 68.0, 256: 88.69, 512: 100.0}
for n in [int(a) for a in sys.argv[1:]] or [6, 7, 8, 9]:
    N = 1 << n
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, N))
    bc0 = ac.adrt(n, x)
    nz = rng.standard_normal(bc0.shape)
    bc1 = bc0 + 0.01 * np.linalg.norm(bc0) / np.linalg.norm(nz) * nz
    # same data in the package's layout: the package's ADRT of x, plus the package-layout image of the same noise
    bp0 = adrt.adrt(x)
    nzp = rng.standard_normal(bp0.shape)
    bp1 = bp0 + 0.01 * np.linalg.norm(bp0) / np.linalg.norm(nzp) * nzp
    pair = lambda v: adrt.utils.truncate(adrt.bdrt(adrt.adrt(v))).sum(axis=-3)
    lo, hi = lam_min[N], 3.4042 * N * N * 1.001
    kap = hi / lo
    rho = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
    D = int(np.ceil(np.log(1e-10 / 2) / np.log(rho)))

    def normal(b, hi=hi, lo=lo, D=D, pair=pair):
        y = adrt.utils.truncate(adrt.bdrt(b)).sum(axis=-3)
        c, a = (hi - lo) / (hi + lo), 2 / (hi + lo)
        xp, xc, om = np.zeros_like(y), a * y, 1.0
        for k in range(1, D):
            om = 1 / (1 - c * c / 2) if k == 1 else 1 / (1 - c * c * om / 4)
            r = y - pair(xc)
            xp, xc = xc, xc + om * a * r + (om - 1) * (xc - xp)
        return xc

    meth = {"normal eq (PyPI adrt)": (normal, bp0, bp1)}
    objs = {
        "L_p cpp": fc.FastLpCpp(n, threads=1),
        "hybrid cpp g1=6": fc.FastHybridCpp(n, threads=1, g1_deg=6),
        "hybrid cpp g1=6 dense16": fc.FastHybridCpp(n, threads=1, g1_deg=6, dense_cap=16),
        "hybrid cpp g1=6 dense8": fc.FastHybridCpp(n, threads=1, g1_deg=6, dense_cap=8),
    }
    for k, P in objs.items():
        meth[k] = (P.apply, bc0, bc1)
    reps = 3 if n < 9 else 1
    T = {k: [] for k in meth}
    for _ in range(reps):
        for k, (f, b0, b1) in meth.items():
            t0 = time.time()
            f(b0)
            T[k].append(time.time() - t0)
    xls_c = None
    e = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    xls = normal(bp1)
    els = e(xls)  # least-squares error on the package-layout noise
    print(
        f"N={N}: load {os.getloadavg()[0]:.2f}; normal-eq degree D={D}; least-squares error at 1% noise {els:.3e}",
        flush=True,
    )
    for k, (f, b0, b1) in meth.items():
        extra = f" D={objs[k].outer_deg} dense rows {objs[k].eng.n_dense}" if k in objs else ""
        print(
            f"   {k:26s}: {min(T[k]):8.3f}s (max {max(T[k]):8.3f}) | LR=I {e(f(b0)):.1e} | 1% err {e(f(b1)):.3e} |{extra}",
            flush=True,
        )
