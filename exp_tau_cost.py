"""Compiled cost versus tau (one core): L_p and hybrid at eps_in = 0.1, outer 1e-10, N = 128, 256; degrees and apply time."""

import time

import numpy as np

import adrtcore as ac
import fastcpp as fc

for n in (7, 8):
    N = 1 << n
    x = np.random.default_rng(0).standard_normal((N, N))
    b = ac.adrt(n, x)
    for tf in (0.2, 1.0, 5.0, 25.0):
        for name, P in (
            ("L_p", fc.FastLpCpp(n, tau_fac=tf, eps_in=0.1, threads=1)),
            ("hybrid", fc.FastHybridCpp(n, tau_fac=tf, eps_in=0.1, threads=1)),
        ):
            P.apply(b)
            ts = []
            for _ in range(2 if n < 8 else 1):
                t0 = time.time()
                y = P.apply(b)
                ts.append(time.time() - t0)
            print(
                f"N={N} tau={tf:g}N {name:6s}: sum d_k/N {P.deg.sum() / N:6.1f} D={P.outer_deg:3d} [{P.lo:.4f},{P.hi:.3f}] | apply {min(ts):7.3f}s | LR=I {np.linalg.norm(y - x) / np.linalg.norm(x):.1e}",
                flush=True,
            )
print("COST DONE", flush=True)
