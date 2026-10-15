"""Compiled (C++ engine, one core) re-measurement of every wall time in the paper, at the settings of each table:
  A  certified bounds (eq. cert): all 2N frequencies at N = 1024, C++ recursion (checked against the numpy reference)
  B  setup-choice table: setup time of L_p for power/Weinstein, explicit/Weinstein, explicit/Weinstein tau=N (eps_in = 1e-10)
  C  fast-hybrid table: apply time at eps_in = 1e-10 for L_p, hybrid G_1^{-1}, hybrid (4I+8Pi)^{-1}
  D  inner-tolerance table: apply time for eps_in = 1e-10, 1e-2, 1e-1 (L_p, hybrid G_1^{-1})
Errors relative to the Moore-Penrose solution (CGLS to 1e-10) on 1% white noise, as in the paper."""

import os
import sys
import time

import numpy as np

import adrtcore as ac
import fastcpp as fc
import fasthybrid as fh
import pinv


def tmin(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.time()
        f()
        ts.append(time.time() - t0)
    return min(ts)


part = sys.argv[1] if len(sys.argv) > 1 else "ABCD"
print(f"host load {os.getloadavg()[0]:.2f}", flush=True)
if "A" in part:
    n = 10
    N = 1 << n
    oms = 2 * np.pi * np.fft.fftfreq(2 * N)
    t = tmin(lambda: fc.cert_lam_max(n, oms, 1, 1), 3)
    cert = fc.cert_lam_max(n, oms, 1, 1)
    ref = np.array([fh.cert_rowsum_levels(n, om, 1) for om in oms[1:N:97]])
    print(
        f"A N={N}: certified bounds for all {2 * N} frequencies: {t:.4f} s (C++, 1 core) | max rel diff vs numpy reference {np.abs(cert[1:N:97] - ref).max() / ref.max():.1e}",
        flush=True,
    )
data = {}


def get(n):
    if n not in data:
        N = 1 << n
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, N))
        b0 = ac.adrt(n, x)
        nz = rng.standard_normal(b0.shape)
        b1 = b0 + 0.01 * np.linalg.norm(b0) / np.linalg.norm(nz) * nz
        xmp, _ = pinv.cgls(n, b1, tol=1e-10, maxit=50000)
        data[n] = (x, b0, b1, np.linalg.norm(xmp - x) / np.linalg.norm(x))
    return data[n]


e = lambda y, x: np.linalg.norm(y - x) / np.linalg.norm(x)
if "B" in part:
    for n in (5, 6, 7):
        N = 1 << n
        for name, kw in (
            ("power / Weinstein", {"lam_bound": "power", "tau_fac": 0.2}),
            ("explicit / Weinstein", {"lam_bound": "explicit", "tau_fac": 0.2}),
            ("explicit / Weinstein, tau=N", {"lam_bound": "explicit", "tau_fac": 1.0}),
        ):
            ts = []
            for _ in range(2):
                t0 = time.time()
                P = fc.FastLpCpp(n, eps_in=1e-10, threads=1, **kw)
                ts.append(time.time() - t0)
            print(
                f"B N={N} {name:28s}: setup {min(ts):7.3f} s | sum d/N {P.deg.sum() / N:6.1f} D={P.outer_deg}",
                flush=True,
            )
if "C" in part:
    for n in (6, 7, 8):
        N = 1 << n
        x, b0, b1, emp = get(n)
        reps = 3 if n < 8 else 2
        for name, ctor in (
            ("L_p", lambda n=n: fc.FastLpCpp(n, eps_in=1e-10, threads=1)),
            ("hybrid G1", lambda n=n: fc.FastHybridCpp(n, eps_in=1e-10, threads=1, precond="g1")),
            ("hybrid 4I+8Pi", lambda n=n: fc.FastHybridCpp(n, eps_in=1e-10, threads=1, precond="lorr")),
        ):
            P = ctor()
            t = tmin(lambda P=P, b0=b0: P.apply(b0), reps)
            print(
                f"C N={N} {name:14s}: apply {t:8.3f} s | sum d/N {P.deg.sum() / N:6.1f} | [{P.lo:.3f},{P.hi:.3f}] D={P.outer_deg} | LR=I {e(P.apply(b0), x):.1e} | err/MP {e(P.apply(b1), x) / emp:.3f}",
                flush=True,
            )
if "D" in part:
    for n in (6, 7, 8):
        N = 1 << n
        x, b0, b1, emp = get(n)
        reps = 3 if n < 8 else 2
        for eps_in in (1e-10, 1e-2, 1e-1):
            for name, ctor in (
                ("L_p", lambda n=n, eps_in=eps_in: fc.FastLpCpp(n, eps_in=eps_in, threads=1)),
                ("hybrid", lambda n=n, eps_in=eps_in: fc.FastHybridCpp(n, eps_in=eps_in, threads=1)),
            ):
                P = ctor()
                t = tmin(lambda P=P, b0=b0: P.apply(b0), reps)
                print(
                    f"D N={N} eps_in={eps_in:.0e} {name:7s}: apply {t:8.3f} s | sum d/N {P.deg.sum() / N:6.1f} D={P.outer_deg} | LR=I {e(P.apply(b0), x):.1e} | err/MP {e(P.apply(b1), x) / emp:.3f}",
                    flush=True,
                )
print("COMPILED TABLES DONE", flush=True)
