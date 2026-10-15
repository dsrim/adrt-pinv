"""Gram kernel of the offset-Fourier butterfly: product formula, resonance at omega=2pi/3, row sums vs. the explicit bound (26).
Run from code/: ../.venv/bin/python exp_kernel_resonance.py"""

"""Check (1) the product formula |M_w(t,t')| = prod_m 2|cos(w e_m/2)|, e_m = alpha_{m+1}-alpha_m,
alpha_m = floor(t'/2^{n-m}) - floor(t/2^{n-m}); (2) per-pair bound |M| <= C N/(w|d|); (3) row sums vs 2N(1+log2N)/w;
(4) lambda_max vs row-sum (Schur) bound."""
import sys

import numpy as np

sys.path.insert(0, ".")
import adrtcore as ac
from fastspec import bfly, bflyT


def prodkernel(n, om):
    N = 1 << n
    t = np.arange(N)
    P = np.ones((N, N))
    aprev = np.zeros((N, N), dtype=int)  # alpha_0 = 0
    for m in range(1, n + 1):
        a = (t[None, :] >> (n - m)) - (t[:, None] >> (n - m))  # alpha_m(t,t')
        e = a - aprev
        P *= 2 * np.abs(np.cos(om * e / 2))
        aprev = a
    return P


# (1) verify against dense A^*A for small n
for n in (3, 4, 5):
    N = 1 << n
    for om in (0.3, 1.0, 2.5):
        s = np.arange(N)
        t = np.arange(N)
        A = np.exp(1j * om * np.array([[ac.delta_s(n, ss, tt) for tt in t] for ss in s]))
        M = A.conj().T @ A
        print(f"n={n} om={om}: max |abs(M) - prod| = {np.max(np.abs(np.abs(M) - prodkernel(n, om))):.2e}")

# (2)-(4)
for n in range(3, 11):
    N = 1 << n
    L = 2 * N
    oms = 2 * np.pi * np.arange(1, N + 1) / L  # omega in (0, pi]
    r_pair = 0.0
    r_row = 0.0
    r_row_small = 0.0
    worst = None
    for om in oms:
        P = prodkernel(n, om)
        d = np.abs(np.arange(N)[None, :] - np.arange(N)[:, None]).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(d > 0, P * om * d / N, 0.0)
        if q.max() > r_pair:
            r_pair = q.max()
            worst = (om, np.unravel_index(q.argmax(), q.shape))
        rs = P.sum(axis=1).max()
        r_row = max(r_row, rs * om / (2 * N * (1 + n)))
    # lambda_max by power iteration at a few frequencies vs row-sum bound
    K = min(64, N)
    z = np.exp(1j * oms[:K])
    rng = np.random.default_rng(0)
    V = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
    for _ in range(60):
        V = bflyT(n, z, bfly(n, z, V))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    lam = np.real(np.sum(np.conj(V) * bflyT(n, z, bfly(n, z, V)), axis=1))
    rows = np.array([prodkernel(n, om).sum(axis=1).max() for om in oms[:K]])
    print(
        f"N={N:5d}: max |M| w|d|/N = {r_pair:.3f} at om={worst[0]:.3f},(t,t')={worst[1]} | "
        f"max rowsum*w/(2N(1+n)) = {r_row:.3f} | max lam_max/rowsum = {np.max(lam / rows):.3f}, "
        f"max lam_max w/(2N(1+n)) = {np.max(lam * oms[:K] / (2 * N * (1 + n))):.3f}"
    )
