import sys
import time

import numpy as np

sys.path.insert(0, ".")
from fastspec import bfly, bflyT


def prodkernel(n, om):
    N = 1 << n
    t = np.arange(N)
    P = np.ones((N, N))
    aprev = np.zeros((N, N), dtype=int)  # alpha_0 = 0
    for m in range(1, n + 1):
        a = (t[None, :] >> (n - m)) - (t[:, None] >> (n - m))
        P *= 2 * np.abs(np.cos(om * (a - aprev) / 2))
        aprev = a
    return P


def cert_rowsum(n, om):
    """Certified upper bound on max_t sum_t' |M_omega(t,t')| via r_m(alpha) <= |1+z^a| r_{m-1}(2a) + max(|1+z^(a+1)| r_{m-1}(2a+1), |1+z^(a-1)| r_{m-1}(2a-1)).
    Level m needs twists |alpha| <= 2^(n-m)-1; level 0 is identically 1."""
    z = np.exp(1j * om)
    A = 1 << n
    r = np.ones(2 * A + 1)
    for m in range(1, n + 1):
        amax = (1 << (n - m)) - 1
        a = np.arange(-amax, amax + 1)
        t0 = np.abs(1 + z**a) * r[2 * a + A]
        tp = np.abs(1 + z ** (a + 1)) * r[2 * a + 1 + A]
        tm = np.abs(1 + z ** (a - 1)) * r[2 * a - 1 + A]
        rn = np.zeros(2 * A + 1)
        rn[a + A] = t0 + np.maximum(tp, tm)
        r = rn
    return r[A]


print(
    "certified Lambda (recursive row-sum bound) vs true max row sum vs lambda_max vs explicit formula (26)", flush=True
)
for n in range(3, 11):
    N = 1 << n
    L = 2 * N
    t0 = time.time()
    oms = 2 * np.pi * np.arange(1, N + 1) / L
    cert = np.array([cert_rowsum(n, om) for om in oms])
    tc = time.time() - t0
    K = min(N, 48)
    sel = np.linspace(0, N - 1, K).astype(int)
    rows = np.array([prodkernel(n, om).sum(axis=1).max() for om in oms[sel]]) if N <= 512 else np.full(K, np.nan)
    z = np.exp(1j * oms[sel])
    rng = np.random.default_rng(0)
    V = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
    for _ in range(80):
        V = bflyT(n, z, bfly(n, z, V))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    lam = np.real(np.sum(np.conj(V) * bflyT(n, z, bfly(n, z, V)), axis=1))
    expl = 2.0 * N * (1 + n) / oms[sel]
    degs = lambda Lam, N=N: np.sum(np.sqrt(1 + Lam / (0.2 * N)))
    print(
        f"N={N:4d}: cert/rowsum max={np.nanmax(cert[sel] / rows):.2f} med={np.nanmedian(cert[sel] / rows):.2f} | cert/lam_max max={np.max(cert[sel] / lam):.2f} med={np.median(cert[sel] / lam):.2f} | cert/explicit max={np.max(cert[sel] / expl):.2f} med={np.median(cert[sel] / expl):.2f} | degree-sum ratio cert/explicit={degs(cert) / degs(np.minimum(2.0 * N * (1 + n) / oms, N**2)):.2f} | cert time {tc:.1f}s",
        flush=True,
    )
