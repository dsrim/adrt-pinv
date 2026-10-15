"""Certified Lambda'_k >= lambda_max(M'_k) for the upper-level Gram M' = A'^* A', A' = B_n ... B_2, by the row-sum recursion
of the companion note started at level 2 with the identity (row sums 1 for every twist) as base:
   r_m(a) <= |1+z^a| r_{m-1}(2a) + max_pm |1+z^(a pm 1)| r_{m-1}(2a pm 1),   m = 2..n,  r_1 = 1.
Checked against the true max row sum of |M'| (dense, N <= 512) and lambda_max(M') (power iteration); degree-sum cost."""

import time

import numpy as np

from fasthybrid import bfly_levels, bflyT_levels, cert_rowsum_levels

print(
    "certified Lambda' (recursion from level 2) vs true max row sum of |M'| vs lambda_max(M') ; degree sums (tau=0.2N, eps=1e-10)",
    flush=True,
)
for n in range(3, 11):
    N = 1 << n
    L = 2 * N
    t0 = time.time()
    oms = 2 * np.pi * np.arange(1, N + 1) / L
    cert = np.array([cert_rowsum_levels(n, om, 2) for om in oms])
    certfull = np.array([cert_rowsum_levels(n, om, 1) for om in oms])
    tc = time.time() - t0
    K = min(N, 48)
    sel = np.linspace(0, N - 1, K).astype(int)
    z = np.exp(1j * oms[sel])
    if N <= 512:
        Aup = np.array([bfly_levels(n, np.full(N, zk), np.eye(N, dtype=complex), 2, n).T for zk in z])  # A'_k (N x N)
        Mup = np.einsum("kji,kjl->kil", np.conj(Aup), Aup)
        rows = np.abs(Mup).sum(axis=2).max(axis=1)
        lam = np.linalg.eigvalsh(Mup)[:, -1]
    else:
        rows = np.full(K, np.nan)
        rng = np.random.default_rng(0)
        V = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
        for _ in range(80):
            V = bflyT_levels(n, z, bfly_levels(n, z, V, 2, n), 2, n)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
        lam = np.real(np.sum(np.conj(V) * bflyT_levels(n, z, bfly_levels(n, z, V, 2, n), 2, n), axis=1))
    degs = lambda Lam, N=N: np.sum(
        np.ceil(np.log(1e-10 / 2) / np.log((np.sqrt(1 + Lam / (0.2 * N)) - 1) / (np.sqrt(1 + Lam / (0.2 * N)) + 1)))
    )
    viol = np.sum(cert[sel] < rows * (1 - 1e-12)) if N <= 512 else 0
    print(
        f"N={N:4d}: cert/rowsum min={np.nanmin(cert[sel] / rows):.3f} med={np.nanmedian(cert[sel] / rows):.2f} (violations {viol}) | cert/lam_max min={np.min(cert[sel] / lam):.2f} med={np.median(cert[sel] / lam):.2f} max={np.max(cert[sel] / lam):.2f}"
        f" | sum d/N: cert' {degs(cert) / N:.0f}, 1.5*power' {degs(1.5 * np.interp(oms, oms[sel], lam)) / N:.0f}, cert full {degs(certfull) / N:.0f} | ratio cert'/cert = {degs(cert) / degs(certfull):.2f} | {tc:.1f}s",
        flush=True,
    )
