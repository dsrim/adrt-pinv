"""Level SVDs in the offset-Fourier domain: transfer matrices between consecutive levels' singular bases,
and the level-product estimate of spec(M_k) vs. the true spectrum."""

import sys

import numpy as np

sys.path.insert(0, ".")


def level_matrix(n, m, z):
    N = 1 << n
    B = np.zeros((N, N), dtype=complex)
    # apply bfly single level m to unit vectors: reuse bfly code structure
    w = 1 << (m - 1)
    nstrips = N // (2 * w)
    for col in range(N):
        g = np.zeros(N, dtype=complex)
        g[col] = 1
        f4 = g.reshape(nstrips, 2, w)
        out = np.empty((nstrips, w, 2), dtype=complex)
        t = np.arange(w)
        zt = z**t
        out[:, :, 0] = f4[:, 0, :] + zt[None, :] * f4[:, 1, :]
        out[:, :, 1] = f4[:, 0, :] + (zt * z)[None, :] * f4[:, 1, :]
        B[:, col] = out.reshape(N)
    return B


n = 6
N = 1 << n
for om in (0.25, 1.0, 2.0):
    z = np.exp(1j * om)
    Bs = [level_matrix(n, m, z) for m in range(1, n + 1)]
    A = np.eye(N, dtype=complex)
    for B in Bs:
        A = B @ A
    lam = np.sort(np.linalg.eigvalsh(A.conj().T @ A))[::-1]
    # level SVDs and transfer matrices T_m = V_m^* U_{m-1}
    svds = [np.linalg.svd(B) for B in Bs]
    align = []
    for m in range(1, n):
        U_prev = svds[m - 1][0]
        V_next = svds[m][2].conj().T
        T = V_next.conj().T @ U_prev  # unitary transfer; identity = perfectly aligned
        # energy of the top-half (large-sigma) left vectors of level m-1 landing in the top-half right vectors of level m
        half = N // 2
        align.append(np.linalg.norm(T[:half, :half], "fro") ** 2 / half)
    # product-of-levels estimate (aligned bases): eigenvalues prod_m sigma_{choice}^2 over all 2^n choice strings
    c2, s2 = 4 * np.cos(om / 4) ** 2, 4 * np.sin(om / 4) ** 2
    est = np.sort(
        np.array([c2 ** (n - j) * s2**j for j in range(n + 1) for _ in range(__import__("math").comb(n, j))])
    )[::-1]
    sig = (2 * np.cos(om / 4)) ** (2 * n)
    print(
        f"omega={om}: lam_max true={lam[0]:.1f}  (2cos w/4)^2n={sig:.1f}  N/omega={N / om:.1f} | #eig > 0.1 lam_max: {np.sum(lam > 0.1 * lam[0])} (cone est N w/pi={N * om / np.pi:.1f}) | "
        f"trace={lam.sum():.0f}"
    )
    print(
        f"   alignment of large-sigma subspaces between consecutive levels (1=aligned, 0.5=random): {np.round(align, 3)}"
    )
    print(f"   true spectrum (top 8): {np.round(lam[:8], 1)}   median {np.median(lam):.2e}  min {lam[-1]:.1e}")
    print(f"   level-product estimate (top 8): {np.round(est[:8], 1)}   median {np.median(est):.2e}  min {est[-1]:.1e}")
