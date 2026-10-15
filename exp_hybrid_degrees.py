import sys
import time

import numpy as np

sys.path.insert(0, ".")
from lab_frame import delta_table
from lab_hybrid import level_matrix

print(
    "upper-level Gram M' = A'^* A' (levels 2..n) vs full Gram M: lambda_max ratio and Chebyshev degree sums (eps=1e-10)"
)
for n in range(3, 9):
    N = 1 << n
    L = 2 * N
    oms = 2 * np.pi * np.fft.fftfreq(L)
    z = np.exp(1j * oms)
    t0 = time.time()
    D = delta_table(n, "digital")
    A = np.exp(1j * oms[:, None, None] * D[None])
    Aup = np.array([np.eye(N, dtype=complex) for _ in z])
    for m in range(2, n + 1):
        Bm = np.array([level_matrix(n, m, zk) for zk in z])
        Aup = np.einsum("kij,kjl->kil", Bm, Aup)
    lam = np.linalg.eigvalsh(np.einsum("kji,kjl->kil", np.conj(A), A))[:, -1]
    lamu = np.linalg.eigvalsh(np.einsum("kji,kjl->kil", np.conj(Aup), Aup))[:, -1]
    line = f"N={N:4d}: lam_max(M')/lam_max(M): min={np.min(lamu / lam):.2f} med={np.median(lamu / lam):.2f} max={np.max(lamu / lam):.2f} (at k=0: {lamu[0] / lam[0]:.2f}; small w: {np.median((lamu / lam)[np.abs(oms) < 0.3]):.2f}; large w: {np.median((lamu / lam)[np.abs(oms) > 2]):.2f})"
    for tf in (0.2, 1.0):
        tau = tf * N
        deg = lambda Lam, tau=tau: np.sum(
            np.maximum(2, np.ceil(np.log(5e-11) / np.log((np.sqrt(1 + Lam / tau) - 1) / (np.sqrt(1 + Lam / tau) + 1))))
        )
        line += f" | tau={tf}N: sum d_k/N full={deg(lam) / N:.0f} upper={deg(lamu) / N:.0f}"
    print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
