"""Outer preconditioners for the hybrid operator G_h = S_(1)^T (I x M'p(M')) S_(1):
   G_1^{-1}            exact level-1 Gram inverse, G_1 = 8I + K (x) J + J (x) K, spec in [4,16]
   (4I + 8 Pi)^{-1}    LORR's truncated level-1 Gram (S_(1)^T M S_(1) = 4I + 8 Pi, Pi = 2x2-cell averaging), Thm 5.1 trick
   none
Generalized condition numbers kappa(P G_h), dense, N = 8..32 (64 optional)."""

import sys
import time

import numpy as np
import scipy.linalg as sla

from lab_hybrid import HybridLab


def Pi_mat(N):
    C = np.kron(np.eye(N // 2), np.ones((2, 2)) / 2)
    return np.kron(C, C)


ns = [int(a) for a in sys.argv[1:]] or [3, 4, 5]
for n in ns:
    N = 1 << n
    for tf in (0.2, 1.0):
        t0 = time.time()
        lab = HybridLab(n, tf)
        Gh, G1 = lab.dense("hyb"), lab.dense("g1")
        GM = 4 * np.eye(N * N) + 8 * Pi_mat(N)
        e0 = np.linalg.eigvalsh(Gh)
        e1 = sla.eigh(Gh, G1, eigvals_only=True)
        eM = sla.eigh(Gh, GM, eigvals_only=True)
        print(
            f"N={N:3d} tau={tf}N: kappa(G_h)={e0[-1] / e0[0]:.2f} | G_1^-1 G_h: [{e1[0]:.4f},{e1[-1]:.4f}] k={e1[-1] / e1[0]:.2f}"
            f" | (4I+8Pi)^-1 G_h: [{eM[0]:.4f},{eM[-1]:.4f}] k={eM[-1] / eM[0]:.2f} | spec G_1 [{np.linalg.eigvalsh(G1)[0]:.4f},{np.linalg.eigvalsh(G1)[-1]:.4f}]  ({time.time() - t0:.0f}s)",
            flush=True,
        )
