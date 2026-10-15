"""Filter on levels m0+1..n only, fusion through S_(1..m0)^T (no explicit preconditioner for m0 >= 2)."""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from lab_hybrid import HybridLab, level_matrix

for n in (3, 4, 5):
    N = 1 << n
    for m0 in (1, 2, 3):
        t0 = time.time()
        lab = HybridLab(n, 0.2)
        L = lab.L
        z = np.exp(1j * lab.oms)
        Blow = np.array([np.eye(N, dtype=complex) for _ in z])
        Aup = np.array([np.eye(N, dtype=complex) for _ in z])
        for m in range(1, n + 1):
            Bm = np.array([level_matrix(n, m, zk) for zk in z])
            if m <= m0:
                Blow = np.einsum("kij,kjl->kil", Bm, Blow)
            else:
                Aup = np.einsum("kij,kjl->kil", Bm, Aup)
        Mup = np.einsum("kji,kjl->kil", np.conj(Aup), Aup)
        lam, V = np.linalg.eigh(Mup)
        lam = np.maximum(lam, 0)
        f = lam / (lam + lab.tau)
        Phiup = np.einsum("kij,kj,klj->kil", V, f, np.conj(V))
        BH = np.conj(np.transpose(Blow, (0, 2, 1)))
        lab.X["m0"] = np.einsum("kij,kjl,klm->kim", BH, Phiup, Blow)
        lab.X["glow"] = np.einsum("kij,kjl->kil", BH, Blow)
        K = lab.dense("m0")
        G = lab.dense("glow")
        import scipy.linalg as sla

        ev = np.linalg.eigvalsh(K)
        evp = sla.eigh(K, G, eigvals_only=True)
        deg = (
            np.sum(
                np.maximum(
                    2,
                    np.ceil(
                        np.log(5e-11)
                        / np.log((np.sqrt(1 + lam[:, -1] / lab.tau) - 1) / (np.sqrt(1 + lam[:, -1] / lab.tau) + 1))
                    ),
                )
            )
            / N
        )
        print(
            f"N={N} m0={m0}: unpreconditioned kappa={ev[-1] / ev[0]:.2f}  preconditioned by (S_(1..m0)^T S_(1..m0))^-1 kappa={evp[-1] / evp[0]:.2f}  sum d_k/N={deg:.0f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
