"""Filter generalization: Phi_q R_q^+ -> r(A^*A) A^*, F = sum_q A^*A r(A^*A); exact left inverse for any r.
Tikhonov r(lam) = 1/(lam+tau). Dense per-frequency, N=8,16: kappa(E^T F E), left-inverse check, rho."""

import sys
import time

import numpy as np

import adrtcore as ac
import specpart

rng = np.random.default_rng(0)


class Tikh(specpart.SpectralPartition):
    def __init__(self, n, tau):
        self.n = n
        N = self.N = 1 << n
        L = self.L = 2 * N
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        D = np.array([[ac.delta_s(n, s, t) for t in range(N)] for s in range(N)], dtype=float)
        self.P = np.empty((L, N, N), dtype=complex)
        self.PAinv = np.empty((L, N, N), dtype=complex)
        for k, om in enumerate(self.oms):
            A = np.exp(1j * om * D)
            M = A.conj().T @ A
            Minv = np.linalg.inv(M + tau * np.eye(N))
            self.P[k] = M @ Minv
            self.PAinv[k] = Minv @ A.conj().T


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]]:
        N = 1 << n
        A = ac.adrt_dense(n)
        smin = np.linalg.svd(A, compute_uv=False)[-1]
        for tau in (0.01 * N, 0.05 * N, 0.2 * N, 1.0 * N):
            t0 = time.time()
            SP = Tikh(n, tau)
            Fw = SP.dense_frame()
            Fw = 0.5 * (Fw + Fw.T)
            w = np.linalg.eigvalsh(Fw)
            Fwi = np.linalg.inv(Fw)
            x = rng.standard_normal((N, N))
            b = ac.adrt(n, x)
            li = np.linalg.norm(SP.apply(b, Fwi) - x) / np.linalg.norm(x)
            Lmat = ac.dense_from_op(lambda bb, SP=SP, Fwi=Fwi: SP.apply(bb, Fwi), (4, 2 * N - 1, N), (N, N))
            rho = np.linalg.norm(Lmat, 2) * smin
            bn = b + 1e-2 * rng.standard_normal((4, 2 * N - 1, N))
            xl = SP.apply(bn, Fwi)
            xmp = (np.linalg.pinv(A) @ bn.ravel()).reshape(N, N)
            print(
                f"N={N} tau={tau:.2f}: kappa(E^T F E)={w[-1] / w[0]:.2f} [{w[0]:.3f},{w[-1]:.3f}] | left-inv check {li:.1e} | rho={rho:.2f} | err spec {np.linalg.norm(xl - x) / np.linalg.norm(x):.2e} vs MP {np.linalg.norm(xmp - x) / np.linalg.norm(x):.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
