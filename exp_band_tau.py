"""Frequency-dependent Tikhonov: tau_k = tau0 in a band |omega_k -+ 2pi/3| <= w, tau elsewhere. lambda_min(F) and degree cost."""

import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
from lab_frame import FrameLab


class FrameLabTau(FrameLab):
    def __init__(self, n, tau_arr, lines="digital"):
        FrameLab.__init__(self, n, 1.0, lines, 2 * (1 << n), "ident")
        f = self.lam / (self.lam + tau_arr[:, None])
        self.f = f
        self.phi = np.einsum("kij,kj,klj->kil", self.V, f, np.conj(self.V))


for n in (5, 6, 7):
    N = 1 << n
    L = 2 * N
    oms = 2 * np.pi * np.fft.fftfreq(L)
    t0 = time.time()
    Lam = np.minimum(float(N) ** 2, 2.0 * N * (1 + n) / np.maximum(np.abs(oms), 1e-9))
    Lam[0] = N**2
    for tau_fac in (1.0, 0.2):
        base = tau_fac * N
        for w_grid, tau0 in ((0, None), (2, 1.0), (4, 1.0), (8, 1.0), (4, N**0.5), (16, N**0.5)):
            tau = np.full(L, base)
            if w_grid > 0:
                band = np.abs(np.abs(oms) - 2 * np.pi / 3) <= w_grid * 2 * np.pi / L + 1e-12
                tau[band] = tau0
            lab = FrameLabTau(n, tau)
            op = spla.LinearOperator(
                (N * N, N * N), matvec=lambda v, lab=lab, N=N: lab.apply(v.reshape(N, N)).ravel(), dtype=float
            )
            w, _ = spla.eigsh(op, k=1, which="SA", tol=1e-6, maxiter=5000)
            wmax, _ = spla.eigsh(op, k=1, which="LA", tol=1e-6, maxiter=5000)
            deg = np.sum(np.sqrt(1 + Lam / tau)) / N
            print(
                f"N={N} tau={tau_fac}N band=+-{w_grid} grid, tau0={tau0}: lam_min(F)={w[0]:.4f} lam_max={wmax[0]:.3f} kappa={wmax[0] / w[0]:.2f}  sum sqrt(1+Lam/tau)/N={deg:.1f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
