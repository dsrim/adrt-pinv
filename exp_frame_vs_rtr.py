import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
import adrtcore as ac
from lab_frame import FrameLab


def lanczos_min(apply, dim, k=1, which="SA"):
    op = spla.LinearOperator((dim, dim), matvec=lambda v: apply(v.reshape(int(np.sqrt(dim)), -1)).ravel(), dtype=float)
    w, U = spla.eigsh(op, k=k, which=which, tol=1e-6, maxiter=3000)
    return w, U


print(
    "lambda_min(R^T R) of the ADRT (adrtcore) vs straight lines, and lambda_min(E^T F_tau E) vs lambda_min(R^T R)/tau"
)
for n in range(3, 9):
    N = 1 << n
    t0 = time.time()
    RtR = lambda x, n=n: ac.adrt_T(n, ac.adrt(n, x))
    w, U = lanczos_min(RtR, N * N)
    wmax, _ = lanczos_min(RtR, N * N, which="LA")
    line = f"N={N:4d}: lam_min(R^TR)={w[0]:.3f} (/N={w[0] / N:.3f}, /N^0.6={w[0] / N**0.6:.3f}) lam_max={wmax[0]:.3e}"
    if N <= 128:
        labS = FrameLab(n, 0.2, "straight", 2 * N, "ident")
        ws, _ = lanczos_min(labS.apply, N * N)
        line += f" | straight: lam_min={ws[0]:.3f} (/N={ws[0] / N:.3f})"
        for tf in (0.2, 1.0):
            lab = FrameLab(n, tf, "digital", 2 * N, "tikhonov")
            wf, Uf = lanczos_min(lab.apply, N * N)
            line += f" | tau={tf}N: lam_min(F)={wf[0]:.4f} vs lam_min(RtR)/tau={w[0] / (tf * N):.4f}"
            # overlap of the minimizers
            line += f" ovl={abs(np.dot(Uf[:, 0], U[:, 0])):.2f}"
    print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
np.save("results/rtr_minvec.npy", U[:, 0])
