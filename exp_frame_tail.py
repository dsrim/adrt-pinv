import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
from lab_frame import FrameLab

print("lower tail of spec(E^T F_tau E), tau=0.2N: number of eigenvalues below thresholds")
for n in (4, 5, 6, 7):
    N = 1 << n
    t0 = time.time()
    lab = FrameLab(n, 0.2, "digital", 2 * N, "tikhonov")
    if N <= 64:
        ev = np.linalg.eigvalsh(lab.dense())
    else:
        op = spla.LinearOperator(
            (N * N, N * N), matvec=lambda v, lab=lab, N=N: lab.apply(v.reshape(N, N)).ravel(), dtype=float
        )
        ev, _ = spla.eigsh(op, k=120, which="SA", tol=1e-5, maxiter=20000)
        ev = np.sort(ev)
    print(
        f"N={N}: lam_min={ev[0]:.4f}; count < 0.5: {np.sum(ev < 0.5)}, < 0.7: {np.sum(ev < 0.7)}, < 0.8: {np.sum(ev < 0.8)}, < 0.9: {np.sum(ev < 0.9)}, < 1.0: {np.sum(ev < 1.0)}, < 1.2: {np.sum(ev < 1.2)}; lowest 12: {np.round(ev[:12], 3)}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
