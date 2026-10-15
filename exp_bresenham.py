import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
from lab_frame import FrameLab

print("lambda_min(R^T R): bresenham (integer-rounded straight lines) vs digital vs straight")
for n in (3, 4, 5, 6, 7):
    N = 1 << n
    t0 = time.time()
    line = f"N={N:4d}:"
    for lines in ("bresenham", "digital", "straight"):
        lab = FrameLab(n, 1.0, lines, 2 * N, "ident")
        op = spla.LinearOperator(
            (N * N, N * N), matvec=lambda v, lab=lab, N=N: lab.apply(v.reshape(N, N)).ravel(), dtype=float
        )
        w, U = spla.eigsh(op, k=1, which="SA", tol=1e-6, maxiter=10000)
        u = U[:, 0].reshape(N, N)
        uh = np.abs(np.fft.fft2(u)) ** 2
        i = uh.argmax()
        a, b = i // N, i % N
        line += f"  {lines}: {w[0]:8.3f} (/N={w[0] / N:.3f}) ghost peak ({a},{b})"
    print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
