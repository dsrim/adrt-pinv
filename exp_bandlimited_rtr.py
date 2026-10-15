"""lambda_min of R^T R restricted to band-limited images (|xi_i|,|xi_j| <= c_f * N/2): does the continuum floor ~2N hold there?"""

import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
import adrtcore as ac

for n in (4, 5, 6, 7):
    N = 1 << n
    t0 = time.time()
    f = np.abs(np.fft.fftfreq(N) * N)
    line = f"N={N:4d}:"
    for cf in (0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        mask = (f[:, None] <= cf * N / 2) & (f[None, :] <= cf * N / 2)
        Pb = lambda v, mask=mask: np.real(np.fft.ifft2(mask * np.fft.fft2(v)))
        C = 20.0 * N

        def mv(v, N=N, Pb=Pb, n=n, C=C):
            X = v.reshape(N, N)
            Xb = Pb(X)
            return (Pb(ac.adrt_T(n, ac.adrt(n, Xb))) + C * (X - Xb)).ravel()

        op = spla.LinearOperator((N * N, N * N), matvec=mv, dtype=float)
        w, _ = spla.eigsh(op, k=1, which="SA", tol=1e-6, maxiter=20000)
        line += f"  cf={cf}: {w[0]:8.2f} (/N={w[0] / N:.3f})"
    print(line + f"  ({time.time() - t0:.0f}s)", flush=True)
