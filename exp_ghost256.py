import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
import adrtcore as ac

G = np.load("results/ghosts.npy", allow_pickle=True).item()
n = 8
N = 256
t0 = time.time()
u0 = np.tile(G[128], (2, 2))  # tiled N=128 ghost (same spatial frequency) as start
op = spla.LinearOperator(
    (N * N, N * N), matvec=lambda v: ac.adrt_T(n, ac.adrt(n, v.reshape(N, N))).ravel(), dtype=float
)
r0 = float(u0.ravel() @ op.matvec(u0.ravel()) / (u0.ravel() @ u0.ravel()))
print(f"N=256: Rayleigh quotient of the tiled N=128 ghost = {r0:.3f}", flush=True)
w, U = spla.eigsh(op, k=1, which="SA", v0=u0.ravel(), ncv=160, tol=1e-5, maxiter=3000)
u = U[:, 0].reshape(N, N)
uh = np.abs(np.fft.fft2(u)) ** 2
i = uh.argmax()
a, b = i // N, i % N
fa = a if a <= N // 2 else N - a
fb = b if b <= N // 2 else N - b
print(
    f"N=256: lam_min(R^TR)={w[0]:.3f} (/N^0.5={w[0] / 16:.3f}); ghost peak ({fa},{fb}), max/N={max(fa, fb) / N:.3f}  ({time.time() - t0:.0f}s)",
    flush=True,
)
np.save("results/ghost256.npy", u)
