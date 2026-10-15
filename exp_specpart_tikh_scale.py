import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import adrtcore as ac
import pinv
from exp_specpart_scale import cheb_solve
from exp_specpart_tikh import Tikh

rng = np.random.default_rng(0)
for n in [int(a) for a in sys.argv[1:]]:
    N = 1 << n
    for tf in (0.2, 1.0):
        t0 = time.time()
        SP = Tikh(n, tf * N)
        ts = time.time() - t0
        Fop = spla.LinearOperator(
            (N * N, N * N), matvec=lambda v, SP=SP, N=N: np.real(SP.frame_window(v.reshape(N, N))).ravel()
        )
        lo = spla.eigsh(Fop, k=1, which="SA", tol=1e-4)[0][0]
        hi = spla.eigsh(Fop, k=1, which="LA", tol=1e-4)[0][0]
        x = rng.standard_normal((N, N))
        b = ac.adrt(n, x)
        bn = b + 1e-2 * rng.standard_normal((4, 2 * N - 1, N))
        y = np.real(SP.quadrant_inverse(b)[:N, :N])
        yn = np.real(SP.quadrant_inverse(bn)[:N, :N])
        op = lambda v, SP=SP: np.real(SP.frame_window(v))
        xmp, h = pinv.cgls(n, bn, tol=1e-10, maxit=20000)
        emp = np.linalg.norm(xmp - x) / np.linalg.norm(x)
        xl = cheb_solve(op, y, lo, hi, 40)
        xln = cheb_solve(op, yn, lo, hi, 40)
        print(
            f"N={N} tau={tf}N: kappa(E^T F E)={hi / lo:.2f} [{lo:.3f},{hi:.3f}] | cheb40 left-inv err {np.linalg.norm(xl - x) / np.linalg.norm(x):.1e} | noisy err {np.linalg.norm(xln - x) / np.linalg.norm(x):.2e} vs MP {emp:.2e} (setup {ts:.0f}s, {time.time() - t0:.0f}s)",
            flush=True,
        )
