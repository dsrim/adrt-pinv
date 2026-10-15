"""Compare the two routes from skimage's radon to ADRT-shaped data (package route vs calibrated direct route) and rerun the
reconstructions with the package route: model mismatch, and errors of L_p (tau = 0.2N, 5N), hybrid and least squares."""

import numpy as np

import adrtcore as ac
import fastcpp as fc
import pinv
from exp_shepp import shepp_logan
from exp_skimage_data import skimage_adrt
from skimage_adrt_pkg import skimage_adrt_pkg

n = 7
N = 1 << n
i, j = np.mgrid[0:N, 0:N]
smooth = (
    np.exp(-(((i - 0.45 * N) ** 2 + (j - 0.42 * N) ** 2) / (0.15 * N) ** 2))
    + 0.7 * np.exp(-(((i - 0.7 * N) ** 2 + (j - 0.65 * N) ** 2) / (0.07 * N) ** 2))
    + 0.5 * np.exp(-(((i - 0.25 * N) ** 2 + (j - 0.72 * N) ** 2) / (0.1 * N) ** 2))
)
for name, x in (("smooth", smooth), ("shepp", -shepp_logan(N))):
    Rx = ac.adrt(n, x)
    mm = lambda b, Rx=Rx: np.linalg.norm(b - Rx) / np.linalg.norm(Rx)
    print(f"{name}: direct calibrated route mismatch {mm(skimage_adrt(x, n, (-1, -1, 0.5))):.3e}", flush=True)
    best = None
    for circle in (False, True):
        for scale in ("none", "cos", "sec"):
            e = mm(skimage_adrt_pkg(x, n, scale, circle))
            print(f"   package route circle={circle} scale={scale}: mismatch {e:.3e}", flush=True)
            if best is None or e < best[0]:
                best = (e, scale, circle)
    print(f"{name}: package route best {best}", flush=True)
for name, x in (("smooth", smooth), ("shepp", -shepp_logan(N))):
    b = skimage_adrt_pkg(x, n, best[1], best[2])
    err = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    nz = np.random.default_rng(0).standard_normal(b.shape)
    nz *= np.linalg.norm(b) / np.linalg.norm(nz)
    for lev in (0.0, 0.01, 0.05):
        bb = b + lev * nz
        xc = pinv.cgls(n, bb, tol=1e-10, maxit=50000)[0]
        ec = err(xc)
        r = [err(fc.FastLpCpp(n, tau_fac=t, eps_in=0.1, threads=1).apply(bb)) for t in (0.2, 5.0)]
        h = [err(fc.FastHybridCpp(n, tau_fac=t, eps_in=0.1, threads=1).apply(bb)) for t in (0.2, 5.0)]
        print(
            f"{name:6s} noise {lev:4.0%}: LS {ec:.3e} | L_p tau=0.2N {r[0]:.3e} ({r[0] / ec:.2f}x), 5N {r[1]:.3e} ({r[1] / ec:.2f}x) | "
            f"hybrid 0.2N {h[0]:.3e} ({h[0] / ec:.2f}x), 5N {h[1]:.3e} ({h[1] / ec:.2f}x)",
            flush=True,
        )
print("ROUTES DONE", flush=True)
