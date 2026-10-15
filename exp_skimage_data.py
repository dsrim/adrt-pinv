"""Inversion of data from another discretization: line integrals computed by scikit-image's `radon` (straight lines,
interpolation) on exactly the lines of the ADRT, then inverted with the ADRT-based inverses (no inverse crime).

For quadrant q we take the quadrant-transformed image T_q x and evaluate the straight-line projection at the slopes
sigma = s/(N-1), s in [N], and at the signed distances of the ADRT lines (row h at column 0 to row h+s at column N-1),
by linear interpolation along skimage's detector; dividing by sqrt(1+sigma^2) converts the arc-length integral to the
ADRT's one-sample-per-column sum.  Sign and center conventions are calibrated on a smooth test image.
Reconstructions: L_p, hybrid (compiled engine), CGLS for the ADRT model, and skimage's FBP on the same lines."""

import itertools
import sys

import numpy as np
from skimage.transform import iradon, radon

import adrtcore as ac
import fastcpp as fc
import pinv
from exp_shepp import shepp_logan


def skimage_adrt(x, n, conv):
    """ADRT-shaped data (adrtcore layout (4, 2N-1, N)) from skimage's radon on the ADRT's straight lines."""
    tsign, dsign, cshift = conv
    N = 1 << n
    s = np.arange(N)
    sig = s / (N - 1.0)
    theta = 90.0 + tsign * np.degrees(np.arctan(sig))
    h = np.arange(-(N - 1), N)[:, None]
    c = (N - 1) / 2.0 + cshift
    dist = (h - c * (1 - sig)[None, :]) / np.sqrt(1 + sig**2)[None, :]  # signed distance of line (h, s) from c
    out = np.zeros((4, 2 * N - 1, N))
    for q in range(4):
        sino = radon(ac.T_apply(q, x), theta=theta, circle=False)  # (detector, angle)
        det = np.arange(sino.shape[0]) - sino.shape[0] // 2
        for j in range(N):
            out[q, :, j] = np.interp(dsign * dist[:, j], det, sino[:, j], left=0.0, right=0.0) / np.sqrt(
                1 + sig[j] ** 2
            )
    hh = np.arange(-(N - 1), N)[:, None]
    out[:, (hh < -s[None, :])] = 0.0  # lines that miss the square
    return out


def calibrate(n):
    N = 1 << n
    i, j = np.mgrid[0:N, 0:N]
    g = np.exp(-(((i - 0.37 * N) ** 2 + (j - 0.58 * N) ** 2) / (0.08 * N) ** 2)) + 0.6 * np.exp(
        -(((i - 0.66 * N) ** 2 + (j - 0.3 * N) ** 2) / (0.12 * N) ** 2)
    )
    ref = ac.adrt(n, g)
    best = None
    for conv in itertools.product((1, -1), (1, -1), (0.0, 0.5, -0.5)):
        e = np.linalg.norm(skimage_adrt(g, n, conv) - ref) / np.linalg.norm(ref)
        print(f"   calibration conv (theta sign, detector sign, center shift) = {conv}: mismatch {e:.3e}", flush=True)
        if best is None or e < best[0]:
            best = (e, conv)
    return best


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    N = 1 << n
    e_cal, conv = calibrate(n)
    print(f"N={N}: chosen convention {conv}, smooth-image mismatch {e_cal:.3e}", flush=True)
    import os

    phantom = os.environ.get("PHANTOM", "shepp")
    if phantom == "smooth":  # smooth test image, different from the calibration image
        i, j = np.mgrid[0:N, 0:N]
        x = (
            np.exp(-(((i - 0.45 * N) ** 2 + (j - 0.42 * N) ** 2) / (0.15 * N) ** 2))
            + 0.7 * np.exp(-(((i - 0.7 * N) ** 2 + (j - 0.65 * N) ** 2) / (0.07 * N) ** 2))
            + 0.5 * np.exp(-(((i - 0.25 * N) ** 2 + (j - 0.72 * N) ** 2) / (0.1 * N) ** 2))
        )
    else:
        x = -shepp_logan(N)
    print(f"phantom: {phantom}", flush=True)
    Rx = ac.adrt(n, x)
    b_sk = skimage_adrt(x, n, conv)
    print(
        f"N={N}: Shepp-Logan model mismatch ||b_skimage - R x|| / ||R x|| = {np.linalg.norm(b_sk - Rx) / np.linalg.norm(Rx):.3e}",
        flush=True,
    )
    rng = np.random.default_rng(0)
    nz = rng.standard_normal(b_sk.shape)
    nz *= np.linalg.norm(b_sk) / np.linalg.norm(nz)
    Lp = fc.FastLpCpp(n, tau_fac=0.2, eps_in=1e-10, threads=1)
    Hy = fc.FastHybridCpp(n, tau_fac=0.2, eps_in=1e-10, threads=1)
    err = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)
    # FBP with skimage on the same straight lines (all four quadrants as angles in [0, 180))
    theta_q = []
    for q in range(4):
        th = 90.0 + conv[0] * np.degrees(np.arctan(np.arange(N) / (N - 1.0)))
        theta_q.append(th)
    results = {}
    for lev in (0.0, 0.01, 0.05):
        b = b_sk + lev * nz
        xl, xh = Lp.apply(b), Hy.apply(b)
        xc, hst = pinv.cgls(n, b, tol=1e-8, maxit=20000)
        # residual of each reconstruction in the ADRT model, for reference
        res = lambda y, b=b: np.linalg.norm(ac.adrt(n, y) - b) / np.linalg.norm(b)
        print(
            f"noise {lev:4.0%}: L_p err {err(xl):.3e} (res {res(xl):.2e}) | hybrid err {err(xh):.3e} (res {res(xh):.2e}) | "
            f"CGLS err {err(xc):.3e} (res {res(xc):.2e}, {len(hst) - 1} its) | ratios to CGLS {err(xl) / err(xc):.3f}, {err(xh) / err(xc):.3f}",
            flush=True,
        )
        results[lev] = (xl, xh, xc)
    # skimage's own filtered back-projection from its own sinogram over 4N angles in [0,180) (no ADRT involved)
    th = np.linspace(0.0, 180.0, 4 * N, endpoint=False)
    for lev in (0.0, 0.01, 0.05):
        sino = radon(x, theta=th, circle=False)
        nzs = rng.standard_normal(sino.shape)
        nzs *= lev * np.linalg.norm(sino) / np.linalg.norm(nzs)
        xf = iradon(sino + nzs, theta=th, circle=False, filter_name="ramp", output_size=N)
        results[("fbp", lev)] = xf
        print(f"noise {lev:4.0%}: skimage FBP ({4 * N} uniform angles, own data) err {err(xf):.3e}", flush=True)
    np.savez(
        f"results/skimage_data_{phantom}_N{N}.npz",
        x=x,
        b_sk=b_sk,
        Rx=Rx,
        **{
            f"{k}_{lev}": v
            for lev, r in results.items()
            if not isinstance(lev, tuple)
            for k, v in zip(("Lp", "hyb", "cgls"), r)
        },
        **{f"fbp_{lev[1]}": v for lev, v in results.items() if isinstance(lev, tuple)},
    )
    print("SKIMAGE DONE", flush=True)
