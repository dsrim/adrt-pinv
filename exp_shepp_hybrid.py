"""Shepp-Logan: original fast oblique inverse L_p (fastspec), fast hybrid (fasthybrid, G_1^{-1} preconditioner) and CGLS
(least squares to 1e-8) at several noise levels; tau = 0.2N.  Figure notes/figures/shepp_hybrid_N{N}.pdf."""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adrtcore as ac
import fasthybrid as fh
import fastspec as fs
import pinv
from exp_shepp import shepp_logan

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    N = 1 << n
    tf = 0.2
    x = -shepp_logan(N)
    rng = np.random.default_rng(0)
    b0 = ac.adrt(n, x)
    noise = rng.standard_normal(b0.shape)
    noise *= np.linalg.norm(b0) / np.linalg.norm(noise)
    levels = [0.0, 0.01, 0.05, 0.1]
    t0 = time.time()
    O = fs.FastSpectralInverse(n, tau_fac=tf, eps=1e-10)
    print(
        f"N={N}: orig setup {time.time() - t0:.0f}s, sum deg/N {O.deg.sum() / N:.1f}, outer deg {O.outer_deg}",
        flush=True,
    )
    t0 = time.time()
    H = fh.FastHybridInverse(n, tau_fac=tf, eps=1e-10, precond="g1")
    print(
        f"N={N}: hybrid setup {time.time() - t0:.0f}s, sum deg/N {H.deg.sum() / N:.1f}, outer [{H.lo:.3f},{H.hi:.3f}] deg {H.outer_deg}",
        flush=True,
    )
    rec = {}
    for lev in levels:
        b = b0 + lev * noise
        O.count = [0]
        t0 = time.time()
        xo = O.apply(b)
        to = time.time() - t0
        co = O.count[0] * 2 * n
        H.count = [0]
        t0 = time.time()
        xh = H.apply(b)
        th = time.time() - t0
        ch = H.count[0]
        xc, hst = pinv.cgls(n, b, tol=1e-8, maxit=20000)
        e = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)
        rec[lev] = (xo, xh, xc, e(xo), e(xh), e(xc), len(hst) - 1)
        print(
            f"  noise {lev:4.0%}: orig err {e(xo):.3e} ({to:.0f}s, {co / (N * N * n):.0f} N^2logN) | hybrid err {e(xh):.3e} ({th:.0f}s, {ch / (N * N * n):.0f} N^2logN) | CGLS err {e(xc):.3e} ({len(hst) - 1} its) | ratios {e(xo) / max(e(xc), 1e-300):.2f}, {e(xh) / max(e(xc), 1e-300):.2f}",
            flush=True,
        )
    fig, axs = plt.subplots(4, len(levels), figsize=(6.4, 6.6))
    vmin, vmax = -0.05, 1.05
    for j, lev in enumerate(levels):
        xo, xh, xc, eo, eh, ec, its = rec[lev]
        b = b0 + lev * noise
        if j == 0:
            axs[0, j].imshow(x, cmap="gray", vmin=vmin, vmax=vmax)
            axs[0, j].set_title("phantom", pad=2)
        else:
            axs[0, j].imshow(b[2], cmap="gray", aspect="auto")
            axs[0, j].set_title(f"data (q.III), noise {lev:.0%}", pad=2)
        axs[1, j].imshow(xo, cmap="gray", vmin=vmin, vmax=vmax)
        axs[1, j].set_title(f"err {eo:.2e}", pad=2)
        axs[2, j].imshow(xh, cmap="gray", vmin=vmin, vmax=vmax)
        axs[2, j].set_title(f"err {eh:.2e}", pad=2)
        axs[3, j].imshow(xc, cmap="gray", vmin=vmin, vmax=vmax)
        axs[3, j].set_title(f"err {ec:.2e} ({its} it)", pad=2)
        for i in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    axs[1, 0].set_ylabel("L_p (original)")
    axs[2, 0].set_ylabel("hybrid")
    axs[3, 0].set_ylabel("CGLS")
    fig.subplots_adjust(wspace=0.04, hspace=0.16, left=0.04, right=0.99, top=0.97, bottom=0.01)
    fig.savefig(f"figures/shepp_hybrid_N{N}.pdf", bbox_inches="tight", pad_inches=0.02)
