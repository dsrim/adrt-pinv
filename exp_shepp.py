"""Shepp-Logan example: data b = R x + noise at several relative noise levels; reconstruction by the
explicit oblique inverse L_p (fastspec) and by CGLS (least squares to 1e-8); image plots + errors."""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adrtcore as ac
import fastspec as fs
import pinv

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


def shepp_logan(N):
    """modified Shepp-Logan phantom (Toft / Kak-Slaney intensities) on [-1,1]^2"""
    E = [
        (1.0, 0.69, 0.92, 0, 0, 0),
        (-0.8, 0.6624, 0.874, 0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0, 18),
        (0.1, 0.21, 0.25, 0, 0.35, 0),
        (0.1, 0.046, 0.046, 0, 0.1, 0),
        (0.1, 0.046, 0.046, 0, -0.1, 0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0, -0.606, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]
    y, x = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
    img = np.zeros((N, N))
    for A, a, b, x0, y0, phi in E:
        p = np.deg2rad(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += A
    return -img[::-1]  # flip so the phantom is upright with row 0 at the top of imshow (sign: display convention)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    N = 1 << n
    tau_fac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    x = -shepp_logan(N)
    rng = np.random.default_rng(0)
    b0 = ac.adrt(n, x)
    noise = rng.standard_normal(b0.shape)
    noise *= np.linalg.norm(b0) / np.linalg.norm(noise)
    levels = [0.0, 0.01, 0.05, 0.1]
    Ps = {}
    for tf in (0.2, 1.0):
        t0 = time.time()
        Ps[tf] = fs.FastSpectralInverse(n, tau_fac=tf, eps=1e-10)
        ts = time.time() - t0
        print(
            f"N={N} tau={tf}N: setup {ts:.0f}s, sum deg {Ps[tf].deg.sum() / N:.0f}N, outer deg {Ps[tf].outer_deg}",
            flush=True,
        )
    rec = {}
    for lev in levels:
        b = b0 + lev * noise
        xo = {tf: Ps[tf].apply(b) for tf in Ps}
        xc, h = pinv.cgls(n, b, tol=1e-8, maxit=5000)
        eo = {tf: np.linalg.norm(xo[tf] - x) / np.linalg.norm(x) for tf in Ps}
        ec = np.linalg.norm(xc - x) / np.linalg.norm(x)
        rec[lev] = (xo, xc, eo, ec, len(h) - 1)
        print(
            f"  noise {lev:4.0%}: oblique tau=0.2N err {eo[0.2]:.3e}, tau=N err {eo[1.0]:.3e} | CGLS err {ec:.3e} ({len(h) - 1} its) | ratios {eo[0.2] / max(ec, 1e-300):.2f}, {eo[1.0] / max(ec, 1e-300):.2f}",
            flush=True,
        )
    # ---- figure: rows = phantom & data / L_p tau=0.2N / L_p tau=N / CGLS ; cols = noise levels
    fig, axs = plt.subplots(4, len(levels), figsize=(6.4, 6.6))
    vmin, vmax = -0.05, 1.05
    for j, lev in enumerate(levels):
        xo, xc, eo, ec, its = rec[lev]
        b = b0 + lev * noise
        if j == 0:
            axs[0, j].imshow(x, cmap="gray", vmin=vmin, vmax=vmax)
            axs[0, j].set_title("phantom", pad=2)
        else:
            axs[0, j].imshow(b[2], cmap="gray", aspect="auto")
            axs[0, j].set_title(f"data (q.III), noise {lev:.0%}", pad=2)
        axs[1, j].imshow(xo[0.2], cmap="gray", vmin=vmin, vmax=vmax)
        axs[1, j].set_title(f"err {eo[0.2]:.2e}", pad=2)
        axs[2, j].imshow(xo[1.0], cmap="gray", vmin=vmin, vmax=vmax)
        axs[2, j].set_title(f"err {eo[1.0]:.2e}", pad=2)
        axs[3, j].imshow(xc, cmap="gray", vmin=vmin, vmax=vmax)
        axs[3, j].set_title(f"err {ec:.2e} ({its} it)", pad=2)
        for i in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    axs[1, 0].set_ylabel("L_p, tau=0.2N")
    axs[2, 0].set_ylabel("L_p, tau=N")
    axs[3, 0].set_ylabel("CGLS")
    fig.subplots_adjust(wspace=0.04, hspace=0.16, left=0.04, right=0.99, top=0.97, bottom=0.01)
    fig.savefig(f"figures/shepp_N{N}.pdf", bbox_inches="tight", pad_inches=0.02)
