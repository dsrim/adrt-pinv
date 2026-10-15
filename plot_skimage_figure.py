"""Figure: data from scikit-image's radon converted to ADRT data by the adrt package's route (skimage_adrt_pkg.py).  For each
phantom (Shepp-Logan, smooth) two rows, without added noise and with 5% white noise: phantom, L_p (tau = 0.2N), L_p (tau = 5N),
and least squares (CGLS) for the ADRT model; 1D middle-row
slices under each image, aligned; one gray scale and one colorbar."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastcpp as fc
import pinv
from exp_shepp import shepp_logan
from skimage_adrt_pkg import skimage_adrt_pkg

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 5.8,
        "axes.titlesize": 5.8,
        "axes.labelsize": 5.8,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 5.0,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    }
)
TEXTWIDTH = 6.4
n = 7
N = 1 << n
i, j = np.mgrid[0:N, 0:N]
phantoms = [
    ("Shepp-Logan", -shepp_logan(N)),
    (
        "smooth",
        np.exp(-(((i - 0.45 * N) ** 2 + (j - 0.42 * N) ** 2) / (0.15 * N) ** 2))
        + 0.7 * np.exp(-(((i - 0.7 * N) ** 2 + (j - 0.65 * N) ** 2) / (0.07 * N) ** 2))
        + 0.5 * np.exp(-(((i - 0.25 * N) ** 2 + (j - 0.72 * N) ** 2) / (0.1 * N) ** 2)),
    ),
]
Lp02 = fc.FastLpCpp(n, tau_fac=0.2, eps_in=0.1, threads=1)
Lp5 = fc.FastLpCpp(n, tau_fac=5.0, eps_in=0.1, threads=1)
fig = plt.figure(figsize=(TEXTWIDTH, 8.4), layout="constrained")
gs = fig.add_gridspec(8, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1, 0.38] * 4)
vmin, vmax = -0.05, 1.05
i0 = N // 2
pairs = []
im = None
row = 0
for name, x in phantoms:
    b = skimage_adrt_pkg(x, n, "cos", False)
    err = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
    rng = np.random.default_rng(0)
    nb = rng.standard_normal(b.shape)
    nb *= np.linalg.norm(b) / np.linalg.norm(nb)
    for lev in (0.0, 0.05):
        bb = b + lev * nb
        recs = [
            ("phantom", x),
            ("L_p, tau=0.2N", Lp02.apply(bb)),
            ("L_p, tau=5N", Lp5.apply(bb)),
            ("least squares", pinv.cgls(n, bb, tol=1e-10, maxit=50000)[0]),
        ]
        for c, (lab, y) in enumerate(recs):
            ax = fig.add_subplot(gs[2 * row, c])
            im = ax.imshow(y, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(lab if c == 0 else f"{lab}\nerror {err(y):.2e}", pad=2)
            if c == 0:
                ax.set_ylabel(f"{name}, {lev:.0%} noise", fontsize=6.0)
                ax.axhline(i0, color="C1", lw=0.6, ls="--")
            sx = fig.add_subplot(gs[2 * row + 1, c])
            pairs.append((ax, sx))
            sx.plot(x[i0], color="0.55", lw=0.7)
            sx.plot(y[i0], color="C0", lw=0.6)
            sx.set_xlim(-0.5, N - 0.5)
            sx.set_ylim(-0.25, 1.25)
            sx.set_xticks([0, N // 2, N - 1])
            if c > 0:
                sx.set_yticklabels([])
            print(f"{name:12s} {lev:4.0%} {lab:16s} err {err(y):.3e}", flush=True)
        row += 1
fig.colorbar(im, cax=fig.add_subplot(gs[:, 4]))
fig.canvas.draw()
fig.set_layout_engine("none")
for a, sx in pairs:
    pa, ps = a.get_position(), sx.get_position()
    sx.set_position([pa.x0, ps.y0, pa.width, ps.height])
fig.savefig(f"figures/skimage_data_N{N}.pdf", pad_inches=0.0)
print("FIGURE DONE")
