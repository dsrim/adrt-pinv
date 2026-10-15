"""Figure (Shepp-Logan): stitched full ADRT data (adrt package layout, quadrant III outlined), reconstructions by L_p, the
hybrid and CGLS at 0/1/5/10% noise, each with a 1D slice through the middle row; one color range and one colorbar.
Same data and settings as Table 9 (exp_shepp_hybrid.py): tau = 0.2N, eps = 1e-10, CGLS to 1e-8; compiled engine (fastcpp)."""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import adrt
import matplotlib.pyplot as plt

import adrtcore as ac
import fastcpp as fc
import pinv
from exp_shepp import shepp_logan

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
n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
N = 1 << n
x = -shepp_logan(N)
rng = np.random.default_rng(0)
b0 = ac.adrt(n, x)
noise = rng.standard_normal(b0.shape)
noise *= np.linalg.norm(b0) / np.linalg.norm(noise)
levels = [0.0, 0.01, 0.05, 0.1]
# ---- exact map from this repository's sinogram layout to the adrt package's, by fingerprints of random images
F = np.random.default_rng(7).standard_normal((3, N, N))
fc_ = np.stack([ac.adrt(n, f).ravel() for f in F], 1)
fp_ = np.stack([adrt.adrt(f).ravel() for f in F], 1)
key = lambda v: tuple(np.round(v, 8))
from collections import defaultdict

gp, gc = defaultdict(list), defaultdict(list)  # lines shared by two quadrants have equal fingerprints:
for i, v in enumerate(fp_):  # pair the entries within each group of equal fingerprints
    if np.abs(v).max() > 1e-12:
        gp[key(v)].append(i)
for i, v in enumerate(fc_):
    if np.abs(v).max() > 1e-12:
        gc[key(v)].append(i)
assert gp.keys() == gc.keys() and all(len(gp[k]) == len(gc[k]) for k in gp), "layouts do not match"
core_idx = np.array([i for k in gc for i in gc[k]])
pkg_idx = np.array([i for k in gc for i in gp[k]])


def to_pkg(b):
    out = np.zeros(4 * (2 * N - 1) * N)
    out[pkg_idx] = b.ravel()[core_idx]
    return out.reshape(4, 2 * N - 1, N)


assert np.allclose(to_pkg(b0), adrt.adrt(x)), "layout map failed"
# quadrant III of this repository (T = identity, index 2) is the package's quadrant index 1 (offsets reversed); mark it by the
# package's own quadrant index, since lines shared by two quadrants make the entry-wise layout map ambiguous between them
q3 = np.zeros((4, 2 * N - 1, N))
q3[1] = 1.0
mask = adrt.utils.stitch_adrt(q3)
assert np.allclose(adrt.adrt(x)[1][::-1], ac.adrt(n, x)[2]), "package quadrant 1 is not quadrant III"


def band_polygon(m):
    """Exact outline of a sheared band in the stitched sinogram: per column the band is a run of rows whose ends move
    linearly with the column, so the outline is the polygon through the four corners (pixel edges at +-0.5)."""
    cols = np.nonzero((m > 0.5).any(axis=0))[0]
    c0, c1 = cols.min(), cols.max()
    top = lambda c: np.nonzero(m[:, c] > 0.5)[0].min()
    bot = lambda c: np.nonzero(m[:, c] > 0.5)[0].max()
    return np.array(
        [
            [c0 - 0.5, top(c0) - 0.5],
            [c1 + 0.5, top(c1) - 0.5],
            [c1 + 0.5, bot(c1) + 0.5],
            [c0 - 0.5, bot(c0) + 0.5],
            [c0 - 0.5, top(c0) - 0.5],
        ]
    )


poly = band_polygon(mask)
# ---- reconstructions
Lp = fc.FastLpCpp(n, tau_fac=0.2, eps_in=1e-10, threads=1)
Hy = fc.FastHybridCpp(n, tau_fac=0.2, eps_in=1e-10, threads=1, precond="g1")
rec = {}
err = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)
for lev in levels:
    b = b0 + lev * noise
    xc, h = pinv.cgls(n, b, tol=1e-8, maxit=20000)
    rec[lev] = {"L_p": Lp.apply(b), "hybrid": Hy.apply(b), "CGLS": xc, "its": len(h) - 1}
    print(
        f"noise {lev:4.0%}: L_p {err(rec[lev]['L_p']):.3e} | hybrid {err(rec[lev]['hybrid']):.3e} | CGLS {err(xc):.3e} ({len(h) - 1} its)",
        flush=True,
    )
# ---- figure: 7 rows (data, then image + slice for each method), 4 columns, one colorbar
vmin, vmax, cmap = -0.05, 1.05, "gray"
i0 = N // 2
smax = adrt.utils.stitch_adrt(to_pkg(b0)).max()
fig = plt.figure(figsize=(TEXTWIDTH, 7.3), layout="constrained")
gs = fig.add_gridspec(7, 5, width_ratios=[1, 1, 1, 1, 0.05], height_ratios=[1.0, 1, 0.38, 1, 0.38, 1, 0.38])
im = None
pairs = []
for j, lev in enumerate(levels):
    ax = fig.add_subplot(gs[0, j])
    if j == 0:
        im = ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axhline(i0, color="C1", lw=0.6, ls="--")
        ax.set_title("phantom (dashed: slice)", pad=2)
    else:
        S = adrt.utils.stitch_adrt(to_pkg(b0 + lev * noise)) / smax
        ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.plot(poly[:, 0], poly[:, 1], color="C1", lw=0.7)
        ax.set_title(f"ADRT data, {lev:.0%} noise", pad=2)
        ys, xs = np.nonzero(mask > 0.5)
        ax.text(xs.mean(), ys.mean(), "III", color="C1", ha="center", va="center", fontsize=6.0, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
for r, name in enumerate(["L_p", "hybrid", "CGLS"]):
    for j, lev in enumerate(levels):
        y = rec[lev][name]
        ax = fig.add_subplot(gs[1 + 2 * r, j])
        ax.imshow(y, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        extra = f" ({rec[lev]['its']} it.)" if name == "CGLS" else ""
        ax.set_title(f"{lev:.0%}: err {err(y):.2e}{extra}", pad=2)
        if j == 0:
            ax.set_ylabel(name, fontsize=6.5)
        sx = fig.add_subplot(gs[2 + 2 * r, j])
        pairs.append((ax, sx))
        sx.plot(x[i0], color="0.55", lw=0.7, label="phantom")
        sx.plot(y[i0], color="C0", lw=0.6, label=name)
        sx.set_ylim(-0.25, 1.25)
        sx.set_xlim(-0.5, N - 0.5)
        sx.set_xticks([0, N // 2, N - 1])
        if j > 0:
            sx.set_yticklabels([])
        if j == 0 and r == 0:
            sx.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2, borderaxespad=0.1)
cax = fig.add_subplot(gs[:, 4])
fig.colorbar(im, cax=cax)
# align each slice horizontally with the image above it (pixel j over abscissa j), after the layout has settled
fig.canvas.draw()
fig.set_layout_engine("none")
for a, sx in pairs:
    pa, ps = a.get_position(), sx.get_position()
    sx.set_position([pa.x0, ps.y0, pa.width, ps.height])
fig.savefig(f"figures/shepp_full_N{N}.pdf", pad_inches=0.0)
print("FIGURE DONE")
