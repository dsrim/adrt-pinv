"""Plots of the per-frequency Chebyshev polynomial filters p_k used in fastspec.FastSpectralInverse."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastspec as fs

# Figures are drawn at the paper's \textwidth (6.4 in) and included without scaling, so these are the printed sizes.
TEXTWIDTH = 6.4
plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 6.7,
        "axes.titlesize": 6.7,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "DejaVu Sans Mono",
        "mathtext.it": "DejaVu Sans Mono:italic",
        "mathtext.bf": "DejaVu Sans Mono:bold",
    }
)
# math symbols are typeset with $...$ but drawn in the same fixed-width font as the text
n = 6
N = 1 << n
L = 2 * N
P = fs.FastSpectralInverse(n, tau_fac=0.2, eps=1e-10)
tau = P.tau


def cheb_poly(lam, lam_max, tau, deg):
    """p(lam) = value after `deg` Chebyshev semi-iteration steps for (lam+tau) z = 1, spec in [tau, lam_max+tau]."""
    c, a = fs.cheb_coeffs(np.array([lam_max]), tau)
    c, a = c[0], a[0]
    x_prev = np.zeros_like(lam)
    x = a * np.ones_like(lam)
    om = 1.0
    for j in range(1, deg):
        om = 1 / (1 - c * c / 2) if j == 1 else 1 / (1 - c * c * om / 4)
        r = 1.0 - (lam + tau) * x
        x_new = x + om * (a * r) + (om - 1) * (x - x_prev)
        x_prev, x = x, x_new
    return x


ks = [0, 1, 4, 16, 64]
labels = {0: r"$\omega=0$", 1: r"$\omega=\pi/64$", 4: r"$\omega=\pi/16$", 16: r"$\omega=\pi/4$", 64: r"$\omega=\pi$"}
# ---- Figure 1: p_k(lambda) vs 1/(lambda+tau), with low-degree polynomials for one frequency to show convergence
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.3), layout="constrained")
lam = np.logspace(-2, np.log10(P.lam_max.max()), 600)
k = 16
lk = lam[lam <= P.lam_max[k]]
for d, col in zip((3, 6, 12, 24), ("0.75", "0.6", "0.45", "0.3")):
    ax.loglog(lk, np.abs(cheb_poly(lk, P.lam_max[k], tau, d)), color=col, lw=0.8, label=f"degree {d}")
ax.loglog(lk, np.abs(cheb_poly(lk, P.lam_max[k], tau, P.deg[k])), color="C3", label=f"degree $d_k={P.deg[k]}$")
ax.loglog(lam, 1 / (lam + tau), "k--", lw=0.8, label=r"$1/(\lambda+\tau)$")
ax.axvline(tau, color="0.6", lw=0.6, ls=":")
ax.text(tau * 1.2, 2e-4, r"$\tau$", fontsize=5.8, color="0.4")
ax.axvline(P.lam_max[k], color="0.6", lw=0.6, ls=":")
ax.text(P.lam_max[k] * 0.22, 2e-4, r"$\Lambda_k$", fontsize=5.8, color="0.4")
ax.set_ylim(1e-4, 0.2)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$|p(\lambda)|$")
ax.legend(frameon=False, handlelength=1.5, borderaxespad=0.2, loc="lower left")
ax.set_title(f"filter $p_k$ at $\\omega=\\pi/4$ ($N={N}$, $\\tau=0.2N$)", pad=3)
fig.savefig("figures/filters_p.pdf", pad_inches=0.0)
plt.close(fig)
# ---- Figure 2: filter response lambda p_k(lambda) (the factor in F_p), and its error vs lambda/(lambda+tau)
fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2), layout="constrained")
for k in ks:
    lk = np.logspace(-2, np.log10(P.lam_max[k]), 600)
    pk = cheb_poly(lk, P.lam_max[k], tau, P.deg[k])
    axs[0].semilogx(lk / tau, lk * pk, label=f"{labels[k]}")
    axs[1].loglog(lk / tau, np.abs(lk * pk - lk / (lk + tau)) + 1e-18, label=f"{labels[k]}")
axs[0].semilogx(lk / tau, lk / (lk + tau), "k--", lw=0.8, label=r"$\lambda/(\lambda+\tau)$")
axs[0].set_xlabel(r"$\lambda/\tau$")
axs[0].set_ylabel(r"$\lambda\,p_k(\lambda)$")
axs[0].set_ylim(-0.05, 1.05)
axs[0].legend(frameon=False, handlelength=1.5, borderaxespad=0.2, loc="lower right")
axs[1].set_xlabel(r"$\lambda/\tau$")
axs[1].set_ylabel(r"$|\lambda\,p_k(\lambda)-\lambda/(\lambda+\tau)|$")
axs[1].axhline(1e-10, color="0.6", lw=0.6, ls=":")
axs[1].text(2e-2, 2e-10, r"$\varepsilon$", fontsize=5.8, color="0.4")
axs[1].set_ylim(1e-17, 1)
fig.savefig("figures/filters_response.pdf", pad_inches=0.0)
plt.close(fig)
# ---- Figure 3: Lambda_k and degrees d_k versus omega_k
fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), layout="constrained")
k = np.arange(1, L // 2 + 1)
om = 2 * np.pi * k / L
axs[0].loglog(om, P.lam_max[k] / N, ".", ms=3, label=r"$\Lambda_k/N$ (used, $\times1.5$ safety)")
axs[0].loglog(om, 1.5 * 2 * np.pi / om, "k--", lw=0.8, label=r"$1.5\cdot2\pi/\omega$")
axs[0].set_xlabel(r"$\omega_k$")
axs[0].set_ylabel(r"$\Lambda_k/N$")
axs[0].legend(frameon=False, handlelength=1.5, borderaxespad=0.2, loc="lower left")
axs[1].semilogx(om, P.deg[k], ".", ms=3, label=f"$d_k$ (sum $={P.deg.sum() / N:.0f}N$)")
axs[1].set_xlabel(r"$\omega_k$")
axs[1].set_ylabel(r"degree $d_k$")
axs[1].legend(frameon=False, handlelength=1.5, borderaxespad=0.2)
axs[1].text(om[0] * 1.1, P.deg[0] * 0.9, f"$d_0={P.deg[0]}$ at $\\omega=0$", fontsize=5.8, color="0.4")
fig.savefig("figures/filters_degrees.pdf", pad_inches=0.0)
plt.close(fig)
print(
    "degrees:",
    {k: int(P.deg[k]) for k in ks},
    "lam_max/N:",
    {k: round(float(P.lam_max[k] / N), 2) for k in ks},
    "outer:",
    P.outer_deg,
)
