"""Figure for the setup-choice comparison (reads setup_compare.json written by exp_setup_compare.py)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 7.2,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.0,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "lines.markersize": 3.5,
    }
)
with open("results/setup_compare.json") as fh:
    res = json.load(fh)
names = []
for v in res.values():
    if v["name"] not in names:
        names.append(v["name"])
Ns = sorted({v["N"] for v in res.values()})
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
markers = "osD^v<>p"
for i, name in enumerate(names):
    rows = [res[f"{N}|{name}"] for N in Ns if f"{N}|{name}" in res]
    n_ = [r["N"] for r in rows]
    axs[0].semilogx(n_, [r["steps"] / 1e3 for r in rows], marker=markers[i], label=name)
    axs[1].semilogx(n_, [r["outer"] for r in rows], marker=markers[i])
    axs[2].loglog(n_, [max(r["lierr"], 1e-16) for r in rows], marker=markers[i])
axs[0].set_xlabel("N")
axs[0].set_ylabel("freq-steps / apply  (x 10^3 N)")
axs[1].set_xlabel("N")
axs[1].set_ylabel("outer degree D")
axs[2].set_xlabel("N")
axs[2].set_ylabel("|L R x - x| / |x|")
axs[2].axhline(1e-10, color="0.6", lw=0.6, ls=":")
for ax in axs:
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.minorticks_off()
fig.legend(
    *axs[0].get_legend_handles_labels(),
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.16),
    handlelength=1.4,
    columnspacing=1.0,
)
fig.subplots_adjust(wspace=0.42, bottom=0.2)
fig.savefig("figures/setup_compare.pdf", bbox_inches="tight", pad_inches=0.02)
print("saved")
