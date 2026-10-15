import sys

import numpy as np

sys.path.insert(0, ".")
from lab_frame import delta_table

print("singular values of A_omega (digital) near omega = 2pi/3")
for n in (5, 6, 7, 8, 9):
    N = 1 << n
    L = 2 * N
    D = delta_table(n, "digital")
    k0 = L / 3.0
    for lab, om in (
        ("exact 2pi/3", 2 * np.pi / 3),
        ("grid floor", 2 * np.pi * np.floor(k0) / L),
        ("grid ceil", 2 * np.pi * np.ceil(k0) / L),
        ("grid-2", 2 * np.pi * (np.floor(k0) - 2) / L),
        ("grid+2", 2 * np.pi * (np.ceil(k0) + 2) / L),
        ("omega=2.0", 2.0),
        ("omega=2.4", 2.4),
    ):
        A = np.exp(1j * om * D)
        sv = np.linalg.svd(A, compute_uv=False)
        print(
            f"  N={N:3d} {lab:12s} om={om:.4f}: sigma_min={sv[-1]:.3f}  #sigma^2<N^0.5: {np.sum(sv**2 < N**0.5):3d}  #sigma^2<4: {np.sum(sv**2 < 4):3d}  #sigma^2<1: {np.sum(sv**2 < 1):3d}  (2sin(om/4))^n={(2 * np.sin(om / 4)) ** n:.3f}  sigma_max^2/N={sv[0] ** 2 / N:.2f}"
        )
print("\nghost offset-frequency profile near 2pi/3 (zero-padded x16), quadrant 2")
import adrtcore as ac

G = np.load("results/ghosts.npy", allow_pickle=True).item()
for N, u in G.items():
    n = int(np.log2(N))
    b = ac.adrt(n, u)[2]
    P = 16 * N
    bf = np.abs(np.fft.fft(b, n=P, axis=0)) ** 2
    prof = bf.sum(axis=1)
    prof /= prof.sum()
    om = 2 * np.pi * np.fft.fftfreq(P)
    d = np.abs(np.abs(om) - 2 * np.pi / 3)
    print(
        f"  N={N}: mass within |om-2pi/3| < 2pi/L*{1}: {prof[d < 2 * np.pi / (2 * N)].sum():.3f}; <2 grid: {prof[d < 2 * 2 * np.pi / (2 * N)].sum():.3f}; <4 grid: {prof[d < 4 * 2 * np.pi / (2 * N)].sum():.3f}; <8 grid: {prof[d < 8 * 2 * np.pi / (2 * N)].sum():.3f}; peak at |om|={abs(om[prof.argmax()]):.4f} (2pi/3={2 * np.pi / 3:.4f})"
    )
