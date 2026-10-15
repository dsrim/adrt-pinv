import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
import adrtcore as ac

out = {}
for n in (5, 6, 7):
    N = 1 << n
    L = 2 * N
    op = spla.LinearOperator(
        (N * N, N * N), matvec=lambda v, n=n, N=N: ac.adrt_T(n, ac.adrt(n, v.reshape(N, N))).ravel(), dtype=float
    )
    w, U = spla.eigsh(op, k=3, which="SA", tol=1e-8, maxiter=5000)
    u = U[:, 0].reshape(N, N)
    out[N] = u
    print(f"N={N}: three smallest eigenvalues of R^TR: {np.round(w, 3)}")
    # 2D spectrum of the ghost
    uh = np.abs(np.fft.fft2(u)) ** 2
    uh /= uh.sum()
    idx = np.argsort(uh.ravel())[::-1][:8]
    print(
        "  top 2D frequencies (a,b)->mass:",
        [
            (
                int(i // N) if i // N <= N // 2 else int(i // N - N),
                int(i % N) if i % N <= N // 2 else int(i % N - N),
                round(float(uh.ravel()[i]), 3),
            )
            for i in idx
        ],
    )
    # radial/angular: mass by |xi| band and by direction
    a = np.fft.fftfreq(N) * N
    A, B = np.meshgrid(a, a, indexing="ij")
    rad = np.hypot(A, B)
    for lo, hi in ((0, N / 8), (N / 8, N / 4), (N / 4, 3 * N / 8), (3 * N / 8, N / 2), (N / 2, N)):
        print(f"  mass in |xi| in [{lo:.0f},{hi:.0f}): {uh[(rad >= lo) & (rad < hi)].sum():.3f}", end="")
    print()
    # which quadrants see it, and at which offset frequencies
    b = ac.adrt(n, u)  # (4, 2N-1, N)
    for q in range(4):
        bq = b[q]
        e = np.sum(bq**2)
        bf = np.abs(np.fft.fft(np.vstack([bq, np.zeros((1, N))]), axis=0)) ** 2  # offset-frequency content (L=2N)
        prof = bf.sum(axis=1)
        prof /= prof.sum()
        k = np.fft.fftfreq(L) * L
        print(
            f"  quadrant {q}: ||R_q u||^2 = {e:.2f};  offset-freq mass |k|<N/8: {prof[np.abs(k) < N / 8].sum():.2f}, N/8..N/4: {prof[(np.abs(k) >= N / 8) & (np.abs(k) < N / 4)].sum():.2f}, N/4..N/2: {prof[(np.abs(k) >= N / 4) & (np.abs(k) < N / 2)].sum():.2f}, >=N/2: {prof[np.abs(k) >= N / 2].sum():.2f}; slope mass s<N/4: {np.sum(bq[:, : N // 4] ** 2) / e:.2f}, mid: {np.sum(bq[:, N // 4 : 3 * N // 4] ** 2) / e:.2f}, s>=3N/4: {np.sum(bq[:, 3 * N // 4 :] ** 2) / e:.2f}"
        )
    # spatial structure: row/column energy profiles, sign pattern
    print("  row-energy profile (8 bins):", np.round(np.add.reduceat((u**2).sum(axis=1), np.arange(0, N, N // 8)), 3))
    print("  col-energy profile (8 bins):", np.round(np.add.reduceat((u**2).sum(axis=0), np.arange(0, N, N // 8)), 3))
    print(
        "  checkerboard correlation <u, (-1)^(i+j) u>/<u,u>:",
        round(float(np.sum(u * u * ((-1.0) ** (np.add.outer(np.arange(N), np.arange(N)))))), 3),
        " row-alternating:",
        round(float(np.sum(u * u * ((-1.0) ** np.arange(N))[:, None])), 3),
        " col-alternating:",
        round(float(np.sum(u * u * ((-1.0) ** np.arange(N))[None, :])), 3),
    )
np.save("results/ghosts.npy", out, allow_pickle=True)
