"""Rigorous lower bound for lambda_min(R^T R) by replacing each quadrant Gram M_omega by sigma_min(A_omega)^2 I:
   R^T R  >=  W := sum_q E^T T_q^T DFT^{-1} diag_k( s_k^2 I ) DFT T_q E,   s_k = sigma_min(A_{omega_k}),
and W = W_v + W_h (vertical/horizontal frequency multipliers on the zero-padded torus, two quadrants each).
Compares lambda_min(W) with lambda_min(R^T R) and prints the profile s_k."""

import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import adrtcore as ac
import fastspec as fs

for n in [int(a) for a in sys.argv[1:]] or [4, 5, 6]:
    N = 1 << n
    L = 2 * N
    oms = 2 * np.pi * np.fft.fftfreq(L)
    z = np.exp(1j * oms)
    I = np.tile(np.eye(N, dtype=complex), (L, 1, 1)).reshape(L * N, N)  # columns of identity per frequency
    A = np.array([fs.bfly(n, np.full(N, zk), np.eye(N, dtype=complex)).T for zk in z])  # A_k (N x N): rows = slopes
    sv = np.linalg.svd(A, compute_uv=False)
    smin = sv[:, -1]
    smax = sv[:, 0]
    w = np.abs(oms)
    print(
        f"N={N}: sigma_min(A_w)^2: at |w|>=2pi/3 min {smin[w >= 2 * np.pi / 3].min() ** 2:.3f}, at |w| in [pi/2,2pi/3) min {smin[(w >= np.pi / 2) & (w < 2 * np.pi / 3)].min() ** 2:.3f}, "
        f"[pi/4,pi/2) {smin[(w >= np.pi / 4) & (w < np.pi / 2)].min() ** 2:.3f}, [pi/8,pi/4) {smin[(w >= np.pi / 8) & (w < np.pi / 4)].min() ** 2:.3f}, <pi/8 {smin[w < np.pi / 8].min() ** 2:.2e}; sigma_max^2 max {smax.max() ** 2:.0f}",
        flush=True,
    )
    s2 = smin**2

    def W_apply(x, L=L, N=N, s2=s2):
        X = np.zeros((L, L))
        X[:N, :N] = x.reshape(N, N)
        Xv = np.real(
            np.fft.ifft(s2[:, None] * np.fft.fft(X, axis=0), axis=0)
        )  # quadrants II, III (vertical offsets; flips do not change |omega|)
        Xh = np.real(np.fft.ifft(s2[None, :] * np.fft.fft(X, axis=1), axis=1))  # quadrants I, IV
        return (2 * (Xv + Xh))[:N, :N].ravel()

    op = spla.LinearOperator((N * N, N * N), matvec=W_apply, dtype=float)
    t0 = time.time()
    try:
        lo = spla.eigsh(op, k=1, which="SA", tol=1e-6, maxiter=5000, ncv=60, return_eigenvectors=False)[0]
    except spla.ArpackNoConvergence as e:
        lo = float(e.eigenvalues[0]) if len(e.eigenvalues) else float("nan")
    # upper bound on lambda_min(W) from a 2D low-pass test image (band |xi| < pi/4 on the padded torus), Rayleigh quotient
    rng = np.random.default_rng(0)
    X = np.zeros((L, L))
    X[:N, :N] = rng.standard_normal((N, N))
    F = np.fft.fft2(X)
    kx = np.abs(2 * np.pi * np.fft.fftfreq(L))
    F[(kx[:, None] >= np.pi / 4) | (kx[None, :] >= np.pi / 4)] = 0
    xt = np.real(np.fft.ifft2(F))[:N, :N].ravel()
    ray = xt @ W_apply(xt) / (xt @ xt)
    print(f"   Rayleigh quotient of W on a |xi|<pi/4 band-limited image: {ray:.4f}")
    # true lambda_min(R^T R)
    rop = spla.LinearOperator(
        (N * N, N * N), matvec=lambda v, n=n, N=N: ac.adrt_T(n, ac.adrt(n, v.reshape(N, N))).ravel(), dtype=float
    )
    lr = (
        spla.eigsh(rop, k=1, which="SA", tol=1e-4, maxiter=5000, return_eigenvectors=False)[0]
        if n <= 6
        else float("nan")
    )
    print(f"   lambda_min(W) = {lo:.4f}   vs   lambda_min(R^T R) = {lr:.3f}   ({time.time() - t0:.0f}s)", flush=True)
