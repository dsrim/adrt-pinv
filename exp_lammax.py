"""Check the spectral bound lambda_max(A_omega^* A_omega) <= C N/|omega| used for the polynomial degrees."""

import sys

import numpy as np

import fastspec as fs

rng = np.random.default_rng(1)
for n in [int(a) for a in sys.argv[1:]]:
    N = 1 << n
    L = 2 * N
    oms = 2 * np.pi * np.fft.fftfreq(L)
    z = np.exp(1j * oms)
    V = rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))
    for _ in range(80):
        V = fs.bflyT(n, z, fs.bfly(n, z, V))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    lam = np.real(np.sum(np.conj(V) * fs.bflyT(n, z, fs.bfly(n, z, V)), axis=1))
    k = np.arange(L)
    nz = k > 0
    ratio = lam[nz] * np.abs(oms[nz]) / N  # should be <= C
    print(
        f"N={N}: lam_max(0)/N^2={lam[0] / N**2:.3f}; max over omega!=0 of lam_max*|omega|/N = {ratio.max():.3f} (2pi={2 * np.pi:.3f}); at omega=pi/N: {lam[1] * oms[1] / N:.3f}; at omega=pi: {lam[L // 2] / N:.3f}",
        flush=True,
    )
