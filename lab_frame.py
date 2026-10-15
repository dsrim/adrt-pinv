"""Laboratory for the frame operator E^T F E of the oblique inverses.

Variants: lines='digital' (ADRT) or 'straight' (pseudo-polar-like: delta_s(t) = s t/(N-1));
L=2N (zero-padded offsets, window E) or L=N (periodic in the offset, no window);
filt='tikhonov' (lambda/(lambda+tau)), 'proj' (1[lambda>=theta]), 'ramp' (|omega| lambda / N), 'ident' (lambda).
Dense N^2 x N^2 matrices for N <= 64; fast apply otherwise.
"""

import numpy as np

import adrtcore as ac


def delta_table(n, lines):
    N = 1 << n
    if lines == "digital":
        return np.array([[ac.delta_s(n, s, t) for t in range(N)] for s in range(N)], dtype=float)
    if lines == "straight":
        s = np.arange(N)[:, None]
        t = np.arange(N)[None, :]
        return s * t / (N - 1.0)
    if lines == "bresenham":
        s = np.arange(N)[:, None]
        t = np.arange(N)[None, :]
        return np.floor(s * t / (N - 1.0) + 0.5)
    raise ValueError(lines)


class FrameLab:
    def __init__(self, n, tau_fac=0.2, lines="digital", L=None, filt="tikhonov", theta=None):
        N = self.N = 1 << n
        self.n = n
        L = self.L = L or 2 * N
        self.tau = tau_fac * N
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        D = delta_table(n, lines)
        self.A = np.exp(1j * self.oms[:, None, None] * D[None, :, :])  # (L, N, N): A_k(s,t)
        M = np.einsum("kst,ksu->ktu", np.conj(self.A), self.A)  # M_k = A_k^* A_k
        lam, V = np.linalg.eigh(M)  # (L,N), (L,N,N)
        lam = np.maximum(lam, 0)
        self.lam, self.V = lam, V
        if filt == "tikhonov":
            f = lam / (lam + self.tau)
        elif filt == "proj":
            f = (lam >= (theta if theta is not None else self.tau)).astype(float)
        elif filt == "ramp":
            f = np.abs(self.oms)[:, None] * lam / N
        elif filt == "ident":
            f = lam.copy()
        else:
            raise ValueError(filt)
        self.f = f
        self.phi = np.einsum("kij,kj,klj->kil", V, f, np.conj(V))  # phi_k = V f V^*

    # quadrant permutations (torus L x L -> strip L x N), as in fastspec
    def Tq(self, q, X):
        N, L = self.N, self.L
        r = (N - 1 - np.arange(L)) % L
        if q == 2:
            return X[:, :N]
        if q == 1:
            return X[r][:, :N]
        if q == 0:
            return X.T[r][:, :N]
        if q == 3:
            return X[r][:, r].T[:, :N]

    def TqT(self, q, S):
        N, L = self.N, self.L
        r = (N - 1 - np.arange(L)) % L
        X = np.zeros((L, L), dtype=S.dtype)
        X[:, :N] = S
        if q == 2:
            return X
        if q == 1:
            return X[r]
        if q == 0:
            return X[r].T
        if q == 3:
            Y = X.T
            Y = Y[:, r]
            return Y[r]

    def apply_quadrant(self, q, x):
        N, L = self.N, self.L
        X = np.zeros((L, L))
        X[:N, :N] = x
        Sf = np.fft.fft(self.Tq(q, X), axis=0)
        Pf = np.einsum("kij,kj->ki", self.phi, Sf)
        return self.TqT(q, np.real(np.fft.ifft(Pf, axis=0)))[:N, :N]

    def apply(self, x, quads=(0, 1, 2, 3)):
        return sum(self.apply_quadrant(q, x) for q in quads)

    def dense(self, quads=(0, 1, 2, 3)):
        N = self.N
        K = np.zeros((N * N, N * N))
        for j in range(N * N):
            e = np.zeros(N * N)
            e[j] = 1
            K[:, j] = self.apply(e.reshape(N, N), quads).ravel()
        return 0.5 * (K + K.T)


def fourier_diag(K, N):
    """Rayleigh quotients of the 2D plane waves e^{2 pi i (a i + b j)/N} on the square: diag of Phi^* K Phi."""
    K4 = K.reshape(N, N, N, N)
    # Phi^* K Phi diagonal: sum_{ij,i'j'} conj(e_xi(ij)) K(ij,i'j') e_xi(i'j') / N^2
    T = np.fft.fft2(
        np.fft.ifft2(K4, axes=(0, 1)), axes=(2, 3)
    )  # (1/N^2) sum_{ij} e^{-i xi ij} K ... sum e^{+i xi' i'j'}
    d = np.array([[np.real(T[a, b, a, b]) for b in range(N)] for a in range(N)])
    return d


if __name__ == "__main__":
    import time

    for n in (3, 4, 5):
        N = 1 << n
        for lines in ("digital", "straight"):
            for L in (2 * N, N):
                t0 = time.time()
                lab = FrameLab(n, 0.2, lines, L)
                K = lab.dense()
                ev, U = np.linalg.eigh(K)
                d = fourier_diag(K, N)
                a, b = np.unravel_index(d.argmin(), d.shape)
                umin = U[:, 0].reshape(N, N)
                uh = np.abs(np.fft.fft2(umin))
                am, bm = np.unravel_index(uh.argmax(), uh.shape)
                print(
                    f"N={N:3d} {lines:8s} L={L:3d}: lam_min={ev[0]:.4f} lam_max={ev[-1]:.3f} | plane-wave Rayleigh min={d.min():.4f} at xi=({a},{b}) "
                    f"| min-eigvec peak freq=({am},{bm}), its edge mass={np.sum(umin[[0, -1], :] ** 2) + np.sum(umin[:, [0, -1]] ** 2):.3f}  ({time.time() - t0:.0f}s)"
                )
