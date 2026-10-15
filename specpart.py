"""
Spectral-partition left inverse (oblique projection onto rg R along the complement of the
quadrant-wise stable eigenspaces), padded-torus construction.

Torus image space: L x L, L = 2N (cyclic in both directions).  For quadrant q the strip image is
    S_q(X) = (T_q X)[:, :N]      (L rows = offsets, N columns = slopes' t-index)
with the torus extensions of the quadrant permutations T_q (row/column reflections mod L, transpose).
The cyclic single-quadrant ADRT of a strip is, per offset frequency omega_k, the N x N butterfly A_k.
    Phi_q  = T_q^{-1} [ F_i^{-1} blockdiag_k(P_k) F_i on the strip, 0 on the other columns ] T_q,
    P_k    = spectral projector of A_k^* A_k onto eigenvalues >= theta,
    F      = sum_q Phi_q   (frame operator),   E = zero-padding of the N x N image window.
Exact identity (for every theta):   L_w R x = x,   L_w := (E^T F E)^{-1} E^T sum_q Phi_q A_q^{-1} iota.
"""

import numpy as np

import adrtcore as ac


class SpectralPartition:
    def __init__(self, n, theta_fac=0.05):
        self.n = n
        N = self.N = 1 << n
        L = self.L = 2 * N
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        D = np.array([[ac.delta_s(n, s, t) for t in range(N)] for s in range(N)], dtype=float)
        self.P = np.empty((L, N, N), dtype=complex)  # spectral projectors
        self.PAinv = np.empty((L, N, N), dtype=complex)  # Phi_k A_k^+ = V_sel diag(1/w_sel) V_sel^* A_k^*
        for k, om in enumerate(self.oms):
            A = np.exp(1j * om * D)
            w, V = np.linalg.eigh(A.conj().T @ A)
            sel = w >= theta_fac * N
            self.P[k] = V[:, sel] @ V[:, sel].conj().T
            self.PAinv[k] = (V[:, sel] / w[sel]) @ V[:, sel].conj().T @ A.conj().T

    # ---- torus quadrant permutations (image L x L -> strip L x N) and their inverses
    def Tq(self, q, X):
        N, L = self.N, self.L
        if q == 2:
            return X[:, :N]  # III: identity
        if q == 1:
            return X[(N - 1 - np.arange(L)) % L][:, :N]  # II : row reflection i -> N-1-i
        if q == 0:
            return X.T[(N - 1 - np.arange(L)) % L][:, :N]  # I  : x(j, N-1-i) = X^T(N-1-i, j)
        if q == 3:
            return X[(N - 1 - np.arange(L)) % L][:, (N - 1 - np.arange(L)) % L].T[:, :N]  # IV : x(N-1-j, N-1-i)

    def TqT(self, q, S):
        """adjoint of Tq: strip L x N -> image L x L (zero on untouched entries)"""
        N, L = self.N, self.L
        X = np.zeros((L, L), dtype=S.dtype)
        X[:, :N] = S
        if q == 2:
            return X
        if q == 1:
            return X[(N - 1 - np.arange(L)) % L]
        if q == 0:
            return X[(N - 1 - np.arange(L)) % L].T
        if q == 3:
            Y = X.T
            Y = Y[:, (N - 1 - np.arange(L)) % L]
            return Y[(N - 1 - np.arange(L)) % L]

    # ---- per-frequency operators on a strip
    def _apply_k(self, S, M):
        Sf = np.fft.fft(S, axis=0)
        Yf = np.einsum("kij,kj->ki", M, Sf)
        return np.fft.ifft(Yf, axis=0)

    def Phi(self, X):
        out = np.zeros_like(X, dtype=complex)
        for q in range(4):
            out += self.TqT(q, self._apply_k(self.Tq(q, X), self.P))
        return out

    def frame_window(self, x):
        """(E^T F E) x for an N x N image"""
        N, L = self.N, self.L
        X = np.zeros((L, L), dtype=complex)
        X[:N, :N] = x
        return self.Phi(X)[:N, :N]

    # ---- data embedding: my layout b[q] (2N-1, N), row h+N-1  ->  cyclic strip (L, N), row h mod L
    def iota(self, bq):
        N, L = self.N, self.L
        S = np.zeros((L, N), dtype=complex)
        for h in range(-(N - 1), N):
            S[h % L] = bq[h + N - 1]
        return S

    def quadrant_inverse(self, b):
        """sum_q Phi_q A_q^{-1} iota b   as an L x L torus image"""
        out = np.zeros((self.L, self.L), dtype=complex)
        for q in range(4):
            S = self.iota(b[q])
            Y = self._apply_k(S, self.PAinv)  # Phi_q A_q^+ (per frequency)
            out += self.TqT(q, Y)
        return out

    def dense_frame(self):
        N = self.N
        return ac.dense_from_op(lambda x: np.real(self.frame_window(x)), (N, N), (N, N))

    def apply(self, b, Fw_inv=None):
        N = self.N
        y = np.real(self.quadrant_inverse(b)[:N, :N])
        if Fw_inv is None:
            Fw_inv = np.linalg.inv(self.dense_frame())
        return (Fw_inv @ y.ravel()).reshape(N, N)
