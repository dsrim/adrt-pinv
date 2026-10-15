"""Hybrid inverse: SPIFE (explicit level-1 Gram inverse) for the cross-quadrant level, polynomial (Tikhonov) filter on the
upper levels m = 2..n.  Per quadrant and offset frequency: A_k = A'_k B_1, A'_k = B_n ... B_2 (upper levels), M' = A'^* A'.

Operators on the square (N^2 x N^2), all of the form  sum_q E^T T_q^T DFT^{-1} blockdiag_k(X_k) DFT T_q E :
   'orig' : X_k = M_k p(M_k)            (frame operator of the note, filter on the full quadrant Gram)
   'hyb'  : X_k = B_1^* M'_k p(M'_k) B_1 (G_h: filter on the upper levels, fused through S_(1)^T)
   'g1'   : X_k = B_1^* B_1             (G_1 = S_(1)^T S_(1), the level-1 Gram; its inverse is LORR's explicit S_(1)^dagger)
   'rtr'  : X_k = M_k                   (R^T R)
The hybrid preconditioned outer operator is Gt = G_1^{-1} G_h (similar to U^T (I x M'p(M')) U, spectrum in [0, 1+eps]).
Inverses (dense, for the accuracy comparison):
   L_orig = (E^T F E)^{-1} E^T sum_q p(M_q) R_q^T iota,   L_hyb = G_h^{-1} S_(1)^T (I x p(M') Q^T) iota.
"""

import numpy as np

import adrtcore as ac
from lab_frame import delta_table


def level_matrix(n, m, z):
    N = 1 << n
    w = 1 << (m - 1)
    nstrips = N // (2 * w)
    B = np.zeros((N, N), dtype=complex)
    t = np.arange(w)
    zt = z**t
    for col in range(N):
        g = np.zeros(N, dtype=complex)
        g[col] = 1
        f4 = g.reshape(nstrips, 2, w)
        out = np.empty((nstrips, w, 2), dtype=complex)
        out[:, :, 0] = f4[:, 0, :] + zt[None, :] * f4[:, 1, :]
        out[:, :, 1] = f4[:, 0, :] + (zt * z)[None, :] * f4[:, 1, :]
        B[:, col] = out.reshape(N)
    return B


class HybridLab:
    def __init__(self, n, tau_fac=0.2, L=None):
        N = self.N = 1 << n
        self.n = n
        L = self.L = L or 2 * N
        self.tau = tau_fac * N
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        z = np.exp(1j * self.oms)
        self.B1 = np.array([level_matrix(n, 1, zk) for zk in z])  # (L,N,N)
        Aup = np.array([np.eye(N, dtype=complex) for _ in z])
        for m in range(2, n + 1):
            Bm = np.array([level_matrix(n, m, zk) for zk in z])
            Aup = np.einsum("kij,kjl->kil", Bm, Aup)
        self.Aup = Aup
        self.A = np.einsum("kij,kjl->kil", Aup, self.B1)
        # check against the closed form
        D = delta_table(n, "digital")
        Aref = np.exp(1j * self.oms[:, None, None] * D[None])
        self.check = np.max(np.abs(self.A - Aref))

        def filt(Mx):
            lam, V = np.linalg.eigh(Mx)
            lam = np.maximum(lam, 0)
            f = lam / (lam + self.tau)
            return np.einsum("kij,kj,klj->kil", V, f, np.conj(V)), np.einsum(
                "kij,kj,klj->kil", V, f / np.maximum(lam, 1e-300) * (lam > 0), np.conj(V)
            )

        M = np.einsum("kji,kjl->kil", np.conj(self.A), self.A)
        Mup = np.einsum("kji,kjl->kil", np.conj(Aup), Aup)
        self.Phi, self.P = filt(M)  # Phi = M p(M), P = p(M)
        self.Phiup, self.Pup = filt(Mup)
        B1H = np.conj(np.transpose(self.B1, (0, 2, 1)))
        self.X = {
            "orig": self.Phi,
            "hyb": np.einsum("kij,kjl,klm->kim", B1H, self.Phiup, self.B1),
            "g1": np.einsum("kij,kjl->kil", B1H, self.B1),
            "rtr": M,
        }
        # backprojection matrices (data strip -> square), per frequency: orig: p(M) A^*;  hyb: B_1^* p(M') A'^*
        AH = np.conj(np.transpose(self.A, (0, 2, 1)))
        AupH = np.conj(np.transpose(Aup, (0, 2, 1)))
        self.BP = {
            "orig": np.einsum("kij,kjl->kil", self.P, AH),
            "hyb": np.einsum("kij,kjl,klm->kim", B1H, self.Pup, AupH),
        }

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

    def iota(self, bq):
        N, L = self.N, self.L
        S = np.zeros((L, N))
        S[np.arange(-(N - 1), N) % L] = bq
        return S

    def apply(self, kind, x):
        N, L = self.N, self.L
        X = np.zeros((L, L))
        X[:N, :N] = x
        out = np.zeros((L, L))
        for q in range(4):
            Sf = np.fft.fft(self.Tq(q, X), axis=0)
            out += self.TqT(q, np.real(np.fft.ifft(np.einsum("kij,kj->ki", self.X[kind], Sf), axis=0)))
        return out[:N, :N]

    def backproject(self, kind, b):
        N, L = self.N, self.L
        out = np.zeros((L, L))
        for q in range(4):
            Bf = np.fft.fft(self.iota(b[q]), axis=0)
            out += self.TqT(q, np.real(np.fft.ifft(np.einsum("kij,kj->ki", self.BP[kind], Bf), axis=0)))
        return out[:N, :N]

    def dense(self, kind):
        N = self.N
        K = np.zeros((N * N, N * N))
        for j in range(N * N):
            e = np.zeros(N * N)
            e[j] = 1
            K[:, j] = self.apply(kind, e.reshape(N, N)).ravel()
        return 0.5 * (K + K.T)


if __name__ == "__main__":
    import time

    import scipy.linalg as sla

    for n in (3, 4, 5, 6):
        N = 1 << n
        for tf in (0.2, 1.0):
            t0 = time.time()
            lab = HybridLab(n, tf)
            K = {k: lab.dense(k) for k in ("orig", "hyb", "g1", "rtr")}
            ev = {k: np.linalg.eigvalsh(K[k]) for k in K}
            evt = sla.eigh(K["hyb"], K["g1"], eigvals_only=True)  # spectrum of G_1^{-1} G_h
            line = (
                f"N={N:3d} tau={tf}N: check A={lab.check:.1e} | orig: [{ev['orig'][0]:.4f},{ev['orig'][-1]:.3f}] k={ev['orig'][-1] / ev['orig'][0]:.2f}"
                f" | G_h: [{ev['hyb'][0]:.4f},{ev['hyb'][-1]:.3f}] k={ev['hyb'][-1] / ev['hyb'][0]:.2f}"
                f" | G_1: [{ev['g1'][0]:.3f},{ev['g1'][-1]:.3f}] k={ev['g1'][-1] / ev['g1'][0]:.2f}"
                f" | hybrid G_1^-1 G_h: [{evt[0]:.4f},{evt[-1]:.4f}] k={evt[-1] / evt[0]:.2f} | lam_min(RtR)/tau={ev['rtr'][0] / lab.tau:.3f}"
            )
            # accuracy: dense left inverses on 1% noise, vs MP
            rng = np.random.default_rng(0)
            x = rng.standard_normal((N, N))
            b0 = ac.adrt(n, x)
            noise = rng.standard_normal(b0.shape)
            noise *= 0.01 * np.linalg.norm(b0) / np.linalg.norm(noise)
            b = b0 + noise
            xo = np.linalg.solve(K["orig"], lab.backproject("orig", b).ravel()).reshape(N, N)
            xh = np.linalg.solve(K["hyb"], lab.backproject("hyb", b).ravel()).reshape(N, N)
            # left-inverse check on noiseless data and MP via CGLS-free dense normal equations
            xo0 = np.linalg.solve(K["orig"], lab.backproject("orig", b0).ravel()).reshape(N, N)
            xh0 = np.linalg.solve(K["hyb"], lab.backproject("hyb", b0).ravel()).reshape(N, N)
            xmp = np.linalg.solve(K["rtr"], ac.adrt_T(n, b).ravel()).reshape(N, N)
            e = lambda y, x=x: np.linalg.norm(y - x) / np.linalg.norm(x)
            line += f" | LR=I: orig {e(xo0):.1e} hyb {e(xh0):.1e} | 1% noise err: MP {e(xmp):.3e} orig {e(xo) / e(xmp):.2f}x hyb {e(xh) / e(xmp):.2f}x  ({time.time() - t0:.0f}s)"
            print(line, flush=True)
