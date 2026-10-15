"""
Optimal-complexity explicit oblique inverse of the ADRT (polynomial filter family).

Per offset frequency omega_k (k < L = 2N) the cyclic single-quadrant ADRT is the butterfly A_k
(N x N); it is applied in O(N log N) by `bfly`.  Let M_k = A_k^* A_k and let p_k be the degree-d_k
Chebyshev polynomial of the shifted inverse (lam + tau)^{-1} on [0, Lambda_k] (Lambda_k >= lambda_max(M_k)).
For ANY family of polynomials the operator

    L_p = (E^T F_p E)^{-1} E^T sum_q  p(M) A^* iota ,      F_p = sum_q  M p(M)

is an exact left inverse of the (windowed) ADRT R.  Choosing d_k ~ sqrt(Lambda_k / tau) gives
sum_k d_k = O(N), hence O(N^2 log N) per application of  sum_q p(M)A^* iota  and of F_p; the outer
inverse (E^T F_p E)^{-1} is a fixed-degree Chebyshev polynomial (kappa(E^T F_p E) <= ~6).
Nothing is iterated to a tolerance: all degrees are fixed in the (data-independent) setup.
"""

import numpy as np


# ------------------------------------------------------------------ butterflies
def bfly(n, z, X):
    """A_omega X for an array of frequencies: X (K, N) over columns t -> (K, N) over slopes s; z = exp(i omega) (K,)."""
    K, N = X.shape
    g = X
    for m in range(1, n + 1):
        w = 1 << (m - 1)
        nstrips = N // (2 * w)
        f4 = g.reshape(K, nstrips, 2, w)
        out = np.empty((K, nstrips, w, 2), dtype=complex)
        t = np.arange(w)
        zt = z[:, None] ** t[None, :]  # (K, w): z^t
        out[:, :, :, 0] = f4[:, :, 0, :] + zt[:, None, :] * f4[:, :, 1, :]
        out[:, :, :, 1] = f4[:, :, 0, :] + (zt * z[:, None])[:, None, :] * f4[:, :, 1, :]
        g = out.reshape(K, N)
    return g


def bflyT(n, z, G):
    """A_omega^* G : (K, N) over slopes -> (K, N) over columns."""
    K, N = G.shape
    f = G
    for m in range(n, 0, -1):
        w = 1 << (m - 1)
        nstrips = N // (2 * w)
        g4 = f.reshape(K, nstrips, w, 2)
        out = np.empty((K, nstrips, 2, w), dtype=complex)
        t = np.arange(w)
        zt = np.conj(z[:, None] ** t[None, :])
        out[:, :, 0, :] = g4[:, :, :, 0] + g4[:, :, :, 1]
        out[:, :, 1, :] = zt[:, None, :] * g4[:, :, :, 0] + (zt * np.conj(z)[:, None])[:, None, :] * g4[:, :, :, 1]
        f = out.reshape(K, N)
    return f


# ----------------------------------------------------- per-frequency polynomials
def cheb_coeffs(lam_max, tau):
    """Chebyshev semi-iteration constants for B = M + tau with spec(M) in [0, lam_max]: interval [tau, lam_max+tau]."""
    lo, hi = tau, lam_max + tau
    return (hi - lo) / (hi + lo), 2.0 / (hi + lo)  # c, a  (per frequency arrays)


def poly_apply(n, z, Y, lam_max, tau, deg, count=None):
    """Z = p_k(M_k) Y_k  for every frequency row k, where p_k is the degree-deg[k] Chebyshev
    semi-iteration polynomial for (M + tau)^{-1} on spec(M) in [0, lam_max[k]].  Rows are processed
    only while deg[k] > step (adaptive cost)."""
    K = Y.shape[0]
    c, a = cheb_coeffs(lam_max, tau)
    Z = np.zeros_like(Y)
    Zprev = np.zeros_like(Y)
    om = np.ones(K)
    Z[:] = a[:, None] * Y  # step 0: x_1 = a y
    dmax = int(deg.max())
    order = np.argsort(-deg)  # rows sorted by decreasing degree
    for j in range(1, dmax):
        act = order[: int(np.sum(deg > j))]
        if act.size == 0:
            break
        omj = np.where(j == 1, 1.0 / (1.0 - c[act] ** 2 / 2.0), 1.0 / (1.0 - c[act] ** 2 * om[act] / 4.0))
        R = Y[act] - (bflyT(n, z[act], bfly(n, z[act], Z[act])) + tau * Z[act])
        if count is not None:
            count[0] += act.size
        Znew = Z[act] + omj[:, None] * (a[act][:, None] * R) + (omj - 1.0)[:, None] * (Z[act] - Zprev[act])
        Zprev[act] = Z[act]
        Z[act] = Znew
        om[act] = omj
    return Z


# ------------------------------------------------------------------- the inverse
class FastSpectralInverse:
    def __init__(self, n, tau_fac=0.2, eps=1e-10, power_iters=40, safety=1.5, outer_deg=None, lam_bound="power"):
        """lam_bound: 'power' (power iteration + safety factor) or 'explicit' (Lambda_k = min(N^2, 2N(1+log2 N)/|omega_k|))."""
        self.n = n
        N = self.N = 1 << n
        L = self.L = 2 * N
        self.tau = tau_fac * N
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        self.z = np.exp(1j * self.oms)
        # --- data-independent spectral bounds Lambda_k >= lambda_max(A_k^* A_k) by power iteration (+ safety)
        if lam_bound == "explicit":
            om = np.abs(self.oms)
            om[0] = np.inf
            self.lam_max = np.minimum(float(N) ** 2, 2.0 * N * (1 + n) / om)
            self.lam_max[0] = float(N) ** 2
        else:
            rng = np.random.default_rng(0)
            V = rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))
            for _ in range(power_iters):
                V = bflyT(n, self.z, bfly(n, self.z, V))
                V /= np.linalg.norm(V, axis=1, keepdims=True)
            W = bflyT(n, self.z, bfly(n, self.z, V))
            self.lam_max = safety * np.real(np.sum(np.conj(V) * W, axis=1))
            self.lam_max = np.maximum(self.lam_max, 1.0)
        # --- per-frequency degrees:  2 (sqrt(k)-1)/(sqrt(k)+1))^d <= eps, k = 1 + Lambda/tau
        kap = 1.0 + self.lam_max / self.tau
        rate = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
        self.deg = np.maximum(2, np.ceil(np.log(eps / 2.0) / np.log(np.maximum(rate, 1e-12)))).astype(int)
        self.count = [0]
        # --- outer spectral interval of E^T F_p E by a few Lanczos steps (data independent), with margins
        import scipy.sparse.linalg as spla

        op = spla.LinearOperator(
            (N * N, N * N), matvec=lambda v: self.frame_window(v.reshape(N, N)).ravel(), dtype=float
        )
        lo, ulo = spla.eigsh(op, k=1, which="SA", tol=1e-3, maxiter=200)
        hi, uhi = spla.eigsh(op, k=1, which="LA", tol=1e-3, maxiter=200)
        lo, hi = lo[0], hi[0]
        # guaranteed enclosure (Weinstein): lambda_min >= theta - ||r||, lambda_max <= theta' + ||r'||
        r_lo = np.linalg.norm(op.matvec(ulo[:, 0]) - lo * ulo[:, 0])
        r_hi = np.linalg.norm(op.matvec(uhi[:, 0]) - hi * uhi[:, 0])
        self.lo_ritz, self.hi_ritz, self.res_lo, self.res_hi = lo, hi, r_lo, r_hi
        self.lo, self.hi = lo - r_lo, hi + r_hi
        kap_o = self.hi / self.lo
        self.outer_deg = (
            outer_deg
            if outer_deg is not None
            else int(np.ceil(np.log(eps / 2.0) / np.log((np.sqrt(kap_o) - 1) / (np.sqrt(kap_o) + 1)))) + 2
        )
        self.count = [0]

    # torus quadrant permutations (image L x L -> strip L x N), same as specpart
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

    # F_p applied to a window image:  E^T sum_q T_q^T [ M p(M) ] T_q E x
    def frame_window(self, x):
        N, L = self.N, self.L
        X = np.zeros((L, L))
        X[:N, :N] = x
        out = np.zeros((L, L))
        for q in range(4):
            Sf = np.fft.fft(self.Tq(q, X), axis=0)
            Pf = poly_apply(self.n, self.z, Sf, self.lam_max, self.tau, self.deg, self.count)
            Mf = bflyT(self.n, self.z, bfly(self.n, self.z, Pf))
            self.count[0] += L
            out += self.TqT(q, np.real(np.fft.ifft(Mf, axis=0)))
        return out[:N, :N]

    # E^T sum_q p(M) A^* iota b
    def backproject(self, b):
        N, L = self.N, self.L
        out = np.zeros((L, L))
        for q in range(4):
            Bf = np.fft.fft(self.iota(b[q]), axis=0)
            Yf = bflyT(self.n, self.z, Bf)
            self.count[0] += L
            Pf = poly_apply(self.n, self.z, Yf, self.lam_max, self.tau, self.deg, self.count)
            out += self.TqT(q, np.real(np.fft.ifft(Pf, axis=0)))
        return out[:N, :N]

    def apply(self, b):
        """x = L_p b  with the outer inverse realized by a fixed-degree Chebyshev polynomial."""
        y = self.backproject(b)
        c = (self.hi - self.lo) / (self.hi + self.lo)
        a = 2.0 / (self.hi + self.lo)
        x_prev = np.zeros_like(y)
        x = a * y
        om = 1.0
        for k in range(1, self.outer_deg):
            om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
            r = y - self.frame_window(x)
            x_new = x + om * (a * r) + (om - 1.0) * (x - x_prev)
            x_prev, x = x, x_new
        return x
