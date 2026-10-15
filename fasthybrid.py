"""Fast hybrid inverse of the ADRT: SPIFE-type explicit treatment of the cross-quadrant level S_(1), polynomial
(Chebyshev / Tikhonov) filters on the upper levels m = 2..n.  Companion of notes/hybrid_level1.tex.

    x = G_h^{-1} S_(1)^T Y,   Y = (I_4 x p(M') Q^T) iota b,   G_h = S_(1)^T (I_4 x M' p(M')) S_(1),

with M'_k = A'^*_k A'_k the Gram of the upper-level butterflies per offset frequency.  Everything is applied through
the offset-Fourier butterflies of fastspec (levels 2..n) and the level-1 butterfly B_1.  The outer solve is a fixed-degree
Chebyshev iteration on P G_h with an explicit SPD preconditioner P:
    'g1'   : P = G_1^{-1},  G_1 = S_(1)^T S_(1) = 8I + K (x) J + J (x) K  (K = tridiag(1,2,1), J = pair swap), spec(G_1) in [4,16];
             applied by a fixed-degree Chebyshev polynomial of the O(N^2) stencil (rate 1/3 per step).
    'lorr' : P = (S_(1)^T M S_(1))^{-1} = (4I + 8 Pi)^{-1} = (I - Pi)/4 + Pi/12, Pi = 2x2-cell averaging;  M drops the
             (h odd, s odd) class of level-1 data in every quadrant (LORR Thm 5.1 projection trick), exact and O(N^2).
    'none' : P = I.
Costs are counted in level-row units: one butterfly level applied to one frequency row (O(N) flops)."""

import numpy as np

import adrtcore as ac
from fastspec import cheb_coeffs


# ------------------------------------------------------------------ butterflies, selected levels
def bfly_levels(n, z, X, m_lo, m_hi, count=None):
    K, N = X.shape
    g = X
    for m in range(m_lo, m_hi + 1):
        w = 1 << (m - 1)
        nstrips = N // (2 * w)
        f4 = g.reshape(K, nstrips, 2, w)
        out = np.empty((K, nstrips, w, 2), dtype=complex)
        zt = z[:, None] ** np.arange(w)[None, :]
        out[:, :, :, 0] = f4[:, :, 0, :] + zt[:, None, :] * f4[:, :, 1, :]
        out[:, :, :, 1] = f4[:, :, 0, :] + (zt * z[:, None])[:, None, :] * f4[:, :, 1, :]
        g = out.reshape(K, N)
    if count is not None:
        count[0] += K * (m_hi - m_lo + 1)
    return g


def bflyT_levels(n, z, G, m_lo, m_hi, count=None):
    K, N = G.shape
    f = G
    for m in range(m_hi, m_lo - 1, -1):
        w = 1 << (m - 1)
        nstrips = N // (2 * w)
        g4 = f.reshape(K, nstrips, w, 2)
        out = np.empty((K, nstrips, 2, w), dtype=complex)
        zt = np.conj(z[:, None] ** np.arange(w)[None, :])
        out[:, :, 0, :] = g4[:, :, :, 0] + g4[:, :, :, 1]
        out[:, :, 1, :] = zt[:, None, :] * g4[:, :, :, 0] + (zt * np.conj(z)[:, None])[:, None, :] * g4[:, :, :, 1]
        f = out.reshape(K, N)
    if count is not None:
        count[0] += K * (m_hi - m_lo + 1)
    return f


def poly_apply_levels(n, z, Y, lam_max, tau, deg, m_lo, m_hi, count=None):
    """Z = p_k(M) Y_k, M = A^* A over levels m_lo..m_hi, p_k the degree-deg[k] Chebyshev polynomial for (M+tau)^{-1} on [0, lam_max[k]]."""
    K = Y.shape[0]
    c, a = cheb_coeffs(lam_max, tau)
    Z = a[:, None] * Y
    Zprev = np.zeros_like(Y)
    om = np.ones(K)
    order = np.argsort(-deg)
    for j in range(1, int(deg.max())):
        act = order[: int(np.sum(deg > j))]
        if act.size == 0:
            break
        omj = np.where(j == 1, 1.0 / (1.0 - c[act] ** 2 / 2.0), 1.0 / (1.0 - c[act] ** 2 * om[act] / 4.0))
        R = Y[act] - (
            bflyT_levels(n, z[act], bfly_levels(n, z[act], Z[act], m_lo, m_hi, count), m_lo, m_hi, count) + tau * Z[act]
        )
        Znew = Z[act] + omj[:, None] * (a[act][:, None] * R) + (omj - 1.0)[:, None] * (Z[act] - Zprev[act])
        Zprev[act] = Z[act]
        Z[act] = Znew
        om[act] = omj
    return Z


# ------------------------------------------------------------------ level-1 Gram and its explicit relatives (image stencils)
def K_rows(x):
    y = 2.0 * x
    y[1:] += x[:-1]
    y[:-1] += x[1:]
    return y


def J_cols(x):
    y = np.empty_like(x)
    y[:, 0::2] = x[:, 1::2]
    y[:, 1::2] = x[:, 0::2]
    return y


def G1_apply(x):
    """G_1 x = 8x + K x J + J x K  (K along rows/columns, J the pair swap of columns/rows)."""

    return 8.0 * x + J_cols(K_rows(x)) + (J_cols(K_rows(x.T))).T


def Pi_apply(x):
    """2x2-cell averaging projector."""
    N = x.shape[0]
    c = x.reshape(N // 2, 2, N // 2, 2).mean(axis=(1, 3))
    return np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)


def lorr_inv(y):
    """(4I + 8 Pi)^{-1} y = (y - Pi y)/4 + Pi y/12."""
    p = Pi_apply(y)
    return (y - p) / 4.0 + p / 12.0


def cheb_solve(op, y, lo, hi, deg):
    """Chebyshev semi-iteration approximating op^{-1} y for spec(op) in [lo, hi] (fixed degree)."""
    c = (hi - lo) / (hi + lo)
    a = 2.0 / (hi + lo)
    x_prev = np.zeros_like(y)
    x = a * y
    om = 1.0
    for k in range(1, deg):
        om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
        r = y - op(x)
        x_new = x + om * (a * r) + (om - 1.0) * (x - x_prev)
        x_prev, x = x, x_new
    return x


def g1_inv(y, deg=14):
    return cheb_solve(G1_apply, y, 4.0, 16.0, deg)


# ------------------------------------------------------------------ certified spectral bounds (row-sum recursion from level m_lo)
def cert_rowsum_levels(n, om, m_lo=2):
    """Certified upper bound on max row sum of |M_omega| for the Gram of levels m_lo..n (identity base at level m_lo-1):
    r_m(a) <= |1+z^a| r_{m-1}(2a) + max_pm |1+z^(a pm 1)| r_{m-1}(2a pm 1), twists |a| <= 2^(n-m)-1 at level m."""
    z = np.exp(1j * om)
    A = 1 << n
    r = np.ones(2 * A + 1)
    for m in range(m_lo, n + 1):
        amax = (1 << (n - m)) - 1
        a = np.arange(-amax, amax + 1)
        t0 = np.abs(1 + z**a) * r[2 * a + A]
        tp = np.abs(1 + z ** (a + 1)) * r[2 * a + 1 + A]
        tm = np.abs(1 + z ** (a - 1)) * r[2 * a - 1 + A]
        rn = np.zeros(2 * A + 1)
        rn[a + A] = t0 + np.maximum(tp, tm)
        r = rn
    return r[A]


# ------------------------------------------------------------------ the inverse
class FastHybridInverse:
    def __init__(
        self,
        n,
        tau_fac=0.2,
        eps=1e-10,
        power_iters=40,
        safety=1.5,
        precond="g1",
        g1_deg=14,
        outer_deg=None,
        lam_max=None,
        lam_bound="power",
    ):
        """lam_bound: 'power' (power iteration x safety) or 'cert' (certified row-sum recursion over levels 2..n)."""
        self.n = n
        N = self.N = 1 << n
        L = self.L = 2 * N
        self.tau = tau_fac * N
        self.precond = precond
        self.g1_deg = g1_deg
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        self.z = np.exp(1j * self.oms)
        self.count = [0]
        up = lambda V: bflyT_levels(n, self.z, bfly_levels(n, self.z, V, 2, n), 2, n)
        if lam_max is None and lam_bound == "cert":
            lam_max = np.array([cert_rowsum_levels(n, om, 2) for om in self.oms])
        if lam_max is None:  # Lambda'_k >= lambda_max(M'_k): power iteration + safety
            rng = np.random.default_rng(0)
            V = rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))
            for _ in range(power_iters):
                V = up(V)
                V /= np.linalg.norm(V, axis=1, keepdims=True)
            lam_max = safety * np.real(np.sum(np.conj(V) * up(V), axis=1))
        self.lam_max = np.maximum(lam_max, 1.0)
        kap = 1.0 + self.lam_max / self.tau
        rate = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
        self.deg = np.maximum(2, np.ceil(np.log(eps / 2.0) / np.log(np.maximum(rate, 1e-12)))).astype(int)
        # outer spectral interval of P G_h (generalized symmetric problem G_h u = theta P^{-1} u), Lanczos + residual enclosure
        import scipy.sparse.linalg as spla

        P, Pinv = self.P, self.Pinv
        A = spla.LinearOperator((N * N, N * N), matvec=lambda v: self.Gh(v.reshape(N, N)).ravel(), dtype=float)
        if precond == "none":
            lo, ulo = spla.eigsh(A, k=1, which="SA", tol=1e-3, maxiter=300)
            hi, uhi = spla.eigsh(A, k=1, which="LA", tol=1e-3, maxiter=300)
        else:
            Mop = spla.LinearOperator((N * N, N * N), matvec=lambda v: Pinv(v.reshape(N, N)).ravel(), dtype=float)
            Minv = spla.LinearOperator((N * N, N * N), matvec=lambda v: P(v.reshape(N, N)).ravel(), dtype=float)
            lo, ulo = spla.eigsh(A, k=1, M=Mop, Minv=Minv, which="SA", tol=1e-3, maxiter=300)
            hi, uhi = spla.eigsh(A, k=1, M=Mop, Minv=Minv, which="LA", tol=1e-3, maxiter=300)
        lo, hi = float(lo[0]), float(hi[0])

        def res(theta, u):  # ||P^{1/2}(G_h u - theta P^{-1} u)|| / ||P^{-1/2} u||  (B-norm Weinstein bound)
            u = u[:, 0]
            r = A.matvec(u) - theta * Mop.matvec(u) if precond != "none" else A.matvec(u) - theta * u
            if precond == "none":
                return np.linalg.norm(r) / np.linalg.norm(u)
            return np.sqrt(abs(r @ Minv.matvec(r))) / np.sqrt(abs(u @ Mop.matvec(u)))

        self.lo_ritz, self.hi_ritz, self.res_lo, self.res_hi = lo, hi, res(lo, ulo), res(hi, uhi)
        self.lo, self.hi = (lo - self.res_lo) * 0.99, (hi + self.res_hi) * 1.01
        kap_o = self.hi / self.lo
        self.outer_deg = (
            outer_deg
            if outer_deg is not None
            else int(np.ceil(np.log(eps / 2.0) / np.log((np.sqrt(kap_o) - 1) / (np.sqrt(kap_o) + 1)))) + 2
        )
        self.count = [0]

    # preconditioner and its inverse
    def P(self, y):
        if self.precond == "g1":
            return g1_inv(y, self.g1_deg)
        if self.precond == "lorr":
            return lorr_inv(y)
        return y

    def Pinv(self, y):
        if self.precond == "g1":
            return G1_apply(y)
        if self.precond == "lorr":
            return 4.0 * y + 8.0 * Pi_apply(y)
        return y

    # torus quadrant permutations, same as fastspec
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

    # G_h x = E^T sum_q T_q^T DFT^{-1} B_1^* M' p(M') B_1 DFT T_q E x
    def Gh(self, x):
        n, N, L, z, c = self.n, self.N, self.L, self.z, self.count
        X = np.zeros((L, L))
        X[:N, :N] = x
        out = np.zeros((L, L))
        for q in range(4):
            Sf = bfly_levels(n, z, np.fft.fft(self.Tq(q, X), axis=0), 1, 1, c)
            Pf = poly_apply_levels(n, z, Sf, self.lam_max, self.tau, self.deg, 2, n, c)
            Mf = bflyT_levels(n, z, bfly_levels(n, z, Pf, 2, n, c), 2, n, c)
            out += self.TqT(q, np.real(np.fft.ifft(bflyT_levels(n, z, Mf, 1, 1, c), axis=0)))
        return out[:N, :N]

    # S_(1)^T Y = E^T sum_q T_q^T DFT^{-1} B_1^* p(M') A'^* DFT iota b_q
    def backproject(self, b):
        n, N, L, z, c = self.n, self.N, self.L, self.z, self.count
        out = np.zeros((L, L))
        for q in range(4):
            Yf = bflyT_levels(n, z, np.fft.fft(self.iota(b[q]), axis=0), 2, n, c)
            Pf = poly_apply_levels(n, z, Yf, self.lam_max, self.tau, self.deg, 2, n, c)
            out += self.TqT(q, np.real(np.fft.ifft(bflyT_levels(n, z, Pf, 1, 1, c), axis=0)))
        return out[:N, :N]

    def apply(self, b):
        """x = G_h^{-1} S_(1)^T Y by a fixed-degree Chebyshev iteration on P G_h."""
        y = self.backproject(b)
        c = (self.hi - self.lo) / (self.hi + self.lo)
        a = 2.0 / (self.hi + self.lo)
        x_prev = np.zeros_like(y)
        x = a * self.P(y)
        om = 1.0
        for k in range(1, self.outer_deg):
            om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
            r = self.P(y - self.Gh(x))
            x_new = x + om * (a * r) + (om - 1.0) * (x - x_prev)
            x_prev, x = x, x_new
        return x


if __name__ == "__main__":
    import time

    from lab_hybrid import HybridLab

    n = 3
    N = 1 << n
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N, N))
    lab = HybridLab(n, 0.2)
    # stencils vs dense
    G1d = lab.dense("g1")
    print("G1 stencil vs dense:", np.abs(G1_apply(x).ravel() - G1d @ x.ravel()).max())
    print("G1^-1 (cheb 14) residual:", np.linalg.norm(G1_apply(g1_inv(x)) - x) / np.linalg.norm(x))
    for pc in ("g1", "lorr", "none"):
        t0 = time.time()
        F = FastHybridInverse(n, 0.2, precond=pc)
        Ghd = lab.dense("hyb")
        print(f"[{pc}] Gh fast vs dense: {np.abs(F.Gh(x).ravel() - Ghd @ x.ravel()).max():.2e}", end=" | ")
        b = ac.adrt(n, x)
        print(f"backproject fast vs lab: {np.abs(F.backproject(b) - lab.backproject('hyb', b)).max():.2e}", end=" | ")
        print(
            f"outer [{F.lo:.4f},{F.hi:.4f}] (ritz [{F.lo_ritz:.4f},{F.hi_ritz:.4f}], res {F.res_lo:.1e},{F.res_hi:.1e}) deg {F.outer_deg} | sum deg/N {F.deg.sum() / N:.1f}",
            end=" | ",
        )
        xr = F.apply(b)
        print(f"L R x - x: {np.linalg.norm(xr - x) / np.linalg.norm(x):.2e}  ({time.time() - t0:.1f}s)")
