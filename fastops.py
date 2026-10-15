"""Optimized engine for the oblique inverses L_p (fastspec) and the hybrid (fasthybrid).  Same operators, faster evaluation:
  1. twiddle tables z^t precomputed per level (fastspec recomputes complex powers in every butterfly call);
  2. frequency rows sorted by inner degree once, so the adaptive Chebyshev loop works on contiguous prefixes (no gather/scatter);
  3. quadrant packing: the per-quadrant operators are real on real strips (degrees symmetrized so p_{-k} = conj p_k), so two
     quadrants are pushed through one complex pass as a + i b and separated by Re/Im afterwards: 2 passes instead of 4;
  4. rows split into balanced chunks (round-robin in degree order) processed by a thread pool (numpy releases the GIL);
  5. optional dense low-frequency table (dense_cap): rows whose inner degree exceeds dense_cap use the exact Tikhonov filter
     (M_k + tau)^{-1}, precomputed by a dense eigendecomposition in the setup and applied as one matrix-vector product.
     Exactness is unaffected (any filter applied consistently gives a left inverse); this cuts both work and parallel depth.
Classes FastLp and FastHybrid mirror fastspec.FastSpectralInverse and fasthybrid.FastHybridInverse."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import fasthybrid as fh
import fastspec as fs


class RowEngine:
    def __init__(self, n, z, deg, lam_max, tau, threads=1, copies=2, dense_cap=None, p_levels=(1, None)):
        self.n = n
        self.N = N = 1 << n
        self.R = copies * len(z)
        self.tau = tau
        Lz = len(z)
        dense = {}
        if dense_cap is not None:  # exact Tikhonov filters for the high-degree rows (shared by copies)
            lo = p_levels[0]
            hi = p_levels[1] or n
            for k in np.nonzero(deg > dense_cap)[0]:
                Ak = (
                    fs.bfly(n, np.full(N, z[k]), np.eye(N, dtype=complex)).T
                    if (lo, hi) == (1, n)
                    else fh.bfly_levels(n, np.full(N, z[k]), np.eye(N, dtype=complex), lo, hi).T
                )
                lam, V = np.linalg.eigh(Ak.conj().T @ Ak)
                lam = np.maximum(lam, 0)
                dense[k] = (V / (lam + tau)) @ V.conj().T
        self.n_dense = len(dense)
        zz, dd, lm = np.tile(z, copies), np.tile(deg, copies), np.tile(lam_max, copies)
        order = np.argsort(-dd, kind="stable")
        T = max(1, threads)
        self.pool = ThreadPoolExecutor(T) if T > 1 else None
        self.chunks = []
        for c in range(T):
            idx = order[c::T]
            zc = zz[idx]
            d = dd[idx].copy()
            cc, a = fs.cheb_coeffs(lm[idx], tau)
            kd = [int(i) % Lz for i in idx if (int(i) % Lz) in dense]  # dense rows come first (largest degrees)
            Fd = np.array([dense[k] for k in kd]) if kd else None
            tw = {m: zc[:, None] ** np.arange(1 << (m - 1))[None, :] for m in range(1, n + 1)}
            self.chunks.append(
                {
                    "idx": idx,
                    "z": zc,
                    "zconj": np.conj(zc),
                    "deg": d,
                    "a": a,
                    "c": cc,
                    "tw": tw,
                    "twc": {m: np.conj(v) for m, v in tw.items()},
                    "Kd": len(kd),
                    "Fd": Fd,
                }
            )
        self.count = [0]

    # ---- per-chunk kernels on the first K rows
    def _B(self, ch, g, lo, hi, K):
        N = self.N
        for m in range(lo, hi + 1):
            w = 1 << (m - 1)
            ns = N // (2 * w)
            f4 = g.reshape(K, ns, 2, w)
            t = f4[:, :, 1, :] * ch["tw"][m][:K, None, :]
            out = np.empty((K, ns, w, 2), dtype=complex)
            out[..., 0] = f4[:, :, 0, :] + t
            t *= ch["z"][:K, None, None]
            out[..., 1] = f4[:, :, 0, :] + t
            g = out.reshape(K, N)
        return g

    def _BT(self, ch, f, lo, hi, K):
        N = self.N
        for m in range(hi, lo - 1, -1):
            w = 1 << (m - 1)
            ns = N // (2 * w)
            g4 = f.reshape(K, ns, w, 2)
            out = np.empty((K, ns, 2, w), dtype=complex)
            out[:, :, 0, :] = g4[..., 0] + g4[..., 1]
            t = g4[..., 1] * ch["zconj"][:K, None, None]
            t += g4[..., 0]
            t *= ch["twc"][m][:K, None, :]
            out[:, :, 1, :] = t
            f = out.reshape(K, N)
        return f

    def _P(self, ch, Y, lo, hi):
        Kd = ch["Kd"]
        if Kd:
            Z = np.empty_like(Y)
            Z[:Kd] = np.matmul(ch["Fd"], Y[:Kd, :, None])[:, :, 0]
            sub = dict(ch)
            sub.update(
                z=ch["z"][Kd:],
                zconj=ch["zconj"][Kd:],
                deg=ch["deg"][Kd:],
                a=ch["a"][Kd:],
                c=ch["c"][Kd:],
                tw={m: v[Kd:] for m, v in ch["tw"].items()},
                twc={m: v[Kd:] for m, v in ch["twc"].items()},
                Kd=0,
            )
            Z[Kd:], s = self._P(sub, Y[Kd:], lo, hi)
            return Z, s + Kd * self.N // max(1, self.n)  # dense matvec ~ N^2 flops = N/n level-rows
        d = ch["deg"]
        a, c = ch["a"], ch["c"]
        tau = self.tau
        Z = a[:, None] * Y
        Zprev = np.zeros_like(Y)
        om = np.ones(len(d))
        steps = 0
        for j in range(1, int(d[0]) if len(d) else 0):
            K = int(np.sum(d > j))
            if K == 0:
                break
            cK = c[:K]
            omj = 1.0 / (1.0 - cK**2 / 2.0) if j == 1 else 1.0 / (1.0 - cK**2 * om[:K] / 4.0)
            ZK = Z[:K]
            Rr = Y[:K] - self._BT(ch, self._B(ch, ZK, lo, hi, K), lo, hi, K) - tau * ZK
            Znew = ZK + omj[:, None] * (a[:K, None] * Rr) + (omj - 1.0)[:, None] * (ZK - Zprev[:K])
            Zprev[:K] = ZK
            Z[:K] = Znew
            om[:K] = omj
            steps += K * 2 * (hi - lo + 1)
        return Z, steps

    def _chunk(self, ch, Y, ops):
        X = Y[ch["idx"]]
        K = X.shape[0]
        steps = 0
        for op, lo, hi in ops:
            if op == "B":
                X = self._B(ch, X, lo, hi, K)
                steps += K * (hi - lo + 1)
            elif op == "BT":
                X = self._BT(ch, X, lo, hi, K)
                steps += K * (hi - lo + 1)
            elif op == "P":
                X, s = self._P(ch, X, lo, hi)
                steps += s
        return X, steps

    def run(self, Y, ops):
        """Y: (R, N) complex rows (copies stacked); ops: list of ('B'|'BT'|'P', m_lo, m_hi), applied left to right."""
        out = np.empty_like(Y)
        res = (
            self.pool.map(lambda ch: self._chunk(ch, Y, ops), self.chunks)
            if self.pool
            else (self._chunk(ch, Y, ops) for ch in self.chunks)
        )
        for ch, (X, s) in zip(self.chunks, res):
            out[ch["idx"]] = X
            self.count[0] += s
        return out


def _sym(lam):
    L = len(lam)
    return np.maximum(lam, lam[(-np.arange(L)) % L])


class _Base:
    PAIRS = ((0, 1), (2, 3))
    engine_cls = RowEngine  # fastcpp.CppEngine substitutes the C++ row kernels

    def _setup_common(self, n, tau_fac, eps_in, threads):
        self.n = n
        N = self.N = 1 << n
        L = self.L = 2 * N
        self.tau = tau_fac * N
        self.threads = threads
        self.oms = 2 * np.pi * np.fft.fftfreq(L)
        self.z = np.exp(1j * self.oms)

    def _degrees(self, lam_max, eps_in, dense_cap=None, p_levels=(1, None)):
        self.lam_max = np.maximum(_sym(lam_max), 1.0)
        kap = 1.0 + self.lam_max / self.tau
        rate = (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)
        self.deg = np.maximum(2, np.ceil(np.log(eps_in / 2.0) / np.log(np.maximum(rate, 1e-12)))).astype(int)
        self.eng = self.engine_cls(
            self.n,
            self.z,
            self.deg,
            self.lam_max,
            self.tau,
            self.threads,
            copies=2,
            dense_cap=dense_cap,
            p_levels=p_levels,
        )

    Tq = fs.FastSpectralInverse.Tq
    TqT = fs.FastSpectralInverse.TqT
    iota = fs.FastSpectralInverse.iota

    def _packed(self, strips_fn, ops, post):
        """strips_fn(q) -> real (L,N) strip; packs quadrant pairs, FFT, runs ops, iFFT, unpacks, applies post(q, real strip)."""
        L, N = self.L, self.N
        Y = np.empty((2 * L, N), dtype=complex)
        for i, (qa, qb) in enumerate(self.PAIRS):
            Y[i * L : (i + 1) * L] = np.fft.fft(strips_fn(qa) + 1j * strips_fn(qb), axis=0)
        Z = self.eng.run(Y, ops)
        out = np.zeros((L, L))
        for i, (qa, qb) in enumerate(self.PAIRS):
            W = np.fft.ifft(Z[i * L : (i + 1) * L], axis=0)
            out += self.TqT(qa, W.real) + self.TqT(qb, W.imag)
        return out[:N, :N]

    def _outer_interval(self, Gop, Pinv=None, P=None, eps=1e-10):
        import scipy.sparse.linalg as spla

        N = self.N
        A = spla.LinearOperator((N * N, N * N), matvec=lambda v: Gop(v.reshape(N, N)).ravel(), dtype=float)
        if Pinv is None:
            lo, ulo = spla.eigsh(A, k=1, which="SA", tol=1e-3, maxiter=300)
            hi, uhi = spla.eigsh(A, k=1, which="LA", tol=1e-3, maxiter=300)
            res = lambda th, u: np.linalg.norm(A.matvec(u[:, 0]) - th * u[:, 0])
        else:
            Mop = spla.LinearOperator((N * N, N * N), matvec=lambda v: Pinv(v.reshape(N, N)).ravel(), dtype=float)
            Minv = spla.LinearOperator((N * N, N * N), matvec=lambda v: P(v.reshape(N, N)).ravel(), dtype=float)
            lo, ulo = spla.eigsh(A, k=1, M=Mop, Minv=Minv, which="SA", tol=1e-3, maxiter=300)
            hi, uhi = spla.eigsh(A, k=1, M=Mop, Minv=Minv, which="LA", tol=1e-3, maxiter=300)

            def res(th, u):
                u = u[:, 0]
                r = A.matvec(u) - th * Mop.matvec(u)
                return np.sqrt(abs(r @ Minv.matvec(r))) / np.sqrt(abs(u @ Mop.matvec(u)))

        lo, hi = float(lo[0]), float(hi[0])
        self.lo, self.hi = (lo - res(lo, ulo)) * 0.99, (hi + res(hi, uhi)) * 1.01
        kap = self.hi / self.lo
        self.outer_deg = int(np.ceil(np.log(eps / 2.0) / np.log((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)))) + 2

    def _outer(self, y, Gop, P=lambda v: v):
        c = (self.hi - self.lo) / (self.hi + self.lo)
        a = 2.0 / (self.hi + self.lo)
        x_prev = np.zeros_like(y)
        x = a * P(y)
        om = 1.0
        for k in range(1, self.outer_deg):
            om = 1.0 / (1.0 - c * c / 2.0) if k == 1 else 1.0 / (1.0 - c * c * om / 4.0)
            r = P(y - Gop(x))
            x_new = x + om * (a * r) + (om - 1.0) * (x - x_prev)
            x_prev, x = x, x_new
        return x


class FastLp(_Base):
    def __init__(
        self, n, tau_fac=0.2, eps_in=0.1, eps=1e-10, threads=8, lam_max=None, power_iters=40, safety=1.5, dense_cap=None
    ):
        self._setup_common(n, tau_fac, eps_in, threads)
        if lam_max is None:
            rng = np.random.default_rng(0)
            V = rng.standard_normal((self.L, self.N)) + 1j * rng.standard_normal((self.L, self.N))
            for _ in range(power_iters):
                V = fs.bflyT(n, self.z, fs.bfly(n, self.z, V))
                V /= np.linalg.norm(V, axis=1, keepdims=True)
            lam_max = safety * np.real(np.sum(np.conj(V) * fs.bflyT(n, self.z, fs.bfly(n, self.z, V)), axis=1))
        self._degrees(lam_max, eps_in, dense_cap, (1, None))
        self._outer_interval(self.frame_window, eps=eps)
        self.eng.count = [0]

    def frame_window(self, x):
        n, L = self.n, self.L
        X = np.zeros((L, L))
        X[: self.N, : self.N] = x
        return self._packed(lambda q: self.Tq(q, X), [("P", 1, n), ("B", 1, n), ("BT", 1, n)], None)

    def backproject(self, b):
        n = self.n
        return self._packed(lambda q: self.iota(b[q]), [("BT", 1, n), ("P", 1, n)], None)

    def apply(self, b):
        return self._outer(self.backproject(b), self.frame_window)


class FastHybrid(_Base):
    def __init__(
        self,
        n,
        tau_fac=0.2,
        eps_in=0.1,
        eps=1e-10,
        threads=8,
        lam_max=None,
        g1_deg=14,
        power_iters=40,
        safety=1.5,
        dense_cap=None,
        precond="g1",
    ):
        self._setup_common(n, tau_fac, eps_in, threads)
        self.g1_deg = g1_deg
        self.precond = precond
        if lam_max is None:
            up = lambda V: fh.bflyT_levels(n, self.z, fh.bfly_levels(n, self.z, V, 2, n), 2, n)
            rng = np.random.default_rng(0)
            V = rng.standard_normal((self.L, self.N)) + 1j * rng.standard_normal((self.L, self.N))
            for _ in range(power_iters):
                V = up(V)
                V /= np.linalg.norm(V, axis=1, keepdims=True)
            lam_max = safety * np.real(np.sum(np.conj(V) * up(V), axis=1))
        self._degrees(lam_max, eps_in, dense_cap, (2, None))
        Pinv = fh.G1_apply if precond == "g1" else (lambda y: 4.0 * y + 8.0 * fh.Pi_apply(y))
        self._outer_interval(self.Gh, Pinv=Pinv, P=self.P, eps=eps)
        self.eng.count = [0]

    def P(self, y):
        return fh.g1_inv(y, self.g1_deg) if self.precond == "g1" else fh.lorr_inv(y)

    def Gh(self, x):
        n, L = self.n, self.L
        X = np.zeros((L, L))
        X[: self.N, : self.N] = x
        return self._packed(
            lambda q: self.Tq(q, X), [("B", 1, 1), ("P", 2, n), ("B", 2, n), ("BT", 2, n), ("BT", 1, 1)], None
        )

    def backproject(self, b):
        n = self.n
        return self._packed(lambda q: self.iota(b[q]), [("BT", 2, n), ("P", 2, n), ("BT", 1, 1)], None)

    def apply(self, b):
        return self._outer(self.backproject(b), self.Gh, self.P)


if __name__ == "__main__":
    import adrtcore as ac

    n = 6
    N = 1 << n
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N, N))
    b = ac.adrt(n, x)
    # correctness against the reference implementations with identical (symmetrized) bounds and degrees
    F = FastLp(n, eps_in=0.1, threads=4)
    R = fs.FastSpectralInverse(n, 0.2, eps=0.1)
    R.lam_max = F.lam_max
    R.deg = F.deg
    print(
        "L_p frame_window vs fastspec:",
        np.abs(F.frame_window(x) - R.frame_window(x)).max() / np.abs(R.frame_window(x)).max(),
    )
    print(
        "L_p backproject vs fastspec:",
        np.abs(F.backproject(b) - R.backproject(b)).max() / np.abs(R.backproject(b)).max(),
    )
    print("L_p LR=I:", np.linalg.norm(F.apply(b) - x) / np.linalg.norm(x), "outer deg", F.outer_deg)
    H = FastHybrid(n, eps_in=0.1, threads=4)
    Rh = fh.FastHybridInverse(n, 0.2, eps=0.1, lam_max=H.lam_max)
    print(
        "hyb Gh vs fasthybrid:",
        np.abs(H.Gh(x) - Rh.Gh(x)).max() / np.abs(Rh.Gh(x)).max(),
        "deg equal:",
        np.array_equal(H.deg, Rh.deg),
    )
    print(
        "hyb backproject vs fasthybrid:",
        np.abs(H.backproject(b) - Rh.backproject(b)).max() / np.abs(Rh.backproject(b)).max(),
    )
    print("hyb LR=I:", np.linalg.norm(H.apply(b) - x) / np.linalg.norm(x), "outer deg", H.outer_deg)
    for cap in (8, 16):
        Fd = FastLp(n, eps_in=0.1, threads=4, dense_cap=cap)
        Hd = FastHybrid(n, eps_in=0.1, threads=4, dense_cap=cap)
        print(
            f"dense_cap={cap}: L_p dense rows {Fd.eng.n_dense}, LR=I {np.linalg.norm(Fd.apply(b) - x) / np.linalg.norm(x):.1e}, D={Fd.outer_deg} | "
            f"hybrid dense rows {Hd.eng.n_dense}, LR=I {np.linalg.norm(Hd.apply(b) - x) / np.linalg.norm(x):.1e}, D={Hd.outer_deg}"
        )
