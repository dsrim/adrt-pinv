"""C++ engine for the oblique inverses: the row kernels of cpp/oblique.cpp (butterflies, adjoints, Chebyshev filter; OpenMP
over frequency rows) behind the RowEngine interface of fastops, so FastLp / FastHybrid run unchanged with engine_cls = CppEngine.
The library is compiled on first use (g++ -O3 -march=native -fopenmp).  FFTs, quadrant packing, the dense low-frequency table,
the G_1^{-1} stencil and the outer iteration stay in numpy."""

import ctypes
import os
import subprocess

import numpy as np

import fastops as fo

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "cpp", "oblique.cpp")
_LIB = os.path.join(_DIR, "cpp", "liboblique.so")


def _load():
    if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
        subprocess.check_call(["g++", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", "-o", _LIB, _SRC])
    lib = ctypes.CDLL(_LIB)
    P = ctypes.c_void_p
    I = ctypes.c_int
    lib.bfly_rows.argtypes = [I, I, P, P, I, I, I, I]
    lib.bfly_rows.restype = None
    lib.cheb_rows.argtypes = [I, I, P, P, P, P, ctypes.c_double, P, P, I, I, I, P]
    lib.cheb_rows.restype = None
    lib.cert_rowsum.argtypes = [I, I, P, I, P, I]
    lib.cert_rowsum.restype = None
    lib.omp_max_threads.restype = I
    return lib


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def _ptr(x):
    return x.ctypes.data_as(ctypes.c_void_p)


class CppEngine(fo.RowEngine):
    """Same construction as RowEngine (one chunk; `threads` = OpenMP threads for the row loops)."""

    def __init__(self, n, z, deg, lam_max, tau, threads=1, copies=2, dense_cap=None, p_levels=(1, None)):
        super().__init__(n, z, deg, lam_max, tau, threads=1, copies=copies, dense_cap=dense_cap, p_levels=p_levels)
        self.omp = max(1, threads)
        lib()
        for ch in self.chunks:
            ch["zc"] = np.ascontiguousarray(ch["z"], dtype=complex)
            ch["degc"] = np.ascontiguousarray(ch["deg"], dtype=np.int32)
            ch["ac"] = np.ascontiguousarray(ch["a"], dtype=float)
            ch["cc"] = np.ascontiguousarray(ch["c"], dtype=float)

    def _B(self, ch, g, lo, hi, K):
        X = np.array(g[:K], dtype=complex, order="C")  # explicit copy: the kernel works in place
        lib().bfly_rows(self.n, K, _ptr(ch["zc"]), _ptr(X), lo, hi, 0, self.omp)
        return X

    def _BT(self, ch, f, lo, hi, K):
        X = np.array(f[:K], dtype=complex, order="C")
        lib().bfly_rows(self.n, K, _ptr(ch["zc"]), _ptr(X), lo, hi, 1, self.omp)
        return X

    def _P(self, ch, Y, lo, hi):
        Kd = ch["Kd"]
        Y = np.ascontiguousarray(Y, dtype=complex)
        K = Y.shape[0]
        Z = np.empty_like(Y)
        if Kd:
            Z[:Kd] = np.matmul(ch["Fd"], Y[:Kd, :, None])[:, :, 0]
        steps = ctypes.c_longlong(0)
        Kr = K - Kd
        if Kr > 0:
            Zr = np.empty((Kr, self.N), dtype=complex)
            lib().cheb_rows(
                self.n,
                Kr,
                _ptr(ch["zc"][Kd:].copy()),
                _ptr(ch["degc"][Kd:].copy()),
                _ptr(ch["ac"][Kd:].copy()),
                _ptr(ch["cc"][Kd:].copy()),
                self.tau,
                _ptr(np.ascontiguousarray(Y[Kd:])),
                _ptr(Zr),
                lo,
                hi,
                self.omp,
                ctypes.byref(steps),
            )
            Z[Kd:] = Zr
        return Z, steps.value + Kd * self.N // max(1, self.n)


def power_lam_max(n, z, m_lo, iters=40, safety=1.5, threads=1):
    """safety x Rayleigh quotient after `iters` power steps of M_k = B^* B (levels m_lo..n), all frequencies, C++ butterflies."""
    N = 1 << n
    K = len(z)
    zc = np.ascontiguousarray(z, dtype=complex)
    V = np.random.default_rng(0).standard_normal((K, N)) + 1j * np.random.default_rng(1).standard_normal((K, N))

    def M(V):
        X = np.array(V, dtype=complex, order="C")
        lib().bfly_rows(n, K, _ptr(zc), _ptr(X), m_lo, n, 0, threads)
        lib().bfly_rows(n, K, _ptr(zc), _ptr(X), m_lo, n, 1, threads)
        return X

    for _ in range(iters):
        V = M(V)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return safety * np.real(np.sum(np.conj(V) * M(V), axis=1))


def cert_lam_max(n, oms, m_lo=1, threads=1):
    out = np.empty(len(oms))
    om = np.ascontiguousarray(oms, dtype=float)
    lib().cert_rowsum(n, len(om), _ptr(om), m_lo, _ptr(out), threads)
    return out


def explicit_lam_max(n, oms):
    N = 1 << n
    w = np.abs(oms)
    lam = np.full(len(oms), float(N) ** 2)
    nz = w > 0
    lam[nz] = np.minimum(float(N) ** 2, 2.0 * N * (1 + n) / w[nz])
    return lam


class FastLpCpp(fo.FastLp):
    """lam_bound: 'power' (C++ power iteration x 1.5), 'explicit' (eq. (Lambda)), 'cert' (C++ row-sum recursion)."""

    engine_cls = CppEngine

    def __init__(self, n, lam_bound="power", **kw):
        L = 2 << n
        z = np.exp(1j * 2 * np.pi * np.fft.fftfreq(L))
        oms = 2 * np.pi * np.fft.fftfreq(L)
        T = kw.get("threads", 1)
        if kw.get("lam_max") is None:
            kw["lam_max"] = {
                "power": lambda: power_lam_max(n, z, 1, threads=T),
                "explicit": lambda: explicit_lam_max(n, oms),
                "cert": lambda: cert_lam_max(n, oms, 1, T),
            }[lam_bound]()
        super().__init__(n, **kw)


class FastHybridCpp(fo.FastHybrid):
    engine_cls = CppEngine

    def __init__(self, n, lam_bound="power", **kw):
        L = 2 << n
        z = np.exp(1j * 2 * np.pi * np.fft.fftfreq(L))
        oms = 2 * np.pi * np.fft.fftfreq(L)
        T = kw.get("threads", 1)
        if kw.get("lam_max") is None:
            kw["lam_max"] = {
                "power": lambda: power_lam_max(n, z, 2, threads=T),
                "cert": lambda: cert_lam_max(n, oms, 2, T),
            }[lam_bound]()
        super().__init__(n, **kw)


if __name__ == "__main__":
    import time

    import adrtcore as ac

    n = 6
    N = 1 << n
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N, N))
    b = ac.adrt(n, x)
    L = 2 * N
    z = np.exp(1j * 2 * np.pi * np.fft.fftfreq(L))
    V = rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))
    import fasthybrid as fh
    import fastspec as fs

    zc = np.ascontiguousarray(z)
    X = V.copy()
    lib().bfly_rows(n, L, _ptr(zc), _ptr(X), 1, n, 0, 1)
    print("bfly   cpp vs fastspec:", np.abs(X - fs.bfly(n, z, V)).max() / np.abs(X).max())
    X = V.copy()
    lib().bfly_rows(n, L, _ptr(zc), _ptr(X), 1, n, 1, 1)
    print("bflyT  cpp vs fastspec:", np.abs(X - fs.bflyT(n, z, V)).max() / np.abs(X).max())
    X = V.copy()
    lib().bfly_rows(n, L, _ptr(zc), _ptr(X), 2, n, 0, 1)
    print("bfly 2..n cpp vs fasthybrid:", np.abs(X - fh.bfly_levels(n, z, V, 2, n)).max() / np.abs(X).max())
    for T in (1, 4):
        R = fo.FastLp(n, eps_in=0.1, threads=1)
        F = FastLpCpp(n, eps_in=0.1, threads=T, lam_max=R.lam_max)  # same bounds, same filters
        print(
            f"T={T}: L_p frame cpp vs fastops: {np.abs(F.frame_window(x) - R.frame_window(x)).max() / np.abs(R.frame_window(x)).max():.1e} | LR=I {np.linalg.norm(F.apply(b) - x) / np.linalg.norm(x):.1e} D={F.outer_deg}"
        )
        Rh = fo.FastHybrid(n, eps_in=0.1, threads=1, dense_cap=8)
        H = FastHybridCpp(n, eps_in=0.1, threads=T, dense_cap=8, lam_max=Rh.lam_max)
        print(
            f"T={T}: hybrid Gh cpp vs fastops: {np.abs(H.Gh(x) - Rh.Gh(x)).max() / np.abs(Rh.Gh(x)).max():.1e} | LR=I {np.linalg.norm(H.apply(b) - x) / np.linalg.norm(x):.1e} D={H.outer_deg} dense rows {H.eng.n_dense}"
        )
    # kernel timing at N=256
    n = 8
    N = 256
    L = 512
    z = np.exp(1j * 2 * np.pi * np.fft.fftfreq(L))
    V = rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))
    X = V.copy()
    zc = np.ascontiguousarray(z)
    for T in (1, 2, 4):
        t0 = time.time()
        for _ in range(20):
            lib().bfly_rows(n, L, _ptr(zc), _ptr(X), 1, n, 0, T)
            lib().bfly_rows(n, L, _ptr(zc), _ptr(X), 1, n, 1, T)
        tc = (time.time() - t0) / 20
        print(f"N=256 butterfly pair on {L} rows: cpp {T} thread(s) {tc * 1e3:.2f} ms", end="")
        if T == 1:
            t0 = time.time()
            for _ in range(5):
                fs.bflyT(n, z, fs.bfly(n, z, V))
            print(f" | numpy {(time.time() - t0) / 5 * 1e3:.2f} ms", end="")
        print(flush=True)
    print("omp max threads:", lib().omp_max_threads())
