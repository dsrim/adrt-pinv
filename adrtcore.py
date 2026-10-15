"""
Core ADRT toolkit following the index conventions of

  [LRR] Li, Ren, Rim, "A range characterization of the single-quadrant ADRT",
        arXiv:2010.05360
  [LORR] Li, Otness, Ren, Rim, "An explicit spectral decomposition of the ADRT",
        arXiv:2412.11151

Single-quadrant data at level m lives on the index set

    E^n_m = { (h, s) : s in [N], -(s mod 2^m) <= h < N },   N = 2^n.

We store such data as a dense array of shape (N + 2^m - 1, N) with
row index  h + (2^m - 1)  so that h = -(2^m-1) is row 0.  Entries that are
outside E^n_m are *structurally zero* (never written to by S_m; and ignored
by S_m^T).  Including them as always-zero rows of the matrix does not change
the Moore-Penrose pseudoinverse.

The full ADRT is  R = S_(n) ... S_(1),  S_(1) = [S_1 T_I ; S_1 T_II ; S_1 T_III ; S_1 T_IV]
with the quadrant permutations T_q of [LORR, eq. (permute)].
"""

import numpy as np


# ----------------------------------------------------------------------------
# quadrant permutations  (T_q f)(i,j), [LORR eq. (permute)]
# ----------------------------------------------------------------------------
def T_apply(q, f):
    """Apply T_q (q = 0..3 for quadrants I..IV) to an N x N image."""
    if q == 0:  # (T_I f)(i,j) = f(j, N-1-i)
        return np.ascontiguousarray(f[:, ::-1].T)
    if q == 1:  # (T_II f)(i,j) = f(N-1-i, j)
        return np.ascontiguousarray(f[::-1, :])
    if q == 2:  # identity
        return np.ascontiguousarray(f)
    if q == 3:  # (T_IV f)(i,j) = f(N-1-j, N-1-i)
        return np.ascontiguousarray(f[::-1, ::-1].T)
    raise ValueError(q)


def T_apply_T(q, g):
    """Apply T_q^T = T_q^{-1} (permutations are orthogonal)."""
    if q == 0:  # inverse of f -> f[:, ::-1].T  is g -> g.T[:, ::-1]
        return np.ascontiguousarray(g.T[:, ::-1])
    if q == 1:
        return np.ascontiguousarray(g[::-1, :])
    if q == 2:
        return np.ascontiguousarray(g)
    if q == 3:  # f[::-1, ::-1].T is an involution
        return np.ascontiguousarray(g.T[::-1, ::-1])
    raise ValueError(q)


# ----------------------------------------------------------------------------
# single-quadrant levels S_m : R^{E^n_{m-1}} -> R^{E^n_m}
# ----------------------------------------------------------------------------
def level_shape(n, m):
    N = 1 << n
    return (N + (1 << m) - 1, N)


def S_apply(n, m, f):
    """S_m [f] = g  (LORR eq. sq_Snm).  f has shape level_shape(n, m-1)."""
    N = 1 << n
    w = 1 << (m - 1)  # input strip width 2^{m-1}
    Hin, Hout = N + w - 1, N + 2 * w - 1
    assert f.shape == (Hin, N)
    nstrips = N // (2 * w)
    # input strips: column = (2*l + c) * w + t   ->  f4[h, l, c, t]
    f4 = f.reshape(Hin, nstrips, 2, w)
    g = np.zeros((Hout, nstrips, w, 2), dtype=f.dtype)  # column = l*2w + 2t + b
    off_in, off_out = w - 1, 2 * w - 1
    for t in range(w):
        for b in (0, 1):
            # g(h, 2t+b) = f0(h, t) + f1(h + t + b, t),  h in [-(2t+b), N)
            # rows of g: h + off_out ; rows of f: h + off_in
            # f0 term: h in [-t, N)
            g[(-t + off_out) : (N + off_out), :, t, b] += f4[(-t + off_in) : (N + off_in), :, 0, t]
            # f1 term: h + t + b in [-t, N)  <=>  h in [-2t-b, N - t - b)
            lo, hi = -2 * t - b, N - t - b
            g[(lo + off_out) : (hi + off_out), :, t, b] += f4[(lo + t + b + off_in) : (hi + t + b + off_in), :, 1, t]
    return g.reshape(Hout, N)


def S_apply_T(n, m, g):
    """S_m^T [g] = f  (back-projection by one level)."""
    N = 1 << n
    w = 1 << (m - 1)
    Hin, Hout = N + w - 1, N + 2 * w - 1
    assert g.shape == (Hout, N)
    nstrips = N // (2 * w)
    g4 = g.reshape(Hout, nstrips, w, 2)
    f4 = np.zeros((Hin, nstrips, 2, w), dtype=g.dtype)
    off_in, off_out = w - 1, 2 * w - 1
    for t in range(w):
        for b in (0, 1):
            f4[(-t + off_in) : (N + off_in), :, 0, t] += g4[(-t + off_out) : (N + off_out), :, t, b]
            lo, hi = -2 * t - b, N - t - b
            f4[(lo + t + b + off_in) : (hi + t + b + off_in), :, 1, t] += g4[(lo + off_out) : (hi + off_out), :, t, b]
    return f4.reshape(Hin, N)


def sq_adrt(n, x):
    """single-quadrant ADRT R^n_n of an N x N image (level-0 data = the image)."""
    g = np.asarray(x, dtype=float)
    for m in range(1, n + 1):
        g = S_apply(n, m, g)
    return g


def sq_adrt_T(n, g):
    f = np.asarray(g, dtype=float)
    for m in range(n, 0, -1):
        f = S_apply_T(n, m, f)
    return f


# ----------------------------------------------------------------------------
# full ADRT  R : R^{N x N} -> R^{4 x (2N-1) x N}
# ----------------------------------------------------------------------------
def adrt(n, x):
    N = 1 << n
    x = np.asarray(x, dtype=float).reshape(N, N)
    out = np.empty((4, 2 * N - 1, N))
    for q in range(4):
        out[q] = sq_adrt(n, T_apply(q, x))
    return out


def adrt_T(n, b):
    N = 1 << n
    b = np.asarray(b, dtype=float).reshape(4, 2 * N - 1, N)
    x = np.zeros((N, N))
    for q in range(4):
        x += T_apply_T(q, sq_adrt_T(n, b[q]))
    return x


def mask_full(n):
    """boolean mask of the structurally nonzero entries of the full ADRT output."""
    N = 1 << n
    M = np.zeros((2 * N - 1, N), dtype=bool)
    for s in range(N):
        M[(N - 1 - s) :, s] = True
    return np.broadcast_to(M, (4, 2 * N - 1, N)).copy()


# ----------------------------------------------------------------------------
# dense matrices (small N) for verification
# ----------------------------------------------------------------------------
def dense_from_op(op, in_shape, out_shape):
    nin = int(np.prod(in_shape))
    nout = int(np.prod(out_shape))
    A = np.zeros((nout, nin))
    e = np.zeros(nin)
    for k in range(nin):
        e[:] = 0
        e[k] = 1
        A[:, k] = op(e.reshape(in_shape)).ravel()
    return A


def adrt_dense(n):
    N = 1 << n
    return dense_from_op(lambda x: adrt(n, x), (N, N), (4, 2 * N - 1, N))


def sq_adrt_dense(n):
    N = 1 << n
    return dense_from_op(lambda x: sq_adrt(n, x), (N, N), (2 * N - 1, N))


# ----------------------------------------------------------------------------
# digital line closed form  (LRR, Cor. after Def. of dual lines)
#   D_n(h, s) = { (h + delta_s(t), t) : t in [N] },
#   delta_s(t) = sum_{k=0}^{n-1} bit_{n-1-k}(s) * floor((t + 2^k) / 2^{k+1})
# ----------------------------------------------------------------------------
def delta_s(n, s, t):
    d = 0
    for k in range(n):
        bit = (s >> (n - 1 - k)) & 1
        if bit:
            d += (t + (1 << k)) >> (k + 1)
    return d


def digital_line(n, h, s):
    N = 1 << n
    return [(h + delta_s(n, s, t), t) for t in range(N)]
