"""
Moore-Penrose pseudoinverse of the full ADRT  R : R^{N x N} -> R^{4 x (2N-1) x N}

    R^+ b = (R^T R)^{-1} R^T b                 (R has full column rank)

computed iteratively.  Every iteration costs one ADRT + one back-projection,
O(N^2 log N), plus the cost of the preconditioner.

  cgls(n, b, ...)        : CGLS on  min ||R x - b||  (mathematically = CG on the
                           normal equations, numerically preferable)
  pcg_normal(n, b, M, ..): PCG on  R^T R x = R^T b  with SPD preconditioner M
"""

import numpy as np

import adrtcore as ac


def _ip(a, b):
    return float(np.vdot(a, b).real)


def cgls(n, b, tol=1e-12, maxit=10000, x0=None, callback=None):
    """CGLS (Hestenes-Stiefel / Paige-Saunders) for min ||R x - b||_2."""
    N = 1 << n
    b = np.asarray(b, dtype=float).reshape(4, 2 * N - 1, N)
    x = np.zeros((N, N)) if x0 is None else np.array(x0, dtype=float)
    r = b - ac.adrt(n, x)
    s = ac.adrt_T(n, r)
    p = s.copy()
    gamma = _ip(s, s)
    g0 = np.sqrt(gamma)
    hist = [g0]
    for it in range(maxit):
        q = ac.adrt(n, p)
        alpha = gamma / _ip(q, q)
        x += alpha * p
        r -= alpha * q
        s = ac.adrt_T(n, r)
        gnew = _ip(s, s)
        hist.append(np.sqrt(gnew))
        if callback is not None:
            callback(it, x, np.sqrt(gnew))
        if np.sqrt(gnew) <= tol * g0:
            break
        p = s + (gnew / gamma) * p
        gamma = gnew
    return x, np.array(hist)


def pcg_normal(n, b, M, tol=1e-12, maxit=10000, x0=None, callback=None, rhs=None):
    """PCG for  G x = R^T b,  G = R^T R, with SPD preconditioner M (callable).
    Convergence measured by the preconditioned residual norm ||r||_M = sqrt(r^T M r)
    (a fixed-size, O(1)-cheap proxy); returns history of ||r||_2 as well."""
    N = 1 << n
    if rhs is None:
        b = np.asarray(b, dtype=float).reshape(4, 2 * N - 1, N)
        rhs = ac.adrt_T(n, b)
    G = lambda v: ac.adrt_T(n, ac.adrt(n, v))
    x = np.zeros((N, N)) if x0 is None else np.array(x0, dtype=float)
    r = rhs - G(x)
    z = M(r)
    p = z.copy()
    rho = _ip(r, z)
    r0 = np.linalg.norm(r)
    hist = [r0]
    for it in range(maxit):
        q = G(p)
        alpha = rho / _ip(p, q)
        x += alpha * p
        r -= alpha * q
        rn = np.linalg.norm(r)
        hist.append(rn)
        if callback is not None:
            callback(it, x, rn)
        if rn <= tol * r0:
            break
        z = M(r)
        rho_new = _ip(r, z)
        p = z + (rho_new / rho) * p
        rho = rho_new
    return x, np.array(hist)


def pinv_dense(n):
    """dense Moore-Penrose pseudoinverse (small N only)"""
    A = ac.adrt_dense(n)
    return np.linalg.pinv(A)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in range(1, 5):
        N = 1 << n
        A = ac.adrt_dense(n)
        Ap = np.linalg.pinv(A)
        b = rng.standard_normal((4, 2 * N - 1, N))
        xd = (Ap @ b.ravel()).reshape(N, N)
        x1, h1 = cgls(n, b, tol=1e-14)
        x2, h2 = pcg_normal(n, b, lambda v: v, tol=1e-14)
        print(
            f"N={N}: |cgls-pinv|/|pinv| = {np.linalg.norm(x1 - xd) / np.linalg.norm(xd):.2e} ({len(h1) - 1} its), "
            f"|pcg-pinv|/|pinv| = {np.linalg.norm(x2 - xd) / np.linalg.norm(xd):.2e} ({len(h2) - 1} its)"
        )
