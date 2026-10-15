import numpy as np

from lab_hybrid import HybridLab


def Kmat(N):
    K = 2 * np.eye(N)
    i = np.arange(N - 1)
    K[i, i + 1] = 1
    K[i + 1, i] = 1
    return K


def Jmat(N):
    J = np.zeros((N, N))
    a = np.arange(0, N, 2)
    J[a, a + 1] = 1
    J[a + 1, a] = 1
    return J


for n in (3, 4):
    N = 1 << n
    lab = HybridLab(n, 0.2)
    G1 = lab.dense("g1")
    K, J = Kmat(N), Jmat(N)
    G1s = 8 * np.eye(N * N) + np.kron(K, J) + np.kron(J, K)  # vec row-major: x(i,j) -> (A kron B) <-> A x B^T
    print(
        f"N={N}: |G1_dense - (8I + K⊗J + J⊗K)| = {np.abs(G1 - G1s).max():.2e}; spec dense [{np.linalg.eigvalsh(G1)[0]:.4f},{np.linalg.eigvalsh(G1)[-1]:.4f}]"
    )
    # structure in LORR's V basis: block-sinusoidal (DST-I on the cell grid x 4 cell patterns)
    h = N // 2
    j = np.arange(1, h + 1)
    k = np.arange(1, h + 1)
    S = np.sqrt(2.0 / (h + 1)) * np.sin(np.pi * np.outer(k, j) / (h + 1))  # orthonormal DST-I (k x j)
    pats = [
        np.array([[1, 0], [0, 1]]),
        np.array([[-1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1], [1, 0]]),
    ]
    V = np.zeros((N * N, 4 * h * h))
    col = 0
    for p, pat in enumerate(pats):
        for k1 in range(h):
            for k2 in range(h):
                img = np.kron(np.outer(S[k1], S[k2]), pat) / np.sqrt(2)
                V[:, col] = img.ravel()
                col += 1
    print(f"   V orthonormal: {np.abs(V.T @ V - np.eye(4 * h * h)).max():.1e}")
    Gv = V.T @ G1s @ V
    # coupling across (k1,k2): zero out the 4x4 blocks with the same (k1,k2) and measure what remains
    idx = np.arange(4 * h * h).reshape(4, h, h)
    off = Gv.copy()
    for k1 in range(h):
        for k2 in range(h):
            ii = idx[:, k1, k2]
            off[np.ix_(ii, ii)] = 0
    print(
        f"   |G1| in V basis: off-(k1,k2)-block norm {np.linalg.norm(off):.3f} of total {np.linalg.norm(Gv):.3f}; max offdiag entry {np.abs(off).max():.3f}"
    )

# torus symbol: G1 on the 2N-torus (cells), 4x4 blocks per cell frequency; min/max eigenvalue
for N in (8, 16, 64, 256):
    h = N // 2
    th = 2 * np.pi * np.arange(h) / h
    lo, hi = np.inf, -np.inf
    for t1 in th:
        for t2 in th:
            # K circulant on (s,d) coordinates per cell: K = 2I + J + Z2, Z2 cross-cell; in cell-Fourier with e^{i t}
            # x_{2a}=(s+d)/sqrt2, x_{2a+1}=(s-d)/sqrt2 ; J: s->s, d->-d ; Z2 quadratic form x_{2a+1}x_{2a+2}: 1/2 (s_a-d_a)(s_{a+1}+d_{a+1})
            def Kc(t):
                T = np.exp(1j * t)
                Z2 = 0.5 * np.array([[T + np.conj(T), T - np.conj(T)], [-(T - np.conj(T)), -(T + np.conj(T))]])
                return 2 * np.eye(2) + np.diag([1, -1]) + Z2

            Jc = np.diag([1.0, -1.0])
            Gs = 8 * np.eye(4) + np.kron(Kc(t1), Jc) + np.kron(Jc, Kc(t2))
            ev = np.linalg.eigvalsh(Gs)
            lo = min(lo, ev[0])
            hi = max(hi, ev[-1])
    print(f"torus N={N}: symbol eigenvalues in [{lo:.4f}, {hi:.4f}]")
