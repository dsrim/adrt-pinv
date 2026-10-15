import itertools

import numpy as np

import adrtcore as ac

n = 3
N = 1 << n


# dense level-1 operator per quadrant: S_1 T_q, output (N+1, N) with row index h+1
def S1q(q):
    cols = []
    for j in range(N * N):
        e = np.zeros(N * N)
        e[j] = 1
        cols.append(ac.S_apply(n, 1, ac.T_apply(q, e.reshape(N, N))).ravel())
    return np.array(cols).T  # ((N+1)N, N^2)


Sq = [S1q(q) for q in range(4)]
S = np.vstack(Sq)
G1 = S.T @ S
h = np.arange(-1, N)[:, None] * np.ones((1, N))
s = np.ones((N + 1, 1)) * np.arange(N)[None, :]
hp, sp = (h % 2).ravel(), (s % 2).ravel()
Pi = np.kron(
    np.kron(np.eye(N // 2), np.ones((2, 2)) / 2), np.kron(np.eye(N // 2), np.ones((2, 2)) / 2)
)  # cell averaging (rows x cols)
# cell-pattern sine basis V (orthonormal), same as before
hN = N // 2
jj = np.arange(1, hN + 1)
kk = np.arange(1, hN + 1)
Sd = np.sqrt(2.0 / (hN + 1)) * np.sin(np.pi * np.outer(kk, jj) / (hN + 1))
pats = [
    np.array([[1, 0], [0, 1]]),
    np.array([[-1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1], [1, 0]]),
]
V = np.zeros((N * N, 4 * hN * hN))
col = 0
for p, pat in enumerate(pats):
    for k1 in range(hN):
        for k2 in range(hN):
            V[:, col] = (np.kron(np.outer(Sd[k1], Sd[k2]), pat) / np.sqrt(2)).ravel()
            col += 1
idx = np.arange(4 * hN * hN).reshape(4, hN, hN)


def offblock(Gv):
    off = Gv.copy()
    for k1 in range(hN):
        for k2 in range(hN):
            ii = idx[:, k1, k2]
            off[np.ix_(ii, ii)] = 0
    return np.abs(off).max()


print("full G1: off-(k1,k2) max", f"{offblock(V.T @ G1 @ V):.3f}")
for dh, ds in itertools.product((0, 1), (0, 1)):
    keep = ~((hp == dh) & (sp == ds))
    M = np.diag(keep.astype(float))
    B = np.vstack([M @ Sq[q] for q in range(4)])
    G = B.T @ B
    sv = np.linalg.svd(B, compute_uv=False)
    ev = np.linalg.eigvalsh(G)
    print(
        f"drop (h%2={dh}, s%2={ds}): rank {np.linalg.matrix_rank(B)}/{N * N}, eig(B^TB) range [{ev[0]:.3f},{ev[-1]:.3f}], "
        f"|B^TB - (4I+8Pi)| = {np.abs(G - (4 * np.eye(N * N) + 8 * Pi)).max():.2e}, off-(k1,k2) block max in V basis {offblock(V.T @ G @ V):.3f}, "
        f"distinct eigs: {np.unique(np.round(ev, 6))[:6]}"
    )
# also: drop class only in the 'extended' row block? try dropping all rows with h = -1 or h = N-1 extra? skip
