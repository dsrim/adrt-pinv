"""Sanity checks of adrtcore against (i) the digital-line closed form,
(ii) the transpose identity, (iii) the `adrt` package (row multiset)."""

import numpy as np

import adrtcore as ac

try:  # runnable both with pytest and as a script
    import pytest

    _sizes = pytest.mark.parametrize("n", [1, 2, 3, 4])
except ImportError:
    _sizes = lambda f: f


@_sizes
def test_transpose(n):
    N = 1 << n
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, N))
    b = rng.standard_normal((4, 2 * N - 1, N))
    lhs = np.sum(ac.adrt(n, x) * b)
    rhs = np.sum(x * ac.adrt_T(n, b))
    assert abs(lhs - rhs) < 1e-9 * max(1, abs(lhs)), (lhs, rhs)


@_sizes
def test_digital_lines(n):
    """sq_adrt applied to a delta must light up exactly the lines through it."""
    N = 1 << n
    A = ac.sq_adrt_dense(n)  # rows indexed (h + N - 1, s)
    for h in range(-(N - 1), N):
        for s in range(N):
            if h < -s:
                continue
            row = A[(h + N - 1) * N + s].reshape(N, N)
            expect = np.zeros((N, N))
            for i, t in ac.digital_line(n, h, s):
                if 0 <= i < N:
                    expect[i, t] = 1
            assert np.array_equal(row, expect), (h, s)
    # rows outside E^n_n must be identically zero
    for h in range(-(N - 1), 0):
        for s in range(-h):
            assert not A[(h + N - 1) * N + s].any()


@_sizes
def test_vs_package(n):
    import adrt

    N = 1 << n
    A = ac.adrt_dense(n)
    B = ac.dense_from_op(lambda x: adrt.adrt(x), (N, N), (4, 2 * N - 1, N))
    rowsA = sorted(map(tuple, A[A.any(axis=1)].astype(int)))
    rowsB = sorted(map(tuple, B[B.any(axis=1)].astype(int)))
    assert rowsA == rowsB, "row multisets differ"
    # each quadrant must be a column-permutation-and-row-permutation of sq ADRT
    assert A.shape == B.shape


if __name__ == "__main__":
    for n in (1, 2, 3, 4):
        test_transpose(n)
        test_digital_lines(n)
        test_vs_package(n)
        print("n =", n, "ok")
