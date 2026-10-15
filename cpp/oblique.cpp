// High-performance kernels for the explicit oblique inverses of the ADRT (companion of fastops.py).
// Per offset-frequency row (length N = 2^n, complex): the single-quadrant butterflies B_m (levels m_lo..m_hi), their
// adjoints, and the Chebyshev semi-iteration z = p_k(M) y for M = B^* B, (M + tau)^{-1} on [0, Lambda_k].
// Rows are independent: OpenMP parallel over rows, each row processed in cache (N complex = 4 KB at N = 256).
// Conventions match fastspec.bfly / bflyT and fasthybrid.bfly_levels exactly (strip pairs, slopes 2t, 2t+1, twiddles z^t, z^{t+1}).
#include <complex>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
typedef std::complex<double> cd;

static inline void twiddles(cd z, int half, cd* pw) {          // pw[t] = z^t, t = 0..half
    pw[0] = cd(1.0, 0.0);
    for (int t = 1; t <= half; ++t) pw[t] = pw[t - 1] * z;
}

// forward levels m_lo..m_hi on one row: in -> out (both length N); uses buf as scratch; returns pointer holding the result
static cd* row_bfly(int n, const cd* pw, cd z, cd* a, cd* b, int m_lo, int m_hi) {
    const int N = 1 << n; cd* in = a; cd* out = b;
    for (int m = m_lo; m <= m_hi; ++m) {
        const int w = 1 << (m - 1), ns = N / (2 * w);
        for (int l = 0; l < ns; ++l) {
            const cd* f0 = in + l * 2 * w; const cd* f1 = f0 + w; cd* o = out + l * 2 * w;
            for (int t = 0; t < w; ++t) { cd u = pw[t] * f1[t]; o[2 * t] = f0[t] + u; o[2 * t + 1] = f0[t] + u * z; }
        }
        cd* tmp = in; in = out; out = tmp;
    }
    return in;
}

// adjoint levels m_hi..m_lo on one row
static cd* row_bflyT(int n, const cd* pw, cd z, cd* a, cd* b, int m_lo, int m_hi) {
    const int N = 1 << n; cd* in = a; cd* out = b; const cd zc = std::conj(z);
    for (int m = m_hi; m >= m_lo; --m) {
        const int w = 1 << (m - 1), ns = N / (2 * w);
        for (int l = 0; l < ns; ++l) {
            const cd* g = in + l * 2 * w; cd* o0 = out + l * 2 * w; cd* o1 = o0 + w;
            for (int t = 0; t < w; ++t) {
                cd g0 = g[2 * t], g1 = g[2 * t + 1];
                o0[t] = g0 + g1;
                o1[t] = std::conj(pw[t]) * (g0 + zc * g1);
            }
        }
        cd* tmp = in; in = out; out = tmp;
    }
    return in;
}

extern "C" {

// X: K rows of length N, in place.  transpose = 0: B_{m_hi} ... B_{m_lo};  1: the adjoint.
void bfly_rows(int n, int K, const cd* z, cd* X, int m_lo, int m_hi, int transpose, int nthreads) {
    const int N = 1 << n;
#pragma omp parallel num_threads(nthreads > 0 ? nthreads : 1)
    {
        std::vector<cd> pw(N / 2 + 2), a(N), b(N);
#pragma omp for schedule(dynamic, 4)
        for (int k = 0; k < K; ++k) {
            twiddles(z[k], N / 2, pw.data());
            std::memcpy(a.data(), X + (size_t)k * N, N * sizeof(cd));
            cd* r = transpose ? row_bflyT(n, pw.data(), z[k], a.data(), b.data(), m_lo, m_hi)
                              : row_bfly(n, pw.data(), z[k], a.data(), b.data(), m_lo, m_hi);
            std::memcpy(X + (size_t)k * N, r, N * sizeof(cd));
        }
    }
}

// Z[k] = p_k(M) Y[k], deg[k] Chebyshev steps with constants a[k], c[k] (fastspec.cheb_coeffs), M = B^* B over levels m_lo..m_hi.
// steps (optional, may be null): accumulated number of (row, level) butterfly applications, for the cost count.
void cheb_rows(int n, int K, const cd* z, const int* deg, const double* a, const double* c, double tau,
               const cd* Y, cd* Z, int m_lo, int m_hi, int nthreads, long long* steps) {
    const int N = 1 << n; long long total = 0;
#pragma omp parallel num_threads(nthreads > 0 ? nthreads : 1) reduction(+:total)
    {
        std::vector<cd> pw(N / 2 + 2), zc(N), zp(N), buf1(N), buf2(N);
#pragma omp for schedule(dynamic, 4)
        for (int k = 0; k < K; ++k) {
            const cd* y = Y + (size_t)k * N; cd* zk = zc.data(); cd* zprev = zp.data();
            const double ak = a[k], ck = c[k]; const int d = deg[k];
            twiddles(z[k], N / 2, pw.data());
            for (int i = 0; i < N; ++i) { zk[i] = ak * y[i]; zprev[i] = cd(0.0, 0.0); }
            double om = 1.0;
            for (int j = 1; j < d; ++j) {
                double omj = (j == 1) ? 1.0 / (1.0 - ck * ck / 2.0) : 1.0 / (1.0 - ck * ck * om / 4.0);
                std::memcpy(buf1.data(), zk, N * sizeof(cd));
                cd* g = row_bfly(n, pw.data(), z[k], buf1.data(), buf2.data(), m_lo, m_hi);
                cd* other = (g == buf1.data()) ? buf2.data() : buf1.data();
                cd* mz = row_bflyT(n, pw.data(), z[k], g, other, m_lo, m_hi);
                for (int i = 0; i < N; ++i) {
                    cd r = y[i] - mz[i] - tau * zk[i];
                    cd znew = zk[i] + omj * (ak * r) + (omj - 1.0) * (zk[i] - zprev[i]);
                    zprev[i] = zk[i]; zk[i] = znew;
                }
                om = omj; total += 2 * (m_hi - m_lo + 1);
            }
            std::memcpy(Z + (size_t)k * N, zk, N * sizeof(cd));
        }
    }
    if (steps) *steps += total;
}

// Certified upper bounds Lambda_k >= lambda_max(M_k) by the row-sum recursion (eq. (cert) of the paper), levels m_lo..n,
// for K frequencies omega[k]; r_m(a) <= |1+z^a| r_{m-1}(2a) + max_pm |1+z^(a pm 1)| r_{m-1}(2a pm 1), r_{m_lo-1} = 1.
void cert_rowsum(int n, int K, const double* omega, int m_lo, double* out, int nthreads) {
    const long A = 1L << n;
#pragma omp parallel num_threads(nthreads > 0 ? nthreads : 1)
    {
        std::vector<double> r(2 * A + 1), rn(2 * A + 1);
#pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) {
            const double om = omega[k];
            auto f = [om](long a) { return 2.0 * std::fabs(std::cos(0.5 * om * (double)a)); };   // |1 + z^a|
            std::fill(r.begin(), r.end(), 1.0);
            for (int m = m_lo; m <= n; ++m) {
                const long amax = (1L << (n - m)) - 1;
                std::fill(rn.begin(), rn.end(), 0.0);
                for (long a = -amax; a <= amax; ++a) {
                    const double t0 = f(a) * r[2 * a + A], tp = f(a + 1) * r[2 * a + 1 + A], tm = f(a - 1) * r[2 * a - 1 + A];
                    rn[a + A] = t0 + (tp > tm ? tp : tm);
                }
                std::swap(r, rn);
            }
            out[k] = r[A];
        }
    }
}

int omp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
}
