"""Compare the setup choices of the explicit oblique inverse:
Lambda_k: power iteration (x1.5)  vs  explicit  min(N^2, 2N(1+log N)/|omega|)
outer interval: Weinstein enclosure | explicit (2/log N, 4.2) | (4/log N, 4.2) | fixed (0.5, 4.2) | overestimate (8/log N, 4.2)
tau: 0.2N, N.   Metrics: degrees, freq-steps per apply, setup time, left-inverse error, noisy error ratio to MP."""

import json
import sys
import time

import numpy as np

import adrtcore as ac
import fastspec as fs
import pinv

rng = np.random.default_rng(0)


def outer_deg(lo, hi, eps=1e-10):
    k = hi / lo
    return int(np.ceil(np.log(eps / 2) / np.log((np.sqrt(k) - 1) / (np.sqrt(k) + 1)))) + 2


CONFIGS = [
    ("power / Weinstein", "power", None, 0.2),
    ("explicit / Weinstein", "explicit", None, 0.2),
    ("explicit / (2/logN,4.2)", "explicit", "2", 0.2),
    ("explicit / (4/logN,4.2)", "explicit", "4", 0.2),
    ("explicit / (0.5,4.2)", "explicit", "0.5", 0.2),
    ("explicit / (8/logN,4.2) [over]", "explicit", "8", 0.2),
    ("explicit / Weinstein, tau=N", "explicit", None, 1.0),
    ("explicit / (1/logN,4.2), tau=N", "explicit", "1", 1.0),
]
res = {}
for n in [int(a) for a in sys.argv[1:]]:
    N = 1 << n
    x = rng.standard_normal((N, N))
    b = ac.adrt(n, x)
    bn = b + 1e-2 * rng.standard_normal((4, 2 * N - 1, N))
    xmp, h = pinv.cgls(n, bn, tol=1e-10, maxit=20000)
    emp = np.linalg.norm(xmp - x) / np.linalg.norm(x)
    cache = {}
    for name, lb, outer, tf in CONFIGS:
        key = (lb, tf)
        if key not in cache:
            t0 = time.time()
            cache[key] = (fs.FastSpectralInverse(n, tau_fac=tf, eps=1e-10, lam_bound=lb), time.time() - t0)
        P, ts = cache[key]
        lo_w, hi_w = P.lo, P.hi
        D_w = P.outer_deg
        if outer is None:
            lo, hi = lo_w, hi_w
            ts_eff = ts
        else:
            lo = float(outer) / n if outer in ("1", "2", "4", "8") else float(outer)
            hi = 4.2
            ts_eff = 0.0 if lb == "explicit" else ts
        P.lo, P.hi = lo, hi
        P.outer_deg = outer_deg(lo, hi)
        P.count = [0]
        t0 = time.time()
        xl = P.apply(b)
        ta = time.time() - t0
        cnt = P.count[0]
        xln = P.apply(bn)
        li = np.linalg.norm(xl - x) / np.linalg.norm(x)
        ratio = np.linalg.norm(xln - x) / np.linalg.norm(x) / emp
        res[f"{N}|{name}"] = {
            "N": N,
            "name": name,
            "sumdeg": float(P.deg.sum() / N),
            "outer": int(P.outer_deg),
            "lo": float(lo),
            "hi": float(hi),
            "steps": float(cnt / N),
            "setup": float(ts_eff),
            "apply": float(ta),
            "lierr": float(li),
            "ratio": float(ratio),
        }
        print(
            f"N={N:3d} {name:34s} sum d_k={P.deg.sum() / N:5.0f}N  D={P.outer_deg:3d} [lo,hi]=[{lo:.3f},{hi:.3f}]  steps/apply={cnt / N:7.0f}N  setup={ts_eff:5.0f}s apply={ta:5.0f}s  left-inv err={li:.1e}  err ratio={ratio:.2f}",
            flush=True,
        )
        P.lo, P.hi, P.outer_deg = lo_w, hi_w, D_w
    with open("results/setup_compare.json", "w") as fh:
        json.dump(res, fh, indent=1)
