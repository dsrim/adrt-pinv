"""Exact arithmetic: characteristic polynomial of the integer matrix R^T R of the full ADRT (N = 4, 8), its factorization
over Q, discriminants, and Galois groups of the low-degree factors (Appendix C of the paper).  Requires sympy."""

import sys
import time

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.galoisgroups import galois_group

import adrtcore as ac

lam = sp.Symbol("lambda")
for n in [int(a) for a in sys.argv[1:]] or [2, 3]:
    N = 1 << n
    t0 = time.time()
    A = ac.adrt_dense(n).astype(np.int64)  # rows: ADRT entries, columns: pixels
    G = A.T @ A
    assert np.array_equal(G, G.T)
    M = DomainMatrix([[sp.ZZ(int(v)) for v in row] for row in G], G.shape, sp.ZZ)
    cp = sp.Poly(M.charpoly(), lam, domain=sp.ZZ)
    c, facs = sp.factor_list(cp.as_expr(), lam)
    print(
        f"N={N}: G = R^T R is {G.shape[0]}x{G.shape[0]}, entries in [{G.min()}, {G.max()}]; charpoly factored in {time.time() - t0:.1f}s",
        flush=True,
    )
    for f, mult in sorted(facs, key=lambda fm: sp.degree(fm[0], lam)):
        d = int(sp.degree(f, lam))
        P = sp.Poly(f, lam)
        line = f"   degree {d:2d} (multiplicity {mult})"
        if d <= 3:
            line += f": {sp.expand(f)}"
        if 2 <= d <= 6:
            disc = sp.discriminant(P)
            sq = sp.sqrt(disc).is_integer if disc > 0 else False
            Gg, _ = galois_group(P, by_name=True)
            line += f" | discriminant {disc} ({'square' if sq else 'not a square'}) | Galois group {Gg.name}"
        print(line, flush=True)
print("GALOIS DONE", flush=True)
