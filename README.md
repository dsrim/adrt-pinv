# adrt-pinv

Code for the paper *Explicit oblique inverses of the approximate discrete Radon transform* (in preparation).

The approximate discrete Radon transform (ADRT) of an N×N image (N = 2^n) has no closed-form Moore–Penrose inverse.
This repository implements the explicit left inverses of the paper, oblique projections onto the range of the ADRT built
from polynomial filters on the butterfly structure of its levels, together with the hybrid that keeps the explicit inverse
of the cross-quadrant level, and every experiment reported in the paper.

## Getting started

Unpack the archive (or clone the repository), then from its root directory:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest test_core.py                     # tests of the ADRT implementation
python fastcpp.py                       # compiles the C++ kernels and checks them against the NumPy engine
```

## Requirements

- Python 3.10 or later with `numpy`, `scipy`, `matplotlib`, `scikit-image` and `adrt` (see `requirements.txt`).
- A C++ compiler with OpenMP (`g++`) for the compiled row kernels. `fastcpp.py` compiles `cpp/oblique.cpp` into
  `cpp/liboblique.so` on first use, with `-O3 -march=native -fopenmp`; the library is machine specific and not tracked.

```
pip install -r requirements.txt
```

## Layout

| path | contents |
|---|---|
| `adrtcore.py` | the ADRT, its levels and its transpose in this repository's sinogram layout |
| `pinv.py` | CGLS for the least-squares (Moore–Penrose) solution |
| `specpart.py` | dense spectral-partition inverse (Section 4) |
| `fastspec.py` | polynomial-filter inverse L_p, NumPy reference (Section 5) |
| `fasthybrid.py` | the hybrid inverse, NumPy reference (Section 6) |
| `fastops.py` | optimized NumPy engine for both (twiddle tables, degree-sorted rows, quadrant packing, dense low-frequency table) |
| `fastcpp.py`, `cpp/oblique.cpp` | C++ row kernels (butterflies, Chebyshev filter; OpenMP over rows) behind the engine; used for all wall times |
| `lab_frame.py`, `lab_hybrid.py` | dense laboratories for the frame operator and the hybrid (small N) |
| `skimage_adrt_pkg.py` | conversion of `skimage.transform.radon` data to ADRT data (Section 7.7) |
| `exp_*.py` | experiments; each prints its results |
| `plot_*.py` | figures of the paper, written to `figures/` |
| `results/` | outputs of the runs reported in the paper |
| `figures/` | the figures of the paper |
| `patches/adrt-gram-freq.patch` | patch to the `adrt` package adding a row-wise normal operator (`adrt.core.gram`), used only by `exp_gram_bench.py` |
| `test_core.py` | tests of `adrtcore.py` against the recursive definition |

`RESULTS.md` lists the script behind each table, figure and numerical statement of the paper.

## Running

Run the scripts from the repository root, for example

```
python exp_fasthybrid.py 6 7 8          # fast hybrid against L_p, N = 64, 128, 256
python plot_shepp_figure.py 7           # Figure 5
python fastcpp.py                       # self-test of the C++ kernels against fastops
```

Notes:

- For timings, pin BLAS to one thread (`OPENBLAS_NUM_THREADS=1`) and use an otherwise idle machine; the paper's timings
  were measured on one core of an exclusive cluster node.
- Some experiments use the output of others: run `exp_ghost.py` before `exp_ghost256.py` and `exp_resonance_svals.py`,
  and `exp_setup_compare.py` before `plot_setup_compare.py`.
- `exp_gram_bench.py` needs the `adrt` package built from source with `patches/adrt-gram-freq.patch` applied
  (`git apply`), compiled with OpenMP (`CFLAGS="-fopenmp -O3"`).
- The larger runs (N ≥ 256) take minutes to hours; the scripts take the sizes as arguments where noted in their docstrings.

## Code style

The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) (configuration in `ruff.toml`):

```
ruff format --check .
ruff check .
```

## License

BSD 3-Clause; see `LICENSE`. The file `patches/adrt-gram-freq.patch` modifies the `adrt` package
(https://pypi.org/project/adrt/), which is distributed under its own BSD 3-Clause license; the patched
package remains subject to that license.

[![DOI](https://zenodo.org/badge/1421438382.svg)](https://doi.org/10.5281/zenodo.23406686)