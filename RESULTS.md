# Results and the scripts behind them

Each row lists, for a result of the paper *Explicit oblique inverses of the approximate discrete Radon transform*,
the scripts that produce it and, where kept, their output in `results/` (or the figure in `figures/`).
Sections refer to the paper. Shared modules are described in `README.md`.

| section | result | scripts | output |
|---|---|---|---|
| 3.2 | Remark 3.2: level decompositions do not compose | `exp_level_transfer.py` | `exp_level_transfer.out` |
| 4.3 | Table 1: spectral partition and L_τ | `exp_specpart_scale.py`, `exp_specpart_tikh.py`, `exp_specpart_tikh_scale.py` | printed |
| 5 | Figures 1–3: polynomial filters | `plot_filters.py` | `figures/filters_*.pdf` |
| 5.1 | explicit bounds Λ_k and the resonance at 2π/3 | `exp_lammax.py`, `exp_kernel_resonance.py` | `exp_kernel_resonance.out` |
| 5.1 | certified bounds Λ_k^cert | `exp_cert_lambda.py` | `certlam.txt` |
| 6.2 | Lemmas 6.2 and 6.3: level-1 Gram matrices | `exp_g1check.py`, `exp_lorr_trunc.py` | printed |
| 6.2 | preconditioners of G_h | `exp_hybrid_precond.py` | `hybrid_precond.txt`, `hybrid_precond64.txt` |
| 6.3 | Table 2: hybrid, dense comparison | `lab_hybrid.py` | `lab_hybrid.txt` |
| 6.3 | Table 3: inner degrees of the hybrid | `exp_hybrid_degrees.py` | `hybrid_degrees.txt` |
| 6.4 | certified upper-level bounds | `exp_cert_lambda_upper.py` | `certlam_upper.txt` |
| 7.1 | Table 4: polynomial-filter inverse | `exp_fastspec_scale.py` | printed |
| 7.2 | Table 5, Figure 4: setup choices | `exp_setup_compare.py`, `plot_setup_compare.py`; setup times: `exp_compiled_tables.py` | `figures/setup_compare.pdf`, `compiled_tables_slurm_3409080.txt` |
| 7.3 | Table 6: fast hybrid | `exp_fasthybrid.py`; wall times: `exp_compiled_tables.py` | `fasthybrid_scale.txt`, `compiled_tables_slurm_3409080.txt` |
| 7.4 | Table 7: inner tolerance | `exp_inner_eps.py`, `exp_inner_eps2.py`; wall times: `exp_compiled_tables.py` | `inner_eps*.txt`, `compiled_tables_slurm_3409080.txt` |
| 7.5 | Table 8: compiled comparison with the normal equations | `exp_hybrid_c2.py`; row-wise normal operator: `exp_gram_bench.py` | `hybrid_c2_slurm_3408424.txt`, `gram_patch2_slurm_3407803.txt` |
| 7.6 | Table 9, Figure 5: Shepp–Logan | `exp_shepp.py`, `exp_shepp_hybrid.py`, `plot_shepp_figure.py` | `shepp_hybrid.txt`, `figures/shepp_full_N128.pdf` |
| 7.7 | Table 10, Figure 6: data from another discretization | `skimage_adrt_pkg.py`, `exp_skimage_data.py`, `exp_skimage_routes.py`, `exp_skimage_tau.py`, `exp_tau_cost.py`, `plot_skimage_figure.py` | `skimage_routes_slurm_3425902.txt`, `skimage_tau_slurm_3425824.txt`, `skimage_data_slurm_3425808.txt`, `tau_cost_slurm_3425849.txt`, `figures/skimage_data_N128.pdf` |
| 7 | Remark 7.1: Chebyshev polynomial of the normal equations | `exp_normal_cheb.py` | `normal_cheb.txt` |
| 8 | straight lines, σ_min of the ADRT, ghosts | `lab_frame.py`, `exp_frame_vs_rtr.py`, `exp_ghost.py`, `exp_ghost256.py`, `exp_resonance_svals.py` | `lab_frame1.txt`, `lab2.txt`, `ghost.txt`, `ghost256.txt` |
| 8 | integer sampling | `exp_bresenham.py`, `exp_bandlimited_rtr.py` | printed |
| 8 | unsuccessful remedies | `exp_band_tau.py`, `exp_frame_tail.py`, `exp_shifted_outer.py` | `jobB.txt … jobG.txt` |
| 8 | per-frequency lower bounds | `exp_diag_bound.py` | `diag_bound.txt` |
| 9 | Remark 9.1: fusing through more levels | `exp_hybrid_m0.py` | `hybrid_m0.txt` |
| B | Proposition B.1: characteristic polynomial of R^T R, Galois groups (N = 4, 8) | `exp_galois.py` (requires `sympy`) | `galois.txt` |
