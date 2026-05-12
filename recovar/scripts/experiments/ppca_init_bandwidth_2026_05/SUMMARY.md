# W-bandlimit test — 2026-05-12 final

Setup: gt_mu_random_w init, q=4, n_iter=5, HEALPix=3, 20k images noise=0.1.

| Dataset | Arm | μCC | W⟨cos⟩ | discrim | iter-5 med pose |
|---|---|---|---|---|---|
| ribo | baseline (no lpmu) | +0.760 | 0.186 | **0.910** | 7.9° |
| ribo | lpmu_R24 only | +0.744 | 0.089 | **0.205** | 133.9° |
| ribo | lpmu_R24 + Wlim_R12 | +0.714 | 0.250 | **0.000** | 142.5° |
| igg | baseline (no lpmu) | +0.760 | 0.269 | **0.760** | 3.7° |
| igg | lpmu_R24 only | +0.269 | 0.085 | **0.137** | 124.9° |
| igg | lpmu_R24 + Wlim_R12 | +0.267 | 0.000 | **0.000** | 127.9° |

## Findings

1. **Baseline (no μ lowpass, free W) is best by a wide margin** — discrim 0.910 ribo / 0.760 igg.
2. **μ low-pass at R=24 voxels alone** (~22Å ribo / ~16Å igg) kills pose recovery: pose stays at random init (~130°) for all 5 iters → discrim drops to 0.20 / 0.14.
3. **μ low-pass + W bandlimit at R=12 voxels** (W forced coarser than μ) collapses W to zero, Pmax → 0, E[z]/GT → NaN, total failure.

## Interpretation

PPCA pose recovery relies on **high-frequency μ content** to discriminate near-identical viewing directions. When μ is band-limited at R_μ ≈ Box/5, the score function is too smooth and poses stay near random init across iterations.

The W bandlimit on top of that is a hard failure mode — applying a Fourier low-pass mask after each M-step strips the W power that PPCA had just learned, leaving W identically zero (visible as WshellRatio[lo,hi]=0,0 from iter 2 onward).

**Practical implication**: a coarse RELION K-class bootstrap (which is what a R≈24 lpmu init would mimic) is NOT recoverable by free-W PPCA at HEALPix-3 with this E-step. Either a finer μ init or a multi-resolution schedule is needed.
