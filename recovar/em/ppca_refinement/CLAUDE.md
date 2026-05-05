# PPCA angle-refinement EM guide

This package is for refinement-first, pose-marginalized PPCA bootstrapped from
K-class or GT volumes. It is not a native InitialModel / VDAM PPCA controller.

Current branch facts:

- K-class dense/local orchestration lives in
  `recovar/em/dense_single_volume/k_class.py`.
- Exact-local support must use `LocalHypothesisLayout` from
  `recovar/em/dense_single_volume/local_layout.py`.
- Parent EM testing rules in `recovar/em/CLAUDE.md` apply. Do not run the full
  RECOVAR long suite for PPCA/EM-only changes.
- `z ~ N(0, I_q)`. Eigenvalue scale lives in `W`.
- `W_prior` is variance-like and separate from mean prior, noise variance, and
  latent prior.

Near-term scope:

1. K-class/GT volume initialization with explicit frame and scaling checks.
2. Dense PPCA score/moment and augmented M-step foundation.
3. PPCA schedule state that reuses K-class pose/current-size evolution but
   requires halfset mean agreement and pose stability before resolution growth.
4. Dataset-backed dense PPCA uses the current dense preprocessing/projection
   helpers, normalizes across rotation blocks, and streams fused pass-2
   backprojection into augmented `[mu, W]` normal equations.
5. Dense PPCA refinement-loop code advances `current_size` only through the
   PPCA halfset gate plus the K-class schedule bridge in
   `recovar/em/dense_single_volume/iteration_loop.py`.
6. Exact-local PPCA consumes `LocalHypothesisLayout`; local pruning must remain
   support-only and must not change the PPCA score expression.
7. The bridge updates production `RefinementState`; callers should switch to
   exact-local PPCA when `bridge.state.do_local_search` is true. Do not add a
   parallel local-search implementation.
