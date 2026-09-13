# Coarse geometry at zero oversampling

RELION's accelerated pass1 builds scorer matrices with device
`make_eulers_3D`, including when adaptive oversampling is zero. Fine scoring
and weighted reconstruction use the distinct host Euler matrix path.
They represent the same intended rotations but differ in float32 arithmetic;
texture interpolation can amplify these small matrix differences.

`_run_relion_iteration_loop` in
`recovar/em/refinement/iteration_loop.py` uses the existing
`sampling._relion_adaptive_pass1_rotations` with canonical source Euler rows
and the separate perturbation matrix. The OS0 result is transported as
`coarse_scoring_rotations`, not substituted for `effective_rotations`.
`half_scoring._score_half_dense` applies it only to the coarse operand of
the soft-Gaussian F32 K1 sparse x-half engine. Fine grids, pose metadata and
M-step rotations retain their original values. CPU fallback remains unchanged.
Nonzero oversampling, local search, firstiter_cc/hard first iteration,
K4, and diagnostic-double behavior are not expanded by this repair.

The stack6 matrix-only discriminator on frozen ac9 and the actual-source
replay on peer commit 84fb5867 agree: centered raw-score maximum absolute
gap versus native falls from .01166 to .001230, and support flips fall
from five to three. The count remains 52884 versus native 52883. The remaining scoring/prior boundary is separate; this change
does not qualify strict parity, full trajectories, FSC or speed.

`tests/unit/test_zero_coarse_geometry.py` executes the production eligibility,
transport and engine-operand expressions with controlled sentinels. It does
not replace an actual-source GPU replay. Native capture and diagnostic receipts
are linked from the coordination status in `docs/development/em_status.md`.
