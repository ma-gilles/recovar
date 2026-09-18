# Zero-oversampling normalization boundary (work in progress)

RELION's accelerated expectation executes two sampling passes, even at adaptive
oversampling zero. In that case it retains the coarse numeric sum and maximum
for subsequent normalization and Pmax, and does not prune fine support again.
Source-pinned diagnosis and counterfactual are in the
[coordination receipt](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_zero_oversampling_counterfactual_20260913.md).

The existing float32 primitive builds raw fine weights as
`expf(score + float32(50 - max_score))`. Its optional coarse denominator is a
numeric value in this maximum-shifted weight convention, not absolute logZ.
Weights are divided by this denominator without re-normalizing retained support.
`keep_all=True` retains finite, positive-weight selected candidates, not padding.
The coarse controller maximum is preserved independently.

[`_relion_pass2_reconstruction_probs_for_mstep`](../../recovar/em/sparse_pass2/sparse_pass2_posterior.py)
now forwards the existing `normalization_sum_weight` and `keep_all` controls to
the float32 primitive, and rejects them on incompatible reconstruction paths.
Defaults preserve the previous computation. Exact forwarding tests are in
[`test_sparse_pass2_relion_f32_posterior.py`](../../tests/unit/test_sparse_pass2_relion_f32_posterior.py).

[`_compute_k_class_significance_batched`](../../recovar/em/scoring/significance.py) now optionally returns
`relion_f32_sum_weight` and `relion_f32_max_posterior`. On the generic selector
it uses the existing F32 primitive on the actual prior-weighted coarse scores,
not exponentiated absolute log evidence. Its existing support, hard assignments,
log evidence and reported Pmax remain unchanged. The specialized F32 selector
reuses its existing sum and posterior instead of recomputing them.

Ordinary K1 sparse dispatch accepts `relion_f32_normalization_sum_weight`.
Both whole-bucket and rotation-chunked paths forward it and retain all positive
selected weights; these feed reconstruction, noise and rotation statistics.
The option rejects nonzero oversampling, double scoring, hard winners and
non-x-half execution. It does not alter the default route when omitted.
CPU tests verify selection preservation for K1/K4, padded tails and cached or
uncached scores. A denominator-doubling test verifies exact halving of actual
weighted inputs to the GPU-only adjoint, noise sumw and rotation statistics
in both bucket modes. The CPU test substitutes only the final CUDA deposition;
it does not qualify the native scatter or reconstructed maps.

[`run_dense_k_class_em_adaptive`](../../recovar/em/classification/k_class.py) activates this state only for zero-oversampling,
soft Gaussian K1 with F32 scoring and sparse x-half reconstruction. Nonzero
oversampling, K4, firstiter_cc/hard winners and double scoring keep their previous
routes. The dense fallback is excluded only for this active sparse-state path.
The coarse winner is mapped through the selected parent rotation and translation
IDs into the existing canonical source Euler rows, without inverse conversion
from rounded rotation matrices. Missing or ambiguous children fail closed.
Both bucket modes publish coarse Pmax and the mapped coarse pose, while using
fine scores for reconstruction weights. For evidence, RELION retains the coarse
numeric sum but uses the fine exponent shift: the bucket publishes
`log(coarse_sum) - float32(50 - fine_max_score)` before the existing fine
score-to-evidence offset. Reusing coarse absolute log evidence would be incorrect.

The controller and bucket tests cover this routing, canonical pose permutation,
exact denominator scaling, and the evidence shift within its publication ULPs.
Actual-source GPU replay and trajectory/FSC checks remain open. No all-float32
completion or speed admission follows from these transport tests.
