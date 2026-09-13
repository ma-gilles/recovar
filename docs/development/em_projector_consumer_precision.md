# EM projector consumer precision

Native reference preparation preserves its established host arithmetic and
complex128 Projector::data. The production single-precision GPU consumer must
not inherit that dtype: doing so disables the complex64 texture projector.
RELION 5.0.1 likewise converts host Complex to XFLOAT in AccProjector::initMdl.

[prepare_scoring_projector](../../recovar/em/refinement/projector_preparation.py)
selects complex64 at the standard refinement dispatch boundary. Explicit
double-projection diagnostics select complex128. Score-only double diagnostics
preserve the supplied projector precision, including already-single captures.
The builder, host references, power spectra, on-disk caches and capture arrays
are unchanged. Existing complex64 device arrays are reused without conversion.

The callers cover adaptive coarse scoring, its first-iteration score probe,
per-class/fused sparse second pass, first-iteration winner-subset second pass,
local K-class execution and direct local K1 dispatch. Coarse and pass-2 precision
are independently selectable: the original host slab remains available when
only pass 2 requests double precision. Conversion is outside per-batch kernels.
Low-level independent replay APIs retain their explicit supplied operands;
this change does not globally narrow every projector API or VDAM preparation.

[Consumer tests](../../tests/unit/test_em_projector_consumer_precision.py) cover
K1/K4 values, source non-mutation, texture eligibility, local dispatch and
diagnostic modes. The
[joint-semantics test](../../tests/unit/test_k_class_joint_semantics.py) verifies
that a float32 coarse consumer does not narrow a double pass-2 source.

This is a precision-boundary repair, not a reduction-order or scoring-formula
rewrite. Production GEMM remains unchanged. CPU dtype/dispatch tests do not
qualify GPU scores, support, accumulators, FSC or autonomous trajectories.
Current-source matched GPU and full K1/exactly-K4 qualification remains required.
