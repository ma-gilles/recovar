# EM half-volume texture staging

The projection route in
[`_project_relion_projector_texture`](../../recovar/em/helpers/projection.py)
reuses [`project_relion_half_capacity`](../../recovar/cuda_backproject.py) without
changing its native kernel. When physical and logical extents coincide, the
kernel can read the supplied half-volume directly instead of constructing a
full cubic JAX buffer and then extracting its positive-frequency half again.

Eligibility is explicit: complex64 PPref, float32 rotations, padding1 or2,
positive model radius, output side twice that radius, exact half-storage shape,
and the existing ABI's physical-size/rotation-count bounds. The kernel's radius
operand is the unchanged logical model radius. Other layouts and precisions
retain their previous staging route. Runtime-radius and image-radius paths are
unchanged. Crop, current-image mask, pixel gather, dense scaling, magnitude
squared and scoring remain in their original order. This is neither a precision
repair nor a scientific-policy change.

The candidate does not persist textures across batches; each call still owns
allocation, staging and destruction inside the existing native transaction.
Historical persistent texture recovery remains separate resource-lifetime work.
No new native symbols, automatic rebuilding or shared-library replacement is
required by this candidate; a qualification must pin a library already exporting
`ProjectRelionHalfRuntime` and fail closed if it is absent.

Focused coverage is in
[`test_em_half_texture_staging.py`](../../tests/unit/test_em_half_texture_staging.py),
with existing ABI/geometry tests in
[`test_relion_projector_capacity.py`](../../tests/unit/test_relion_projector_capacity.py).
CPU tests validate routing and unchanged masks/gathers/scaling; GPU tests compare
all float32 words with the previous full-staging path. CPU tests do not establish
kernel equivalence or performance.

Qualification receipts live under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_half_staging_gate_20260913`.
The preceding historical operator comparison is in
`em_projector_history_ab_20260913`: historical direct-half/persistent primitives
were faster with bitwise-identical values on one reconstructed real10073
reference and128 saved rotations. Those ratios do not measure this candidate,
an E-step, an exact historical reference replay or a complete trajectory.
Required follow-ups remain current-source fixed-state outputs/support/accumulators,
short batch-size timing, fast CPU/GPU guards and the K1/exactly-K4 scientific
ladder before quality/performance acceptance. Preserve production GEMM.
