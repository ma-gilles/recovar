# EM half-volume texture staging

The projection route in
[`_project_relion_projector_texture`](../../recovar/em/helpers/projection.py)
reuses [`project_relion_half_capacity`](../../recovar/cuda_backproject.py) without
changing its native kernel. The kernel reads the supplied half-volume directly
instead of constructing a
full cubic JAX buffer and then extracting its positive-frequency half again.

Eligibility is explicit: complex64 PPref, float32 rotations, padding1 or2,
positive model radius, positive even output side at most4096, exact half-storage shape,
and the existing ABI's physical-size/rotation-count bounds. The kernel's radius
operand is the unchanged logical model radius. Image output size is independent
of that radius; no model padding or reconstruction at a different radius is
introduced. Other layouts and precisions
retain their previous staging route. Runtime-radius and image-radius paths are
unchanged. Crop, current-image mask, pixel gather, dense scaling, magnitude
squared and scoring remain in their original order. This is neither a precision
repair nor a scientific-policy change.

The reconciliation candidate additionally supports direct half-storage textures
for large fixed geometry and a persistent texture owned by a bucketed K1 call.
Eligible host complex64 projectors are uploaded once; the owner closes before
deferred backprojection replay and finalization, with dispatch cleanup covering
exceptions. Double scoring retains its previous route. Persistent textures
cannot be combined with runtime capacity or radius overrides. The existing
capacity route and native ABI remain unchanged.

The additional routes require the isolated half-texture and persistent-texture
entry points, pinned in the reconciliation's `cuda_texture_v1` library.
Ownership and routing coverage lives in
[`test_relion_persistent_texture.py`](../../tests/unit/test_relion_persistent_texture.py).
Incremental GPU comparisons establish bitwise projection and squared-amplitude
equality for tested inputs. The large-grid routing guard does not allocate a
full box and does not establish full-size memory or trajectory qualification.

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

The unequal-extent extension was first checked without source edits in
`em_half_staging_unequal_probe_20260913`: all13 A100 cases had bitwise-identical
projections and magnitudes squared, including model radius100/output202 and
smaller output80/198. On one reconstructed real reference, warm output202
projection/abs2 medians were4.058→1.957ms (8 rotations),4.449→2.879ms (128)
and6.427→4.675ms (1024). These are synchronized substep timings, not full
iteration or dataset speed. The standalone source repair excludes the separate
projector precision fix. Source-specific CPU/GPU qualification is recorded in
`em_half_staging_unequal_gate_20260913`; whole-engine quality remains open.
