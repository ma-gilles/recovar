# Local-search implementation lessons

The current implementation lives in [local/](../recovar/em/local/); refinement
scheduling lives in [refinement/](../recovar/em/refinement/). Follow the
[EM validation ladder](development/em_parity_runbook.md#validation-ladder)
for current checks. Historical timings below are not current qualification.

## Projection deduplication

Per-bucket projection deduplication previously increased a real 5k exact-local
run from about 76.7s to 126.9s. After reconstruction gating, the measured
projection duplicate factor was only about 1.004–1.005. Keep direct projection
unless a new workload demonstrates enough duplication to justify the gathers.

## Packed-half adjoint

Packed half-image rows require the direct `half_volume=True` adjoint or its
VJP equivalent. For out-of-plane rotations, the old
`half_image=True, half_volume=False` comparison produced a non-Hermitian full
volume and was not a valid reference. Preserve the native packed-half contract;
do not restore full-volume accumulation followed by folding as a parity fix.
The historical direct-half path reached about 20.8s per warm 5k/128 iteration;
that measurement does not qualify the current source or larger workloads.
