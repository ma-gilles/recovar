# VDAM RELION projector stable-capacity CPU proof

## Decision

Proceed only with a **center-padded logical projector plus the original logical
`r_max`**. That construction is bitwise identical across all 32 logical sizes
observed in the frozen GF46 0--200 trajectory, including projection into the
larger physical output box followed by a coordinate-preserving logical crop.

Do not rebuild the C++ projector at the physical bucket size. Although all
texels inside the logical sphere are bitwise identical, the larger rebuild
populates an outer Fourier annulus that trilinear interpolation reads at the
logical boundary. Its maximum projection relative L2 is `9.534610e-02` even
when the logical `r_max` is retained. This is about 7,200 times the largest
`1.31763e-05` within-mode map-repeat distance in the earlier GF46 stable-shape
trajectory and is not numerical noise.

Do not substitute the physical bucket's `r_max` either. With logical texels
center-padded by zeros, changing only the cutoff reaches `1.092200e-01`
relative L2. With both a physical rebuild and physical cutoff, the maximum is
`4.213960e-01`.

## Result

| Construction | All 32 pairs bitwise | Max relative L2 | Max abs |
|---|---:|---:|---:|
| Physical rebuild + logical `r_max` | no | 9.534610e-02 | 3.240971e-01 |
| Physical rebuild + physical `r_max` | no | 4.213960e-01 | 5.825108e-01 |
| Center-padded logical + logical `r_max` | **yes** | **0** | **0** |
| Center-padded logical + physical `r_max` | no | 1.092200e-01 | 2.837264e-01 |

The fixture covers 32 logical sizes and 14 physical capacity classes using six
fixed general rotations per pair. RELION's C++ binding builds every PPref in
complex128; the projection test casts it to the complex64 dtype used by VDAM
and invokes RECOVAR's production JAX RELION interpolator on `TFRT_CPU_0`.

The complete per-pair table is emitted by:

```bash
JAX_PLATFORMS=cpu \
RECOVAR_RELION_BIND_BUILD_DIR=/path/to/relion_bind \
pixi run python scripts/prove_vdam_projector_capacity.py \
  --output-json /scratch/projector-proof/report.json \
  --output-markdown /scratch/projector-proof/report.md
```

The focused pytest gate is:

```bash
JAX_PLATFORMS=cpu \
RECOVAR_RELION_BIND_BUILD_DIR=/path/to/relion_bind \
pixi run pytest tests/unit/initial_model/test_vdam_projector_capacity_proof.py \
  --run-slow -v -o log_cli=true -o log_cli_level=INFO
```

It completed `4 passed in 25.26s`. No broad RECOVAR suite was run.

## Why the constructions differ

For a logical radius `rL` and physical radius `rP`, the physical rebuild agrees
bitwise where `r² <= rL²` but adds nonzero texels where
`rL² < r² <= rP²`. Trilinear samples whose coordinates remain inside the
logical sphere can still read neighboring texels from that annulus. A
center-padded logical map preserves the original zero boundary stencil.

The cutoff must also remain logical. A larger physical projection has extra
image-plane pixels, and floating-point rotations around the logical radial
boundary must be tested against the same `rL`; using `rP` changes both the
model-sphere and output-disk decisions.

## Exact next production seam

1. Keep `compute_fourier_transform_map(..., current_size=logical_size)` in
   `reference_to_relion_projector_half_maps`; center-pad its returned PPref to
   the stable physical capacity without running a second C++ FFT.
2. Carry two distinct concepts through the shared projector helpers: static
   physical storage/output capacity and the runtime logical `r_max`.
3. Add a runtime-radius CUDA texture-projector entry point. The current generic
   projector encodes `max_r2_x4` as an FFI attribute, while
   `_relion_projector_texture_enabled` requires PPref shape to equal the shape
   implied by `r_max`; both currently force the logical cutoff back into the
   compilation key. The new path must accept a scalar logical radius operand
   and validate that the PPref shape is a sufficient aligned capacity.
4. Remove `relion_projector_r_max` from `run_local_bucket_big_jit`'s static
   arguments only after that runtime CUDA contract is covered by a focused
   logical-versus-capacity GPU parity test. Keep the physical output size
   static and keep logical pixel loop bounds runtime-bound.

This is the smallest seam that can collapse projector-driven compilation
without silently falling back from the fast texture path or changing RELION's
logical projection.

## Provenance

| Field | Value |
|---|---|
| Integration base | `09b868b9c66195cd2d5a48a67ca7c3ba7f902ec4` |
| Worktree branch | `codex/vdam-projector-capacity-proof-20260903` |
| Host | `della-mol.princeton.edu` |
| JAX / NumPy | `0.9.0.1` / `2.2.6` |
| RELION binding SHA-256 | `9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980` |
