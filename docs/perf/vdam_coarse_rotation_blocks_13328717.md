# VDAM selected coarse-rotation-block primitive: job 13328717

Status: **GO for hybrid integration experiments; not enabled in production.**

The primitive rescored selected aligned 16-rotation blocks while preserving the
existing rectangular scorer's 128-thread translation/lane mapping, pixel
order, direct residual-square update, and per-thread atomic additions. It is a
shared `recovar.cuda_backproject` primitive, not a VDAM-only duplicate.

## Sealed run

- Source commit: `695a629fa70ee951734d98728bb3daffcc53bd88`
- Source tree: `04a3ca49be3547cc8329d6c27c7b1aa2acf30c3e`
- Slurm job: `13328717` (`COMPLETED`, exit `0:0`, 54 seconds)
- GPU: NVIDIA H100 80GB HBM3, UUID
  `GPU-ddb1592d-744e-ea56-d0a3-aec6e7c97d10`
- Candidate CUDA binary SHA-256:
  `664c741946c16df13b59ae41c6a98949c5463beddbe33193e66948ae87a91de0`
- Result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_rotation_blocks_h100_695a629fa_20260901`

## Focused checks

| Gate | Result |
|---|---:|
| CPU wrapper, symbol, source-topology, dtype/fail-closed checks | 13 passed |
| Existing full rectangular CUDA atomic-envelope test | passed |
| Selected-block CUDA atomic-envelope/padding test | passed |

The selected-block GPU test covers two physical images with different image,
weight, and initial-diff2 operands; reversed source-block order; lookup holes;
a valid final partial block; reserved `-1` padding (`+inf`); and malformed
negative/out-of-range IDs (`qNaN`). Every valid score falls within the same
enumerated legal atomic-order envelope as the established full rectangular
primitive.

## Binary resource audit

The previously qualified full rectangular binary and the new full rectangular
kernel both use `REG:80 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:608` on `sm_90`.
The new selected-block kernel uses `REG:80 STACK:0 SHARED:0 LOCAL:0` and
`CONSTANT[0]:624`. Factoring the common scorer into a forced-inline helper did
not change register count or occupancy resources, but its SASS is not
byte-identical. A focused old/new warmed kernel-runtime comparison remains a
required gate before claiming zero performance change to the existing full
rectangular path.

## Deliberate limitations

- The primitive is not wired into production selection or posterior code.
- IDs are local to the passed reference slab. Production integration must use
  aligned blocks or a compact, explicitly mapped union of global source blocks.
- `-1` is the only padding ID; any other invalid ID emits NaN and must trigger
  full-direct fallback.
- A partial final source block currently follows RECOVAR's existing
  16-register rectangular-tail behavior. RELION's distinct one-Euler tail
  topology needs its own implementation and gate before a general default.
- Posterior-band blocks alone are insufficient: the hybrid selector must also
  certify and rescore raw pre-prior winner blocks before recomputing RELION's
  float32 coarse offset, posterior, support, and discrete state.
