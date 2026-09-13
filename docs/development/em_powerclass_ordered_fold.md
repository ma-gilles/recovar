# Ordered powerClass block fold

Private performance candidate on frozen82399613b; no quality or speed admission.

The 250-row box380 saved trace executes the high-resolution power helper14 times.
Its ascending sum has568 blocks, hence7952 iterations and15904 tiny add kernels
(loop body plus induction). The same-thread host XLA module ranges total .0683s;
this is not a full critical-path or speedup prediction. The existing per-row CUB
posterior allocator and repeated projector texture staging are separate costs.

Only the last block-sum fold in
`recovar.em.sparse_pass2.sparse_pass2_scoring._relion_cuda_powerclass_highres_xi2_half`
changes dispatch. F32/custom-CUDA uses
`recovar.cuda_backproject.relion_ordered_sum_f32`, implemented in
`recovar/cuda/relion_scoring.cuh`. One thread per image executes
`total = __fadd_rn(total, block_sum)` in ascending block order from positive zero.
No parallel tree, atomics, FMA, pixel conversion, normalization, cutoff, or
within-block reduction change. CPU, non-CUDA and diagnostic F64 retain the JAX
reference. The existing native atomic powerClass function is NOT a substitute:
it has different pixel FMA and inter-block ordering.

Qualification requires direct F32 wordwise fold tests, complete helper comparison
against the unchanged GPU JAX recurrence, then matched actual-input score/support
and E/M replay. Source/binary-pinned short timing comes before any trajectory
admission. Full K1 and exactly-K4 float32 quality and performance gates remain open.
