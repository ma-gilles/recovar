# hp4 pose-vs-Pmax gap (2026-04-21)

Context: fair 5k replay against RELION on
`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized`,
output in
`_agent_scratch/multi_iter_full_hp4_v2_gt`,
Slurm job `7214098`.

## Main observation

At hp4 local search, `ave_Pmax` still lags RELION materially, but the pose
metrics and GT/FSC metrics are already very close:

- Final merged corr vs GT:
  - recovar: `0.962863`
  - RELION: `0.961718`
- Final merged FSC threshold crossings:
  - both stay above `0.143` through shell `41`
  - both cross `0.5` at shell `37`
- Iter 3 pose metrics (RELION iter 6 / recovar idx 2):
  - full-angle mean `0.6959°`
  - view-direction mean `0.5971°`
  - in-plane mean `0.2849°`
  - translation mean `0.6143 px`
- Iter 4 pose metrics (RELION iter 7 / recovar idx 3):
  - full-angle mean `0.6888°`
  - view-direction mean `0.5915°`
  - in-plane mean `0.2809°`
  - translation mean `0.3499 px`

But `ave_Pmax` still shows a large gap:

- RELION iter 6: `0.9284`
- recovar idx 2: `0.8424`
- RELION iter 7: `0.9502`
- recovar idx 3: `0.8539`

## Interpretation

`ave_Pmax` is still useful as a posterior-sharpness metric, but it is not a
reliable proxy for pose accuracy once hp4 is active. On this benchmark:

- pose accuracy is already sub-degree
- GT/FSC is essentially tied or slightly better for recovar
- the remaining gap is in posterior concentration / scoring statistics

So future hp4 debugging should not treat a large `ave_Pmax` gap as evidence of
catastrophically wrong angles by itself. Check:

- `pose_comparison_iter*.npz`
- `gt_comparison_final.npz`
- translation error quantiles
- shell stats (`tau2`, `sigma2`, `data_vs_prior`)

before concluding that local search is still geometrically wrong.

## Concrete bugs found after this baseline

Two real oversampled-local-search bugs were identified immediately after this
run and fixed in code:

1. Local-search translation priors were centered at `+old_offset` instead of
   `-old_offset`.
2. Oversampled local-search hard assignments were decoded against the coarse
   grid instead of the fine local-search grid.

Those fixes are expected to matter most on the 500-image `noise=0.1` / `os=1`
benchmark, but may also tighten the 5k hp4 posterior gap.
