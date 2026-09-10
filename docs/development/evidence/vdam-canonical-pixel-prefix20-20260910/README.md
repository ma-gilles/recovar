# Frozen canonical-pixel real-prefix evidence

Source `5ca9c8fff30be6fbdf77471cff793388bffffe00`, job **13664081**:
EMPIAR-10076, 10,000 particles, 256², K1 InitialModel, seed29, original200
schedule stopped after20. Native/candidate used one H100 UUID. This is a
prefix comparison, not natural200 or auto-refine convergence evidence.

The integrator rehashed the sealed source and evidence inputs, recomputed all21
saved cross-FSC AUCs and the worst raw-map shell curve, and compared initial-map
bytes. Every map condition passes: minimum AUC **0.9999968977569093** at20;
initial maps are bitwise exact. Canonical pixel metadata is qualified for this
frozen-source real-prefix scope. No ground truth or timing ratio is available.

Strict state remains different:57 coarse-count disagreements, first at3;
294 selected-row Pmax gaps≥1e-3, first at13. These are4,000 selected-row
comparisons across20 updates. Count3 is73 versus72 at input row1579. No raw
margin waiver follows from this uninstrumented run. Later diagnostic explanations
require their own source/input/margin audit. Existing map gates were unchanged.

[Machine-readable admission](result.json) records verified pins and exact scope.
[Producer report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_canonical_pixel_integrated_prefix20_20260909/RESULTS.md)
contains commands, source/native manifests, lifecycle, complete curves and state.
[Integrator audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/canonical_pixel_prefix_admission_20260910/review.py)
is read-only on frozen inputs; outputs remain in its separate artifact root.
[Executed command and source receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/canonical_pixel_prefix_admission_20260910/verification/receipt.json).
The CPU audit completed in11.4s without GPU/Slurm work or source changes.

Reproduce from the active cleanup checkout, with preserved inputs available:

```bash
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u VIRTUAL_ENV \
  CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  .pixi/envs/default/bin/python \
  /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/canonical_pixel_prefix_admission_20260910/review.py
```

The audit regenerates its own result file, not producer artifacts. Missing or
changed input/source pins fail rather than silently using another checkpoint.
This evidence does not qualify later cleanup commits, full-production-F32,
100k/256, exactlyK4, robustness or shared downstream behavior.
