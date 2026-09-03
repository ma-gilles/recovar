# EMPIAR-10202 set-6 K=1 checkpoint equivalence

This record compares the first four matching numbered checkpoints from two
full-particle, box-800, I1 RECOVAR refinements.  It is an in-progress
trajectory-equivalence gate, not a final RECOVAR-versus-RELION resolution
result.  The control uses commit
`6e414838463e30e977114f3f5ffe93a2e232766c` (tree
`70834de2d8394815380d2894acf7cadc9299809f`); the compact candidate uses
commit `8069ac01508d57bfd74a7686930ebcea66b6e328` (tree
`bc7220a4dde5c85e4e615cae3113a05b042655fc`).  Both refinement jobs use one
H100, the deposited set-6 particles, poses and CTFs, the same initial
reference, and the same I1 refinement options.

The streamed auditor at commit
`8ff746f0c24b4871eb056f8df0576492fde2a07c` separates two questions:

1. **Strict execution equivalence** requires all recorded discrete state to
   be exact and bounds every floating product's relative L2 difference.
2. **Scientific indicators** report coarse/fine pose agreement, half-map FSC
   crossing shells and curve error, and regularized half-map agreement.  A
   strict rejection therefore remains visible instead of being relabeled as a
   pass merely because the reconstructed maps agree.

## Completed checkpoint comparisons

The checkpoint index below is zero-based; it corresponds to numbered
iterations one through four in the refinement log.

| Checkpoint | Audit job | Strict execution | Minimum pose agreement | FSC 0.5 / 0.143 shell, control = candidate | FSC RMSE | Minimum map correlation | Maximum map relative L2 |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | `13366560` | accepted | 1.000000 | 32 / 32 | 0 | 0.9999999999999966 | 7.96845e-8 |
| 1 | `13366561` | rejected | 0.9993446 | 69 / 86 | 1.48676e-5 | 0.9999999996200842 | 2.75174e-5 |
| 2 | `13366580` | rejected | 0.9983614 | 89 / 146 | 1.18175e-4 | 0.9999999132360765 | 4.15955e-4 |
| 3 | `13367158` | rejected | 0.9971818 | 102 / 157 | 2.27516e-4 | 0.9999988345090234 | 1.52466e-3 |

All floating artifacts were finite.  Checkpoint 0 retained exact discrete
execution state.  The later strict rejections are real: a small number of
coarse and fine pose decisions diverged as floating-point perturbations
accumulated.  Through checkpoint 3, however, both FSC thresholds cross at the
same shell, map correlation remains above 0.9999988, and the FSC-curve RMSE
remains `2.28e-4` or smaller.  This is positive evidence that compact batching
has not degraded early reconstruction quality; it does not replace the final
masked/unmasked half-map comparison after both trajectories complete.

The four audit jobs requested and received exactly
`cpu=4,mem=64G,node=1,billing=16`, ran on the `cpu` partition, and completed
`0:0` in 2:07, 3:15, 3:51, and 3:39.  CPU placement is intentional: the auditor
memory-maps and streams completed arrays and performs no refinement or GPU
calculation.

## Evidence and reproduction

The disposable comparison root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_checkpoint_science_8ff746f0c_20260902`
and carries a `SAFE_TO_DELETE` marker.  Its launcher SHA-256 is
`773836c939bd07b4f7568edc2cd4ac7d50f420ce2be06820b7a1f2c107259b5c`.
The result JSON SHA-256 values are:

- checkpoint 0: `90db22f20d3b21bf9787d98d6a2708f0fa8a402c2c786d35f255f81adf23974a`;
- checkpoint 1: `c552c84c7911166baff2d2f52dbc69cae883ae914805edd0fa70237e097e5945`;
- checkpoint 2: `52a2f02d9bb59e9deadc78218155d994a1b0e6f1ffda3a1afab9ff235b95d077`;
- checkpoint 3: `e5dd0e147d9ab3dd6a66f15c57198477d66bb924af1763626fab3f603eeeb2ff`.

The immutable input roots are:

- control intermediates:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_full_recovar_release_6e4148384_20260902/outputs/intermediates`;
- compact-candidate intermediates:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_full_recovar_compact_8069ac015_20260902/outputs/intermediates`.

Repeat any completed checkpoint in a new output path with:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
.pixi/envs/default/bin/python scripts/audit_em_k1_checkpoint_equivalence.py \
  --baseline-intermediates /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_full_recovar_release_6e4148384_20260902/outputs/intermediates \
  --candidate-intermediates /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_set6_i1_full_recovar_compact_8069ac015_20260902/outputs/intermediates \
  --iteration-zero-based 2 \
  --max-relative-l2 1e-6 \
  --output /absolute/new/output/iteration-002-equivalence.json
```

Exit status 0 means strict execution equivalence was accepted; status 2 means
the comparison completed but the strict gate rejected it.  Inspect
`science_indicators` in either case.  Missing, malformed, or non-finite
artifacts fail closed.  The focused unit gate is:

```bash
pixi run pytest -q tests/unit/test_audit_em_k1_checkpoint_equivalence.py
```

Additional numbered checkpoints will be audited with the same contract as the
two full trajectories advance.
