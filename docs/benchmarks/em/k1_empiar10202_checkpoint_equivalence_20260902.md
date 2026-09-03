# EMPIAR-10202 set-6 K=1 checkpoint equivalence

This record compares the first six matching numbered checkpoints from two
full-particle, box-800, I1 RECOVAR refinements.  It is an in-progress
trajectory-equivalence gate, not a final RECOVAR-versus-RELION resolution
result.  The control uses commit
`6e414838463e30e977114f3f5ffe93a2e232766c` (tree
`70834de2d8394815380d2894acf7cadc9299809f`); the compact candidate uses
commit `8069ac01508d57bfd74a7686930ebcea66b6e328` (tree
`bc7220a4dde5c85e4e615cae3113a05b042655fc`).  Both refinement jobs use one
H100, the deposited set-6 particles, poses and CTFs, the same initial
reference, and the same I1 refinement options.

The streamed auditor introduced at commit
`8ff746f0c24b4871eb056f8df0576492fde2a07c` separates two questions. Commit
`2f6759608c82356dd5c24e7b149002df8cc84f28` additionally recognizes a
matching empty local-rotation array as an exact structural sentinel; it still
fails closed for a one-sided empty array, a shape mismatch, or a dtype
mismatch.

1. **Strict execution equivalence** requires all recorded discrete state to
   be exact and bounds every floating product's relative L2 difference.
2. **Scientific indicators** report coarse/fine pose agreement, half-map FSC
   crossing shells and curve error, and regularized half-map agreement.  A
   strict rejection therefore remains visible instead of being relabeled as a
   pass merely because the reconstructed maps agree.

## Completed checkpoint comparisons

The checkpoint index below is zero-based; it corresponds to numbered
iterations one through six in the refinement log.

| Checkpoint | Audit job | Strict execution | Minimum pose agreement | FSC 0.5 / 0.143 shell, control = candidate | FSC RMSE | Minimum map correlation | Maximum map relative L2 |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | `13366560` | accepted | 1.000000 | 32 / 32 | 0 | 0.9999999999999966 | 7.96845e-8 |
| 1 | `13366561` | rejected | 0.9993446 | 69 / 86 | 1.48676e-5 | 0.9999999996200842 | 2.75174e-5 |
| 2 | `13366580` | rejected | 0.9983614 | 89 / 146 | 1.18175e-4 | 0.9999999132360765 | 4.15955e-4 |
| 3 | `13367158` | rejected | 0.9971818 | 102 / 157 | 2.27516e-4 | 0.9999988345090234 | 1.52466e-3 |
| 4 | `13368412` | rejected | 0.9941666 | 102 / 156 | 3.95167e-4 | 0.9999974684818770 | 2.24688e-3 |
| 5 | `13370903` | rejected | 0.9863023 | 149 / 181 | 3.82489e-4 | 0.9999868676649892 | 5.11901e-3 |

All floating artifacts were finite.  Checkpoint 0 retained exact discrete
execution state.  The later strict rejections are real: a small number of
coarse and fine pose decisions diverged as floating-point perturbations
accumulated.  Through checkpoint 5, however, both FSC thresholds cross at the
same shell in the control and candidate.  At checkpoint 5, half-map
correlations are `0.9999869` and `0.9999893`, relative L2 differences are
`0.0051190` and `0.0046228`, the nonzero-shell FSC RMSE is `3.82489e-4`, and
the maximum shellwise FSC difference is `0.00198059`.  This is strong positive
evidence that compact batching has not degraded reconstruction quality through
iteration 6, despite rejecting strict numerical trajectory identity.  It does
not replace the final masked/unmasked half-map comparison after both
trajectories complete.

The six authoritative audit jobs requested and received exactly
`cpu=4,mem=64G,node=1,billing=16`, ran on the `cpu` partition, and completed
`0:0` in 2:07, 3:15, 3:51, 3:39, 4:48, and 5:26.  CPU placement is
intentional: the auditor memory-maps and streams completed arrays and performs
no refinement or GPU calculation.  Job `13370903` used 22,111,680 KiB batch
MaxRSS and exact requested/allocated resources on `della-i13n10`.

The unchanged pre-fix checkpoint-5 audit job `13370307` exited before the
comparison because the original harness treated the legitimate matching
`(0,3,3)` `float32` rotation sentinels as malformed.  Empty-aware proof job
`13370471` completed `0:0`; the committed-script job `13370903` then reproduced
that report byte-for-byte.  These two harness events are not scientific
failures and did not alter either running refinement.

## Evidence and reproduction

The disposable comparison root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_checkpoint_science_8ff746f0c_20260902`
and carries a `SAFE_TO_DELETE` marker.  Its launcher SHA-256 is
`773836c939bd07b4f7568edc2cd4ac7d50f420ce2be06820b7a1f2c107259b5c`.
It contains the authoritative checkpoint 0--4 results, whose JSON SHA-256
values are:

- checkpoint 0: `90db22f20d3b21bf9787d98d6a2708f0fa8a402c2c786d35f255f81adf23974a`;
- checkpoint 1: `c552c84c7911166baff2d2f52dbc69cae883ae914805edd0fa70237e097e5945`;
- checkpoint 2: `52a2f02d9bb59e9deadc78218155d994a1b0e6f1ffda3a1afab9ff235b95d077`;
- checkpoint 3: `e5dd0e147d9ab3dd6a66f15c57198477d66bb924af1763626fab3f603eeeb2ff`;
- checkpoint 4: `f654efdedb5943f3a528ae37c160d35fde376a365f6fce17c58abc897fa467f2`.

The authoritative checkpoint-5 audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it005_committed_audit_2f6759608_20260902`.
Its result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it005_committed_audit_2f6759608_20260902/outputs/iteration-005-committed-equivalence.json`,
SHA-256
`b7254fa0c2f38a2a619afd93a6a17eaa178d399f0dbbd8170f3b052151baae2d`.
It carries a `SAFE_TO_DELETE` marker.  The committed auditor SHA-256 is
`a8b280b94bb0eee6cb829adc7b3cb46d748f18ca6fb0dcca9846e85459ced5a0`;
the Slurm launcher SHA-256 is
`d17143309cd68516441cf2daadc526bebbccfcd12593d092157115d38f751dd2`.

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
  --iteration-zero-based 5 \
  --max-relative-l2 1e-6 \
  --output /absolute/new/output/iteration-005-equivalence.json
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
