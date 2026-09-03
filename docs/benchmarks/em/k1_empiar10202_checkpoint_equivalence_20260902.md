# EMPIAR-10202 set-6 K=1 checkpoint equivalence

This record compares the first eight matching numbered checkpoints from two
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
iterations one through eight in the refinement log.

| Checkpoint | Audit job | Strict execution | Minimum pose agreement | FSC 0.5 / 0.143 shell, control = candidate | FSC RMSE | Minimum map correlation | Maximum map relative L2 |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | `13366560` | accepted | 1.000000 | 32 / 32 | 0 | 0.9999999999999966 | 7.96845e-8 |
| 1 | `13366561` | rejected | 0.9993446 | 69 / 86 | 1.48676e-5 | 0.9999999996200842 | 2.75174e-5 |
| 2 | `13366580` | rejected | 0.9983614 | 89 / 146 | 1.18175e-4 | 0.9999999132360765 | 4.15955e-4 |
| 3 | `13367158` | rejected | 0.9971818 | 102 / 157 | 2.27516e-4 | 0.9999988345090234 | 1.52466e-3 |
| 4 | `13368412` | rejected | 0.9941666 | 102 / 156 | 3.95167e-4 | 0.9999974684818770 | 2.24688e-3 |
| 5 | `13370903` | rejected | 0.9863023 | 149 / 181 | 3.82489e-4 | 0.9999868676649892 | 5.11901e-3 |
| 6 | `13372565` | rejected | 0.9743086 | 150 / 183 | 1.89246e-3 | 0.9999485706015202 | 1.012996e-2 |
| 7 | `13374865` | rejected | 0.9564818 | 159 / 192 | 1.02527e-3 | 0.9999027478619030 | 1.393356e-2 |

All floating artifacts were finite.  Checkpoint 0 retained exact discrete
execution state.  The later strict rejections are real: a growing minority of
coarse and fine pose decisions diverged as floating-point perturbations
accumulated.  Through checkpoint 7, however, both FSC thresholds cross at the
same shell in the control and candidate.  At checkpoint 7, half-map
correlations are `0.9999027` and `0.9999241`, relative L2 differences are
`0.0139336` and `0.0123057`, the nonzero-shell FSC RMSE is `0.00102527`, and
the maximum shellwise FSC difference is `0.00600702`.  Coarse/fine pose
agreement is `0.9564818` for half 1 and `0.9623124` for half 2.  The largest
floating-array relative L2 difference (`0.0788678`) remains in the half-1
`Ft_y` accumulator, not a map.  This is strong positive evidence that compact
batching has not degraded reconstruction quality through numbered iteration
8, despite rejecting strict numerical trajectory identity.  The growing drift
is retained in the ledger and does not replace the final masked/unmasked
half-map comparison after both trajectories complete.

The eight authoritative audit jobs requested and received exactly
`cpu=4,mem=64G,node=1,billing=16`, ran on the `cpu` partition, and completed
`0:0` in 2:07, 3:15, 3:51, 3:39, 4:48, 5:26, 3:09, and 2:57.  CPU placement is
intentional: the auditor memory-maps and streams completed arrays and performs
no refinement or GPU calculation.  Job `13370903` used 22,111,680 KiB batch
MaxRSS on `della-i13n10`; job `13372565` used 8,063,012 KiB
`/usr/bin/time` maximum RSS (Slurm batch MaxRSS 35,789,204 KiB) with exact
requested/allocated resources on `della-h16n18`.  Job `13374865` used
8,178,116 KiB `/usr/bin/time` maximum RSS (Slurm batch MaxRSS 36,043,200 KiB)
on `della-h14n1` with the same exact requested/allocated resources.

The unchanged pre-fix checkpoint-5 audit job `13370307` exited before the
comparison because the original harness treated the legitimate matching
`(0,3,3)` `float32` rotation sentinels as malformed.  Empty-aware proof job
`13370471` completed `0:0`; the committed-script job `13370903` then reproduced
that report byte-for-byte.  These two harness events are not scientific
failures and did not alter either running refinement.

## Corrected/no-pad versus combined candidate checkpoints

A separate full-particle experiment compares the corrected/no-padding source
at commit `b31f7bb3a88b96885e568fa4a12d5ec265ab4aab` with the combined
owned-mean plus persistent-texture source at commit
`37f640c7e1553ce2b6ed95b061aed4c9e16552b8`.  This comparison is distinct from
the release-versus-compact experiment above and does not isolate either
candidate change.  At numbered iteration 7
(zero-based checkpoint 6), the controller metadata, logged resolution
(`4.23` A), FSC crossing shells (`150` at 0.5 and `183` at 0.143), and next
quantized size (`498`) agree.  The execution state is not exact: pose
agreement is `97.2408%` and `97.3914%` for the two halves, and every one of the
42 translation-grid scalars differs by at most `0.00340033` pixels.

The regularized half maps remain close but measurably different.  Their
centered correlations are `0.999945606` and `0.999950657`, with relative L2
differences `0.0104176` and `0.00992269`.  The nonzero-shell FSC RMSE is
`0.000848545`, the maximum absolute shellwise FSC delta is `0.00324598`, and
the mean candidate-minus-baseline FSC delta is `+0.0000933168`.  Persistent
texture took `591.2` seconds versus `589.3` seconds (`+0.322%`), with an HBM
peak of `39,075` MiB versus `39,073` MiB.  The correct conclusion at this
checkpoint is therefore **science-close with measurable divergence**, not
execution equivalence.  Final acceptance still depends on the completed
masked and unmasked half-map comparison.

CPU audit job `13381829` completed `0:0` in 6:25 with exact requested and
allocated `cpu=4,mem=32G,node=1,billing=8`, no GPU, and no exclusive
allocation.  Its comparator exit status 2 is the expected result for a valid
nonexact comparison.  Preceding job `13381813` failed after three seconds on
an obsolete audit-repository HEAD pin, before launching the comparator; it
has no scientific result.  The sealed semantic summary is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint7_deep_corrected_vs_ptex_20260903T0610/checkpoint7_deep_corrected_vs_ptex.json`
(SHA-256
`ab85f97d7dde5592d4bf1e7b0a97ce5b551b9e18ff821350e0fa001556192e51`).
The root carries a complete manifest with SHA-256
`ccc3978c30dbaf11c5bd958fa29a99b734a71d10b3532231fa935f1431e810fc`.

At numbered iteration 12, the intermediate science indicators have not
collapsed, but the trajectory divergence is no longer small enough to carry
the candidate.  The two FSC curves still cross 0.5 and 0.143 at the same
shells 186 and 221, and the controller still chooses next size 570.  However,
coarse and fine pose agreement is only `87.0494%/87.0486%`, half-map
relative-L2 differences are `0.032883/0.032615`, and `Ft_y` relative-L2
differences are `0.176446/0.176019`.  FSC RMSE is `0.00181986` and maximum
shellwise difference is `0.00837472`.  At numbered iteration 13 the FSC 0.5
crossing differs by one shell and the next-size decisions split to 574 versus
576.  This rejects the combined candidate rather than extending the earlier
"science-close" observation into an acceptance.

Activation telemetry is equally important to interpreting the result.  The
intended exact-local persistent-texture route emits zero activation lines;
all 24 candidate device-signature records say `active=false`.  Both runs use
the same pre-existing sparse-pass-2 persistent-texture path.  The comparison
therefore neither qualifies nor causally blames exact-local persistent
texture.  It also cannot isolate the owned-mean transfer from the wrapper and
fallback changes in the child commit.

Audit job `13382769` completed `0:0` in 9:01 with exact CPU-only resources.
The sealed semantic JSON and validated manifest are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint12_size564_deep_corrected_vs_ptex_20260903T0736/checkpoint12_size564_deep_corrected_vs_ptex.json`
(SHA-256
`68f7da89d582788e03b5453ac51d0b79f2671b5b03f53e0272ef3008ab05a278`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint12_size564_deep_corrected_vs_ptex_20260903T0736/MANIFEST.sha256`
(SHA-256
`faf5eef53f9eb20a02110a6de4c7b424324587927460f3d14c7286566ee31b7f`).
Local deliverable commit `a8a3c7eb1` removes both candidate changes; the
audit remains as a fail-closed regression record.

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

The authoritative checkpoint-6 audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it006_committed_audit_47f8fe79d_20260903`.
Its result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it006_committed_audit_47f8fe79d_20260903/outputs/iteration-006-committed-equivalence.json`,
SHA-256
`b286db9ac8a8c0ac3775745a9c28554ff148199f38648bd40a829949aa34305e`.
It carries a `SAFE_TO_DELETE` marker.  The Slurm launcher SHA-256 is
`58ff042385a6aab87473ebc664e0cd2127d5e03d05bed780e9d3981f765e60f8`;
the stable candidate file-ledger SHA-256 is
`63b78ee0a48b1ab9728fcafc8807d772418af5a99a79ca3d8307af1a5b9b72e9`.

The authoritative checkpoint-7 audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it007_committed_audit_1dd067bd7_20260903`.
Its result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/empiar10202_it007_committed_audit_1dd067bd7_20260903/outputs/iteration-007-committed-equivalence.json`,
SHA-256
`fc41fe9c4b1ba5875ecec0860a1d8e03a3a355012accf5e401c7ceedeb3ebd52`.
The run and runtime roots both carry `SAFE_TO_DELETE` markers.  The committed
auditor SHA-256 is
`a8b280b94bb0eee6cb829adc7b3cb46d748f18ca6fb0dcca9846e85459ced5a0`;
the Slurm launcher SHA-256 is
`f8e7bfccc8610de0bf6effb7c7c813e6dcc40447e86db65cabf07b15bcafa34e`.
The baseline and candidate file-ledger SHA-256 values are
`124befbc8ade7c80583da88fd5983ca2cabf80fa294adc739057ee866752f324`
and
`f40ade466237d7ae0e643b39472ac57d45b562ca31fcb7d32327c68bd8139bab`,
respectively.

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
  --iteration-zero-based 7 \
  --max-relative-l2 1e-6 \
  --output /absolute/new/output/iteration-007-equivalence.json
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
