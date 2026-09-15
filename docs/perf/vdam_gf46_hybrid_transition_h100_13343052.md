# GF46 integrated-hybrid checkpoint diagnostic — job 13343052

> **Complete GPU data; not sealed promotion evidence.** All six requested
> arms finished on one H100. The runner then exited 2 because its v1 analyzer
> incorrectly equated RELION's persisted cutoff rank with the inclusive
> threshold support. A source-pinned v2 rerun is required.

## Result at a glance

| Gate | Result |
|---|---:|
| Frozen transition | GF46 iteration `180 -> 181`, K=1, 1,000 particles |
| Hybrid novel ordered-support rows | **0** |
| Direct repeat support variation | 1 row (row 412), one inclusive tie |
| Persisted cutoff/discrete/STAR/model identity | **exact** |
| Map/model numerical classification | all hybrid-repeat and crossed pairs within `2x` pooled direct-repeat envelope |
| Median warm wall | `21.267882 -> 4.852816 s` (**4.383x**) |
| Median expectation | `20.640134 -> 4.172020 s` (**4.947x**) |
| Median pass 1 | `18.787344 -> 2.309944 s` (**8.133x**) |
| Median pass 2 | `1.110826 -> 1.121583 s` (`0.990x`) |
| Median peak RSS | `3.325 -> 3.508 GiB` |

The hybrid selected exact source16 rescoring for all 6,000 audited profile
images and used zero whole-batch fallbacks. The audit hybrid cold and warm
supports were identical to the direct cold support. Direct warm admitted one
additional threshold-tied ID in its diagnostic support, while its serialized
cutoff counts and every downstream discrete state remained identical.

## Provenance and classification

- Slurm job: `13343052`; H100 node `della-h19g1`.
- Source: `58d868eed043b93e16387eb6fc32bf61c9612a31`.
- Tree: `d4a520d3699fe67a2ce272c4f0d0724d7ccab025`.
- Source manifest: `a7c62e24d342bc209e39cf6c96bf26ebedb86b20f30e6af2477298a750ec63f9`.
- Result root: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_transition_13343052_58d868eed`.
- Slurm state: `FAILED 2:0` only after all arms completed; analyzer setup
  failure, not a CUDA/science/timing-arm failure.

The frozen v3 scores remain correctness **2/20** and runtime **0/20**. This
one-transition diagnostic cannot enable the feature, qualify no-growth, or
replace a full trajectory.
