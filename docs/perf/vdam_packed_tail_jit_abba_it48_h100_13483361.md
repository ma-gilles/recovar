# Packed VDAM tail JIT crossed H100 diagnostic (job 13483361)

## Decision

**Steady-state performance: NO-GO. Science diagnostic: PASS. Cold compilation:
directional win, not a promotion.**

The default-off packed-tail JIT at commit
`818e3bbf0b8f7311c9fe1155edbdd8e57bb96118` preserves the tested it48 K=1
science, but it does not reduce warm end-to-end time. Both same-H100 paired
comparisons are slower with the JIT enabled. The median paired saving is
`-0.011473 s` (`-0.8925%`), so the candidate fails the preregistered requirements
that both pairs improve, the median improvement reach 5%, and the saving exceed
the repeat span.

The candidate does reduce fresh-process compilation time by `4.2593 s`
(`14.63%`) at this boundary. That is a useful attribution result, but it does
not justify a larger warm trajectory or a formal accumulator-promotion gate.
The implementation remains isolated, explicit opt-in, and default off.

## Run identity

- Slurm job: `13483361` (`COMPLETED`, exit `0:0`, 2 minutes 19 seconds)
- Order: control / JIT / JIT / control
- Node: `della-h19g2`
- Physical GPU UUID for all four arms:
  `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85`
- Immutable source head:
  `818e3bbf0b8f7311c9fe1155edbdd8e57bb96118`
- Source tree: `1928a89fea36daf3b6a01b9f066a3f18760b3a4d`
- Persistent arm-runner SHA256:
  `bbc2dc6fc976a42be34e0272857cdacd68d6ef281297927d380bba67dd2bd88c`
- Analysis SHA256:
  `1ebd003f20aa3594d713ba1ba96536d900b47b106b2c085f48fb28f9003d74de`
- Evidence-manifest SHA256:
  `a6ecab5801125ece75cbbed5a52cac9b9b081c6273ce52f0d17105d7de5e2308`
- Output root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_packed_tail_abba_818e3bbf0_20260905T155129Z`

Every arm used a fresh process and a private JAX cache. The selector-off arms
record zero packed-tail JIT calls. The selector-on arms record five calls, one
per live bucket, and both requested/effective route fields agree in cold and
warm metadata.

## Timing

| Arm | Selector | Cold wall (s) | Warm wall (s) | Cold expectation (s) | Cold packed tail (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control 1 | 0 | 29.150226 | 1.314856 | 26.036281 | 9.017047 |
| JIT 1 | 1 | 24.849457 | 1.316602 | 21.304882 | 4.301702 |
| JIT 2 | 1 | 24.878083 | 1.304357 | 21.350465 | 4.269615 |
| control 2 | 0 | 29.095942 | 1.283157 | 25.980003 | 8.956828 |

Warm paired savings are `-0.001746 s` (`-0.1328%`) and `-0.021200 s`
(`-1.6521%`). The maximum within-mode warm repeat span is `0.031698 s`; the
median candidate warm wall is `0.8832%` slower than the median control wall.

Cold medians move as follows:

| Cold metric | Control median (s) | JIT median (s) | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall | 29.123084 | 24.863770 | -14.63% |
| Expectation | 26.008142 | 21.327674 | -18.00% |
| Pass 2 | 15.433839 | 10.684300 | -30.77% |
| Packed-tail telemetry | 8.986937 | 4.285659 | -52.31% |

This separation explains the initial one-arm smoke: fusing the tail collapses
compile/dispatch setup, but after both routes are compiled the eager shared
primitives are already just as fast. The optimization is therefore not the
steady-state GPU-utilization lever needed to close the RELION gap.

## Science diagnostic

All eight hard/discrete metadata fields and the complete particle data STAR
are exactly equal in every cold and warm arm. The fields are selected particle
IDs, best rotation IDs and matrices, translations, class assignments, maximum
posterior values, pose assignments, and half-set class assignments.

The empirical repeat envelope is the largest of eight within-selector and
cold/warm map comparisons: map relative L2 `3.0172801e-9`. All four paired
control/JIT map comparisons are inside it:

| Comparison | Map relative L2 | FSC AUC | Within repeat envelope |
| --- | ---: | ---: | --- |
| control 1 cold / JIT 1 cold | `2.3979297e-9` | `0.9999999999999372` | yes |
| control 1 warm / JIT 1 warm | `2.2126309e-9` | `0.9999999999999462` | yes |
| control 2 cold / JIT 2 cold | `1.8360422e-9` | `0.9999999999999633` | yes |
| control 2 warm / JIT 2 warm | `1.8265394e-9` | `0.9999999999999639` | yes |

This lightweight diagnostic deliberately does not claim model-STAR spectra or
raw accumulator promotion. Those would be mandatory in a formal gate if the
candidate had passed performance. Because warm performance is a clear no-go,
no additional GPU qualification is warranted.

## Setup-only failed submission

Job `13483304` failed in 2 seconds before creating an output root or running
science because the outer script referred to a login-node-local `/tmp` arm
runner. Job `13483361` fixed this by hashing and copying a persistent shared
runner into the run provenance directory before executing any arm. No output,
timing, or science from `13483304` is reused.
