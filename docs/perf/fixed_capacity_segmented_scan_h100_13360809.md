# Fixed-capacity segmented scan boundary — H100 job 13360809

## Decision

The segmented scan is an exact reusable execution boundary, but it is not a
material VDAM performance lever. Retain the implementation as infrastructure;
do not promote it as the pass-2 speed fix.

## Qualification

| Field | Value |
|---|---|
| Source | `560c1a6c754328140613fa5a2e3aea7b3e17afe3` |
| Slurm | `13360809` (`COMPLETED`, exit `0:0`) |
| Hardware | `della-h21g4`, NVIDIA H100 80GB HBM3, `GPU-099c0d77-bb85-f2e9-f628-148b733c9176` |
| Gate | `recovar.fixed_capacity_local_score_gate.v7` |
| Precision lanes | float32 and float64 |
| Exactness | Mature calls, whole boundary, uniform scan, and production segmented scan all exact |
| Default promotion | No |

The production executor was `chronological_uniform_scan_segments`. Every
sealed call retained the mature EM operand, carry, output, and chronology
contract.

## Mechanism timing

| Calls | Synchronized calls (ms) | Segmented scan (ms) | Scan speedup | First scan (s) |
|---:|---:|---:|---:|---:|
| 2 | 1.809 | 1.979 | 0.914x | 0.00497 |
| 8 | 4.870 | 4.599 | 1.059x | 1.496 |
| 16 | 8.779 | 7.468 | 1.176x | 1.553 |

The largest warm benefit is only 1.176x for 16 tiny calls, with a roughly
1.5-second first-compile cost. GF46 uses only a few much larger pass-2 calls,
so launch removal cannot explain or close the remaining runtime gap.

## Provenance

- Run JSON SHA-256: `4035dfc21008b751d9c9cd9b905cfcf49bf6e90c7de0dda3ae380b2edb6c207c`
- Harness result SHA-256: `eeea60ed2e54a95e529cca04494932d9dabf7f9321bd29e14a19b49011efb6e9`
- JUnit SHA-256: `7858d9af9c3386e8bbaee33ec465f59ffb91ef5528650ec3441e14565a531b68`
- Source manifest SHA-256: `ce02c3a50c46952244705369befba9f33546c67553d784007d4051b8d6d37323`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/fixed_capacity_segmented_scan_h100_560c1a6c7_20260902T2355Z`

