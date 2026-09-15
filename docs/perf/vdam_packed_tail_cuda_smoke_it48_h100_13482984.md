# Packed VDAM tail JIT CUDA route smoke (job 13482984)

## Decision

**CUDA route and science smoke: PASS. Performance promotion: NOT CLAIMED.**

This completed job proves that local commit
`818e3bbf0b8f7311c9fe1155edbdd8e57bb96118` compiles and executes the
default-off packed-tail JIT on an H100. It is not a crossed same-allocation
timing gate, and the submit-script entry in its static manifest names Slurm's
ephemeral spool copy. Therefore its timing is observational only and cannot
promote the selector.

## Run identity

- Slurm job: `13482984` (`COMPLETED`, exit `0:0`, 33 seconds)
- Node: `della-h19g2`
- Immutable detached source head: `818e3bbf0b8f7311c9fe1155edbdd8e57bb96118`
- Source tree: `1928a89fea36daf3b6a01b9f066a3f18760b3a4d`
- Output root: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_packed_tail_cuda_smoke_818e3bbf0_20260905T1538Z`
- Checkpoint optimiser SHA256: `addfec8d36581f9c2e02504b29bbca979f9222132eb1a60ccd534a0ed3564e25`
- Particle stack SHA256: `804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9`

The runtime reported a GPU backend with slicing and the custom CUDA extension
enabled. The packed-tail selector was effective for all five physical buckets.
Cold execution created two `_run_packed_source_vdam_tail` cache objects, matching
the two physical image-shape classes (42 and 32); warm execution reused them.

## Science checks

All cold and warm maps are finite and nonzero. The control is the sealed exact-
CTF host-native it48 run from job `13421292`.

| Comparison | map nL2 | FSC AUC | minimum non-DC FSC | particle STAR | 8 discrete fields |
| --- | ---: | ---: | ---: | --- | --- |
| control cold / control warm | `1.9055940e-9` | `0.9999999999999609` | `0.9999999999993720` | exact | exact |
| candidate cold / candidate warm | `1.9606710e-9` | `0.9999999999999576` | `0.9999999999993207` | exact | exact |
| control cold / candidate cold | `1.4722775e-9` | `0.9999999999999762` | `0.9999999999996229` | exact | exact |
| control warm / candidate warm | `2.0785819e-9` | `0.9999999999999530` | `0.9999999999992634` | exact | exact |
| worst of all six pairs | `2.1892564e-9` | `0.9999999999999488` | `0.9999999999991634` | exact | exact |

The worst map delta is below the sealed control repeat envelope
(`2.6815105e-9`). The eight exact metadata fields are selected particle IDs,
best rotation IDs and matrices, translations, class assignments, maximum
posterior values, pose assignments, and half-set class assignments.

The model STAR is intentionally not called bitwise exact. Its map pathname is
output-root-specific, and three continuous spectra carry tiny repeat-scale
rounding changes. The largest cross-run normalized differences are
`1.6632e-8` for `rlnReferenceSigma2`, `1.5621e-8` for `rlnSsnrMap`, and
`1.5636e-9` for `rlnSigma2Noise`. These are recorded, not waived; the crossed
ABBA gate must judge them against its own repeats.

## Observational timing only

- cold profile wall: `24.902579176 s`
- warm profile wall: `1.212870127 s`
- packed-tail telemetry total: `4.276882172 s`
- packed-tail calls: `5`

These numbers show that the route is viable but do not establish a causal
speedup. The next performance decision requires control/candidate/candidate/
control execution on one H100 with private caches and repeat-bounded science.

## Provenance limitation and required repair

Every persistent entry in `provenance/static_inputs.sha256` still verifies.
The only post-job failure is
`/var/spool/slurmd/job13482984/slurm_script`, whose bytes had the preregistered
SHA256 `8746c4a95a67db967d19bd8399ad726d9e2b9e0a90e102b1d05ce2117ae91970`
but were removed by Slurm after completion. The production gate must first copy
the submit script into its output provenance directory and hash that persistent
copy. No completed science needs to be rerun to establish this smoke result.
