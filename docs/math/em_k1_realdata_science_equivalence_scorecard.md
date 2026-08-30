# K=1 real-data science-equivalence scorecard

This fixed scorecard is separate from strict full-spectrum RELION numerical
parity. Comparable unmasked, unaligned within-engine half-map FSC is always
mandatory. Cross-engine equivalence can pass directly in the canonical frame
or through one pinned continuous proper-SO(3) rotation and translation fitted
from low-frequency merged maps and applied unchanged to both split halves.
Reflection, density-sign, and scale fitting are forbidden. A common-mask FSC
is reported as supporting evidence only and can never rescue a failure.

## Frozen primary gates

| Gate | Threshold |
| --- | ---: |
| Half-map resolution ratio | <= 1.05 |
| Half-FSC curve RMSE in the jointly resolved band | <= 0.02 |
| Absolute half-FSC band-AUC difference | <= 0.02 |
| Merged cross-engine band FSC-AUC, raw canonical **or** proper-rigid route | >= 0.95 |
| Each cross-engine half-map band FSC-AUC on the same route | >= 0.90 |

The joint band is shells 1 through one shell before the earlier first
three-shell-sustained crossing below `1/7`. Resolution uses
`box_size * voxel_size_angstrom / crossing_shell`.

## Frozen calibration replay

The completed 10073 and 10345 runs calibrate the metric only. They ran a
descendant commit and do not count in the PR #158 scoring denominator.

| Case | Status | Half RMSE | Half AUC delta | Merged band AUC | Minimum half band AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| `empiar-10073-native-c1` | pass | 0.002646 | 0.001260 | 0.969136 | 0.940802 |
| `empiar-10345-native-c1` | pass | 0.003242 | 0.000165 | 0.975610 | 0.965089 |

## Fixed scoring case

Equivalence and absolute high-resolution achievement are reported
separately. The target passes overall only when both pass.

| Case | Overall | Equivalence | Route | High resolution | RECOVAR half FSC (A) | RELION half FSC (A) | Raw merged AUC | Proper merged AUC | Half RMSE |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `empiar-10202-set06-k1-I1` | pending | -- | `--` | pending | -- | -- | -- | -- | -- |

The independent high-resolution gate requires **each** engine's unmasked
half-map FSC resolution to be <= 3.0 A. For context,
EMD-9012 records 1.86 A deposited validation and
1.94 A EMDB-calculated unmasked half-map resolution.
These reference values provide context; they are not substituted for either
engine's measured result.

The deposited 30,515-particle half split is already frozen: 15,258 in
half 1 and 15,257 in half 2, hashed as one `uint8` `rlnRandomSubset`
value per source-STAR row. STAR normalization must preserve particle
order, deposited halves, Euler angles, and origins.
The complete 78,118,401,024-byte stack is pinned by SHA-256 prefix
`8eecf0fb` (not merely by its MRC header).

The older `particles.native.star` artifact with SHA-256 prefix
`c13cb927` is explicitly rejected because it rerandomized halves and
dropped deposited pose/shift metadata.

This deposited/FREALIGN frame requires explicit `I1`; bare RELION `I`
canonicalizes to `I2` and is forbidden here. The frozen operator sequence
is identity followed by RELION `SymList` order; its rounded little-endian
`(left, right)` float64 digest begins `093a0876`.

The preparation contract is fully frozen: normalized STAR SHA-256 prefix
`d66afb30`; RELION/RECOVAR initial-map file prefixes `4f83710c` and
`d77516a0`; exact shared canonical-array prefix `b617f90d`. There are no
pending preparation hashes. The subject commit must match exactly;
ancestry is insufficient. The case remains pending only until both full
refinements and their sealed FSC analysis complete.

## Diagnostics

The pinned producer searches a HEALPix order-1 proper-rotation grid, refines
at order 2, and then continuously refines a rotation vector and subpixel
translation on a 65-cubed compact fit using full-box shells through 32.
Its single transform is reported and
applied unchanged to merged, half-1, and half-2 maps. Aligned unmasked FSC
may replace only failed raw cross-engine gates; it cannot change the three
mandatory half-map-quality gates.

The producer also constructs one engine-symmetric soft mask from the two
aligned merged maps, hashes it, and applies it identically to both engines.
Masked within-engine and cross-engine FSC values are included in the report
but are never read by any acceptance gate, so masking cannot conceal poor
independent half-map quality.
All six diagnostic input-map hashes must exactly match the six corresponding
hashes in the external collector. Production evidence also binds and hashes
the launch manifest, finalizer command and canonical argv, and a separate
immutable execution envelope; any supplied binding is checked fail-closed.

## Code references

- `scripts/summarize_em_k1_realdata_science_equivalence.py`: scorecard validation, FSC band metrics, provenance gates, and rendering.
- `scripts/collect_em_k1_science_diagnostics.py`: continuous proper-SO(3)+translation fitting and common-mask FSC artifacts.
- `tests/unit/test_summarize_em_k1_realdata_science_equivalence.py`: deterministic metric, provenance, calibration, and non-rescue tests.
- `/home/mg6942/mytigress/RECOVAR_RELION_EM_COMPARISON/scripts/collect_metrics.py`: external signed-FSC artifact collector pinned by SHA-256 in the manifest.
