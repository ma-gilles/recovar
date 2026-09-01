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

The completed 10073, 10345, and 10097 runs calibrate the metric only. They
ran descendant commits and do not count in the PR #158 scoring denominator.
A calibration status is determined only by the mandatory unmasked, unaligned
within-engine half-map gates; cross-engine qualification is reported separately.

| Case | Half-map calibration | Half RMSE | Half AUC delta | Raw merged/min-half AUC | Proper merged/min-half AUC | Qualified route |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `empiar-10073-native-c1` | pass | 0.002646 | 0.001260 | 0.969136/0.940802 | -- | `raw_canonical` |
| `empiar-10345-native-c1` | pass | 0.003242 | 0.000165 | 0.975610/0.965089 | -- | `raw_canonical` |
| `empiar-10097-native-c1` | pass | 0.004926 | 0.001501 | 0.935736/0.885803 | 0.935427/0.885517 | `none_unqualified` |

These are deliberately parity-calibration refinements, not reproductions of
the deposited publication workflows. Both engines crossed the
three-consecutive-shell unmasked half-map FSC 1/7 threshold at shell 81 on
10073 and shell 49 on 10345. Under the frozen resolution formula, these are
6.568 A and 8.235 A, respectively. On 10097, RECOVAR and RELION cross at
shells 44 and 45 (7.622 A and 7.452 A). These values are worse than the
3.7 A deposited
resolution for [EMD-8012](https://www.ebi.ac.uk/emdb/EMD-8012) and the 3.51 A
focused resolution for [EMD-20795](https://www.ebi.ac.uk/emdb/EMD-20795),
which also deposits a 3.8 A sharpened full-complex map.

The absolute gap is expected from the frozen calibration protocol. For 10073,
normalization intentionally drops the supplied refined Euler angles and
origins. The 10345 source STAR has no Euler columns and only zero or invalid
in-plane origins. Both engines therefore start from the same newly generated
de-novo K=1 model, and the reported maps and FSCs are unmasked, unsharpened,
and unpostprocessed. The deposited 10073 workflow instead used EMD-2966
low-pass filtered to 60 A. The exact published 10345 complex did not use 3D
classification, but its final maps used non-uniform and local-resolution
refinement, local-resolution estimation, sharpening, and local filtering;
the deposited primary map is a focused refinement. C1 is the appropriate
symmetry for 10073 and 10345 and is not the cause of that gap.

The 10073 and 10345 cases establish that RECOVAR and RELION reach essentially
the same reconstruction under the matched protocol. The 10097 within-engine
half-map comparison also passes strongly, but its raw cross-engine AUCs
(0.935736 merged; 0.885803/0.888938 halves) miss the frozen cross-engine
gates. Corrected job 13276576 tested the allowed proper-SO(3)+translation
route after explicitly adding canonical identity to the HEALPix seed set.
It fitted only a 0.244-degree rotation and 0.084-voxel translation, but its
0.935427/0.885517/0.888483 aligned AUCs still miss the same gates. The route
is therefore recorded as unqualified and does not rescue 10097; a small
global rigid drift does not explain the residual cross-engine difference.
Job 13275901 is retained only as a superseded audit artifact because its
seed search omitted identity and selected a false distant orientation.

These calibration results do not establish
that this intentionally stripped-down protocol reproduces the published
reconstruction. Absolute high-resolution achievement is tested separately by
the frozen 10202 case below.

## Supporting RELION corrected-masked FSC

These measurements are supporting-only: they do not enter any acceptance
gate, cannot rescue an unmasked failure, and do not change the scoring
denominator. Each mask was generated only from the RELION merged map and
then passed byte-for-byte to both engines' independent half-map postprocess.

| Dataset | RECOVAR corrected masked (A; shell) | RELION corrected masked (A; shell) | Curve RMSE | AUC delta | Mask SHA-256 prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| 10073 | 4.156; 128 | 4.092; 130 | 0.004180 | 0.001600 | `49ba576a4c44` |
| 10345 | 5.240; 77 | 5.240; 77 | 0.004790 | 0.000313 | `14f572e49552` |
| 10097 | 5.782; 58 | 5.684; 59 | 0.008370 | 0.002113 | `b47439f30167` |

RELION first low-pass filtered each merged map to 15 A, then used
`relion_mask_create --extend_inimask 5 --width_soft_edge 8`. Both
postprocess calls used `--force_mask --skip_fsc_weighting --low_pass 0
--randomize_at_fsc 0.8 --random_seed 42`. Exact per-dataset argv, input
hashes, mask thresholds, and output hashes are sealed in the aggregate
artifact with SHA-256 `0c6930e801b2d4364fdde8d69d3982250b397f20c6bf17caecc9b6862f0f28ad`.

Producer jobs were `13273806` for 10073/10345 and `13274377` for
10097; requested and allocated resources matched. The first job's nonzero
state occurred only after its two retained datasets, at the later 10097
component audit. The isolated second job reused the literal audited 10097
mask. Regenerating that mask changed only MRC header-statistic bytes 249,
250, and 253; the voxel payload was identical.

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
