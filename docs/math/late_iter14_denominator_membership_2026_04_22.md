# Late Iter 13 -> 14 Denominator Membership Findings

## Scope

This note records the late-iteration RECOVAR-vs-RELION parity debugging
results for the corrected accelerated reruns on the 5k benchmark at
RELION iter `13 -> 14`.

The goal here is not to match old RECOVAR behavior. The goal is to
explain the current RECOVAR-vs-RELION `Pmax` gap in terms of:

- candidate support membership
- denominator membership
- score geometry on the shared candidate set

## Corrected truth source

RELION dump targeting was corrected to match by stack index from
`rlnImageName`, not by internal `part_id`.

Corrected late accelerated rerun targets:

- stack `422` -> dataset row `421`
- stack `1094` -> dataset row `1093`

Artifacts:

- RELION operands:
  - `_agent_scratch/relion_truthdump_it014_stack422_acc_v2/relion_operands.npz`
  - `_agent_scratch/relion_truthdump_it014_stack1094_acc_v2/relion_operands.npz`
- RELION reruns:
  - `_agent_scratch/relion_truthdump_it014_rerun_stack422_acc_v2/run_it014_data.star`
  - `_agent_scratch/relion_truthdump_it014_rerun_stack1094_acc_v2/run_it014_data.star`
- RECOVAR score dumps:
  - `_agent_scratch/late_it014_pairdump_hspecon_support_v1/score_dumps/it000_half1/group_0001_images_421.npz`
  - `_agent_scratch/late_it014_pairdump_hspecon_support_v1/score_dumps/it000_half2/group_0001_images_1093.npz`
  - `_agent_scratch/late_it014_pairdump_hspecoff_v2/score_dumps/it000_half1/group_0001_images_421.npz`
  - `_agent_scratch/late_it014_pairdump_hspecoff_v2/score_dumps/it000_half2/group_0001_images_1093.npz`
- Analysis outputs:
  - `_agent_scratch/analysis/late_it014_stack422_hspecon_v2.json`
  - `_agent_scratch/analysis/late_it014_stack1094_hspecon_v2.json`
  - `_agent_scratch/analysis/late_it014_stack422_hspecoff_v2.json`
  - `_agent_scratch/analysis/late_it014_stack1094_hspecoff_v2.json`
  - `_agent_scratch/analysis/late_it014_denominator_membership_summary_v2.json`

## RELION semantics

On the actual local accelerated rerun path:

- `Pmax == exp_max_weight / exp_sum_weight`

So the late mismatch is not a raw-vs-normalized `Pmax` semantics issue.

## Rerun caveat

The corrected accelerated reruns are self-consistent, but they still do
not exactly reproduce the original reference `run_it014_data.star`.

- stack `422`: rerun `0.255284...`, reference `0.2246`
- stack `1094`: rerun `0.241808...`, reference `0.248135`

Use the corrected reruns as the current particle-level truth source for
candidate-set debugging, but do not assume they have already closed the
reference-run provenance gap.

## Support identity results

For both corrected late particles, RECOVAR and RELION match on:

- factorized local mode
- selected direction set
- selected psi set
- angular prior values
- translation grid center / old-offset behavior
- best `(rotation, translation)` candidate

This closes the earlier suspicion that these particular late particles
were diverging because of non-factorized local support or mismatched
angular priors.

## RELION denominator / threshold / reconstruction membership

The v2 accelerated dumps explicitly record three membership concepts:

- `D_RELION`: in the denominator used for `sum_weight`
- `S_RELION`: in the fine-threshold subset
- `R_RELION`: in the reconstruction set

For the two corrected late reruns:

- stack `422`: `D = S = R = 53`
- stack `1094`: `D = S = R = 48`

So for these late reruns there is no extra post-threshold
renormalization effect to explain the `Pmax` mismatch. The relevant
denominator difference is that RELION materializes a much smaller
candidate set than RECOVAR’s full grouped local factorized grid.

## Quantitative results

### Half-spectrum on

RELION rerun:

- stack `422`: `0.255380`
- stack `1094`: `0.241936`

RECOVAR full local denominator:

- stack `422`: `0.178093`
- stack `1094`: `0.259605`

RECOVAR masked to `D_RELION`:

- stack `422`: `0.245492`
- stack `1094`: `0.269317`

RECOVAR full-support mass inside `D_RELION`:

- stack `422`: `0.725455`
- stack `1094`: `0.963939`

Equivalent outside mass:

- stack `422`: `27.45%`
- stack `1094`: `3.61%`

Interpretation:

- stack `422`: denominator membership explains most of the late gap
- stack `1094`: denominator membership explains little; score geometry on
  the shared set still matters

### Half-spectrum off

RELION rerun:

- stack `422`: `0.255380`
- stack `1094`: `0.241936`

RECOVAR full local denominator:

- stack `422`: `0.367423`
- stack `1094`: `0.543866`

RECOVAR masked to `D_RELION`:

- stack `422`: `0.400056`
- stack `1094`: `0.544265`

Interpretation:

- `half_spectrum_scoring=off` is far too sharp late
- masking to `D_RELION` does not rescue that path

## Shared-set score geometry

On `half_spectrum_scoring=on`, the shared-set score fits are strong but
not exact:

- stack `422`: corr `0.9879`, slope `0.9879`, `R^2 = 0.9759`,
  mean abs delta error `0.2869`
- stack `1094`: corr `0.9722`, slope `0.8986`, `R^2 = 0.9451`,
  mean abs delta error `0.5180`

Interpretation:

- stack `422`: shared-set score geometry is already close enough that the
  main issue is denominator membership
- stack `1094`: a real shared-set score mismatch remains

## Code-path implication

Public RECOVAR already has sparse candidate-mask machinery in the
adaptive oversampling path:

- `helpers/oversampling.py`
- `find_significant_mask(...)`
- `compute_pass2_stats_sparse(...)`

But the grouped exact local-search path currently calls `run_em(...)`
without a `rotation_translation_mask`, so it normalizes over the full
factorized local grid for the image/group.

That makes the likely next implementation target:

- a diagnostic RECOVAR mode that evaluates or normalizes on a RELION-like
  materialized denominator set

This should still be treated as a diagnostic, not a production parity
fix, until the shared-set score mismatch and rerun-vs-reference mismatch
are both better understood.
