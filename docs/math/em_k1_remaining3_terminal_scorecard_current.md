# Fixed K=1 score: 31/34

| Done | Case | Fixture | Numbered min | Final merged | GT delta | Status |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| [ ] | 4 | high_noise_100k_g256_white_noise3_bf80 | 0.999723064486 | 0.992612675922 | +0.003809471595 | fail |
| [ ] | 5 | very_high_noise_100k_g256_white_noise10_bf80 | 0.999954595534 | 0.989369975606 | +0.000453798863 | fail |
| [ ] | 10 | high_res_anisotropic_100k_g384_radial_noise3_bf0 | 0.999993234192 | 0.994309183056 | +0.000127645088 | fail |

Thresholds are frozen at merged FSC-AUC >= 0.995, merged GT delta >= -0.002, and exact numbered topology.

## Case 10 causal localization (2026-08-28)

The remaining case-10 error is upstream pose-state drift, not a dominant final
reconstruction or spectrum error:

| Final replay arm | Signed merged FSC-AUC vs RELION | Gate |
| --- | ---: | --- |
| sealed RECOVAR boundary | 0.993882650175 | fail |
| image correction only | 0.993889044117 | fail |
| scale correction only | 0.993936530446 | fail |
| image + scale correction | 0.993943852388 | fail |
| spectrum only | 0.993904048101 | fail |
| reference only | 0.994191466771 | fail |
| reference + spectra | 0.994212581017 | fail |
| RELION tau only | 0.994488754974 | fail |
| RELION pose/prior state | 0.997558008678 | **pass** |
| exact joint final boundary | 0.997959585936 | **pass** |

The exact-state iteration-2 replay contains the same 576/576 active fine
candidate tuples as native RELION for source row 449, with exact rotation and
translation coordinates, exact orientation priors, identical 183-candidate
significant support, and the same support membership.  Native RELION's top two
fine candidates have exactly equal float32 combined log weight.  RECOVAR's
centered pre-prior scores differ by 8.23e-5 RMS (1.99e-4 maximum after removing
the candidate-independent raw-score offset), making the RELION-first tied
candidate one float32 ULP worse and changing only the hard winner.  The next
discriminator compares the two tied tuples' projected reference, unshifted and
shifted image, correction weight, high-resolution sum, and 256-lane reduction
operands pixel by pixel.

Evidence roots:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_pose_prior_oracle_retry1_20260828T0250ET/analysis/canonical_target_metrics.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_final_boundary_factorial_20260828T0335ET/analysis/with_intensity.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it2_exactstate_row00449_coarse_fine_retry1_20260828T1125ET/analysis/K1_CASE10_ROW000449_IT2_FINE_STAGES.json`
