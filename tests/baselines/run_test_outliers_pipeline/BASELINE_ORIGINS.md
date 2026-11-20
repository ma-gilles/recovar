# Outlier Pipeline Baseline Origins

## SPA baselines (from OLD ~/recovar)
- `long_generated/all_scores.json` — generated from OLD ~/recovar code
- `tiny_baseline.json` — generated from OLD ~/recovar code

## ET baselines (from current branch)
- `long_generated/all_scores_cryo_et.json` — regenerated from branch
  `claude/clean-embedding-ordering_allpass` (2026-03-24). NOT from ~/recovar.
  Reason: embedding ordering change (sorted-original vs halfset-concatenated)
  causes KMeans to produce different clusters, giving different but equally
  valid outlier classification. Old baseline was incompatible with new ordering.

## Cryo-ET baselines on dosed tilt series (2026-10-18)
The simulator used to drop the per-tilt dose: it appended dose and tilt as CTF columns
11-12 while the generators already return 11 columns, so cryo-ET fixtures had dose 0
(`rlnMicrographPreExposure=0`). With the fix, tilts carry 1.5-19.5 e/A^2, which makes the
data harder. These cryo-ET baselines were regenerated on the dosed data from the control
cryoet-dev `1af04bd53` + the dose fix (`bb37e9097`), H100, custom CUDA library:
- `../run_test_all_metrics/long_generated/all_scores_cryo_et.json`: job 15788276; legacy
  alias keys (`constrasts_*`, `contrasts_*`, `pcs_relative_variance_*`) set to the values
  of the metrics they alias.

The two outlier-pipeline cryo-ET baselines were then regenerated again with junk detection
in round 1 only (78eafa9af), from the same control plus that commit (`bf539fd2d`):
- `long_generated/all_scores_cryo_et.json` (this folder): per-metric median of four runs,
  rounded to 4 decimals (Slurm job 15875161). Round-2 precision spans 0.723-0.768.
- `../pipeline_with_indices/cryo_et_baseline.json`: per-metric median of four runs (Slurm
  job 15875162). Round-2 particle precision spans 0.478-0.485.

Before 78eafa9af, junk detection in round 2 put its cutoff at the largest bend of a smooth
score curve and flagged anywhere from 10% to 80% of clusters run to run. The SPA outlier
and indices baselines still come from old ~/recovar; the SPA tests pass against them with
the change (job 15875163).

Old ~/recovar cannot produce these: it ignores the STAR dose and applies its own fixed
per-tilt dose model.

## All other regression baselines
All non-outlier baselines (compute_state, metrics, PDB trajectory, etc.)
remain from OLD ~/recovar and should NOT be regenerated from new code.
