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

## All other regression baselines
All non-outlier baselines (compute_state, metrics, PDB trajectory, etc.)
remain from OLD ~/recovar and should NOT be regenerated from new code.
