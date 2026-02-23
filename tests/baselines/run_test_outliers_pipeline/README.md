## run_test_outliers_pipeline baselines

This folder stores versioned detection-metric snapshots for the outlier
pipeline regression tests.

### Tiny self-contained baseline (`tiny_baseline.json`)

Captured by running:
```bash
OUTLIERS_WRITE_TINY_BASELINE=1 pytest tests/integration/test_run_test_outliers_pipeline_regression.py \
    -m "integration and slow" --run-integration --run-slow -v
```

This requires no external volumes or GPU.  It runs the full
`pipeline_with_outliers` on a 500-image, grid_size=32 simulator dataset.
The baseline stores per-round precision, recall, F1, and inlier/outlier counts.

### Long dataset baseline

To capture a new long-run baseline:
```bash
OUTLIERS_VOLUMES_DIR=/path/to/volumes/vol \
OUTLIERS_BASELINE_JSON=tests/baselines/run_test_outliers_pipeline/long_baseline.json \
OUTLIERS_WRITE_BASELINE=1 \
pytest tests/integration/test_run_test_outliers_pipeline_regression.py \
    -m "integration and slow and gpu" --run-integration --run-slow --run-gpu -v \
    -k test_outliers_pipeline_regression_against_baseline
```

### Running regression checks

```bash
# Tiny (no GPU needed, ~5-30 min):
OUTLIERS_USE_GPU="" pytest tests/integration/test_run_test_outliers_pipeline_regression.py \
    -m "integration and slow" --run-integration --run-slow -v \
    -k test_outliers_pipeline_tiny_regression

# Long (GPU, ~1-2 h):
OUTLIERS_VOLUMES_DIR=/path/to/volumes/vol \
OUTLIERS_BASELINE_JSON=tests/baselines/run_test_outliers_pipeline/long_baseline.json \
pytest tests/integration/test_run_test_outliers_pipeline_regression.py \
    -m "integration and slow and gpu" --run-integration --run-slow --run-gpu -v
```

### Metric schema

Every baseline JSON contains these keys (all numeric, scalar float):

```
outlier_recall_round_N       – fraction of true outliers detected  (higher=better)
outlier_precision_round_N    – fraction of detected that are true  (higher=better)
outlier_f1_round_N           – harmonic mean of recall/precision   (higher=better)
inlier_count_round_N         – number of images kept as inliers
outlier_count_round_N        – number of images flagged as outliers
true_outlier_count           – ground-truth outlier count from simulation
total_images                 – total image count

# Cryo-ET only (particle-level):
particle_recall_round_N
particle_precision_round_N
particle_f1_round_N
particle_inlier_count_round_N
particle_outlier_count_round_N
true_particle_outlier_count
total_particles
```

### Notes

- Baselines committed to git must be produced from a **clean** (unmodified)
  version of the code.
- Include `metadata.json` alongside each baseline JSON (see `run_test_all_metrics/`
  for the pattern: git commit, run args, date).
- The `tiny_baseline.json` should be regenerated whenever the simulator or
  the outlier detection logic changes in a way that legitimately shifts
  detection accuracy.
