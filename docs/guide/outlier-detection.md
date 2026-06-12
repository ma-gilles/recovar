# Outlier Detection

RECOVAR includes tools for detecting junk particles and outliers in your dataset.

!!! tip "Manual outlier detection via k-means"
    A common approach is to run `recovar analyze` with k-means clustering, then inspect the PC scatter plots for isolated clusters. The [Tutorial](tutorial.md#step-2-analyze-results) demonstrates this on EMPIAR-10076: cluster 0 (1.3% of particles) is visibly separated from the main body and is removed using `extract_image_subset_from_kmeans` before re-running the pipeline.

## Junk particle detection

```bash
recovar junk_particle_detection output -o junk_output
```

This analyzes the pipeline output to identify particles that are likely junk (ice, aggregates, etc.) based on their fit to the model. Output is organized into `plots/` and `data/` subdirectories. Use `--save-all-plots` for a full diagnostic plot dump (default: just indices and summary).

## Outlier detection

```bash
recovar outlier_detection output -o outlier_output
```

Identifies statistical outliers in the dataset. Like junk detection, output uses `plots/` and `data/` subdirectories, with `--save-all-plots` for full diagnostics.

## Pipeline with outliers

For a combined workflow that runs the pipeline and outlier detection together:

```bash
recovar pipeline_with_outliers particles.star -o output --mask mask.mrc
```

## Using results

Both commands output indices of detected outliers/junk under their `data/` subdirectory.
`outlier_detection` writes the combined indices to
`data/combined_results/`, with one file per zdim used (default `zdim=4`):

- `combined_image_outliers_4.pkl` — image-level outlier indices (always written)
- `combined_image_inliers_4.pkl` — the complementary inlier indices
- `combined_particle_outliers_4.pkl` / `combined_particle_inliers_4.pkl` — particle-level indices (written for tilt-series, or whenever particle outliers are found)

The inlier file is the most convenient input to `--ind` for a clean re-run.
You can also filter a star file directly:

```python
import pickle, starfile

outlier_dir = "outlier_output/data/combined_results"
with open(f"{outlier_dir}/combined_image_outliers_4.pkl", "rb") as f:
    outlier_idx = pickle.load(f)

data = starfile.read("particles.star")
mask = ~data["particles"].index.isin(outlier_idx)
data["particles"] = data["particles"][mask]
starfile.write(data, "particles_cleaned.star")
```

!!! tip "Re-run on the clean subset"
    Pass the inlier indices straight to the pipeline instead of editing the star file:
    ```bash
    recovar pipeline particles.star -o output_clean --mask mask.mrc \
        --ind outlier_output/data/combined_results/combined_image_inliers_4.pkl
    ```
