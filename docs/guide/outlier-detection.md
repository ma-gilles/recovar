# Outlier Detection

RECOVAR includes tools for detecting junk particles and outliers in your dataset.

!!! tip "Manual outlier detection via k-means"
    A common approach is to run `recovar analyze` with k-means clustering, then inspect the PC scatter plots for isolated clusters. The [Tutorial](tutorial.md#step-2-analyze-results) demonstrates this on EMPIAR-10076: cluster 0 (1.3% of particles) is visibly separated from the main body and is removed using `extract_image_subset_from_kmeans` before re-running the pipeline.

## Junk particle detection

```bash
recovar junk_particle_detection output -o junk_output
```

This analyzes the pipeline output to identify particles that are likely junk (ice, aggregates, etc.) based on their fit to the model.

## Outlier detection

```bash
recovar outlier_detection output -o outlier_output
```

Identifies statistical outliers in the dataset.

## Pipeline with outliers

For a combined workflow that runs the pipeline and outlier detection together:

```bash
recovar pipeline_with_outliers particles.star -o output --mask mask.mrc
```

## Using results

Both commands output indices of detected outliers/junk. You can use these to filter your dataset:

```python
import pickle, starfile

with open("outlier_output/outlier_indices.pkl", "rb") as f:
    outlier_idx = pickle.load(f)

data = starfile.read("particles.star")
mask = ~data["particles"].index.isin(outlier_idx)
data["particles"] = data["particles"][mask]
starfile.write(data, "particles_cleaned.star")
```
