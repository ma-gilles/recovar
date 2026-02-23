## run_test_all_metrics baselines

This folder stores versioned `all_scores.json` snapshots used for long regression checks.

### How to create deterministic input volumes

```bash
python scripts/generate_fixed_test_volumes.py \
  --output-prefix /scratch/gpfs/AMITS/mg6942/recovar_fixed_vols/vol \
  --n-volumes 50 \
  --grid-size 128 \
  --voxel-size 4.25
```

### How to capture a baseline into git

```bash
python scripts/capture_metrics_baseline.py \
  --scores-json /path/to/all_scores.json \
  --name clean_YYYYMMDD \
  --volumes-prefix /scratch/gpfs/AMITS/mg6942/recovar_fixed_vols/vol \
  --run-args "--grid-size 128 --n-images 50000 --noise-level 1.0 --contrast-std 0.1"
```

Each baseline folder contains:
- `all_scores.json`: captured metrics dictionary
- `metadata.json`: source path, git commit, run args, and notes
