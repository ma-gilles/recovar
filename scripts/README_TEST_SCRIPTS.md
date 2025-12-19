# Test Scripts Usage Guide

This directory contains scripts for creating test datasets and running the recovar pipeline for performance profiling and testing.

## Quick Start

### Option 1: Using Presets (Easiest)

```bash
# Make scripts executable (first time only)
chmod +x scripts/run_test.sh scripts/run_test_presets.sh

# Quick small test (128x100k images)
./scripts/run_test_presets.sh small-all

# Or run steps separately
./scripts/run_test_presets.sh small-create    # ~30 seconds
./scripts/run_test_presets.sh small-pipeline  # ~30 minutes

# Large test (256x300k images)
./scripts/run_test_presets.sh large-all       # Create + pipeline
./scripts/run_test_presets.sh large-create    # ~3 minutes  
./scripts/run_test_presets.sh large-pipeline  # ~6 hours
```

### Option 2: Using Main Script (More Flexible)

```bash
# General syntax
./scripts/run_test.sh <action> <image_size> <n_images>

# Create datasets
./scripts/run_test.sh create 128 100000   # Small dataset
./scripts/run_test.sh create 256 300000   # Large dataset
./scripts/run_test.sh create 192 200000   # Custom size

# Run pipeline on existing dataset
./scripts/run_test.sh pipeline 128 100000
./scripts/run_test.sh pipeline 256 300000

# Create and run pipeline in one command
./scripts/run_test.sh all 128 100000
./scripts/run_test.sh all 256 300000
```

## Actions

| Action     | Description                                    | Time Estimate        |
|------------|------------------------------------------------|----------------------|
| `create`   | Generate test dataset only                     | 30 sec - 3 min       |
| `pipeline` | Run recovar pipeline on existing dataset       | 30 min - 6 hours     |
| `all`      | Create dataset and run pipeline                | 31 min - 6 hours     |

## Dataset Presets

### Small Dataset (128x100k)
- **Image size:** 128x128
- **Number of images:** 100,000
- **Creation time:** ~30 seconds
- **Pipeline time:** ~30 minutes
- **Use case:** Quick testing, debugging, profiling

### Large Dataset (256x300k)
- **Image size:** 256x256
- **Number of images:** 300,000
- **Creation time:** ~3 minutes
- **Pipeline time:** ~6 hours
- **Use case:** Realistic performance testing, production simulation

## Dataset Locations

Datasets are created in `/workspace/data-<size>-<nimages>/`:
- Small: `/workspace/data-128-100000/`
- Large: `/workspace/data-256-300000/`
- Custom: `/workspace/data-<your_size>-<your_nimages>/`

## Examples

### Quick End-to-End Test
```bash
./scripts/run_test_presets.sh small-all
```

### Profile Small Dataset Pipeline
```bash
# Create dataset once
./scripts/run_test.sh create 128 100000

# Profile with nsys
nsys profile -o profile_128_100k \
  ./scripts/run_test.sh pipeline 128 100000
```

### Batch Processing
```bash
# Create multiple datasets
for size in 128 192 256; do
  ./scripts/run_test.sh create $size 100000
done

# Run pipelines
for size in 128 192 256; do
  ./scripts/run_test.sh pipeline $size 100000
done
```

### Custom Dataset Size
```bash
# Create intermediate size dataset
./scripts/run_test.sh create 192 150000

# Run pipeline
./scripts/run_test.sh pipeline 192 150000
```

## Profiling with Nsight Systems

```bash
# Profile small dataset
./scripts/run_test.sh create 128 100000
nsys profile -o recovar_profile_128_100k \
  --trace=cuda,nvtx \
  ./scripts/run_test.sh pipeline 128 100000

# Profile large dataset  
./scripts/run_test.sh create 256 300000
nsys profile -o recovar_profile_256_300k \
  --trace=cuda,nvtx \
  ./scripts/run_test.sh pipeline 256 300000
```

## Environment Variables

You can customize the base directory by editing `run_test.sh`:

```bash
# Default: /workspace
BASE_DIR="/workspace"

# To change, edit line 14 in run_test.sh:
BASE_DIR="/custom/path"
```

## Troubleshooting

### Dataset not found error
```bash
Error: Dataset not found at /workspace/data-128-100000/test_dataset
```
**Solution:** Create the dataset first:
```bash
./scripts/run_test.sh create 128 100000
```

### Permission denied
```bash
-bash: ./scripts/run_test.sh: Permission denied
```
**Solution:** Make script executable:
```bash
chmod +x scripts/run_test.sh scripts/run_test_presets.sh
```

### Out of memory
**Solution:** Use smaller dataset or adjust batch sizes in the code

## See Also

- `PERFORMANCE_ANALYSIS.md` - Detailed performance analysis and optimization guide
- Nsight Systems documentation: https://docs.nvidia.com/nsight-systems/

