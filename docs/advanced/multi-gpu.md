# Multi-GPU

RECOVAR can use multiple GPUs to speed up the pipeline.

## Usage

```bash
# Use all available GPUs
recovar pipeline particles.star -o output --mask mask.mrc --multi-gpu

# Use a specific number of GPUs
recovar pipeline particles.star -o output --mask mask.mrc --multi-gpu --n-gpus 4
```

## Memory management

Control GPU memory usage per device:

```bash
# Limit to 8 GB per GPU
recovar pipeline particles.star -o output --mask mask.mrc --gpu-gb 8
```

This is useful on shared clusters or when GPUs have limited memory.

## Environment variables

JAX respects standard CUDA environment variables:

```bash
# Use specific GPUs (e.g., GPUs 0 and 2)
CUDA_VISIBLE_DEVICES=0,2 recovar pipeline particles.star -o output --mask mask.mrc --multi-gpu

# Disable GPU memory preallocation (useful on shared machines)
XLA_PYTHON_CLIENT_PREALLOCATE=false recovar pipeline ...
```
