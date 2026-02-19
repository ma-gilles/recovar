# Performance Analysis Results

## Summary Files
- `performance_summary.md` - Detailed performance breakdown with IO analysis
- `nvtx_summary.txt` - Full NVTX range statistics
- `gpu_kernel_summary.txt` - GPU kernel performance
- `cuda_api_summary.txt` - CUDA API statistics  
- `memory_transfer_summary.txt` - Memory transfer details

## Key Findings

### IO Bottlenecks
- **16.2% of total time** spent on data transfers:
  - `data_io:collate_to_jax` (8.1%) - 218,914 instances, 42ms avg
  - `data_io:numpy_to_jax_transfer` (8.1%) - 218,914 instances, 42ms avg
- **Memory transfers**: 48.5% Host-to-Device, 41.5% Device-to-Device
- **7.3M Host-to-Device transfers** - suggests many small operations

### Top Compute Bottlenecks
1. `principal_components:estimate_principal_components` - 12.0%
2. `compute_regularized_covariance_columns_in_batch` - 7.6%
3. `compute_both_H_B` - 7.4%
4. `compute_H_B:accumulate_H_B` - 5.6% (592k instances - good parallelization candidate)

## Opening the Profile in GUI

### Why isn't the cluster default nsys (e.g. 2025.2) used?
Profiling runs **inside the container**. The container uses whatever `nsys` is on its PATH (from the image or from the bind-mounted `/opt/nvidia/nsight-systems`). That was 2025.3.2, so the `.nsys-rep` files were created in that format. The cluster default (`nsys` / `nsys-ui` when you log in) is the node’s install (e.g. 2025.2) and is **not** what runs inside the container.

To create profiles that open with the cluster default GUI, submit with the cluster’s `nsys` path:
```bash
# Find cluster nsys (e.g. 2025.2), then submit:
export NSYS_BIN=/opt/nvidia/nsight-systems/2025.2/target-linux-x64/nsys   # adjust path
./crun_recovar_workload_della.sh profile-1gpu-256
```
The script passes `NSYS_BIN` into the container and puts that directory on PATH so the job uses that version.

### Current Issue
Existing profiles were created with Nsight Systems 2025.3.2 (container’s nsys), so opening with cluster default (e.g. 2025.2) can show a version mismatch.

### Solutions

#### Option 1: Use the version that created the report (e.g. 2025.3.2)
```bash
/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/nsys-ui \
  /tigress/CRYOEM/singerlab/mg6942/recovar_profiling/data/data-256-300000/test_dataset/recovar_1gpu_profile_4355455.nsys-rep
```

#### Option 2: Download and Open Locally
1. Download the profile file to a local machine
2. Install Nsight Systems 2025.3.2 or newer on that machine
3. Open with: `nsys-ui <downloaded_file>.nsys-rep`

#### Option 3: Rerun with Compatibility Flags
If the GUI still doesn't work, we can rerun with explicit format flags. However, the current profile is valid (stats work fine), so this is likely a GUI bug.

### Alternative: Use Command-Line Analysis
All statistics are available via command-line:
```bash
# NVTX ranges
nsys stats --report nvtx_sum <profile>.nsys-rep

# GPU kernels
nsys stats --report gpukernsum <profile>.nsys-rep

# Memory transfers
nsys stats --report cuda_gpu_mem_time_sum <profile>.nsys-rep
```

## Profile Location
```
/tigress/CRYOEM/singerlab/mg6942/recovar_profiling/data/data-256-300000/test_dataset/
├── recovar_1gpu_profile_4355455.nsys-rep (9.8GB)
├── recovar_1gpu_profile_4355455.sqlite
├── recovar_2gpu_profile_4355456.nsys-rep (10GB)
└── recovar_4gpu_profile_4355457.nsys-rep (11GB)
```

### Profiles created with cluster default nsys (2025.5.2)
These should open with the cluster default `nsys-ui` when jobs complete.
- **Scratch**: job **4685319** → `.../data-256-300000/test_dataset/recovar_1gpu_profile_4685319.nsys-rep`
- **Tigress**: job **4685320** → `.../data-256-300000/test_dataset/recovar_1gpu_profile_4685320.nsys-rep`

**Logs (on scratch):**  
`/scratch/gpfs/AMITS/mg6942/recovar_profiling/output/slurm-4685319.{out,err}` and `slurm-4685320.{out,err}`
