# RECOVAR Example Scripts

SLURM batch scripts for running RECOVAR on three benchmark datasets from the paper. Each script runs the full pipeline followed by analysis at two latent dimensions (zdim=10, 20).

## Datasets

| Script | EMPIAR | Sample | Particles | Runtime |
|--------|--------|--------|-----------|---------|
| `run_10076.slurm` | [10076](https://www.ebi.ac.uk/empiar/EMPIAR-10076/) | Bacterial 50S ribosome | 131,899 | ~2-4 hours |
| `run_10180.slurm` | [10180](https://www.ebi.ac.uk/empiar/EMPIAR-10180/) | Pre-catalytic spliceosome | 327,490 | ~6-12 hours |
| `run_10028.slurm` | [10028](https://www.ebi.ac.uk/empiar/EMPIAR-10028/) | Plasmodium 80S ribosome | 105,247 | ~2-4 hours |

## Usage

Submit a job:

```bash
sbatch examples/run_10076.slurm
```

## Adapting for your system

Each script has variables at the top that you should modify:

```bash
PYTHON=/path/to/your/python          # Python with recovar installed
DATA=/path/to/particle/data          # Where your input files are
OUT=/path/to/output                  # Where to write results
```

You may also need to change the SLURM directives (`#SBATCH` lines) for your cluster's partition names, account, and GPU configuration.

## What each script does

1. Prints environment info (git commit, JAX backend, GPU)
2. Runs `recovar pipeline` with appropriate inputs
3. Runs `recovar analyze` at zdim=10 (20 k-means clusters, 2 trajectories)
4. Runs `recovar analyze` at zdim=20 (20 k-means clusters, 2 trajectories)

## Input data

- **10076 & 10028**: Use pre-downsampled `.mrcs` files with poses and CTF as `.pkl` files
- **10180**: Uses `.mrcs` with poses/CTF from `recovar/assets/` and a filtered particle index
- **10028**: Extracts poses from a cryoSPARC `.cs` file as a preprocessing step (the `.cs` file has `alignments3D/pose` but the image paths don't resolve on the current system)
