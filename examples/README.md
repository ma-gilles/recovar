# RECOVAR Example Scripts

SLURM batch scripts for running RECOVAR on two benchmark datasets from the paper.

## Datasets

| Script | EMPIAR | Sample | Particles | Workflow |
|--------|--------|--------|-----------|----------|
| `run_10076.slurm` | [10076](https://www.ebi.ac.uk/empiar/EMPIAR-10076/) | Bacterial 50S ribosome | 131,899 | Pipeline → analyze → extract subset → re-run |
| `run_10180.slurm` | [10180](https://www.ebi.ac.uk/empiar/EMPIAR-10180/) | Pre-catalytic spliceosome | 327,490 | Pipeline → analyze → density → trajectory |

## Usage

```bash
sbatch examples/run_10076.slurm
sbatch examples/run_10180.slurm
```

## What each script does

### EMPIAR-10076

1. Run `recovar pipeline` (no contrast correction)
2. Run `recovar analyze` (zdim=20, 20 clusters, 2 trajectories)
3. Extract a clean particle subset by excluding outlier cluster 0
4. Re-run `recovar pipeline` on the cleaned subset
5. Re-run `recovar analyze` on the subset

### EMPIAR-10180

1. Run `recovar pipeline` with filtered particle index and contrast correction
2. Run `recovar analyze` (zdim=4, 20 clusters, 2 trajectories)
3. Estimate conformational density (`estimate_conformational_density`)
4. Compute a trajectory through the density landscape (`compute_trajectory`)

## Adapting for your system

Each script has variables at the top:

```bash
PYTHON=/path/to/your/python          # Python with recovar installed
DATA=/path/to/particle/data          # Where your input files are
OUT=/path/to/output                  # Where to write results
```

Change the `#SBATCH` directives for your cluster's partition names, account, and GPU configuration.
