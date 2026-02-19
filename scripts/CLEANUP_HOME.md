# Clean up data stored in home directory

The following paths under your repo (home) were used before we switched to scratch. You can remove them to free quota.

## Locations and sizes (approx.)

| Location | Size | Contents |
|----------|------|----------|
| **`.pixi/`** (repo root) | **~6.5 GB** | Pixi environments/cache (now duplicated on scratch) |
| **`scripts/output/`** | **~706 MB** | Slurm logs (slurm-*.out, slurm-*.err), some CSV |
| **`scripts/job_scripts/`** | ~1.5 MB | Generated batch scripts (many old runs) |

**Total in home from this workflow: ~7.2 GB**

## How to clean

### Option A: Remove everything (frees ~7.2 GB)

```bash
cd /home/mg6942/heterogeneity_dev

# Remove pixi env in home (you use the primed env on scratch now)
rm -rf .pixi

# Remove old Slurm logs and job scripts
rm -rf scripts/output/*
rm -rf scripts/job_scripts/*
```

### Option B: Move logs to scratch first, then remove

```bash
cd /home/mg6942/heterogeneity_dev
WORK=/scratch/gpfs/AMITS/mg6942/recovar_profiling

# Move existing output and job_scripts to scratch (for archive)
mkdir -p "$WORK/output_archive" "$WORK/job_scripts_archive"
mv scripts/output/* "$WORK/output_archive/" 2>/dev/null || true
mv scripts/job_scripts/* "$WORK/job_scripts_archive/" 2>/dev/null || true

# Remove .pixi in home
rm -rf .pixi
```

### Option C: Remove only the largest (.pixi) to free 6.5 GB

```bash
cd /home/mg6942/heterogeneity_dev
rm -rf .pixi
```

## After cleanup

- New runs use **RECOVAR_WORK_BASE** (scratch), so job scripts and Slurm output go to:
  - `/scratch/gpfs/AMITS/mg6942/recovar_profiling/job_scripts`
  - `/scratch/gpfs/AMITS/mg6942/recovar_profiling/output`
- Pixi uses the primed env on scratch; no need for `.pixi` in the repo.
