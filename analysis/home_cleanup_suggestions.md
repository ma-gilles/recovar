# Home directory cleanup suggestions

Summary of what could be **removed** or **moved** to free space in `/home/mg6942`.  
Disk: GPFS ~5TB total, ~3.1TB free (40% used). Sizes below are approximate.

---

## Safe to remove (names or purpose suggest disposable)

| Item | Size | Why |
|------|------|-----|
| **recovar_test_delete_this/** | **156 MB** | Name says "delete_this" – test pipeline output. |
| **recovar/todel.mrc** | 65 MB | Filename suggests temporary. |
| **recovar/core.1116610** | 159 MB | Core dump (crash file); safe to delete. |
| **ucsf-chimerax_1.9ubuntu24.04_amd64.deb** | **420 MB** | Installer; remove after ChimeraX is installed. |
| **Miniconda3-latest-Linux-x86_64.sh** | 74 MB | Installer; remove if conda is already set up. |
| **slurm-55614345.out** | 2.6 MB | Single Slurm log; remove if no longer needed. |

**Subtotal (quick wins): ~876 MB**

---

## Consider removing (test / scratch dirs)

| Item | Size | Notes |
|------|------|--------|
| **recovartest/** | 625 MB | Test project; delete if obsolete. |
| **recovartest2/** | 387 MB | Test project; delete if obsolete. |
| **recovar_test/** | **3.4 GB** | Large test tree; archive or delete if not needed. |
| **cauchy_test/** | 389 MB | Test dir; remove if no longer used. |
| **test/** | 90 MB | Generic test dir. |
| **slurmo/** | 71 MB | Many `.out`/`.err` Slurm logs; prune or remove. |

**Subtotal (if all removed): ~4.9 GB**

---

## Good candidates to move (not delete)

| Item | Size | Suggestion |
|------|------|------------|
| **heterogeneity_dev/recovar.sif** | **3.7 GB** | Singularity image. Move to project scratch or shared app location if your cluster has one; keep a symlink in the repo if needed. |
| **recovar/** (whole dir) | 5.3 GB | Data/volumes (many `.mrc`). Move to scratch or project storage if you have quota pressure; keep only symlinks or a small “current” set in home. |
| **PPCA-EM-Notes/** | 2.5 GB | Large `.npy` (e.g. `estimation_scenarios_l1_sweep_full_avgmodes.npy` ~451 MB). Move to project/scratch if acceptable for your workflow. |

---

## Cache / dotdirs (optional cleanup)

| Item | Size | Notes |
|------|------|--------|
| **~/.cache** | ~774 MB | Pip, numba, fontconfig, mozilla, etc. Safe to clear; apps will repopulate as needed. |
| **~/.local** | ~8.6 GB | Includes pip/conda envs, icons, etc. Prune old envs or unused apps only if you know what you’re doing. |
| **~/.cursor-server** | ~3.3 GB | Cursor IDE server. Leave unless you’re sure you can reinstall. |

**Suggested:** run `rm -rf ~/.cache/*` to free ~774 MB with low risk.

---

## Suggested order of operations

1. **Delete obvious junk** (no move):  
   `recovar_test_delete_this/`, `recovar/core.1116610`, `recovar/todel.mrc`, ChimeraX `.deb`, Miniconda `.sh`, `slurm-55614345.out` → **~876 MB**.
2. **Clear cache:**  
   `rm -rf ~/.cache/*` → **~774 MB**.
3. **Remove test dirs** if you don’t need them:  
   `recovartest/`, `recovartest2/`, `recovar_test/`, `cauchy_test/`, `test/`, and/or `slurmo/` → up to **~4.9 GB**.
4. **Move large assets** if you have scratch/project space:  
   `heterogeneity_dev/recovar.sif`, `recovar/`, or large `PPCA-EM-Notes` files.

If you tell me which of these you want to do (e.g. “only safe deletes” or “also remove recovar_test and recovartest”), I can give exact commands for your home dir.
