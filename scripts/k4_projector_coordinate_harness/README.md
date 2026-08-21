# K=4 particle-3591 projector coordinate harness

This diagnostic freezes the first divergent K=4 boundary at iteration 1,
particle 3591, selected class 2. It is intentionally smaller than an EM run:
one H100 evaluates eight exact float32 Euler matrices at the 840 current-size
Fourier pixels against one exact complex64 `PPref`.

The fail-closed outlier identity is fine candidate 18 (hidden 899570), rotation
row 4, translation 42, RELION pixel 242 (y=11, x=11), which maps to RECOVAR
window column 641.

The harness first records the float32 arrays produced by RECOVAR's
full-volume-to-compact staging kernel before `cudaMemcpy3D` and compares them
with RELION's directly staged `PPref` arrays. Only then does it compare current
RECOVAR expression order, exact RELION source order, explicit FMA orders, and
explicit noncontracted multiply/add. It also samples the adjacent lower and
upper 1/256 y-coordinate bins around the target's near-half-step texture
coordinate. Every variant writes all 8-by-840 coordinates and complex
projections; the validator uses exact and array-error metrics, never
correlation.

Prepare a disposable package after loading CUDA 12.6:

```bash
source /etc/profile.d/modules.sh
module purge
module load cudatoolkit/12.6
python scripts/k4_projector_coordinate_harness/prepare.py \
  --artifact-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p3591_cross_engine_identity6_globalwinner_h100_prepared_20260715_134127 \
  --operand-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p3591_relion_fine_operands_sm90_prepared_20260715_151500 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/REPLACE_WITH_DATED_RUN \
  --python /absolute/path/to/.pixi/envs/default/bin/python
```

The prepared root contains its exact `sbatch --parsable` command. Preparation
builds an SM90-only binary and performs no GPU work or Slurm submission.

Limitations: this is a frozen one-particle replay, not a production-register
tap or an EM quality test. A coordinate match classifies the first projector
boundary but does not by itself establish a map-quality result; maps remain
gated by FSC/FSC-AUC.
