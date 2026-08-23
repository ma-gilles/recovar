#!/usr/bin/env bash
#SBATCH --job-name=k1-c22-rec-bpm
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_all_20260810T1600ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_all_20260810T1600ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:45:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_DIFF_SHA256:?submission must pin worktree diff hash}"

ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_all_20260810T1600ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_rotation_mass_all_20260810T1600ET
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
LAUNCHER=${REPO}/scripts/run_k1_case22_recovar_membership_all.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = fa60832431c638e969a0e08ef9de5b83191eca51
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = edaad0aab3d96b09ba111def1b2a3bb1c89d1179221187df267755c575ac9ffc
test -z "$(find "${ROOT}/membership" "${ROOT}/output" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR=${ROOT}/membership
# The replay starts from RELION iteration 1; the physical iteration-2 update is
# labelled local iteration 1 by run_multi_iter_parity.
export RECOVAR_BPREF_MEMBERSHIP_DUMP_ITERATION=1
export RECOVAR_BPREF_MEMBERSHIP_DUMP_HALF=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
unset RECOVAR_K1_COARSE_GAUSSIAN_FFI RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
unset RECOVAR_K1_RELION_F32_COARSE_SUPPORT

cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert pathlib.Path(jax.__file__).resolve().is_relative_to(root/'.pixi/envs/default'); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

KEEP_STACK_INDICES=$("${PYTHON}" - "${RELION}/run_it001_data.star" <<'PY'
import sys
import starfile

particles = starfile.read(sys.argv[1])["particles"]
indices = [
    int(str(image_name).split("@", 1)[0]) - 1
    for image_name, subset in zip(
        particles["rlnImageName"], particles["rlnRandomSubset"], strict=True
    )
    if int(subset) == 1
]
assert len(indices) == 1490 and len(set(indices)) == len(indices)
print(",".join(map(str, indices)))
PY
)

COMMAND=(
  "${PYTHON}" -m scripts.run_multi_iter_parity
  --relion_dir "${RELION}" --data_star "${DATA}/particles.star"
  --iter 1 --max_iter 1 --continuous-relion-noise-state --skip_final_iteration
  --image_batch_size 187 --rotation_block_size 8192 --image-fourier-backend relion_cuda
  --keep_stack_indices "${KEEP_STACK_INDICES}"
  --output_dir "${ROOT}/output" --save_intermediates_dir "${ROOT}/output/intermediates"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION}/run_it001_optimiser.star" > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

start=$(date +%s)
"${COMMAND[@]}" > "${ROOT}/output/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

"${PYTHON}" - "${ROOT}/membership" "${RELION}/run_it001_data.star" <<'PY'
import sys
from pathlib import Path

import numpy as np
import starfile

directory = Path(sys.argv[1])
paths = sorted(directory.glob("bpref_membership_*.npz"))
assert paths
seen = set()
for path in paths:
    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["schema"].item()) == "recovar-bpref-rotation-mass-v2"
        assert int(payload["iteration"]) == 1
        assert int(payload["half"]) == 1
        assert int(payload["current_size"]) == 60
        stacks = np.asarray(payload["stack_indices_1based"], dtype=np.int64)
        counts = np.asarray(payload["actual_counts"], dtype=np.int64)
        candidate_count = np.asarray(payload["candidate_translation_count"], dtype=np.int64)
        reconstruction = np.asarray(payload["reconstruction_rotation_mass"])
        significant_count = np.asarray(payload["significant_translation_count"], dtype=np.int64)
        assert stacks.size == counts.size == candidate_count.shape[0] == reconstruction.shape[0]
        assert candidate_count.shape == reconstruction.shape == significant_count.shape
        for row, stack in enumerate(stacks.tolist()):
            assert stack not in seen
            seen.add(stack)
            assert 0 < counts[row] <= candidate_count.shape[1]
            assert not np.any(candidate_count[row, counts[row]:])
            assert not np.any(reconstruction[row, counts[row]:])
            assert not np.any(significant_count[row, counts[row]:])
particles = starfile.read(sys.argv[2])["particles"]
expected = {
    int(str(image_name).split("@", 1)[0])
    for image_name, subset in zip(
        particles["rlnImageName"], particles["rlnRandomSubset"], strict=True
    )
    if int(subset) == 1
}
assert len(seen) == 1490 and seen == expected
print({"shards": len(paths), "particles": len(seen), "local_iteration": 1, "physical_iteration": 2})
PY
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/output/run.log"
printf '{"job_id":"%s","wall_s":%d,"physical_iteration":2,"half":1,"particles":1490,"exact_bpref_operands":true,"all_particle_membership":true}\n' "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${ROOT}/membership" -type f -name '*.npz' -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/membership_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
