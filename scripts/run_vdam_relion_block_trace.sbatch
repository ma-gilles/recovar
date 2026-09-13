#!/usr/bin/env bash
#SBATCH --job-name=vdam-relion-block-trace
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-relion-block-trace-%j.out

set -euo pipefail

: "${TRACE_ROOT:?set an empty disposable trace root containing SAFE_TO_DELETE}"
: "${RELION_SOURCE:?set the isolated RELION block-trace source worktree}"
: "${RELION_BINARY:?set the qualified block-trace relion_refine executable}"
: "${EXPECTED_RELION_HEAD:?pin the block-trace source commit}"
: "${EXPECTED_RELION_BINARY_SHA256:?pin the trace executable hash}"
: "${EXPECTED_REPO_HEAD:?pin the RECOVAR capture/validator commit}"
: "${EXPECTED_PHYSICAL_GPU_UUID:?pin the exact native-reference physical GPU}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
FIXTURE_DIR=${FIXTURE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf47/repeat-01/vdam-gf47/data}
REFERENCE_DIR=${REFERENCE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf47/repeat-01/vdam-gf47/relion}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
RELION_DIR=${TRACE_ROOT}/relion
ANALYSIS_DIR=${TRACE_ROOT}/analysis
PROVENANCE_DIR=${TRACE_ROOT}/provenance
WORKER_TRACE_FILE=${TRACE_ROOT}/worker_trace_it001.tsv
BLOCK_TRACE_FILE=${TRACE_ROOT}/block_trace_it001.bin

test -f "${TRACE_ROOT}/SAFE_TO_DELETE"
test -x "${RELION_BINARY}" -a -x "${PIXI_PY}"
test -s "${FIXTURE_DIR}/particles.star"
test -s "${FIXTURE_DIR}/fixture_materialization.json"
test -s "${REFERENCE_DIR}/run_it001_class001.mrc"
test "$(git -C "${RELION_SOURCE}" rev-parse HEAD)" = "${EXPECTED_RELION_HEAD}"
test -z "$(git -C "${RELION_SOURCE}" status --porcelain=v1 --untracked-files=no)"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_BINARY_SHA256}"
test "$(sha256sum "${FIXTURE_DIR}/particles.star" | awk '{print $1}')" = \
    64a2cfb5a0d6827944716b0de7e4a8ea13faccead2206ad0a4b422f98dd44897
test "$(sha256sum "${FIXTURE_DIR}/fixture_materialization.json" | awk '{print $1}')" = \
    3c8280cdf7705028f7423e41dedd53b3f9894dd750d5fba44eea0e243a29ef5a
test -z "$(find "${RELION_DIR}" "${ANALYSIS_DIR}" "${PROVENANCE_DIR}" \
    -mindepth 1 -print -quit 2>/dev/null || true)"
mkdir -p "${RELION_DIR}" "${ANALYSIS_DIR}" "${PROVENANCE_DIR}"

source /etc/profile.d/modules.sh
set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in
        RELION_DUMP_*|RELION_P1_*|RELION_DISPATCH_LOG|RELION_FORCE_*)
            unset "${variable_name}"
            ;;
    esac
done < <(env)
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export RELION_VDAM_WORKER_LOG=${WORKER_TRACE_FILE}
export RELION_VDAM_WORKER_TRACE_ITER=1
export RELION_VDAM_BLOCK_TRACE=${BLOCK_TRACE_FILE}
export RELION_VDAM_BLOCK_TRACE_ITER=1
export RELION_VDAM_BLOCK_TRACE_CAPACITY=1000000

physical_gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
printf '%s\n' "${physical_gpu_uuid}" > "${PROVENANCE_DIR}/physical_gpu_uuid.txt"
if [[ "${physical_gpu_uuid}" != "${EXPECTED_PHYSICAL_GPU_UUID}" ]]; then
    echo "Expected physical GPU ${EXPECTED_PHYSICAL_GPU_UUID}, got ${physical_gpu_uuid}" >&2
    exit 3
fi
nvidia-smi -q > "${PROVENANCE_DIR}/nvidia_smi.txt"

command=(
    "${RELION_BINARY}"
    --o "${RELION_DIR}/run"
    --iter 200 --grad --denovo_3dref --grad_write_iter 1
    --i "${FIXTURE_DIR}/particles.star" --ctf --K 1 --sym C1
    --flatten_solvent --zero_mask --dont_combine_weights_via_disc
    --pool 3 --pad 1 --particle_diameter 200 --oversampling 1
    --healpix_order 1 --offset_range 6 --offset_step 2 --auto_sampling
    --tau2_fudge 4 --random_seed 29 --j 8 --gpu 0
)
printf '%q ' "${command[@]}" > "${PROVENANCE_DIR}/command.sh"
printf '\n' >> "${PROVENANCE_DIR}/command.sh"
start_epoch=$(date +%s)
(
    cd "${FIXTURE_DIR}"
    "${command[@]}"
) > "${RELION_DIR}/runner.stdout" 2> "${RELION_DIR}/runner.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${WORKER_TRACE_FILE}"
test -s "${BLOCK_TRACE_FILE}"
test -s "${RELION_DIR}/run_it001_class001.mrc"
test -s "${RELION_DIR}/run_it200_class001.mrc"
"${PIXI_PY}" -m scripts.build_vdam_worker_schedule \
    --trace "${WORKER_TRACE_FILE}" --iteration 1 --n-particles 200 \
    --dataset-particles 3000 \
    --n-threads 8 --pool-size 24 \
    --output-npz "${ANALYSIS_DIR}/worker_schedule_it001.npz" \
    --output-json "${ANALYSIS_DIR}/worker_schedule_it001.json"
"${PIXI_PY}" -m scripts.build_vdam_block_chronology \
    --capture "${BLOCK_TRACE_FILE}" --iteration 1 --n-particles 200 \
    --n-threads 8 --n-classes 1 --sm-count 132 \
    --output-npz "${ANALYSIS_DIR}/block_chronology_it001.npz" \
    --output-json "${ANALYSIS_DIR}/block_chronology_it001.json"

set +e
"${PIXI_PY}" -m scripts.audit_vdam_kclass_trajectory \
    --candidate-dir "${RELION_DIR}" --reference-dir "${REFERENCE_DIR}" \
    --K 1 --checkpoint 1 --minimum-fsc-auc 0.999 \
    --minimum-assignment-accuracy 0.999 \
    --output-json "${ANALYSIS_DIR}/trace_map_it001.json" \
    --output-shells-npz "${ANALYSIS_DIR}/trace_map_it001_shells.npz"
map_status=$?
"${PIXI_PY}" -m scripts.audit_vdam_particle_state_trajectory \
    --recovar-dir "${RELION_DIR}" --relion-dir "${REFERENCE_DIR}" \
    --iterations 1 --output "${ANALYSIS_DIR}/trace_particles_it001.json"
particle_status=$?
set -e

"${PIXI_PY}" - "${PROVENANCE_DIR}" "${wall_s}" "${map_status}" "${particle_status}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {
    "schema": "recovar.vdam_relion_block_trace_job.v1",
    "wall_s": int(sys.argv[2]),
    "map_exit_status": int(sys.argv[3]),
    "particle_exit_status": int(sys.argv[4]),
}
report["result"] = "pass" if not report["map_exit_status"] and not report["particle_exit_status"] else "fail"
(root / "trace_job.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
sha256sum "${RELION_BINARY}" "${WORKER_TRACE_FILE}" "${BLOCK_TRACE_FILE}" \
    "${ANALYSIS_DIR}"/* > "${PROVENANCE_DIR}/results.sha256"
(( map_status == 0 && particle_status == 0 ))
