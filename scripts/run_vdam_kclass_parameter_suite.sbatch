#!/usr/bin/env bash
#SBATCH --job-name=vdam-kclass-params
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%4
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-kclass-params-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
K="${K:-4}"
FIXTURE_DIR="${FIXTURE_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex}"

cd "${REPO_ROOT}"
test -z "$(git status --porcelain)"
HEAD_SHORT="$(git rev-parse --short=8 HEAD)"

case "${SLURM_ARRAY_TASK_ID}" in
    0) CASE_NAME="seed7"; EXTRA_ARGS=(--random-seed 7) ;;
    1) CASE_NAME="hp0"; EXTRA_ARGS=(--healpix-order 0) ;;
    2) CASE_NAME="tau2_2"; EXTRA_ARGS=(--tau2-fudge 2) ;;
    3) CASE_NAME="tau2_8"; EXTRA_ARGS=(--tau2-fudge 8) ;;
    4) CASE_NAME="offset_r4_s1"; EXTRA_ARGS=(--offset-range 4 --offset-step 1) ;;
    5) CASE_NAME="diam140"; EXTRA_ARGS=(--particle-diameter 140) ;;
    6) CASE_NAME="hp2"; EXTRA_ARGS=(--healpix-order 2) ;;
    7) CASE_NAME="os2"; EXTRA_ARGS=(--oversampling 2) ;;
    8) CASE_NAME="os0"; EXTRA_ARGS=(--oversampling 0) ;;
    *) echo "unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${OUTPUT_BASE}/vdam_k${K}_${CASE_NAME}_${HEAD_SHORT}/pair"
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "refusing to reuse output root: ${OUTPUT_ROOT}" >&2
    exit 2
fi

exec .pixi/envs/default/bin/python scripts/run_vdam_kclass_parity_pair.py \
    --fixture-dir "${FIXTURE_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --K "${K}" \
    --nr-iter 8 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    "${EXTRA_ARGS[@]}"
