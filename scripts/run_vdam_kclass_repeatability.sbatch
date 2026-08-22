#!/usr/bin/env bash
#SBATCH --job-name=vdam-kclass-repeat
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-kclass-repeat-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
K="${K:-4}"
FIXTURE_DIR="${FIXTURE_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex}"

cd "${REPO_ROOT}"
test -z "$(git status --porcelain)"
HEAD_SHORT="$(git rev-parse --short=8 HEAD)"
PANEL_ROOT="${OUTPUT_BASE}/vdam_k${K}_repeatability_${HEAD_SHORT}"
if [[ -e "${PANEL_ROOT}" ]]; then
    echo "refusing to reuse repeatability root: ${PANEL_ROOT}" >&2
    exit 2
fi
mkdir -p "${PANEL_ROOT}"
touch "${PANEL_ROOT}/SAFE_TO_DELETE"

for REPEAT_INDEX in 1 2; do
    .pixi/envs/default/bin/python scripts/run_vdam_kclass_parity_pair.py \
        --fixture-dir "${FIXTURE_DIR}" \
        --output-root "${PANEL_ROOT}/repeat-${REPEAT_INDEX}/pair" \
        --K "${K}" \
        --nr-iter 8 \
        --threads "${SLURM_CPUS_PER_TASK}"
done

.pixi/envs/default/bin/python scripts/audit_vdam_kclass_repeatability.py \
    --first-pair-root "${PANEL_ROOT}/repeat-1/pair" \
    --second-pair-root "${PANEL_ROOT}/repeat-2/pair" \
    --output-json "${PANEL_ROOT}/repeatability_report.json" \
    --output-shells-npz "${PANEL_ROOT}/repeatability_shellwise_fsc.npz"
