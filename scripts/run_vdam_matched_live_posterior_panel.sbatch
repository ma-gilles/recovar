#!/usr/bin/env bash
#SBATCH --job-name=vdam-live-post
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:30:00

set -euo pipefail

: "${VDAM_PANEL_ROOT:?set a fresh SAFE_TO_DELETE panel root}"
: "${VDAM_REPO_ROOT:?set the immutable science worktree}"
: "${VDAM_EXPECTED_HEAD:?pin the immutable science commit}"
: "${VDAM_DATA_DIR:?set the frozen materialized VDAM fixture directory}"
: "${VDAM_RELION_BINARY:?set the isolated instrumented RELION binary}"
: "${VDAM_CUDA_LIBRARY:?set the qualified CUDA library}"
: "${VDAM_EXPECTED_CUDA_SHA256:?pin the qualified CUDA library}"
: "${VDAM_EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding}"

REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
NATIVE_ATOMIC_SOFTMASK=${VDAM_CANDIDATE_NATIVE_ATOMIC_SOFTMASK:-0}
test "${REPEAT_COUNT}" -ge 2
case "${NATIVE_ATOMIC_SOFTMASK}" in
  0|1) ;;
  *) echo "VDAM_CANDIDATE_NATIVE_ATOMIC_SOFTMASK must be 0 or 1" >&2; exit 2 ;;
esac
test -f "${VDAM_PANEL_ROOT}/SAFE_TO_DELETE"
test -z "$(find "${VDAM_PANEL_ROOT}" -mindepth 1 -maxdepth 1 \
  ! -name SAFE_TO_DELETE \
  ! -name "slurm-${SLURM_JOB_ID}.out" \
  ! -name "slurm-${SLURM_JOB_ID}.err" \
  -print -quit)"
cd "${VDAM_REPO_ROOT}"
test "$(git rev-parse HEAD)" = "${VDAM_EXPECTED_HEAD}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"
test -s "${VDAM_DATA_DIR}/particles.star"
test -x "${VDAM_RELION_BINARY}"
test -f "${VDAM_CUDA_LIBRARY}"
test "$(sha256sum "${VDAM_CUDA_LIBRARY}" | awk '{print $1}')" = \
  "${VDAM_EXPECTED_CUDA_SHA256}"

PIXI_PY=${VDAM_REPO_ROOT}/.pixi/envs/default/bin/python
RELION_BIND_BINARY=${VDAM_REPO_ROOT}/recovar/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
test -x "${PIXI_PY}"
test -s "${RELION_BIND_BINARY}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${VDAM_EXPECTED_RELION_BIND_SHA256}"

physical_gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
case "${physical_gpu_uuid}" in
  GPU-*) ;;
  *) echo "failed to resolve one physical GPU UUID: ${physical_gpu_uuid}" >&2; exit 2 ;;
esac

for repeat in $(seq 1 "${REPEAT_COUNT}"); do
  repeat_tag=$(printf '%02d' "${repeat}")
  output_root=${VDAM_PANEL_ROOT}/repeat-${repeat_tag}
  mkdir -p "${output_root}/native_panels"
  touch "${output_root}/SAFE_TO_DELETE"
  env \
    RELION_ACC_DUMP_BPREF_TOPOLOGY="${output_root}/provenance/bpref_topology.tsv" \
    RELION_ACC_DUMP_BPREF_PANEL_DIR="${output_root}/native_panels" \
    OUTPUT_ROOT="${output_root}" \
    DATA_DIR="${VDAM_DATA_DIR}" \
    EXPECTED_REPO_HEAD="${VDAM_EXPECTED_HEAD}" \
    EXPECTED_CUDA_SHA256="${VDAM_EXPECTED_CUDA_SHA256}" \
    EXPECTED_RELION_BIND_SHA256="${VDAM_EXPECTED_RELION_BIND_SHA256}" \
    VDAM_RELION_BINARY="${VDAM_RELION_BINARY}" \
    RECOVAR_CUDA_LIB_SOURCE="${VDAM_CUDA_LIBRARY}" \
    REPO_ROOT="${VDAM_REPO_ROOT}" \
    TARGET_ITERATION=1 \
    NR_ITER_SCHEDULE=200 \
    RELION_THREADS=8 \
    RANDOM_SEED=0 \
    TARGET_GPU_UUID="${physical_gpu_uuid}" \
    VDAM_NATIVE_ONLY=1 \
    bash scripts/run_vdam_fullschedule_mstep_boundary.sbatch \
      > "${output_root}/driver.stdout" \
      2> "${output_root}/driver.stderr"
  test -s "${output_root}/provenance/bpref_topology.tsv"
  test -s "${output_root}/relion/run_it001_data.star"
done

target_indices=$(
  "${PIXI_PY}" - "${VDAM_PANEL_ROOT}/repeat-01" <<'PY'
import sys
from pathlib import Path

from scripts.run_vdam_worker_private_host_replay import _load_native_panels

root = Path(sys.argv[1])
panels = _load_native_panels(
    root / "native_panels",
    root / "relion" / "run_it001_data.star",
    iteration=1,
)
print(",".join(str(identity) for identity in sorted(panels)))
PY
)
target_count=$(awk -F, '{print NF}' <<< "${target_indices}")
test "${target_count}" -eq 200

shared_jax_cache=${VDAM_PANEL_ROOT}/jax_cache
mkdir -p "${shared_jax_cache}"
for repeat in $(seq 1 "${REPEAT_COUNT}"); do
  repeat_tag=$(printf '%02d' "${repeat}")
  output_root=${VDAM_PANEL_ROOT}/candidate-repeat-${repeat_tag}
  posterior_root=${output_root}/posterior
  recovar_root=${output_root}/recovar
  runtime_root=${output_root}/runtime
  mkdir -p "${posterior_root}" "${recovar_root}" "${runtime_root}" "${output_root}/provenance"
  touch "${output_root}/SAFE_TO_DELETE"

  unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORMS JAX_PLATFORM_NAME
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export RECOVAR_EXPECTED_REPO_ROOT=${VDAM_REPO_ROOT}
  export RECOVAR_CUDA_LIB=${VDAM_CUDA_LIBRARY}
  export JAX_COMPILATION_CACHE_DIR=${shared_jax_cache}
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  export TMPDIR=${runtime_root}/tmp
  export PIXI_HOME=${runtime_root}/pixi_home
  export RATTLER_CACHE_DIR=${runtime_root}/rattler_cache
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_DIR=${posterior_root}
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_GLOBAL_INDICES=${target_indices}
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_ITERATION=1
  export RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_SCORES=1
  export RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION=${NATIVE_ATOMIC_SOFTMASK}
  mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

  start_epoch=$(date +%s)
  "${PIXI_PY}" -m scripts.run_ab_initio \
    --i "${VDAM_DATA_DIR}/particles.star" \
    --o "${recovar_root}/run" \
    --nr_iter 200 \
    --grad_write_iter 1 \
    --K 1 \
    --tau2_fudge 4 \
    --sym C1 \
    --do_run_C1 1 \
    --particle_diameter 200 \
    --random_seed 0 \
    --healpix_order 1 \
    --oversampling 1 \
    --offset_range 6 \
    --offset_step 2 \
    --padding_factor 1 \
    --image_batch_size 500 \
    --datadir "${VDAM_DATA_DIR}" \
    --gpu 0 \
    --require_custom_cuda \
    --diagnostic_stop_after_iteration 1 \
      > "${output_root}/runner.stdout" \
      2> "${output_root}/runner.stderr"
  wall_s=$(( $(date +%s) - start_epoch ))
  test -s "${recovar_root}/run_it001_class001.mrc"

  "${PIXI_PY}" - "${posterior_root}" "${target_indices}" <<'PY'
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = {int(value) for value in sys.argv[2].split(",")}
observed = set()
for path in sorted(root.glob("local_fused_posterior_it001_image_*.npz")):
    with np.load(path, allow_pickle=False) as payload:
        required = {"selected_global_image_indices", "posterior", "pass2_scores_total"}
        missing = required.difference(payload.files)
        if missing:
            raise SystemExit(f"posterior dump lacks fields {sorted(missing)}: {path}")
        identity = np.asarray(payload["selected_global_image_indices"], dtype=np.int64)
        if identity.shape != (1,):
            raise SystemExit(f"posterior dump identity topology changed: {path}")
        observed.add(int(identity[0]))
if observed != expected:
    raise SystemExit(
        f"posterior identity closure failed: missing={sorted(expected - observed)} "
        f"extra={sorted(observed - expected)}"
    )
PY
  printf '{"schema":"recovar.vdam_live_posterior_repeat.v1","repeat":%d,"science_head":"%s","physical_gpu_uuid":"%s","particle_count":%d,"wall_s":%d,"native_atomic_softmask":%d}\n' \
    "${repeat}" "${VDAM_EXPECTED_HEAD}" "${physical_gpu_uuid}" \
    "${target_count}" "${wall_s}" "${NATIVE_ATOMIC_SOFTMASK}" \
    > "${output_root}/provenance.json"
done

"${PIXI_PY}" - "${VDAM_PANEL_ROOT}/submission_provenance.json" \
  "${VDAM_EXPECTED_HEAD}" "${physical_gpu_uuid}" "${REPEAT_COUNT}" \
  "${VDAM_EXPECTED_CUDA_SHA256}" "${NATIVE_ATOMIC_SOFTMASK}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

output, head, gpu_uuid, repeats, cuda_sha, native_atomic_softmask = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "schema": "recovar.vdam_live_posterior_panel_submission.v1",
            "submitted_at": datetime.now().astimezone().isoformat(),
            "science_head": head,
            "physical_gpu_uuid": gpu_uuid,
            "execution_policy": (
                "one allocation, one physical H100, sequential native and live candidate arms"
            ),
            "repeat_count_per_engine": int(repeats),
            "cuda_library_sha256": cuda_sha,
            "candidate_native_atomic_softmask": native_atomic_softmask == "1",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

mkdir -p "${VDAM_PANEL_ROOT}/analysis"
"${PIXI_PY}" -m scripts.audit_vdam_live_posterior_repeat_panel \
  --panel-root "${VDAM_PANEL_ROOT}" \
  --repeat-count "${REPEAT_COUNT}" \
  --iteration 1 \
  --output "${VDAM_PANEL_ROOT}/analysis/live_posterior_repeat_panel.json" \
  > "${VDAM_PANEL_ROOT}/analysis/live_posterior_repeat_panel.stdout"

sha256sum \
  scripts/run_vdam_matched_live_posterior_panel.sbatch \
  scripts/audit_vdam_live_posterior_repeat_panel.py \
  "${VDAM_PANEL_ROOT}/submission_provenance.json" \
  "${VDAM_PANEL_ROOT}/analysis/live_posterior_repeat_panel.json" \
  > "${VDAM_PANEL_ROOT}/matched_live_posterior_evidence.sha256"
for repeat in $(seq 1 "${REPEAT_COUNT}"); do
  repeat_tag=$(printf '%02d' "${repeat}")
  sha256sum \
    "${VDAM_PANEL_ROOT}/repeat-${repeat_tag}/provenance/bpref_topology.tsv" \
    "${VDAM_PANEL_ROOT}/repeat-${repeat_tag}/relion/run_it001_data.star" \
    "${VDAM_PANEL_ROOT}/candidate-repeat-${repeat_tag}/provenance.json" \
    >> "${VDAM_PANEL_ROOT}/matched_live_posterior_evidence.sha256"
done
test "$(git rev-parse HEAD)" = "${VDAM_EXPECTED_HEAD}"
test -z "$(git status --porcelain=v1 --untracked-files=no)"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = \
  "${physical_gpu_uuid}"
touch "${VDAM_PANEL_ROOT}/MATCHED_LIVE_POSTERIOR_SUCCESS"
