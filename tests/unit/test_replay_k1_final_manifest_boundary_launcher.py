from __future__ import annotations

from pathlib import Path

LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "replay_k1_final_manifest_boundary.sbatch"
)


def test_launcher_requires_explicit_boundary_inputs() -> None:
    source = LAUNCHER.read_text()

    for name in (
        "REPO",
        "CASE_ROOT",
        "SOURCE_RUN",
        "INIT_RELION_ITERATION",
        "HEALPIX_ORDER",
        "REFINEMENT_SEED",
        "CUDA_LIB",
        "BIND_BUILD_DIR",
    ):
        assert f': "${{{name}:?' in source

    assert "REPO=/scratch/" not in source
    assert "SOURCE_RUN=/scratch/" not in source


def test_launcher_accepts_explicit_boundary_overrides() -> None:
    source = LAUNCHER.read_text()

    assert "SOURCE_MANIFESTS=${SOURCE_MANIFESTS_OVERRIDE:-${SOURCE_RUN}/output/intermediates}" in source
    assert "SOURCE_RESULTS=${SOURCE_RESULTS_OVERRIDE:-${SOURCE_RUN}/output/refinement_results.npz}" in source
    assert '--diagnostic-final-manifest-dir "${SOURCE_MANIFESTS}"' in source
    assert '--diagnostic-final-source-results "${SOURCE_RESULTS}"' in source


def test_launcher_isolates_cuda_rebuilds_from_pinned_input() -> None:
    source = LAUNCHER.read_text()

    assert "RUN_CUDA_LIB=${RUN_CUDA_DIR}/libcuda_backproject.so" in source
    assert 'install -m 0555 "${CUDA_LIB}" "${RUN_CUDA_LIB}"' in source
    assert "export RECOVAR_CUDA_LIB=${RUN_CUDA_LIB}" in source
    assert 'test "$(sha256sum "${CUDA_LIB}"' in source
    assert "cuda_after_run_${SLURM_JOB_ID}.sha256" in source


def test_launcher_gates_replay_maps_with_signed_fsc_auc() -> None:
    source = LAUNCHER.read_text()

    assert "final_map_replay_inertness.json" in source
    assert 'for name in ("final_half1.mrc", "final_half2.mrc", "final_merged.mrc")' in source
    assert 'row["fsc_auc_non_dc"] >= threshold' in source
    assert '"signed shellwise FSC and normalized non-DC FSC-AUC; no correlation"' in source
