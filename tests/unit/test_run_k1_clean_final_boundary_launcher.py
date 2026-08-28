from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_k1_clean_final_boundary.sbatch"
)


def test_clean_final_boundary_launcher_seals_inputs_and_final_policy() -> None:
    source = LAUNCHER.read_text()

    for name in (
        "REPO",
        "CASE_ROOT",
        "GRID_SIZE",
        "HEALPIX_ORDER",
        "REFINEMENT_SEED",
        "CUDA_LIB",
        "BIND_BUILD_DIR",
        "RELION_SRC_DIR",
        "EXPECTED_REPO_HEAD",
        "EXPECTED_SOURCE_DIFF_SHA256",
        "EXPECTED_CUDA_LIB_SHA256",
        "EXPECTED_INPUTS_SHA256",
    ):
        assert f': "${{{name}:?' in source

    assert "--max_iter 999" in source
    assert "--apply-initial-lowpass" in source
    assert "RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR" in source
    assert "unset RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER RECOVAR_FINAL_ALL_DATA_GRID_CORRECT" in source
    assert 'results["git_dirty_count"]' in source
    assert 'results["final_all_data_ran"]' in source
