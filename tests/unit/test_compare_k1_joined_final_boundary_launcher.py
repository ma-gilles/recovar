from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compare_k1_case10_joined_final_boundary.sbatch"
)


def test_joined_final_boundary_launcher_requires_case_specific_provenance() -> None:
    source = LAUNCHER.read_text()

    for name in (
        "REPO",
        "CASE_ROOT",
        "SOURCE_RECOVAR",
        "FINAL_RELION_ITERATION",
        "RECOVAR_CAPTURE_ROOT",
        "RELION_CAPTURE_ROOT",
        "RECOVAR_CAPTURE_JOB_ID",
        "RELION_CAPTURE_JOB_ID",
    ):
        assert f': "${{{name}:?' in source

    assert "recovar_k1_native_units_instrumentation_20260826" not in source
    assert "10_high_res_anisotropic_100k_g384_radial_noise3_bf0" not in source
    assert "mstep_it018_rank1_half0_c0_pre_reconstruct" not in source


def test_joined_final_boundary_launcher_checks_both_producers_and_iteration() -> None:
    source = LAUNCHER.read_text()

    assert 'SUCCESS_${RECOVAR_CAPTURE_JOB_ID}' in source
    assert 'SUCCESS_${RELION_CAPTURE_JOB_ID}' in source
    assert 'printf \'%03d\' "${FINAL_RELION_ITERATION}"' in source
    assert 'rg -q "^iter=${FINAL_RELION_ITERATION}$"' in source
    assert 'threshold = 0.999999' in source
