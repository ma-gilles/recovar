from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_k1_case26_ordered_exact_it2.sbatch"


def _launcher_text() -> str:
    return LAUNCHER.read_text()


def test_gaussian_coarse_override_survives_environment_setup():
    text = _launcher_text()
    export = "export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1"
    assert export in text
    assert "unset RECOVAR_K1_COARSE_GAUSSIAN_FFI" not in text[text.index(export) :]


def test_direct_noise_only_enables_required_atomic_scale_path():
    text = _launcher_text()
    implication = """if [[ "${WAVG_ATOMIC_DIRECT_NOISE_ONLY}" = 1 ]]; then
    WAVG_ATOMIC_SCALE_AA=1
fi"""
    assert implication in text
    assert "export RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY=1" in text
    assert "export RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA=1" in text


def test_candidate_flags_fail_closed_for_state_swap_replay():
    text = _launcher_text()
    assert '[[ "${GAUSSIAN_COARSE}${FINE_ROTATION_EXECUTION_ORDER}${WAVG_ATOMIC_SCALE_AA}" != 000 ]]' in text
    assert "fresh K=1 candidate flags cannot be combined with state-swap replay" in text


def test_full_case_mode_runs_final_without_grid_correction():
    text = _launcher_text()
    assert "2|3|999" in text
    assert 'if [[ "${MAX_ITER}" != 999 ]]; then\n    COMMAND+=(--skip_final_iteration)\nfi' in text
    assert 'assert bool(np.asarray(result["final_all_data_ran"]).item())' in text
    assert 'assert not bool(np.asarray(result["final_all_data_grid_correct"]).item())' in text
    assert 'if [[ "${MAX_ITER}" = 999 ]]; then\n    FSC_SCOPE_ARGS=()\nfi' in text
    assert "unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER" in text
