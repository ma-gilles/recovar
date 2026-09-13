from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "run_vdam_repeat_distribution_summary.sbatch"
)


def test_distribution_summary_is_restart_safe_and_preserves_failed_evidence():
    text = RUNNER.read_text()
    required = (
        "${EXPECTED_ANALYZER_HEAD:?pin the analysis source head}",
        "${EXPECTED_SCIENCE_HEAD:?pin the trajectory source head}",
        'status --porcelain=v1 --untracked-files=no',
        'test -s "${repeat_root}/trajectory_audit.json"',
        'test -s "${repeat_root}/relion/relion.timing.json"',
        'test -s "${repeat_root}/recovar/recovar.timing.json"',
        'VDAM_NATIVE_ONLY_PANEL_ROOT=${VDAM_NATIVE_ONLY_PANEL_ROOT:-}',
        'test -f "${native_root}/SCIENCE_COMPLETED"',
        'additional_native_args+=(--additional-native-root "${native_root}")',
        "scripts.audit_vdam_repeat_panel",
        "scripts.audit_vdam_repeat_state_panel",
        '"map_native_radius_distribution_result"',
        '"schedule_distribution_result"',
        '"${REPO_ROOT}/scripts/audit_vdam_candidate_state_envelope.py"',
        '"${REPO_ROOT}/scripts/audit_vdam_sampling_trajectory.py"',
        "set +e",
        'touch "${ANALYSIS_ROOT}/ANALYSIS_FAILED"',
        "sha256sum",
        "#SBATCH --partition=cpu",
    )
    missing = [token for token in required if token not in text]
    assert not missing, f"repeat distribution summary lost contract tokens: {missing}"
