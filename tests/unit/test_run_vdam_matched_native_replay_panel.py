from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_vdam_matched_native_replay_panel.sbatch"
)


def _source() -> str:
    return SCRIPT.read_text()


def test_matched_panel_keeps_all_arms_in_one_gpu_allocation() -> None:
    source = _source()
    assert "physical_gpu_uuid=$(nvidia-smi --query-gpu=uuid" in source
    assert 'TARGET_GPU_UUID="${physical_gpu_uuid}"' in source
    assert 'VDAM_TARGET_GPU_UUID="${physical_gpu_uuid}"' in source
    assert "for repeat in $(seq 1 \"${REPEAT_COUNT}\")" in source
    assert "for kind in private shared" in source
    assert "\nsbatch " not in source


def test_matched_panel_runs_native_before_private_and_shared_replay() -> None:
    source = _source()
    native = source.index("bash scripts/run_vdam_fullschedule_mstep_boundary.sbatch")
    replay = source.index("bash scripts/run_vdam_worker_private_host_replay.sbatch")
    audit = source.index("scripts.audit_vdam_native_bpref_repeat_panel")
    assert native < replay < audit
    assert "VDAM_NATIVE_ONLY=1" in source
    assert "VDAM_MERGE_CALLBACKS=1" in source
    assert 'VDAM_SHARED_ACCUMULATORS="${shared_accumulators}"' in source


def test_matched_panel_seals_source_binaries_and_both_reports() -> None:
    source = _source()
    for token in (
        "VDAM_EXPECTED_HEAD",
        "VDAM_EXPECTED_CUDA_SHA256",
        "VDAM_EXPECTED_RELION_BIND_SHA256",
        "VDAM_EXPECTED_EXACT_PTX_SHA256",
        "git status --porcelain=v1 --untracked-files=no",
        "recovar.vdam_replay_repeat_panel_submission.v2",
        "replay_repeat_distribution.json",
        "native_vs_replay_bpref_repeat.json",
        "matched_panel_evidence.sha256",
        "MATCHED_PANEL_SUCCESS",
    ):
        assert token in source
