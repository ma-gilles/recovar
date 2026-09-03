import json
from pathlib import Path

import pytest

from scripts import audit_em_k1_box800_local_padding_gate as audit

pytestmark = pytest.mark.unit


def _ledger(*, commit: str, output_dir: Path, total_time_s: float) -> dict:
    return {
        "adaptive_oversampling": 1,
        "auto_local_healpix_order": 3,
        "current_sizes": [62],
        "data_dir": "/data/empiar10202/smoke",
        "diagnostic_single_half": True,
        "git_commit": commit,
        "global_profile_rows": [],
        "healpix_order": 3,
        "image_shape": [800, 800],
        "initial_pose_source_path": "/data/seed_box800_single_half.npz",
        "initial_pose_source_requested": "none",
        "initial_pose_source_resolved": "diagnostic_npz",
        "initial_pose_source_sha256": "a" * 64,
        "local_profile_rows": [],
        "max_significants": -1,
        "max_significants_resolution": {
            "active_max_significants": -1,
            "do_grad": False,
            "gradient_refine": False,
            "maximum_significants_argument": None,
            "source": "cli_override",
            "target_iteration": 1,
        },
        "n_images": 64,
        "output_dir": str(output_dir),
        "profile_only": True,
        "setup_phase_seconds": {"before_iterations": 0.5},
        "stop_after_local_search_score_only": True,
        "symmetry": {
            "family": "icosahedral",
            "label": "I1",
            "operator_count": 60,
            "operator_sha256": "b" * 64,
        },
        "timing_dir": str(output_dir / "timing"),
        "timing_rows": [],
        "timing_summary": {"n_rows": 0},
        "total_time_s": total_time_s,
        "volume_shape": [800, 800, 800],
        "voxel_size": 0.788,
        "wall_times_trajectory": [total_time_s - 0.7],
    }


def _log_text(
    *,
    total_time_s: float,
    fine_rotation_block_size: int = 128,
    fine_significant_samples: int = 11057,
    parent_mask_max: int = 109,
    fatal_line: str | None = None,
) -> str:
    lines = [
        "RELION EM batch sizing: requested image_batch_size=64 rotation_block_size=8192; "
        "using image_batch_size=24 rotation_block_size=148 (n_rot=148 n_trans=84 K=1, "
        "score_budget=200.0M floats, score_pixels=1532, mode=compact_k1_relion_score_bpref_overlap)",
        "Local search batch sizing: cone_radius=0.393 rad, est_cone_rots=74, eff_n_rot=148 "
        "n_trans=84 -> image_batch_size=24, rotation_block_size=148",
        "RELION EM batch sizing: requested image_batch_size=64 rotation_block_size=8192; "
        "using image_batch_size=29 rotation_block_size=41 (n_rot=72 n_trans=21 K=1, "
        "score_budget=200.0M floats, score_pixels=320800, mode=compact_k1_relion_score_bpref_overlap)",
        "RELION EM batch sizing: requested image_batch_size=64 rotation_block_size=8192; "
        "using image_batch_size=64 rotation_block_size=72 (n_rot=72 n_trans=21 K=1, "
        "score_budget=200.0M floats, score_pixels=1532, mode=compact_k1_relion_score_bpref_overlap)",
        "Exact local bucket loop done: chunks=2/2 images=32/32 wall=5.0s images/s=6.4",
        "Exact local significant-support summary: chunks=2 big_jit_buckets=2 sparse_big_jit_buckets=2 "
        "reconstruction_rows=184 padded_rows=8192 significant_samples=1368 "
        "mean_reconstruction_rows_per_image=5.75 mean_significant_samples_per_image=42.75",
        "RELION local adaptive pass 2 mask: parent significant samples median=31 "
        f"max={parent_mask_max}; fine valid candidates median=1008 max=3488",
        "RELION EM batch sizing: requested image_batch_size=64 rotation_block_size=8192; "
        f"using image_batch_size=24 rotation_block_size={fine_rotation_block_size} "
        f"(n_rot={fine_rotation_block_size} n_trans=84 K=1, score_budget=200.0M floats, "
        "score_pixels=1532, mode=compact_k1_relion_score_bpref_overlap)",
        "Local search caller-qualified batch sizing: requested image_batch_size=24 rotation_block_size=148; "
        f"using image_batch_size=24 rotation_block_size={fine_rotation_block_size} "
        f"(local_rot_max={fine_rotation_block_size} n_trans=84)",
        "Exact local bucket loop done: chunks=4/4 images=32/32 wall=14.5s images/s=2.2",
        "Exact local significant-support summary: chunks=4 big_jit_buckets=4 sparse_big_jit_buckets=4 "
        f"reconstruction_rows=556 padded_rows=1904 significant_samples={fine_significant_samples} "
        "mean_reconstruction_rows_per_image=17.38 mean_significant_samples_per_image=345.53",
        f"Stopping after local-search diagnostic at iteration 1: profiles=0 score_only=True wall={total_time_s - 0.7:.1f}s",
        f"Refinement complete in {total_time_s:.1f}s (1 iterations)",
    ]
    if fatal_line is not None:
        lines.append(fatal_line)
    return "\n".join(lines) + "\n"


def _write_hbm(path: Path, values: list[int], uuid: str) -> None:
    rows = ["timestamp,index,uuid,memory_used_mib,memory_free_mib,gpu_utilization_percent"]
    rows.extend(
        f"2026-09-02T00:00:{index:02d}-04:00,0,{uuid},{value},{81072 - value},0" for index, value in enumerate(values)
    )
    path.write_text("\n".join(rows) + "\n")


def _write_run(
    root: Path,
    *,
    commit: str,
    total_time_s: float,
    hbm_values: list[int],
    fine_rotation_block_size: int = 128,
    fine_significant_samples: int = 11057,
    parent_mask_max: int = 109,
    fatal_line: str | None = None,
) -> dict[str, Path]:
    root.mkdir()
    paths = {
        "ledger": root / "benchmark_ledger.json",
        "log": root / "run.err",
        "hbm": root / "hbm.csv",
        "completed": root / "COMPLETED",
    }
    paths["ledger"].write_text(json.dumps(_ledger(commit=commit, output_dir=root, total_time_s=total_time_s)) + "\n")
    paths["log"].write_text(
        _log_text(
            total_time_s=total_time_s,
            fine_rotation_block_size=fine_rotation_block_size,
            fine_significant_samples=fine_significant_samples,
            parent_mask_max=parent_mask_max,
            fatal_line=fatal_line,
        )
    )
    _write_hbm(paths["hbm"], hbm_values, f"GPU-{commit}")
    paths["completed"].touch()
    return paths


def _audit_pair(baseline: dict[str, Path], candidate: dict[str, Path], **kwargs) -> dict:
    return audit.run_audit(
        baseline_ledger=baseline["ledger"],
        candidate_ledger=candidate["ledger"],
        baseline_log=baseline["log"],
        candidate_log=candidate["log"],
        baseline_hbm=baseline["hbm"],
        candidate_hbm=candidate["hbm"],
        baseline_completed=baseline["completed"],
        candidate_completed=candidate["completed"],
        **kwargs,
    )


def test_padding_gate_accepts_semantic_match_and_reports_reductions(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path / "baseline",
        commit="baseline",
        total_time_s=650.344,
        hbm_values=[6523, 37773, 38059],
    )
    candidate = _write_run(
        tmp_path / "candidate",
        commit="candidate",
        total_time_s=124.897,
        hbm_values=[6523, 20123, 6809],
    )

    report = _audit_pair(
        baseline,
        candidate,
        minimum_hbm_reduction_mib=17000,
        minimum_wall_reduction_s=500,
    )

    assert report["summary"] == {
        "semantic_equivalence_accepted": True,
        "candidate_completion_accepted": True,
        "hbm_gate_accepted": True,
        "wall_gate_accepted": True,
        "accepted": True,
    }
    assert report["performance"]["hbm"]["reduction_mib"] == 38059 - 20123
    assert report["performance"]["wall"]["reduction_s"] == 650.344 - 124.897
    assert [row["candidate_wall_time_s"] for row in report["performance"]["exact_local_bucket_timings"]] == [
        5.0,
        14.5,
    ]
    assert len(report["artifacts"]["candidate"]["log"]["sha256"]) == 64
    assert report["failures"] == []


def test_padding_gate_rejects_selected_batch_plan_drift(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", commit="baseline", total_time_s=650.0, hbm_values=[100, 400])
    candidate = _write_run(
        tmp_path / "candidate",
        commit="candidate",
        total_time_s=120.0,
        hbm_values=[100, 200],
        fine_rotation_block_size=127,
    )

    report = _audit_pair(baseline, candidate)

    assert report["summary"]["accepted"] is False
    assert report["comparisons"]["selected_batch_plans"]["exact"] is False
    assert any("selected batch plans" in failure for failure in report["failures"])


def test_padding_gate_rejects_support_and_adaptive_mask_drift(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", commit="baseline", total_time_s=650.0, hbm_values=[100, 400])
    candidate = _write_run(
        tmp_path / "candidate",
        commit="candidate",
        total_time_s=120.0,
        hbm_values=[100, 200],
        fine_significant_samples=11058,
        parent_mask_max=110,
    )

    report = _audit_pair(baseline, candidate)

    assert report["summary"]["accepted"] is False
    assert report["comparisons"]["significant_support"]["exact"] is False
    assert report["comparisons"]["adaptive_masks"]["exact"] is False


def test_padding_gate_rejects_fatal_candidate_log_and_performance_regression(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", commit="baseline", total_time_s=100.0, hbm_values=[100, 200])
    candidate = _write_run(
        tmp_path / "candidate",
        commit="candidate",
        total_time_s=101.0,
        hbm_values=[100, 201],
        fatal_line="Traceback (most recent call last):",
    )

    report = _audit_pair(baseline, candidate)

    assert report["summary"]["accepted"] is False
    assert report["summary"]["candidate_completion_accepted"] is False
    assert report["summary"]["hbm_gate_accepted"] is False
    assert report["summary"]["wall_gate_accepted"] is False
    assert report["completion"]["candidate"]["fatal_matches"][0]["kind"] == "python_traceback"
