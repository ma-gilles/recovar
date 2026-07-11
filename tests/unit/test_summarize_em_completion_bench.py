from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "summarize_em_completion_bench.py"
SPEC = importlib.util.spec_from_file_location("summarize_em_completion_bench", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarizer
SPEC.loader.exec_module(summarizer)


def _write_k1_runtime_default_fixture(recovar_dir: Path, log_text: str) -> None:
    recovar_dir.mkdir()
    np.savez(
        recovar_dir / "refinement_results.npz",
        n_classes=np.asarray(1),
        tau2_fudge=np.asarray(1.0),
        healpix_order=np.asarray(3.0),
        coarse_healpix_order=np.asarray(3.0),
        finest_healpix_order=np.asarray(4.0),
        adaptive_oversampling=np.asarray(1.0),
        particle_diameter_ang=np.asarray(200.0),
        firstiter_cc=np.asarray(True),
        apply_initial_lowpass=np.asarray(True),
        n_images=np.asarray(4),
        half1_indices=np.asarray([0, 2]),
        half2_indices=np.asarray([1, 3]),
    )
    (recovar_dir / "run_full_refinement.log").write_text(log_text)


def test_k1_runtime_defaults_accept_current_sparse_backend_log(tmp_path):
    recovar_dir = tmp_path / "recovar"
    _write_k1_runtime_default_fixture(
        recovar_dir,
        (
            "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
            "(oversampling=1, pass2_backend=sparse, skip_significance_pruning=False)\n"
        ),
    )

    runtime_defaults = summarizer._check_k1_refinement_runtime_defaults(recovar_dir)

    assert runtime_defaults["status"] == "ok"
    assert runtime_defaults["values"]["k1_adaptive_fine_pass_route"] is True
    assert runtime_defaults["failures"] == []


def test_k1_runtime_defaults_accept_legacy_sparse_backend_log(tmp_path):
    recovar_dir = tmp_path / "recovar"
    _write_k1_runtime_default_fixture(
        recovar_dir,
        "RELION adaptive K=1 routing through sparse run_dense_k_class_em_adaptive\n",
    )

    runtime_defaults = summarizer._check_k1_refinement_runtime_defaults(recovar_dir)

    assert runtime_defaults["status"] == "ok"
    assert runtime_defaults["values"]["k1_adaptive_fine_pass_route"] is True
    assert runtime_defaults["failures"] == []


def test_k1_runtime_defaults_reject_dense_backend_log(tmp_path):
    recovar_dir = tmp_path / "recovar"
    _write_k1_runtime_default_fixture(
        recovar_dir,
        (
            "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
            "(oversampling=1, pass2_backend=dense)\n"
        ),
    )

    runtime_defaults = summarizer._check_k1_refinement_runtime_defaults(recovar_dir)

    assert runtime_defaults["status"] == "failed"
    assert runtime_defaults["values"]["k1_adaptive_fine_pass_route"] is False
    assert "pass2_backend=sparse" in runtime_defaults["failures"][0]


def test_completion_status_requires_recovar_metrics():
    assert (
        summarizer._completion_status_from_metrics(
            {"relion_merged_vs_gt": {"fsc_auc": 0.7}},
            recovar_metric_keys=("recovar_merged_vs_gt",),
            notes=[],
        )
        == "pending"
    )
    assert (
        summarizer._completion_status_from_metrics(
            {
                "relion_merged_vs_gt": {"fsc_auc": 0.7},
                "recovar_merged_vs_gt": {"fsc_auc": 0.72},
            },
            recovar_metric_keys=("recovar_merged_vs_gt",),
            notes=[],
        )
        == "ok"
    )


def test_completion_status_reports_failed_loads():
    assert (
        summarizer._completion_status_from_metrics(
            {},
            recovar_metric_keys=("recovar_vs_gt",),
            notes=["failed to load K=4 RECOVAR class 1 at /tmp/missing.mrc: bad file"],
        )
        == "failed"
    )


def test_k1_final_all_data_guard_accepts_completed_final_pass(tmp_path):
    recovar_dir = tmp_path / "recovar"
    recovar_dir.mkdir()
    np.savez(
        recovar_dir / "refinement_results.npz",
        final_all_data_ran=np.asarray(True),
        fsc_final_all_data=np.asarray([1.0, 0.8, 0.5], dtype=np.float32),
    )

    metadata = summarizer._k1_final_all_data_metadata(recovar_dir)

    assert metadata["status"] == "ok"
    assert metadata["final_all_data_ran"] is True
    assert metadata["fsc_final_all_data_present"] is True
    assert metadata["fsc_final_all_data_finite"] is True
    assert metadata["failures"] == []


def test_k1_final_all_data_guard_rejects_pre_final_maps(tmp_path):
    recovar_dir = tmp_path / "recovar"
    recovar_dir.mkdir()
    np.savez(
        recovar_dir / "refinement_results.npz",
        final_all_data_ran=np.asarray(False),
        fsc_final_all_data=np.asarray([1.0, 0.8, 0.5], dtype=np.float32),
    )

    metadata = summarizer._k1_final_all_data_metadata(recovar_dir)

    assert metadata["status"] == "ok"
    assert metadata["final_all_data_ran"] is False
    assert "final_all_data_ran is false" in metadata["failures"]


def test_map_metrics_reports_sign_invariant_fsc_auc_for_global_flip():
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(8, 8, 8))

    metrics = summarizer.map_metrics(volume, -volume)

    assert metrics["corr"] < -0.999
    assert metrics["abs_corr"] > 0.999
    assert metrics["fsc_auc"] < -0.999
    assert metrics["fsc_auc_sign_flipped"] > 0.999
    assert metrics["fsc_auc_sign_invariant"] > 0.999
    assert metrics["sign_invariant_best_sign"] == -1


def test_map_metrics_reports_integer_shift_to_align_lhs_to_rhs():
    rng = np.random.default_rng(1)
    lhs = rng.normal(size=(8, 8, 8))
    rhs = np.roll(lhs, shift=(1, -2, 3), axis=(0, 1, 2))

    metrics = summarizer.map_metrics(lhs, rhs)

    assert metrics["integer_shift_lhs_to_rhs_zyx"] == [1, -2, 3]
    assert abs(metrics["integer_shift_norm_voxels"] - np.sqrt(14.0)) < 1e-12


def test_finite_max_ignores_nan_values():
    assert summarizer.finite_max(float("nan"), -0.25) == -0.25
    assert np.isnan(summarizer.finite_max(float("nan"), float("inf")))


def test_particle_metrics_prefers_final_all_data_pose_keys_and_orders_by_image_name(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    fixture_dir = tmp_path / "fixture"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    fixture_dir.mkdir()
    (relion_dir / "run_it015_data.star").write_text("# fake\n")
    (relion_dir / "run_data.star").write_text("# fake\n")
    (fixture_dir / "particles.star").write_text("# fake\n")

    np.savez(
        recovar_dir / "refinement_results.npz",
        n_images=np.asarray(2),
        voxel_size=np.asarray(2.0),
        fsc_final_all_data=np.asarray([1.0, 0.9, 0.7, 0.4], dtype=np.float32),
        final_all_data_sampling_perturbation=np.asarray(-0.22, dtype=np.float32),
        final_all_data_sampling_perturbation_applied=np.asarray(True),
        final_all_data_sampling_relion_iteration=np.asarray(17, dtype=np.int32),
        final_all_data_sampling_star=np.asarray("/relion/run_sampling.star"),
        final_all_data_sampling_star_source=np.asarray("final"),
        final_all_data_sampling_offset_range=np.asarray(0.675, dtype=np.float32),
        final_all_data_sampling_offset_step=np.asarray(0.45, dtype=np.float32),
        final_all_data_grid_correct=np.asarray(False),
        final_all_data_gridding_correct=np.asarray("radial"),
        half1_indices=np.asarray([0, 1], dtype=np.int64),
        half2_indices=np.asarray([], dtype=np.int64),
        pmax_per_image_iter_015=np.asarray([0.4, 0.2], dtype=np.float32),
        pmax_final_all_data_by_image=np.asarray([0.9, 0.8], dtype=np.float32),
        best_rotation_eulers_final_all_data_by_image=np.zeros((2, 3), dtype=np.float32),
        best_rotation_eulers_final_by_image=np.full((2, 3), 30.0, dtype=np.float32),
        best_translations_final_all_data_by_image=np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        best_translations_final_by_image=np.asarray([[11.0, 0.0], [10.0, 0.0]], dtype=np.float32),
    )

    dataset_df = pd.DataFrame({"rlnImageName": ["2@particles.mrcs", "1@particles.mrcs"]})
    relion_iter_df = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [90.0, 90.0],
            "rlnAngleTilt": [0.0, 0.0],
            "rlnAnglePsi": [0.0, 0.0],
            "rlnOriginXAngst": [20.0, 22.0],
            "rlnOriginYAngst": [0.0, 0.0],
            "rlnMaxValueProbDistribution": [0.2, 0.4],
        }
    )
    relion_final_df = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [0.0, 0.0],
            "rlnAngleTilt": [0.0, 0.0],
            "rlnAnglePsi": [0.0, 0.0],
            "rlnOriginXAngst": [0.0, 2.0],
            "rlnOriginYAngst": [0.0, 0.0],
            "rlnMaxValueProbDistribution": [0.8, 0.9],
        }
    )

    def fake_star_particles(path):
        path = Path(path)
        if path.name == "particles.star":
            return dataset_df
        if path.name == "run_data.star":
            return relion_final_df
        return relion_iter_df

    monkeypatch.setattr(summarizer, "_star_particles", fake_star_particles)

    metrics = summarizer._particle_metrics(
        recovar_dir=recovar_dir,
        relion_dir=relion_dir,
        fixture_dir=fixture_dir,
        relion_iter=15,
    )

    assert metrics is not None
    assert metrics["relion_data_star"].endswith("run_it015_data.star")
    assert metrics["relion_final_data_star"].endswith("run_data.star")
    assert metrics["final_all_data_fsc"]["npz_key"] == "fsc_final_all_data"
    assert metrics["pmax"]["npz_key"] == "pmax_final_all_data_by_image"
    assert metrics["pmax"]["relion_data_star"].endswith("run_data.star")
    assert abs(metrics["pmax"]["relion_mean"] - float(np.mean([0.9, 0.8]))) < 1e-7
    assert metrics["pmax"]["abs_diff"]["max"] < 1e-7
    assert metrics["pose_rotation_deg"]["npz_key"] == "best_rotation_eulers_final_all_data_by_image"
    assert metrics["pose_rotation_deg"]["relion_data_star"].endswith("run_data.star")
    assert metrics["pose_rotation_deg"]["angle_error"]["max"] == 0.0
    assert metrics["translation_px"]["npz_key"] == "best_translations_final_all_data_by_image"
    assert metrics["translation_px"]["relion_data_star"].endswith("run_data.star")
    assert metrics["translation_px"]["l2_error"]["max"] == 0.0
    assert abs(metrics["final_all_data_sampling"]["perturbation"] - float(np.float32(-0.22))) < 1e-8
    assert metrics["final_all_data_sampling"]["applied"] is True
    assert metrics["final_all_data_sampling"]["relion_iteration"] == 17
    assert metrics["final_all_data_sampling"]["sampling_star"] == "/relion/run_sampling.star"
    assert metrics["final_all_data_sampling"]["sampling_star_source"] == "final"
    assert abs(metrics["final_all_data_sampling"]["offset_range_px"] - float(np.float32(0.675))) < 1e-8
    assert abs(metrics["final_all_data_sampling"]["offset_step_px"] - float(np.float32(0.45))) < 1e-8
    assert metrics["final_all_data_reconstruction"]["grid_correct"] is False
    assert metrics["final_all_data_reconstruction"]["gridding_correct"] == "radial"


def test_particle_metrics_prefers_by_image_pmax_history(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    fixture_dir = tmp_path / "fixture"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    fixture_dir.mkdir()
    (relion_dir / "run_it005_data.star").write_text("# fake\n")
    (fixture_dir / "particles.star").write_text("# fake\n")

    np.savez(
        recovar_dir / "refinement_results.npz",
        n_images=np.asarray(3),
        half1_indices=np.asarray([2, 0], dtype=np.int64),
        half2_indices=np.asarray([1], dtype=np.int64),
        pmax_per_image_iter_005=np.asarray([9.0, 8.0, 7.0], dtype=np.float32),
        pmax_per_image_by_image_iter_005=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )

    dataset_df = pd.DataFrame({"rlnImageName": ["1@particles.mrcs", "2@particles.mrcs", "3@particles.mrcs"]})
    relion_iter_df = pd.DataFrame(
        {
            "rlnImageName": ["3@particles.mrcs", "1@particles.mrcs", "2@particles.mrcs"],
            "rlnMaxValueProbDistribution": [0.3, 0.1, 0.2],
        }
    )

    def fake_star_particles(path):
        return dataset_df if Path(path).name == "particles.star" else relion_iter_df

    monkeypatch.setattr(summarizer, "_star_particles", fake_star_particles)

    metrics = summarizer._particle_metrics(
        recovar_dir=recovar_dir,
        relion_dir=relion_dir,
        fixture_dir=fixture_dir,
        relion_iter=5,
    )

    assert metrics is not None
    assert metrics["pmax"]["npz_key"] == "pmax_per_image_by_image_iter_005"
    assert metrics["pmax"]["abs_diff"]["max"] < 1e-7


def test_k1_summary_reports_relion_final_map_separately_from_half_average(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    fixture_dir = tmp_path / "fixture"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    fixture_dir.mkdir()
    for path in [
        recovar_dir / "final_merged.mrc",
        recovar_dir / "final_half1.mrc",
        recovar_dir / "final_half2.mrc",
        relion_dir / "run_class001.mrc",
        relion_dir / "run_half1_class001_unfil.mrc",
        relion_dir / "run_half2_class001_unfil.mrc",
        fixture_dir / "reference_gt.mrc",
    ]:
        path.write_text("stub\n")

    volumes = {
        recovar_dir / "final_merged.mrc": np.asarray([10.0, 10.0]),
        recovar_dir / "final_half1.mrc": np.asarray([11.0, 11.0]),
        recovar_dir / "final_half2.mrc": np.asarray([12.0, 12.0]),
        relion_dir / "run_class001.mrc": np.asarray([20.0, 20.0]),
        relion_dir / "run_half1_class001_unfil.mrc": np.asarray([30.0, 30.0]),
        relion_dir / "run_half2_class001_unfil.mrc": np.asarray([50.0, 50.0]),
        fixture_dir / "reference_gt.mrc": np.asarray([100.0, 100.0]),
    }

    def fake_load(path):
        return volumes[Path(path)]

    def fake_map_metrics(lhs, rhs, **_kwargs):
        return {
            "lhs_mean": float(np.mean(lhs)),
            "rhs_mean": float(np.mean(rhs)),
            "fsc_auc": float(np.mean(rhs)),
        }

    monkeypatch.setattr(summarizer, "_load_recovar_volume", fake_load)
    monkeypatch.setattr(summarizer, "_load_relion_volume", fake_load)
    monkeypatch.setattr(summarizer, "map_metrics", fake_map_metrics)
    monkeypatch.setattr(summarizer, "_particle_metrics", lambda **_kwargs: None)

    summary = summarizer.summarize_k1(recovar_dir, relion_dir, fixture_dir)

    metrics = summary["metrics"]
    assert metrics["relion_merged_vs_gt"]["lhs_mean"] == 20.0
    assert metrics["relion_halfavg_vs_gt"]["lhs_mean"] == 40.0
    assert metrics["recovar_merged_vs_relion_merged"]["rhs_mean"] == 20.0
    assert metrics["recovar_merged_vs_relion_halfavg"]["rhs_mean"] == 40.0


def test_k1_summary_treats_pre_final_all_data_map_as_missing_required_product(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    fixture_dir = tmp_path / "fixture"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    fixture_dir.mkdir()
    for path in [
        recovar_dir / "final_merged.mrc",
        recovar_dir / "final_half1.mrc",
        recovar_dir / "final_half2.mrc",
        relion_dir / "run_class001.mrc",
        relion_dir / "run_half1_class001_unfil.mrc",
        relion_dir / "run_half2_class001_unfil.mrc",
        fixture_dir / "reference_gt.mrc",
    ]:
        path.write_text("stub\n")
    np.savez(
        recovar_dir / "refinement_results.npz",
        n_classes=np.asarray(1),
        tau2_fudge=np.asarray(1.0),
        healpix_order=np.asarray(3.0),
        coarse_healpix_order=np.asarray(3.0),
        finest_healpix_order=np.asarray(4.0),
        adaptive_oversampling=np.asarray(1.0),
        particle_diameter_ang=np.asarray(200.0),
        firstiter_cc=np.asarray(True),
        apply_initial_lowpass=np.asarray(True),
        n_images=np.asarray(4),
        half1_indices=np.asarray([0, 2]),
        half2_indices=np.asarray([1, 3]),
        final_all_data_ran=np.asarray(False),
        fsc_final_all_data=np.asarray([1.0, 0.9, 0.6], dtype=np.float32),
    )
    (recovar_dir / "run_full_refinement.log").write_text(
        "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
        "(oversampling=1, pass2_backend=sparse, skip_significance_pruning=False)\n"
    )

    monkeypatch.setattr(summarizer, "_load_recovar_volume", lambda path: np.asarray([1.0, 2.0]))
    monkeypatch.setattr(summarizer, "_load_relion_volume", lambda path: np.asarray([1.0, 2.0]))
    monkeypatch.setattr(
        summarizer,
        "map_metrics",
        lambda _lhs, _rhs, **_kwargs: {"fsc_auc": 0.9, "corr": 1.0},
    )
    monkeypatch.setattr(summarizer, "_particle_metrics", lambda **_kwargs: None)

    summary = summarizer.summarize_k1(recovar_dir, relion_dir, fixture_dir)

    assert summary["status"] == "ok"
    assert summary["final_all_data"]["final_all_data_ran"] is False
    assert "K=1 final all-data guard: final_all_data_ran is false" in summary["notes"]
    assert summarizer._section_has_missing_required_products(summary) is True


def test_k1_summary_allows_missing_final_all_data_when_relion_also_stopped_at_max_iter(
    tmp_path, monkeypatch
):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    fixture_dir = tmp_path / "fixture"
    recovar_dir.mkdir()
    relion_dir.mkdir()
    fixture_dir.mkdir()
    for path in [
        recovar_dir / "final_merged.mrc",
        recovar_dir / "final_half1.mrc",
        recovar_dir / "final_half2.mrc",
        relion_dir / "run_it003_half1_class001.mrc",
        relion_dir / "run_it003_half2_class001.mrc",
        fixture_dir / "reference_gt.mrc",
    ]:
        path.write_text("stub\n")
    np.savez(
        recovar_dir / "refinement_results.npz",
        n_classes=np.asarray(1),
        tau2_fudge=np.asarray(1.0),
        healpix_order=np.asarray(3.0),
        coarse_healpix_order=np.asarray(3.0),
        finest_healpix_order=np.asarray(4.0),
        adaptive_oversampling=np.asarray(1.0),
        particle_diameter_ang=np.asarray(200.0),
        firstiter_cc=np.asarray(True),
        apply_initial_lowpass=np.asarray(True),
        n_images=np.asarray(4),
        n_iterations=np.asarray(3),
        convergence_has_converged=np.asarray(False),
        half1_indices=np.asarray([0, 2]),
        half2_indices=np.asarray([1, 3]),
        final_all_data_ran=np.asarray(False),
    )
    (recovar_dir / "run_full_refinement.log").write_text(
        "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
        "(oversampling=1, pass2_backend=sparse, skip_significance_pruning=False)\n"
    )

    monkeypatch.setattr(summarizer, "_load_recovar_volume", lambda path: np.asarray([1.0, 2.0]))
    monkeypatch.setattr(summarizer, "_load_relion_volume", lambda path: np.asarray([1.0, 2.0]))
    monkeypatch.setattr(
        summarizer,
        "map_metrics",
        lambda _lhs, _rhs, **_kwargs: {"fsc_auc": 0.9, "corr": 1.0},
    )
    monkeypatch.setattr(summarizer, "_particle_metrics", lambda **_kwargs: None)

    summary = summarizer.summarize_k1(recovar_dir, relion_dir, fixture_dir)

    assert summary["status"] == "ok"
    assert summary["final_all_data"]["final_all_data_ran"] is False
    assert (
        "optional K=1 final all-data guard: RECOVAR ended at max_iter without convergence "
        "and RELION fixture has no run_it004 products"
    ) in summary["notes"]
    assert not any(note.startswith("K=1 final all-data guard:") for note in summary["notes"])
    assert summarizer._section_has_missing_required_products(summary) is False


def test_k1_relion_final_map_path_ignores_iteration_half_maps(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    (relion_dir / "run_it003_half1_class001.mrc").write_text("half1\n")
    (relion_dir / "run_it003_half2_class001.mrc").write_text("half2\n")

    assert summarizer._k1_relion_final_map_path(relion_dir) is None

    merged = relion_dir / "run_it002_class001.mrc"
    merged.write_text("merged\n")
    (relion_dir / "run_it004_half2_class001.mrc").write_text("newer half\n")

    assert summarizer._k1_relion_final_map_path(relion_dir) == merged


def test_k4_relion_path_ignores_iteration_half_maps(tmp_path):
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    (relion_dir / "run_it008_half1_class003.mrc").write_text("half\n")

    assert summarizer._k4_relion_path(relion_dir, 3) is None

    merged = relion_dir / "run_it007_class003.mrc"
    merged.write_text("merged\n")

    assert summarizer._k4_relion_path(relion_dir, 3) == merged


def test_markdown_reports_k1_fsc_auc_correctness_gate_before_metric_details():
    summary = {
        "k1": {
            "status": "ok",
            "metadata": {},
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {"fsc_auc": 0.71, "corr": 0.9},
                "relion_merged_vs_gt": {"fsc_auc": 0.7, "corr": 0.95},
            },
            "notes": [],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    markdown = summarizer.build_markdown(summary)

    assert "| Correctness Gate | RECOVAR GT FSC AUC | RELION GT FSC AUC | Delta |" in markdown
    assert "| merged_vs_gt | 0.71 | 0.7 | 0.01 |" in markdown
    assert markdown.index("| Correctness Gate |") < markdown.index("| Comparison | FSC AUC | Corr |")


def test_markdown_reports_k4_mean_fsc_auc_correctness_gate():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "metadata": {},
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.62, "mean_corr": 0.9},
                "relion_vs_gt": {"mean_fsc_auc": 0.64, "mean_corr": 0.91},
            },
            "notes": [],
        },
    }

    markdown = summarizer.build_markdown(summary)

    assert "| Correctness Gate | RECOVAR GT FSC AUC | RELION GT FSC AUC | Delta |" in markdown
    assert "| class_mean_vs_gt | 0.62 | 0.64 | -0.02 |" in markdown
    assert markdown.index("| Correctness Gate |") < markdown.index("| Comparison | Permutation | Mean FSC AUC |")


def test_k4_permutation_prefers_fsc_auc_when_fsc_is_available(monkeypatch):
    # Correlation favors the identity assignment, while FSC-AUC favors a swap.
    # K4 GT comparisons must optimize the metric used for acceptance.
    score_table = {
        (0, 0): {"corr": 0.99, "fsc_auc": 0.10},
        (0, 1): {"corr": 0.10, "fsc_auc": 0.80},
        (1, 0): {"corr": 0.10, "fsc_auc": 0.80},
        (1, 1): {"corr": 0.99, "fsc_auc": 0.10},
    }

    def fake_map_metrics(lhs, rhs, **_kwargs):
        return dict(score_table[(int(lhs[0]), int(rhs[0]))])

    monkeypatch.setattr(summarizer, "map_metrics", fake_map_metrics)

    summary = summarizer.best_permutation_summary(
        [np.asarray([0]), np.asarray([1])],
        [np.asarray([0]), np.asarray([1])],
        rhs_label="gt",
        include_fsc=True,
    )

    assert summary["permutation_score_key"] == "fsc_auc"
    assert summary["permutation_lhs_to_rhs"] == [1, 0]
    assert summary["mean_fsc_auc"] == 0.80


def test_completion_metadata_keeps_k4_sparse_env_provenance(tmp_path):
    scratch_dir = tmp_path / "scratch"
    recovar_dir = scratch_dir / "k4_100k256_recovar"
    recovar_dir.mkdir(parents=True)
    (scratch_dir / "submission.env").write_text(
        "\n".join(
            [
                "RUN_K4_FUSED_SPARSE_PASS2=1",
                "EM_COMPLETION_TIMING_PROBE=1",
                "K4_MEM=128G",
                "K4_TIME_LIMIT=04:00:00",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES=19",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION=2.0",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE=4096",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP=pair_sparse",
                "RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES=4294967296",
                "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE=512",
                "RECOVAR_SPARSE_KCLASS_GROUP_TIMING=1",
                "RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS=1",
            ]
        )
        + "\n"
    )

    metadata = summarizer._completion_metadata(recovar_dir, "k4")

    assert metadata["env"]["RUN_K4_FUSED_SPARSE_PASS2"] == "1"
    assert metadata["env"]["EM_COMPLETION_TIMING_PROBE"] == "1"
    assert metadata["env"]["K4_MEM"] == "128G"
    assert metadata["env"]["K4_TIME_LIMIT"] == "04:00:00"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES"] == "19"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION"] == "2.0"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE"] == "4096"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP"] == "pair_sparse"
    assert metadata["env"]["RECOVAR_SPARSE_PASS2_MAX_PROJECTION_GATHER_BYTES"] == "4294967296"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE"] == "512"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_GROUP_TIMING"] == "1"
    assert metadata["env"]["RECOVAR_SPARSE_KCLASS_RECTANGULAR_ACTIVE_ROWS"] == "1"


def test_required_k4_fails_when_gt_fsc_auc_is_below_relion():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.60},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": [],
        },
    }

    summarizer._mark_required_failures(summary, ("k4",), fsc_auc_parity_tol=1e-4)

    assert summary["k4"]["status"] == "failed"
    assert "GT FSC-AUC correctness gate failed" in summary["k4"]["notes"][-1]
    assert "RECOVAR=0.6" in summary["k4"]["notes"][-1]
    assert "RELION=0.62" in summary["k4"]["notes"][-1]


def test_required_k4_passes_when_gt_fsc_auc_is_within_tolerance():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.61995},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": [],
        },
    }

    summarizer._mark_required_failures(summary, ("k4",), fsc_auc_parity_tol=1e-4)

    assert summary["k4"]["status"] == "ok"
    assert summary["k4"]["notes"] == []


def test_required_k4_allows_optional_final_all_data_diagnostic_note():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.64},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": [
                "optional K=4 particle metrics: missing RECOVAR final all-data FSC; "
                "RECOVAR ended at max_iter without convergence and RELION fixture has no run_it016 products"
            ],
        },
    }

    summarizer._mark_required_failures(summary, ("k4",), fsc_auc_parity_tol=1e-4)

    assert summary["k4"]["status"] == "ok"
    assert len(summary["k4"]["notes"]) == 1


def test_required_k1_allows_optional_final_all_data_diagnostic_note():
    summary = {
        "k1": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {"fsc_auc": 0.64},
                "relion_merged_vs_gt": {"fsc_auc": 0.62},
            },
            "notes": [
                "optional K=1 particle metrics: missing RECOVAR final all-data FSC; "
                "RECOVAR ended at max_iter without convergence and RELION fixture has no run_it004 products"
            ],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    summarizer._mark_required_failures(summary, ("k1",), fsc_auc_parity_tol=1e-4)

    assert summary["k1"]["status"] == "ok"
    assert len(summary["k1"]["notes"]) == 1


def test_required_k1_allows_relion_final_map_halfavg_fallback_notes():
    summary = {
        "k1": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {"fsc_auc": 0.64},
                "relion_merged_vs_gt": {"fsc_auc": 0.62},
                "recovar_merged_vs_relion_merged": {"fsc_auc": 0.99},
                "recovar_merged_vs_relion_halfavg": {"fsc_auc": 0.99},
                "relion_halfavg_vs_gt": {"fsc_auc": 0.62},
            },
            "notes": [
                "missing K=1 RELION final merged map",
                "K=1 RELION final merged map missing; recovar_merged_vs_relion_merged uses half-map average",
                "K=1 RELION final merged map missing; relion_merged_vs_gt uses half-map average",
                "using RELION slurm_walltime.json",
            ],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    summarizer._mark_required_failures(summary, ("k1",), fsc_auc_parity_tol=1e-4)

    assert summary["k1"]["status"] == "ok"
    assert len(summary["k1"]["notes"]) == 4


def test_required_k1_noctf_sign_ambiguity_uses_sign_invariant_gate():
    summary = {
        "k1": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {
                    "fsc_auc": -0.01,
                    "fsc_auc_sign_invariant": 0.70,
                },
                "relion_merged_vs_gt": {
                    "fsc_auc": 0.02,
                    "fsc_auc_sign_invariant": 0.68,
                },
            },
            "sign_ambiguity": {"allow_global_sign": True, "reason": "dataset_params_option=noctf"},
            "notes": [
                "K=1 no-CTF fixture: global sign is ambiguous; required GT FSC-AUC gate uses "
                "sign-invariant FSC-AUC while signed metrics are still reported"
            ],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    summarizer._mark_required_failures(summary, ("k1",), fsc_auc_parity_tol=1e-4)

    assert summary["k1"]["status"] == "ok"
    rows = summarizer._correctness_gate_rows(
        "k1",
        summary["k1"]["metrics"],
        sign_invariant_gt=summarizer._section_allows_global_sign(summary["k1"]),
    )
    assert rows == [("merged_vs_gt_sign_invariant", 0.70, 0.68, 0.019999999999999907)]


def test_required_k1_normal_fixture_still_uses_signed_gate():
    summary = {
        "k1": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {
                    "fsc_auc": -0.01,
                    "fsc_auc_sign_invariant": 0.70,
                },
                "relion_merged_vs_gt": {
                    "fsc_auc": 0.02,
                    "fsc_auc_sign_invariant": 0.68,
                },
            },
            "sign_ambiguity": {"allow_global_sign": False},
            "notes": [],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    summarizer._mark_required_failures(summary, ("k1",), fsc_auc_parity_tol=1e-4)

    assert summary["k1"]["status"] == "failed"
    assert "merged_vs_gt" in summary["k1"]["notes"][-1]
    assert "RECOVAR=-0.01" in summary["k1"]["notes"][-1]


def test_timing_probe_status_cannot_satisfy_required_k4():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "metadata": {"env": {"EM_COMPLETION_TIMING_PROBE": "1"}},
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.64},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": [],
        },
    }

    summarizer._annotate_timing_probe_status(summary)

    assert summary["k4"]["status"] == "timing_probe"
    assert "not a correctness acceptance run" in summary["k4"]["notes"][-1]
    assert summarizer._section_has_missing_required_products(summary["k4"]) is True
    markdown = summarizer.build_markdown(summary)
    assert "Status: `timing_probe`" in markdown
    assert "K=4 timing probe only" in markdown

    summarizer._mark_required_failures(summary, ("k4",), fsc_auc_parity_tol=1e-4)

    assert summary["k4"]["status"] == "failed"
    assert "required artifacts or metrics are missing" in summary["k4"]["notes"][-1]


def test_timing_probe_false_env_keeps_acceptance_status():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "metadata": {"env": {"EM_COMPLETION_TIMING_PROBE": "0"}},
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.64},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": [],
        },
    }

    summarizer._annotate_timing_probe_status(summary)

    assert summary["k4"]["status"] == "ok"
    assert summary["k4"]["notes"] == []


def test_required_k4_still_fails_nonoptional_missing_note():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_vs_gt": {"mean_fsc_auc": 0.64},
                "relion_vs_gt": {"mean_fsc_auc": 0.62},
            },
            "notes": ["K=4 particle metrics: missing RECOVAR final all-data FSC"],
        },
    }

    summarizer._mark_required_failures(summary, ("k4",), fsc_auc_parity_tol=1e-4)

    assert summary["k4"]["status"] == "failed"
    assert "required artifacts or metrics are missing" in summary["k4"]["notes"][-1]


def test_required_k1_still_fails_nonoptional_missing_note():
    summary = {
        "k1": {
            "status": "ok",
            "timing": {},
            "metrics": {
                "recovar_merged_vs_gt": {"fsc_auc": 0.64},
                "relion_merged_vs_gt": {"fsc_auc": 0.62},
            },
            "notes": ["K=1 particle metrics: missing RECOVAR final all-data FSC"],
        },
        "k4": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
    }

    summarizer._mark_required_failures(summary, ("k1",), fsc_auc_parity_tol=1e-4)

    assert summary["k1"]["status"] == "failed"
    assert "required artifacts or metrics are missing" in summary["k1"]["notes"][-1]


def test_relion_iteration_exists_detects_post_max_iter_products(tmp_path):
    assert summarizer._relion_iteration_exists(tmp_path, 16) is False

    (tmp_path / "run_it016_data.star").write_text("# fake\n")

    assert summarizer._relion_iteration_exists(tmp_path, 16) is True


def test_sparse_group_timing_aggregate_reports_stage_totals_and_mode_breakdown():
    events = [
        {
            "mode": "compact_pair",
            "score_s": 6.0,
            "mstep_noise_stats_s": 9.0,
            "mstep_weighted_sums_s": 2.0,
            "mstep_adjoint_s": 3.0,
            "noise_s": 1.0,
            "stats_s": 1.5,
            "total_profiled_s": 18.0,
            "wall_s": 20.0,
        },
        {
            "mode": "compact_pair",
            "score_s": 3.0,
            "mstep_noise_stats_s": 6.0,
            "mstep_weighted_sums_s": 1.0,
            "mstep_adjoint_s": 2.0,
            "noise_s": 0.5,
            "stats_s": 0.5,
            "total_profiled_s": 10.0,
            "wall_s": 10.0,
        },
        {
            "mode": "rectangular",
            "score_s": 5.0,
            "mstep_noise_stats_s": 4.0,
            "total_profiled_s": 11.0,
            "wall_s": 12.0,
        },
    ]

    aggregate = summarizer._sparse_group_timing_aggregate(events)

    assert aggregate is not None
    assert aggregate["sparse_kclass_group_timing_count"] == 3
    assert aggregate["sparse_kclass_group_wall_total_s"] == 42.0
    assert aggregate["sparse_kclass_group_score_total_s"] == 14.0
    assert aggregate["sparse_kclass_group_mstep_noise_stats_total_s"] == 19.0
    assert aggregate["sparse_kclass_group_score_fraction_of_group_wall"] == 14.0 / 42.0
    assert aggregate["sparse_kclass_group_mstep_noise_stats_fraction_of_group_wall"] == 19.0 / 42.0

    by_mode = aggregate["sparse_kclass_group_by_mode"]
    assert by_mode["compact_pair"]["count"] == 2
    assert by_mode["compact_pair"]["wall_total_s"] == 30.0
    assert by_mode["compact_pair"]["score_total_s"] == 9.0
    assert by_mode["compact_pair"]["score_fraction_of_wall"] == 9.0 / 30.0
    assert by_mode["rectangular"]["count"] == 1


def test_markdown_includes_sparse_group_timing_mode_summary():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "metadata": {},
            "timing": {
                "sparse_kclass_group_timing_count": 1,
                "sparse_kclass_group_wall_total_s": 10.0,
                "sparse_kclass_group_score_total_s": 3.0,
                "sparse_kclass_group_score_fraction_of_group_wall": 0.3,
                "sparse_group_timing_events": [
                    {
                        "iteration": 4,
                        "current_size": 256,
                        "mode": "compact_pair",
                        "group_key": "bucket_size",
                        "group_value": 8192,
                        "score_s": 3.0,
                        "mstep_noise_stats_s": 5.0,
                        "total_profiled_s": 9.0,
                        "wall_s": 10.0,
                    }
                ],
                "sparse_kclass_group_by_mode": {
                    "compact_pair": {
                        "count": 1,
                        "wall_total_s": 10.0,
                        "score_total_s": 3.0,
                        "mstep_noise_stats_total_s": 5.0,
                        "score_fraction_of_wall": 0.3,
                        "mstep_noise_stats_fraction_of_wall": 0.5,
                    }
                },
            },
            "metrics": {},
            "notes": [],
        },
    }

    markdown = summarizer.build_markdown(summary)

    assert "sparse_kclass_group_score_fraction_of_group_wall" in markdown
    assert "| K4 Group Mode | Count | Wall s |" in markdown
    assert "compact_pair" in markdown


def test_global_profile_aggregate_reports_time_totals_and_phase_counts():
    rows = [
        {
            "phase": "iteration",
            "k_class_enabled": True,
            "sparse_kclass_fused_s": 10.0,
            "sparse_kclass_compact_pair_plan_s": 2.0,
        },
        {
            "phase": "final_all_data",
            "k_class_enabled": True,
            "sparse_kclass_fused_s": 4.0,
            "sparse_kclass_compact_pair_plan_s": 0.5,
        },
        {
            "phase": "iteration",
            "k_class_enabled": False,
            "dense_score_s": 3.0,
        },
    ]

    aggregate = summarizer._global_profile_aggregate(rows)

    assert aggregate is not None
    assert aggregate["global_profile_row_count"] == 3
    assert aggregate["global_profile_kclass_row_count"] == 2
    assert aggregate["global_profile_phase_counts"] == {"iteration": 2, "final_all_data": 1}
    assert aggregate["global_profile_time_totals_s"]["sparse_kclass_fused_s"] == 14.0
    assert aggregate["global_profile_time_totals_s"]["sparse_kclass_compact_pair_plan_s"] == 2.5
    assert aggregate["global_profile_top_time_totals_s"][0] == {
        "key": "sparse_kclass_fused_s",
        "seconds": 14.0,
    }


def test_markdown_includes_global_profile_timing_table():
    summary = {
        "k1": {"status": "skipped", "timing": {}, "metrics": {}, "notes": []},
        "k4": {
            "status": "ok",
            "metadata": {},
            "timing": {
                "global_profile_row_count": 2,
                "global_profile_kclass_row_count": 2,
                "global_profile_top_time_totals_s": [
                    {"key": "sparse_kclass_fused_s", "seconds": 14.0},
                    {"key": "sparse_kclass_compact_pair_plan_s", "seconds": 2.5},
                ],
            },
            "metrics": {},
            "notes": [],
        },
    }

    markdown = summarizer.build_markdown(summary)

    assert "global_profile_row_count" in markdown
    assert "| Global Profile Key | Total s |" in markdown
    assert "sparse_kclass_fused_s" in markdown
