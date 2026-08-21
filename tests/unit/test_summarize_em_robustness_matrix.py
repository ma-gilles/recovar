from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "summarize_em_robustness_matrix.py"
SPEC = importlib.util.spec_from_file_location("summarize_em_robustness_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarizer
SPEC.loader.exec_module(summarizer)


def test_summarizes_k1_table_complete_failed_and_pending(tmp_path):
    root = tmp_path / "em_k1_robustness"
    root.mkdir()
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std|case_root|case_job_id",
                f"1|baseline|3000|128|1.0|white|uniform|1|80|0|0|0.7|-|01:00:00|128G|500|0|0|0|0|{root / 'cases' / '1_baseline'}|111",
                f"2|pending|3000|128|1.0|white|uniform|2|80|0|0|0.7|-|01:00:00|128G|500|0|0|0|0|{root / 'cases' / '2_pending'}|222",
                f"4|dryrun|3000|128|1.0|white|uniform|4|80|0|0|0.7|-|01:00:00|128G|500|0|0|0|0|{root / 'k1_matrix_dryrun' / 'cases' / '4_dryrun'}|em_k1_matrix_4_dryrun.sh",
                f"10|failed|3000|128|1.0|white|uniform|10|80|0|0|0.7|-|01:00:00|128G|500|0|0|0|0|{root / 'cases' / '10_failed'}|333",
            ]
        )
        + "\n"
    )

    complete = root / "cases" / "1_baseline"
    (complete / "recovar").mkdir(parents=True)
    (complete / "case_config.json").write_text(json.dumps({"index": 1, "name": "baseline"}))
    (complete / "recovar" / "slurm_walltime.json").write_text(
        json.dumps({"slurm_job_id": "111", "external_wall_s": 12.5, "exit_status": 0})
    )
    (complete / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "ok",
                    "metrics": {
                        "recovar_merged_vs_gt": {"fsc_auc": 0.75, "corr": 0.91},
                        "relion_merged_vs_gt": {"fsc_auc": 0.73, "corr": 0.89},
                    },
                    "timing": {
                        "recovar_walltime_s": 12.5,
                        "relion_walltime_s": 20.0,
                        "recovar_iteration_rows": [{"res_ang": 4.25}],
                    },
                    "notes": [],
                }
            }
        )
    )
    np.savez(
        complete / "recovar" / "refinement_results.npz",
        n_iterations=np.asarray(13),
        convergence_iteration=np.asarray(13),
        convergence_has_converged=np.asarray(True),
        fsc_final_all_data=np.asarray([1.0, 0.9, 0.7, 0.4], dtype=np.float32),
        best_rotation_eulers_final_all_data_by_image=np.zeros((2, 3), dtype=np.float32),
        best_translations_final_all_data_by_image=np.zeros((2, 2), dtype=np.float32),
    )

    failed = root / "cases" / "10_failed"
    (failed / "recovar").mkdir(parents=True)
    (failed / "recovar" / "slurm_walltime.json").write_text(
        json.dumps({"slurm_job_id": "333", "external_wall_s": 4, "exit_status": 137})
    )
    (failed / "recovar" / "run_full_refinement.log").write_text(
        "start\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert [case.case_id for case in cases] == ["1", "2", "4", "10"]
    assert cases[0].status == "ok"
    assert cases[0].wall_s == 12.5
    assert cases[0].relion_wall_s == 20.0
    assert cases[0].fsc_auc_vs_gt == 0.75
    assert cases[0].relion_fsc_auc_vs_gt == 0.73
    assert abs(cases[0].fsc_auc_delta_vs_relion - 0.02) < 1e-12
    assert cases[0].final_resolution_A == 4.25
    assert cases[0].map_corr_vs_gt == 0.91
    assert cases[0].recovar_convergence_has_converged is True
    assert abs(cases[0].recovar_final_all_data_fsc_auc - 0.675) < 1e-6
    assert cases[0].recovar_has_final_all_data_poses is True
    assert cases[1].status == "pending"
    assert cases[2].status == "dryrun"
    assert cases[3].status == "failed"
    assert "exit_status=137" in cases[3].failure_reason
    assert cases[3].failure_log == str(failed / "recovar" / "run_full_refinement.log")
    assert "CUDA out of memory" in cases[3].log_excerpt


def test_live_slurm_header_overrides_stale_table_job_id(tmp_path):
    root = tmp_path / "em_k1_robustness"
    root.mkdir()
    case_root = root / "cases" / "1_baseline"
    case_root.mkdir(parents=True)
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std|case_root|case_job_id",
                f"1|baseline|3000|128|1.0|white|uniform|1|80|0|0|0.7|-|01:00:00|128G|500|0|0|0|0|{case_root}|111",
            ]
        )
        + "\n"
    )
    (root / "em_k1_matrix_1_baseline.out").write_text(
        "=== em_k1_matrix_1_baseline ===\n"
        "Slurm job: 444\n"
        "Host: della-test\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].job_id == "444"
    assert cases[0].status == "running"


def test_per_case_summary_drops_stale_aggregate_runtime_default_notes(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "11_small_baseline"
    case.mkdir(parents=True)
    (root / "k1_robustness_matrix_summary.json").write_text(
        json.dumps(
            [
                {
                    "case_root": str(case),
                    "status": "failed",
                    "recovar_vs_gt_fsc_auc": 0.62,
                    "relion_vs_gt_fsc_auc": 0.63,
                    "notes": [
                        "K=1 runtime default guard: k1_adaptive_fine_pass_route=False",
                        "K=1 was selected as required, but required artifacts or metrics are missing",
                    ],
                }
            ]
        )
    )
    (case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "failed",
                    "metrics": {
                        "recovar_merged_vs_gt": {"fsc_auc": 0.621},
                        "relion_merged_vs_gt": {"fsc_auc": 0.633},
                    },
                    "notes": [
                        "K=1 GT FSC-AUC correctness gate failed for merged_vs_gt: "
                        "RECOVAR=0.621, RELION=0.633, delta=-0.012, tolerance=0.0001"
                    ],
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "failed"
    assert cases[0].fsc_auc_vs_gt == 0.621
    assert cases[0].relion_fsc_auc_vs_gt == 0.633
    assert abs(cases[0].fsc_auc_delta_vs_relion + 0.012) < 1e-12
    assert len(cases[0].notes) == 1
    assert cases[0].notes[0].startswith("K=1 GT FSC-AUC correctness gate failed")
    assert cases[0].failure_reason == cases[0].notes[0]


def test_per_case_noctf_summary_uses_sign_invariant_auc_when_marked(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "6_noctf"
    case.mkdir(parents=True)
    (case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "ok",
                    "sign_ambiguity": {"allow_global_sign": True, "reason": "dataset_params_option=noctf"},
                    "metrics": {
                        "recovar_merged_vs_gt": {
                            "fsc_auc": -0.005,
                            "fsc_auc_sign_invariant": 0.005,
                            "corr": -0.2,
                            "abs_corr": 0.2,
                        },
                        "relion_merged_vs_gt": {
                            "fsc_auc": 0.004,
                            "fsc_auc_sign_invariant": 0.004,
                        },
                    },
                    "notes": [
                        "K=1 no-CTF fixture: global sign is ambiguous; required GT FSC-AUC gate uses sign-invariant FSC-AUC while signed metrics are still reported"
                    ],
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "ok"
    assert cases[0].fsc_auc_vs_gt == 0.005
    assert cases[0].relion_fsc_auc_vs_gt == 0.004
    assert abs(cases[0].fsc_auc_delta_vs_relion - 0.001) < 1e-12
    assert cases[0].map_corr_vs_gt == 0.2


def test_per_case_normal_summary_keeps_signed_auc(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "1_ctf"
    case.mkdir(parents=True)
    (case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "failed",
                    "metrics": {
                        "recovar_merged_vs_gt": {
                            "fsc_auc": -0.005,
                            "fsc_auc_sign_invariant": 0.005,
                            "corr": -0.2,
                            "abs_corr": 0.2,
                        },
                        "relion_merged_vs_gt": {
                            "fsc_auc": 0.004,
                            "fsc_auc_sign_invariant": 0.004,
                        },
                    },
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "failed"
    assert cases[0].fsc_auc_vs_gt == -0.005
    assert cases[0].relion_fsc_auc_vs_gt == 0.004
    assert abs(cases[0].fsc_auc_delta_vs_relion + 0.009) < 1e-12
    assert cases[0].map_corr_vs_gt == -0.2


def test_recovar_final_all_data_ran_false_is_not_missing_metadata(tmp_path):
    root = tmp_path / "em_k4_mini"
    case = root / "cases" / "5_k4"
    (case / "recovar").mkdir(parents=True)
    (case / "case_config.json").write_text(json.dumps({"index": 5, "name": "k4", "n_classes": 4}))
    (case / "summary_metrics.json").write_text(
        json.dumps({"kclass": {"status": "ok", "metrics": {"recovar_mean_vs_gt": {"fsc_auc": 0.2}}}})
    )
    np.savez(
        case / "recovar" / "refinement_results.npz",
        n_iterations=np.asarray(8),
        convergence_has_converged=np.asarray(False),
        final_all_data_ran=np.asarray(False),
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].recovar_final_all_data_ran is False
    assert cases[0].recovar_final_all_data_fsc_auc is None
    assert cases[0].recovar_has_final_all_data_poses is False
    assert "missing RECOVAR fsc_final_all_data" not in cases[0].notes
    assert "missing RECOVAR final all-data pose arrays" not in cases[0].notes
    assert (
        "RECOVAR final_class maps are last numbered iteration maps; final all-data did not run"
        in cases[0].notes
    )


def test_kclass_final_all_data_without_convergence_is_diagnostic(tmp_path):
    root = tmp_path / "em_k4_mini"
    case = root / "cases" / "5_k4"
    (case / "recovar").mkdir(parents=True)
    (case / "case_config.json").write_text(json.dumps({"index": 5, "name": "k4", "n_classes": 4}))
    (case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k4": {
                    "status": "ok",
                    "metrics": {
                        "recovar_vs_gt": {"mean_fsc_auc": 0.12},
                        "relion_vs_gt": {"mean_fsc_auc": 0.19},
                    },
                }
            }
        )
    )
    np.savez(
        case / "recovar" / "refinement_results.npz",
        n_iterations=np.asarray(8),
        convergence_has_converged=np.asarray(False),
        final_all_data_ran=np.asarray(True),
        fsc_final_all_data=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        best_rotation_eulers_final_all_data_by_image=np.zeros((2, 3), dtype=np.float32),
        best_translations_final_all_data_by_image=np.zeros((2, 2), dtype=np.float32),
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "diagnostic_ok"
    assert cases[0].recovar_final_all_data_ran is True
    assert cases[0].recovar_convergence_has_converged is False
    assert any("final all-data ran without convergence" in note for note in cases[0].notes)
    assert any("iteration-cap K-class GT FSC-AUC evidence" in note for note in cases[0].notes)
    assert not any("RELION-vs-GT metrics may use final maps" in note for note in cases[0].notes)


def test_discovery_ignores_generated_summaries_directory(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "1_baseline"
    case.mkdir(parents=True)
    (case / "summary_metrics.json").write_text(
        json.dumps({"k1": {"status": "ok", "metrics": {"recovar_merged_vs_gt": {"fsc_auc": 0.7}}}})
    )
    generated = root / "summaries" / "robustness_after_regen"
    generated.mkdir(parents=True)
    (generated / "summary.md").write_text("# generated monitor report\n")

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert [case.case_root for case in cases] == [case]


def test_cli_writes_k4_markdown_and_json(tmp_path):
    root = tmp_path / "em_k4_mini"
    case = root / "cases" / "3_radial"
    case.mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|init_radius|noise_scale_std|contrast_std|percent_outliers|put_extra_particles|image_offset_n_std|time_limit|mem|case_root|script|job_id",
                f"3|radial|3000|128|3.0|radial1|uniform|3|80|10|0|0|0|0|0|02:00:00|128G|{case}|{root / 'jobs' / 'em_k4_mini_3_radial.sh'}|444",
            ]
        )
        + "\n"
    )
    (case / "case_config.json").write_text(json.dumps({"index": 3, "name": "radial", "n_classes": 4}))
    (case / "relion_ref").mkdir()
    (case / "relion_ref" / "slurm_walltime.json").write_text(json.dumps({"external_wall_s": 41, "exit_status": 0}))
    (case / "kclass_gt_fsc_auc.json").write_text(json.dumps({"mean_fsc_auc_1_nyquist": 0.62}))
    (case / "relion_kclass_gt_fsc_auc.json").write_text(json.dumps({"mean_fsc_auc_1_nyquist": 0.59}))
    (case / "kclass_gt_fsc.json").write_text(
        json.dumps(
            {
                "primary": {
                    "per_class": [
                        {"corr": 0.8, "resolution_0143_A": 7.0, "fsc_vs_gt": [1.0, 0.7, 0.5]},
                        {"corr": 0.6, "resolution_0143_A": 9.0, "fsc_vs_gt": [1.0, 0.4, 0.2]},
                    ]
                }
            }
        )
    )
    (case / "relion_kclass_gt_fsc.json").write_text(
        json.dumps(
            {
                "primary": {
                    "per_class": [
                        {"corr": 0.75, "resolution_0143_A": 7.5, "fsc_vs_gt": [1.0, 0.65, 0.45]},
                        {"corr": 0.55, "resolution_0143_A": 9.5, "fsc_vs_gt": [1.0, 0.35, 0.15]},
                    ]
                }
            }
        )
    )

    out_md = tmp_path / "summary.md"
    out_json = tmp_path / "summary.json"
    assert summarizer.main([str(root), "--output-markdown", str(out_md), "--output-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text())
    assert payload["schema"] == "em_robustness_matrix_summary_v1"
    assert payload["cases"][0]["case_id"] == "3"
    assert payload["cases"][0]["n_classes"] == 4
    assert payload["cases"][0]["status"] == "ok"
    assert payload["cases"][0]["fsc_auc_vs_gt"] == 0.62
    assert payload["cases"][0]["relion_wall_s"] == 41
    assert payload["cases"][0]["relion_fsc_auc_vs_gt"] == 0.59
    assert abs(payload["cases"][0]["fsc_auc_delta_vs_relion"] - 0.03) < 1e-12
    assert payload["cases"][0]["final_resolution_A"] == 8.0
    assert payload["cases"][0]["map_corr_vs_gt"] == 0.7
    markdown = out_md.read_text()
    assert "RECOVAR FSC AUC vs GT" in markdown
    assert "RELION FSC AUC vs GT" in markdown
    assert "Delta vs RELION" in markdown


def test_kclass_curve_only_reports_use_integrated_fsc_auc_before_assignment_score(tmp_path):
    root = tmp_path / "em_kclass_curve_only"
    case = root / "cases" / "1_k2"
    case.mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|script|job_id",
                f"1|k2|10000|128|{case}|{root / 'jobs' / 'job.sh'}|555",
            ]
        )
        + "\n"
    )
    (case / "case_config.json").write_text(json.dumps({"index": 1, "name": "k2", "n_classes": 2}))
    (case / "kclass_gt_fsc.json").write_text(
        json.dumps(
            {
                "primary": {
                    "best_mean_fsc_1_8": 0.1,
                    "per_class": [
                        {"corr": 0.8, "resolution_0143_A": 7.0, "fsc_vs_gt": [1.0, 0.2, 0.6, 0.6]},
                        {"corr": 0.7, "resolution_0143_A": 8.0, "fsc_vs_gt": [1.0, 0.4, 0.6, 0.8]},
                    ],
                }
            }
        )
    )
    (case / "relion_kclass_gt_fsc.json").write_text(
        json.dumps(
            {
                "primary": {
                    "best_mean_fsc_1_8": 0.95,
                    "per_class": [
                        {"corr": 0.75, "resolution_0143_A": 7.5, "fsc_vs_gt": [1.0, 0.9, 0.2, 0.2]},
                        {"corr": 0.65, "resolution_0143_A": 8.5, "fsc_vs_gt": [1.0, 0.8, 0.2, 0.2]},
                    ],
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "ok"
    assert abs(cases[0].fsc_auc_vs_gt - 0.55) < 1e-12
    assert abs(cases[0].relion_fsc_auc_vs_gt - 0.3625) < 1e-12
    assert cases[0].fsc_auc_vs_gt > cases[0].relion_fsc_auc_vs_gt


def test_cli_deduplicates_repeated_root_arguments(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "1_baseline"
    case.mkdir(parents=True)
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"1|baseline|3000|128|{case}|111",
            ]
        )
        + "\n"
    )

    out_md = tmp_path / "summary.md"
    out_json = tmp_path / "summary.json"
    assert (
        summarizer.main(
            [
                str(root),
                str(root),
                "--output-markdown",
                str(out_md),
                "--output-json",
                str(out_json),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text())
    assert payload["scratch_roots"] == [str(root.resolve())]
    assert out_md.read_text().count(f"- `{root.resolve()}`") == 1


def test_slurm_cancelled_accounting_marks_case_cancelled(tmp_path, monkeypatch):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "2_cancelled"
    root.mkdir()
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"2|cancelled|3000|128|{case}|222",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(summarizer, "collect_slurm_accounting", lambda _cases: {"222": ("CANCELLED by 230216", "0:0")})

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "cancelled"
    assert cases[0].slurm_state == "CANCELLED by 230216"
    assert cases[0].slurm_exit_code == "0:0"
    assert "Slurm job 222 was cancelled" in cases[0].notes


def test_relion_bind_error_cpp_build_line_is_not_failure(tmp_path, monkeypatch):
    root = tmp_path / "em_kclass"
    case = root / "cases" / "6_ribo_k4_50k"
    script = root / "jobs" / "em_kclass_matrix_6_ribo_k4_50k.sh"
    case.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|script|job_id",
                f"6|ribo_k4_50k|50000|256|{case}|{script}|666",
            ]
        )
        + "\n"
    )
    (root / "em_kclass_matrix_6_ribo_k4_50k.out").write_text(
        "\n".join(
            [
                "[ 55%] Building CXX object CMakeFiles/_relion_bind_core.dir/src/error.cpp.o",
                "[100%] Built target _relion_bind_core",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(summarizer, "collect_slurm_accounting", lambda _cases: {})

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status in {"pending", "running"}
    assert cases[0].failure_reason is None
    assert cases[0].failure_log is None


def test_k4_aggregate_summary_promotes_relion_gt_fsc_auc(tmp_path):
    root = tmp_path / "em_k4_mini"
    case = root / "cases" / "3_radial"
    case.mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|script|job_id",
                f"3|radial|3000|128|{case}|{root / 'jobs' / 'em_k4_mini_3_radial.sh'}|444",
            ]
        )
        + "\n"
    )
    (root / "k4_mini_summary.json").write_text(
        json.dumps(
            [
                {
                    "index": 3,
                    "case": "radial",
                    "status": "completed",
                    "job_id": "444",
                    "path": str(case),
                    "wall_s": 31.0,
                    "relion_wall_s": 41.0,
                    "mean_fsc_auc_1_nyquist": 0.62,
                    "mean_fsc_auc_1_16": 0.61,
                    "relion_mean_fsc_auc_1_nyquist": 0.59,
                    "relion_mean_fsc_auc_1_16": 0.58,
                }
            ]
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "ok"
    assert cases[0].fsc_auc_vs_gt == 0.62
    assert cases[0].relion_fsc_auc_vs_gt == 0.59


def test_k4_summary_metrics_mean_fsc_auc_is_primary_quality_metric(tmp_path):
    root = tmp_path / "em_k4_mini"
    case = root / "cases" / "3_radial"
    case.mkdir(parents=True)
    (case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k4": {
                    "status": "completed",
                    "metrics": {
                        "recovar_vs_gt": {"mean_fsc_auc": 0.72, "mean_corr": 0.81},
                        "relion_vs_gt": {"mean_fsc_auc": 0.69, "mean_corr": 0.78},
                    },
                    "notes": [],
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "ok"
    assert cases[0].fsc_auc_vs_gt == 0.72
    assert cases[0].relion_fsc_auc_vs_gt == 0.69
    assert cases[0].map_corr_vs_gt == 0.81


def test_summary_metrics_prefers_completed_k4_over_pending_k1_section(tmp_path):
    root = tmp_path / "em_k4_completion"
    root.mkdir()
    (root / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "pending",
                    "metrics": {},
                    "timing": {"relion_walltime_s": 12695},
                    "notes": ["K=1 runtime default guard: missing refinement_results.npz"],
                },
                "k4": {
                    "status": "ok",
                    "metrics": {
                        "recovar_vs_gt": {"mean_fsc_auc": 0.285, "mean_corr": 0.71},
                        "relion_vs_gt": {"mean_fsc_auc": 0.262, "mean_corr": 0.69},
                    },
                    "timing": {"recovar_walltime_s": 17105, "relion_walltime_s": 7771},
                    "notes": [],
                },
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "ok"
    assert cases[0].fsc_auc_vs_gt == 0.285
    assert cases[0].relion_fsc_auc_vs_gt == 0.262
    assert cases[0].wall_s == 17105
    assert cases[0].relion_wall_s == 7771
    assert cases[0].notes == []


def test_nondefault_completion_failure_is_diagnostic_failed(tmp_path):
    root = tmp_path / "em_k4_highmem_budget_patch_20260629"
    root.mkdir()
    (root / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k4": {
                    "status": "failed",
                    "metrics": {"relion_vs_gt": {"mean_fsc_auc": 0.262}},
                    "notes": [
                        "K=4 runtime default guard: missing refinement_results.npz",
                        "K=4 was selected as required, but required artifacts or metrics are missing",
                    ],
                }
            }
        )
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "diagnostic_failed"
    assert "missing refinement_results.npz" in cases[0].failure_reason


def test_diag_root_running_case_is_diagnostic_running(tmp_path):
    root = tmp_path / "em_kclass_k2_diag_mstep"
    case = root / "cases" / "1_ribo_k2"
    (case / "recovar").mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|script|job_id",
                f"1|ribo_k2|10000|128|{case}|{root / 'jobs' / 'job.sh'}|444",
            ]
        )
        + "\n"
    )
    (case / "recovar" / "run_full_refinement.log").write_text(
        "2026-06-29 === RELION Iteration 1/20: current_size=40\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "diagnostic_running"


def test_dedupe_keeps_default_and_diagnostic_probe_separate(tmp_path):
    default_root = tmp_path / "em_kclass_remaining_axes"
    probe_root = tmp_path / "em_kclass_case6_true_forced_sparse2"
    dense_probe_root = tmp_path / "em_kclass_case6_dense_pass2_probe"
    lazy_probe_root = tmp_path / "em_kclass_case6_lazy_mask_retry2"
    default_case = default_root / "cases" / "6_ribo_k4"
    probe_case = probe_root / "cases" / "6_ribo_k4"
    dense_probe_case = dense_probe_root / "cases" / "6_ribo_k4"
    lazy_probe_case = lazy_probe_root / "cases" / "6_ribo_k4"
    default_case.mkdir(parents=True)
    probe_case.mkdir(parents=True)
    dense_probe_case.mkdir(parents=True)
    lazy_probe_case.mkdir(parents=True)
    for root, case, job_id in (
        (default_root, default_case, "111"),
        (probe_root, probe_case, "222"),
        (dense_probe_root, dense_probe_case, "333"),
        (lazy_probe_root, lazy_probe_case, "444"),
    ):
        (root / "case_table.tsv").write_text(
            "\n".join(
                [
                    "index|name|n_images|grid|case_root|script|job_id",
                    f"6|ribo_k4|50000|256|{case}|{root / 'jobs' / 'job.sh'}|{job_id}",
                ]
            )
            + "\n"
        )
        (case / "recovar").mkdir()
        (case / "recovar" / "run_full_refinement.log").write_text(
            "2026-06-29 === RELION Iteration 1/4: current_size=40\n"
        )

    cases = summarizer.discover_cases(
        [default_root, probe_root, dense_probe_root, lazy_probe_root],
        max_excerpt_lines=8,
        dedupe_case_reruns=True,
    )
    roots = {(case.status, case.scratch_root.name) for case in cases}

    assert roots == {
        ("running", default_root.name),
        ("diagnostic_running", lazy_probe_root.name),
    }


def test_relion_active_case_is_reported_running(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "1_baseline"
    case.mkdir(parents=True)
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"1|baseline|100000|256|{case}|111",
            ]
        )
        + "\n"
    )
    (case / "relion_autorefine.log").write_text(
        "\n".join(
            [
                "Auto-refine: Iteration= 4",
                "Auto-refine: Resolution= 11.3333 (no gain for 0 iter)",
                "Expectation iteration 4",
                "4.52/11.77 min ............................................................ done",
            ]
        )
        + "\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].relion_iteration == 4
    assert cases[0].relion_resolution_A == 11.3333
    assert cases[0].relion_latest_progress == "4.52/11.77 min"


def test_duplicate_case_prefers_later_requested_root(tmp_path):
    old_root = tmp_path / "old_matrix"
    new_root = tmp_path / "rerun_matrix"
    old_case = old_root / "cases" / "15_small_outliers"
    new_case = new_root / "cases" / "15_small_outliers"
    old_case.mkdir(parents=True)
    (new_case / "recovar").mkdir(parents=True)
    (old_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"15|small_outliers|3000|128|{old_case}|111",
            ]
        )
        + "\n"
    )
    (new_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"15|small_outliers|3000|128|{new_case}|222",
            ]
        )
        + "\n"
    )
    (old_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "failed",
                    "notes": ["old failed run"],
                }
            }
        )
    )
    (new_case / "recovar" / "run_full_refinement.log").write_text(
        "=== RELION Iteration 3/10: current_size=64\n"
    )

    cases = summarizer.discover_cases([old_root, new_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_id == "15"
    assert cases[0].case_name == "small_outliers"
    assert cases[0].case_root == new_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "running"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_duplicate_case_prefers_active_rerun_over_older_completed_metrics(tmp_path):
    old_root = tmp_path / "old_completed"
    active_root = tmp_path / "active_rerun"
    old_case = old_root / "cases" / "1_baseline"
    active_case = active_root / "cases" / "1_baseline"
    old_case.mkdir(parents=True)
    (active_case / "recovar").mkdir(parents=True)

    (old_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"1|baseline|10000|128|{old_case}|111",
            ]
        )
        + "\n"
    )
    (active_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"1|baseline|10000|128|{active_case}|222",
            ]
        )
        + "\n"
    )
    (old_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "kclass": {
                    "status": "ok",
                    "metrics": {"recovar_mean_vs_gt": {"fsc_auc": 0.3}},
                }
            }
        )
    )
    (active_case / "recovar" / "run_full_refinement.log").write_text(
        "=== RELION Iteration 3/5: current_size=48\n"
    )

    cases = summarizer.discover_cases([old_root, active_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_root == active_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "running"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_duplicate_case_prefers_pending_rerun_over_older_completed_metrics(tmp_path):
    old_root = tmp_path / "old_completed"
    pending_root = tmp_path / "pending_rerun"
    old_case = old_root / "cases" / "5_large_grid"
    pending_case = pending_root / "cases" / "5_large_grid"
    old_case.mkdir(parents=True)
    pending_root.mkdir(parents=True)

    (old_root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_classes|n_images|grid|noise_level|noise_model|poses|seed|pdb_bfactor|init_radius|noise_scale_std|contrast_std|volume_radius|image_offset_n_std|percent_outliers|max_iter|class_distribution|time_limit|mem|case_root|script|job_id",
                f"5|large_grid|4|50000|256|1|white|uniform|2805|80|10|0|0|0.7|0|0|8|uniform|18:00:00|500G|{old_case}|{old_root / 'jobs' / 'old.sh'}|111",
            ]
        )
        + "\n"
    )
    (pending_root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_classes|n_images|grid|noise_level|noise_model|poses|seed|pdb_bfactor|init_radius|noise_scale_std|contrast_std|volume_radius|image_offset_n_std|percent_outliers|max_iter|class_distribution|time_limit|mem|case_root|script|job_id",
                f"5|large_grid|4|50000|256|1|white|uniform|2805|80|10|0|0|0.7|0|0|20|uniform|24:00:00|500G|{pending_case}|{pending_root / 'jobs' / 'pending.sh'}|222",
            ]
        )
        + "\n"
    )
    (old_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "kclass": {
                    "status": "ok",
                    "metrics": {"recovar_mean_vs_gt": {"fsc_auc": 0.19}},
                }
            }
        )
    )

    cases = summarizer.discover_cases([old_root, pending_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_root == pending_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "pending"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_duplicate_case_prefers_later_pending_rerun_over_stale_failure(tmp_path):
    stale_root = tmp_path / "stale_failure"
    pending_root = tmp_path / "pending_rerun"
    stale_case = stale_root / "cases" / "22_severe_outliers"
    pending_case = pending_root / "cases" / "22_severe_outliers"
    stale_case.mkdir(parents=True)
    pending_root.mkdir(parents=True)

    (stale_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"22|severe_outliers|3000|128|{stale_case}|111",
            ]
        )
        + "\n"
    )
    (pending_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"22|severe_outliers|3000|128|{pending_case}|222",
            ]
        )
        + "\n"
    )
    (stale_case / "summary_metrics.json").write_text(
        json.dumps({"k1": {"status": "failed", "notes": ["old RELION baseline missing"]}})
    )

    cases = summarizer.discover_cases([stale_root, pending_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_root == pending_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "pending"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_duplicate_case_prefers_active_rerun_over_later_stale_failure(tmp_path):
    active_root = tmp_path / "active_rerun"
    stale_root = tmp_path / "stale_failure"
    active_case = active_root / "cases" / "5_very_high_noise"
    stale_case = stale_root / "cases" / "5_very_high_noise"
    (active_case / "recovar").mkdir(parents=True)
    (stale_case / "recovar").mkdir(parents=True)

    (active_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"5|very_high_noise|100000|256|{active_case}|222",
            ]
        )
        + "\n"
    )
    (stale_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"5|very_high_noise|100000|256|{stale_case}|111",
            ]
        )
        + "\n"
    )
    (active_case / "recovar" / "run_full_refinement.log").write_text(
        "=== RELION Iteration 2/16: current_size=64\n"
    )
    (stale_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "failed",
                    "notes": ["old SIGBUS run"],
                }
            }
        )
    )

    cases = summarizer.discover_cases([active_root, stale_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_root == active_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "running"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_duplicate_case_prefers_active_central_log_rerun_over_stale_failure(tmp_path):
    stale_root = tmp_path / "stale_failure"
    active_root = tmp_path / "active_rerun"
    stale_case = stale_root / "cases" / "32_mid_kent"
    active_case = active_root / "cases" / "32_mid_kent"
    stale_case.mkdir(parents=True)
    active_case.mkdir(parents=True)

    (stale_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"32|mid_kent|10000|128|{stale_case}|111",
            ]
        )
        + "\n"
    )
    (active_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"32|mid_kent|10000|128|{active_case}|222",
            ]
        )
        + "\n"
    )
    (stale_case / "summary_metrics.json").write_text(
        json.dumps({"k1": {"status": "failed", "notes": ["old quality failure"]}})
    )
    script = active_root / "jobs" / "em_k1_matrix_32_mid_kent.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")
    (active_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id|script",
                f"32|mid_kent|10000|128|{active_case}|222|{script}",
            ]
        )
        + "\n"
    )
    (active_root / "em_k1_matrix_32_mid_kent.out").write_text("Installing collected packages: recovar\n")

    cases = summarizer.discover_cases([stale_root, active_root], max_excerpt_lines=8, dedupe_case_reruns=True)

    assert len(cases) == 1
    assert cases[0].case_root == active_case
    assert cases[0].job_id == "222"
    assert cases[0].status == "running"
    assert "omitted 1 duplicate row" in "; ".join(cases[0].notes)


def test_oom_guard_path_in_build_log_does_not_mark_active_case_failed(tmp_path, monkeypatch):
    root = tmp_path / "em_k1_case22_oom_guard_retry"
    case = root / "cases" / "22_small_severe_outliers"
    case.mkdir(parents=True)
    script = root / "jobs" / "em_k1_matrix_22_small_severe_outliers.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id|script",
                f"22|small_severe_outliers|3000|128|{case}|10500285|{script}",
            ]
        )
        + "\n"
    )
    (root / "em_k1_matrix_22_small_severe_outliers.out").write_text(
        "/usr/local/cuda/bin/nvcc -o "
        f"{root}/cuda/libcuda_backproject.so cuda_backproject.cu\n"
    )
    monkeypatch.setattr(summarizer, "collect_slurm_accounting", lambda _cases: {})

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert summarizer.first_failure_line([f"nvcc -o {root}/cuda/libcuda_backproject.so"]) is None
    assert summarizer.first_failure_line(["RESOURCE_EXHAUSTED: Out of memory while allocating"]) is not None
    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].failure_reason is None


def test_recovar_active_case_reports_iteration_progress(tmp_path):
    root = tmp_path / "em_k1_robustness"
    case = root / "cases" / "1_baseline"
    (case / "recovar").mkdir(parents=True)
    (root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|case_job_id",
                f"1|baseline|100000|256|{case}|111",
            ]
        )
        + "\n"
    )
    (case / "recovar" / "run_full_refinement.log").write_text(
        "\n".join(
            [
                "=== RELION Iteration 2/15: current_size=100, healpix_order=3, local_search=False ===",
                "Sparse pass-2 bucket group start: bucket_size=16 chunks=909 images=49969",
                "Sparse pass-2 bucket group done: bucket_size=16 chunks=909 images=49969 wall=132.9s images/s=376.0",
                "Sparse pass-2 (bucketed): 49969 images, 909 buckets, 132.91s E+M",
                "RELION Iteration 2: current_size=100, pixel_res=15.12, res=16.5 A, ave_Pmax=1.0000, healpix_order=3, converged=False, time=123.4s",
            ]
        )
        + "\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].recovar_iteration == 2
    assert cases[0].recovar_total_iterations == 15
    assert cases[0].recovar_current_size == 100
    assert cases[0].recovar_last_iteration_time_s == 123.4
    assert cases[0].recovar_latest_stage == "iter 2/15 complete in 123.4s"


def test_recovar_active_kclass_case_reports_sparse_fused_progress(tmp_path):
    root = tmp_path / "em_kclass"
    case = root / "cases" / "5_ribo_k4_50k"
    (case / "recovar").mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|job_id|K",
                f"5|ribo_k4_50k|50000|256|{case}|555|4",
            ]
        )
        + "\n"
    )
    (case / "recovar" / "run_full_refinement.log").write_text(
        "\n".join(
            [
                "=== RELION Iteration 8/8: current_size=72, healpix_order=1, local_search=False ===",
                "Sparse fused K-class pass-2 bucket group done: mode=compact_pair pair_bucket_size=1024 chunks=149 images=11485 wall=61.7s images/s=186.3",
                "Sparse fused K-class pass-2 bucket group start: mode=compact_pair pair_bucket_size=4096 chunks=996 images=19025",
            ]
        )
        + "\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].recovar_iteration == 8
    assert cases[0].recovar_total_iterations == 8
    assert cases[0].recovar_latest_stage == "sparse pass-2 bucket 4096: 996 chunks/19025 images"


def test_recovar_active_kclass_case_reports_sparse_fused_done(tmp_path):
    root = tmp_path / "em_kclass"
    case = root / "cases" / "5_ribo_k4_50k"
    (case / "recovar").mkdir(parents=True)
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|job_id|K",
                f"5|ribo_k4_50k|50000|256|{case}|555|4",
            ]
        )
        + "\n"
    )
    (case / "recovar" / "run_full_refinement.log").write_text(
        "\n".join(
            [
                "=== RELION Iteration 8/8: current_size=72, healpix_order=1, local_search=False ===",
                "Sparse fused K-class pass-2: 50000 images, 4 classes, 5730 buckets, 988.64s E+M; median local rot=96",
            ]
        )
        + "\n"
    )

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].recovar_latest_stage == "sparse pass-2 done: 50000 images/5730 buckets/988.64s"


def test_kclass_recovar_metric_only_stays_running_when_relion_eval_expected(tmp_path, monkeypatch):
    root = tmp_path / "em_kclass"
    case = root / "cases" / "5_ribo_k4_50k"
    case.mkdir(parents=True)
    (case / "relion_ref").mkdir()
    (root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|job_id|K",
                f"5|ribo_k4_50k|50000|256|{case}|555|4",
            ]
        )
        + "\n"
    )
    (case / "kclass_gt_fsc.json").write_text(
        json.dumps(
            {
                "primary": {
                    "per_class": [
                        {"corr": 0.7, "resolution_0143_A": 8.0, "fsc_vs_gt": [1.0, 0.6, 0.3]},
                    ]
                }
            }
        )
    )
    monkeypatch.setattr(summarizer, "collect_slurm_accounting", lambda _cases: {"555": ("RUNNING", "0:0")})

    cases = summarizer.discover_cases([root], max_excerpt_lines=8)

    assert len(cases) == 1
    assert cases[0].status == "running"
    assert cases[0].fsc_auc_vs_gt is not None
    assert cases[0].relion_fsc_auc_vs_gt is None
    assert cases[0].fsc_auc_delta_vs_relion is None


def test_supersedes_never_started_duplicate_when_completed_copy_exists(tmp_path):
    ok_root = tmp_path / "ok_root"
    failed_root = tmp_path / "failed_root"
    missing_root = tmp_path / "missing_root"
    ok_case = ok_root / "cases" / "26_tiny_severe"
    failed_case = failed_root / "cases" / "26_tiny_severe"
    missing_case = missing_root / "cases" / "26_tiny_severe"
    missing_script = missing_root / "jobs" / "em_k4_mini_26_tiny_severe.sh"
    ok_case.mkdir(parents=True)
    failed_case.mkdir(parents=True)
    missing_script.parent.mkdir(parents=True)

    (ok_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std|case_root|case_job_id",
                f"26|tiny_severe|1000|128|5.0|radial1|nonuniform|26|80|0|0|0.7|-|02:00:00|96G|250|0|0|0|0|{ok_case}|local",
            ]
        )
        + "\n"
    )
    (failed_root / "selected_cases.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|noise_level|noise_model|dataset_params_option|seed|pdb_bfactor|noise_scale_std|contrast_std|volume_radius|relion_bg_radius_px|time_limit|mem|streaming_chunk|streaming_mmap|percent_outliers|put_extra_particles|image_offset_n_std|case_root|case_job_id",
                f"26|tiny_severe|1000|128|5.0|radial1|nonuniform|26|80|0|0|0.7|-|02:00:00|96G|250|0|0|0|0|{failed_case}|10286798",
            ]
        )
        + "\n"
    )
    (ok_case / "recovar").mkdir()
    (ok_case / "recovar" / "slurm_walltime.json").write_text(json.dumps({"external_wall_s": 10, "exit_status": 0}))
    (ok_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "ok",
                    "metrics": {"recovar_merged_vs_gt": {"fsc_auc": 0.5}},
                    "notes": [],
                }
            }
        )
    )
    (failed_case / "summary_metrics.json").write_text(
        json.dumps(
            {
                "k1": {
                    "status": "failed",
                    "metrics": {},
                    "notes": ["K=1 runtime default guard: missing refinement_results.npz"],
                }
            }
        )
    )
    (failed_case / "recovar").mkdir()
    (failed_case / "recovar" / "slurm_walltime.json").write_text(
        json.dumps({"slurm_job_id": "10286798", "external_wall_s": 20, "exit_status": 1})
    )
    (failed_case / "recovar" / "run_full_refinement.log").write_text(
        "start\nRESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.66GiB\n"
    )
    (missing_root / "k4_mini_summary.json").write_text(
        json.dumps(
            [
                {
                    "index": 26,
                    "name": "tiny_severe",
                    "case_root": str(missing_case),
                    "status": "missing_metrics",
                    "job_id": "10294045",
                }
            ]
        )
    )
    (missing_root / "case_table.tsv").write_text(
        "\n".join(
            [
                "index|name|n_images|grid|case_root|script|job_id",
                f"26|tiny_severe|1000|128|{missing_case}|{missing_script}|10294045",
            ]
        )
        + "\n"
    )
    (missing_root / "em_k4_mini_26_tiny_severe.err").write_text(
        "error: package directory 'recovar/relion_bind/build/CMakeFiles/CMakeScratch' does not exist\n"
    )

    cases = summarizer.discover_cases([ok_root, failed_root, missing_root], max_excerpt_lines=8)
    statuses = {str(case.case_root): case.status for case in cases}

    assert statuses[str(ok_case)] == "ok"
    assert statuses[str(failed_case)] == "superseded"
    assert statuses[str(missing_case)] == "superseded"
