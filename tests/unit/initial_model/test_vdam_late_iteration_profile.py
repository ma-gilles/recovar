import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.data_io.image_loader import ImageLoader
from scripts.run_vdam_late_iteration_profile import (
    _all_optimized_q32_environment,
    _capture_raw_image_cache_loads,
    _process_resource_delta,
    _profile_metadata,
    _recovar_argv,
    _validate_profile_contract_environment,
    _validate_profile_environment,
    _validate_profile_execution_contract,
)
from scripts.summarize_vdam_nsys_sqlite import summarize

ROOT = Path(__file__).resolve().parents[3]


class _AuditLoader(ImageLoader):
    def _load(self, indices):
        return np.ones((len(indices), self.image_size, self.image_size), dtype=self._dtype)


def test_late_profile_cache_audit_records_load_all_and_restores_method():
    loader = _AuditLoader(num_images=3, image_size=2, dtype=np.float32)
    original = ImageLoader.load_all

    with _capture_raw_image_cache_loads(True) as events:
        loader.load_all()

    assert ImageLoader.load_all is original
    assert events is not None and len(events) == 1
    event = events[0]
    assert {
        key: event[key]
        for key in (
            "loader_type",
            "num_images",
            "image_size",
            "dtype",
            "estimated_bytes",
            "cached_before",
            "cached_after",
            "cached_nbytes",
        )
    } == {
        "loader_type": f"{_AuditLoader.__module__}.{_AuditLoader.__qualname__}",
        "num_images": 3,
        "image_size": 2,
        "dtype": "<f4",
        "estimated_bytes": 48,
        "cached_before": False,
        "cached_after": True,
        "cached_nbytes": 48,
    }
    assert event["elapsed_s"] >= 0.0
    assert event["current_rss_after_bytes"] - event["current_rss_before_bytes"] == event[
        "current_rss_delta_bytes"
    ]
    assert event["high_water_rss_after_bytes"] - event["high_water_rss_before_bytes"] == event[
        "high_water_rss_delta_bytes"
    ]
    assert event["high_water_rss_delta_bytes"] >= 0
    assert event["cached_shape"] == [3, 2, 2]
    assert event["cached_dtype"] == "<f4"
    assert event["cached_c_contiguous"] is True
    assert event["cached_writeable"] is True
    assert event["loader_topology"] == {
        "mapped_rows": 0,
        "mapped_files": [],
        "mapped_file_count": 0,
        "mapping_unique_index_count": 0,
        "mapping_min_index": None,
        "mapping_max_index": None,
        "mapping_is_unique": True,
        "mapping_is_contiguous_set": False,
        "mapping_is_strictly_ascending": True,
        "mapping_mrc_indices_sha256": hashlib.sha256(b"").hexdigest(),
        "leaf_loader_count": 0,
        "leaf_loaders": [],
        "leaf_cached_before": [],
        "leaf_cached_after": [],
    }


def test_late_profile_cache_audit_can_be_disabled():
    original = ImageLoader.load_all
    with _capture_raw_image_cache_loads(False) as events:
        assert events is None
    assert ImageLoader.load_all is original


def test_late_profile_metadata_requires_exactly_one_diagnostic_iteration(tmp_path):
    prefix = tmp_path / "warm" / "run"
    prefix.parent.mkdir()
    meta_path = Path(f"{prefix}_it181_recovar_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "current_size": 128,
                "healpix_order": 3,
                "n_rotations": 294912,
                "n_translations": 116,
                "subset_size": 1000,
                "selected_particle_ids": list(range(1000)),
                "random_perturbation": 0.1,
                "vdam_iteration_profile_summary": {"expectation_time_s": 1.25},
            }
        )
    )
    Path(f"{prefix}_diagnostic_continuation.json").write_text(
        json.dumps(
            {
                "classification": "diagnostic_performance_only",
                "iteration": 180,
            }
        )
    )

    report = _profile_metadata(prefix, 181)

    assert report["schedule"]["n_rotations"] == 294912
    assert report["schedule"]["subset_size"] == 1000
    assert report["iteration_profile"]["expectation_time_s"] == 1.25

    metadata = json.loads(meta_path.read_text())
    del metadata["subset_size"]
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="lacks required schedule fields"):
        _profile_metadata(prefix, 181)

    metadata["subset_size"] = None
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="subset_size is invalid"):
        _profile_metadata(prefix, 181)

    metadata["subset_size"] = 1000.0
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="subset_size is invalid"):
        _profile_metadata(prefix, 181)

    metadata["subset_size"] = 999
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="does not match selected_particle_ids"):
        _profile_metadata(prefix, 181)

    metadata["subset_size"] = 1000
    meta_path.write_text(json.dumps(metadata))
    Path(f"{prefix}_it182_recovar_meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exactly one iteration"):
        _profile_metadata(prefix, 181)


def test_nsys_sqlite_summary_reports_invocations_shapes_and_busy_fraction(tmp_path):
    sqlite_path = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(sqlite_path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1, 'coarse_kernel'), (2, 'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER,
            end INTEGER,
            deviceId INTEGER,
            shortName INTEGER,
            gridX INTEGER,
            gridY INTEGER,
            gridZ INTEGER,
            blockX INTEGER,
            blockY INTEGER,
            blockZ INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (10, 30, 0, 1, 8, 1, 1, 256, 1, 1),
            (20, 40, 0, 1, 8, 1, 1, 256, 1, 1);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER,
            end INTEGER,
            nameId INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 100, 2);
        """
    )
    connection.commit()
    connection.close()

    report = summarize(sqlite_path)

    assert report["capture_span_source"] == "cuda_api"
    assert report["capture_span_ns"] == 100
    assert report["devices"]["0"]["kernel_count"] == 2
    assert report["devices"]["0"]["gpu_busy_ns"] == 30
    assert report["devices"]["0"]["gpu_idle_ns_within_capture"] == 70
    assert report["devices"]["0"]["gpu_busy_fraction_within_capture"] == pytest.approx(0.3)
    assert report["kernels"][0]["name"] == "coarse_kernel"
    assert report["kernels"][0]["count"] == 2
    assert report["kernel_signatures"][0]["shape"] == {
        "gridX": 8,
        "gridY": 1,
        "gridZ": 1,
        "blockX": 256,
        "blockY": 1,
        "blockZ": 1,
    }


def test_late_profile_resource_delta_reports_io_and_cpu_counters():
    before = {
        "user_cpu_s": 1.0,
        "system_cpu_s": 2.0,
        "minor_faults": 3,
        "major_faults": 4,
        "input_blocks": 5,
        "output_blocks": 6,
        "voluntary_context_switches": 7,
        "involuntary_context_switches": 8,
        "proc_io": {"read_bytes": 9, "syscr": 10},
    }
    after = {
        "user_cpu_s": 1.5,
        "system_cpu_s": 2.25,
        "minor_faults": 13,
        "major_faults": 4,
        "input_blocks": 15,
        "output_blocks": 16,
        "voluntary_context_switches": 17,
        "involuntary_context_switches": 18,
        "proc_io": {"read_bytes": 109, "syscr": 30},
    }

    delta = _process_resource_delta(before, after)

    assert delta["user_cpu_s"] == pytest.approx(0.5)
    assert delta["input_blocks"] == 10
    assert delta["proc_io"] == {"read_bytes": 100, "syscr": 20}


def test_late_profile_only_passes_nondefault_candidate_options(tmp_path):
    args = SimpleNamespace(
        checkpoint_iteration=180,
        input_star=tmp_path / "particles.star",
        nr_iter=200,
        random_seed=29,
        image_batch_size=500,
        data_dir=tmp_path,
        checkpoint_optimiser=tmp_path / "run_it180_optimiser.star",
        exact_local_bucket_radix=4,
        exact_local_physical_order_chunk_size=0,
        stable_fourier_window_shapes=False,
    )

    control = _recovar_argv(args=args, output_prefix=tmp_path / "control" / "run")
    assert "--exact-local-bucket-radix" not in control
    assert "--exact-local-physical-order-chunk-size" not in control

    args.exact_local_bucket_radix = 2
    args.exact_local_physical_order_chunk_size = 220
    candidate = _recovar_argv(args=args, output_prefix=tmp_path / "candidate" / "run")
    assert candidate[-4:] == [
        "--exact-local-bucket-radix",
        "2",
        "--exact-local-physical-order-chunk-size",
        "220",
    ]


def test_late_profile_environment_rejects_implicit_fused_gemm_conflict():
    with pytest.raises(
        ValueError,
        match=(
            "RECOVAR_K1_COARSE_FUSED_PROJECTOR=<effective default>.*"
            "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION=<effective default>"
        ),
    ):
        _validate_profile_environment(
            {"RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1"},
        )


@pytest.mark.parametrize(
    ("name", "enabled_value"),
    [
        ("RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION", "1"),
        ("RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL", "1"),
        ("RECOVAR_K1_COARSE_MULTISTREAM_WORKERS", "8"),
        ("RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE", "1"),
    ],
)
def test_late_profile_environment_rejects_every_explicit_gemm_selector(
    name,
    enabled_value,
):
    environment = {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        name: enabled_value,
    }
    with pytest.raises(ValueError, match=name):
        _validate_profile_environment(environment)


def test_late_profile_environment_accepts_explicit_q32_gemm_selectors():
    environment = {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
    }

    report = _validate_profile_environment(environment)

    assert report["resolved_backend"] == "gemm_macro"
    assert report["gemm_macro_requested"] is True
    assert all(
        value in {False, 0}
        for name, value in report["effective_selectors"].items()
        if name
        not in {
            "RECOVAR_K1_COARSE_GAUSSIAN_FFI",
            "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF",
        }
    )


def _complete_q32_profile_meta() -> dict:
    compact = {
        "enabled": True,
        "default_enabled": False,
        "compact_posterior_enabled": True,
        "compact_posterior_default_enabled": False,
        "selected_score_layout": "fixed_capacity_source16",
        "positive_only_scan_role": "correctness_oracle_not_runtime",
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "batch_count": 1,
        "certificate_chunk_count_per_batch": 1,
        "certificate_chunk_rows": 4608,
        "selected_rescore_batch_count": 1,
        "fallback_batch_count": 0,
        "selected_rescore_image_count": 200,
        "fallback_image_count": 0,
        "selected_source16_block_count": 800,
        "selected_exact_candidate_count": 473_600,
        "selected_score_table_capacity_candidates": 18_944_000,
        "dense_global_score_table_capacity_candidates": 85_248_000,
        "selected_score_table_capacity_bytes_f32": 75_776_000,
        "dense_global_score_table_capacity_bytes_f32": 340_992_000,
        "selected_to_dense_score_table_capacity_fraction": 2.0 / 9.0,
        "topology_full_to_compact_sha256": "a" * 64,
        "input_image_batch_size": 500,
        "requested_hybrid_image_batch_size": 200,
        "effective_image_batch_size": 200,
        "streamed_certificate_candidate_count_at_effective_batch": 200
        * 4608
        * 37,
        "actual_image_batch_sizes": [200],
        "physical_image_batch_sizes": [200],
    }
    exact = {
        "skip_generic_default_enabled": False,
        "skip_generic_requested": True,
        "skip_generic_effective": True,
        "exact_coarse_operands_effective": True,
        "exact_compact_preprocess_default_enabled": False,
        "exact_compact_preprocess_requested": True,
        "exact_compact_preprocess_effective": True,
        "generic_score_preprocess_count": 0,
        "exact_source_preprocess_count": 1,
        "generic_ctf_evaluation_count": 0,
        "generic_full_translation_count": 0,
        "generic_assembly_count": 0,
        "exact_assembly_count": 1,
        "translate_score_call_site_count": 1,
        "translate_score_call_count": 1,
        "downstream_operand_source": "exact_source_star",
        "diagnostic_operand_source": "exact_source_star",
        "raw_score_capture_changed": False,
        "generic_fallback_policy": "raise_before_generic_score_fallback",
        "skipped_generic_outputs": [
            "coarse_gaussian_shifted_corrected",
            "coarse_gaussian_pixel_weight",
            "coarse_gaussian_unshifted_corrected",
        ],
    }
    local = {
        "flat_local_rows_enabled": True,
        "stable_flat_row_capacity_enabled": True,
        "packed_local_projection_enabled": True,
        "defer_packed_vdam_enabled": True,
        "packed_vdam_reuses_flat_score_projection": True,
        "packed_final_noise_enabled": True,
        "packed_vdam_avoids_dense_noise_rows": True,
        "packed_final_noise_preserves_dense_scalar_order": True,
        "sum_packed_final_noise_rows": 800,
        "stable_fourier_window_shapes": True,
        "stable_fourier_window_quantum": 32,
        "logical_current_size": 84,
        "physical_current_size": 96,
        "logical_reconstruction_pixels": 2835,
        "physical_reconstruction_pixels": 3691,
        "n_windowed": 3690,
        "n_projection_windowed": 3691,
        "big_jit_projection_pixels": 3691,
        "chunk_flat_score_rows": [61_440],
        "chunk_padded_rotations": [61_440],
        "chunk_planned_padded_rotations": [61_440],
        "chunk_reconstruction_rows": [377],
        "chunk_nonzero_posterior_rows": [377],
        "sum_flat_score_rows": 61_440,
        "sum_padded_rows": 61_440,
        "sum_planned_padded_rows": 61_440,
        "sum_reconstruction_rows": 377,
        "sum_nonzero_posterior_rows": 377,
        "fused_pair_fine_score_enabled": False,
        "fused_pair_fine_score_default_enabled": False,
        "fused_pair_fine_uses_shared_compact_order": False,
        "fused_pair_fine_avoids_pair_pixel_gathers": False,
        "fused_pair_fine_restores_dense_posterior_order": False,
        "chunk_fused_pair_capacities": [],
        "chunk_fused_pair_counts": [],
        "chunk_fused_pair_dense_capacities": [],
        "sum_fused_pair_candidates": 0,
        "sum_fused_pair_capacity": 0,
        "sum_fused_pair_dense_capacity": 0,
        "fused_pair_valid_fraction_of_dense": 0.0,
        "fused_pair_padded_fraction_of_dense": 0.0,
        "coarse_gaussian_gemm_hybrid": compact,
        "exact_coarse_operand_assembly": exact,
    }
    return {
        "requested_stable_fourier_window_shapes": True,
        "effective_stable_fourier_window_shapes": True,
        "requested_stable_flat_row_capacity": True,
        "effective_stable_flat_row_capacity": True,
        "requested_fused_pair_fine_score": False,
        "effective_fused_pair_fine_score": False,
        "n_translations": 37,
        "halfset_0_profile_summary": local,
    }


def test_late_profile_complete_q32_contract_is_fail_closed():
    environment = _all_optimized_q32_environment()
    contract = _validate_profile_execution_contract(
        _complete_q32_profile_meta(),
        contract_mode="all_optimized_q32",
        image_shape=(128, 128),
        environment=environment,
    )

    stable = contract["all_optimized"]["stable_fourier"]
    assert contract["profile_exact"] is True
    assert stable["profiles"]["halfset_0_profile_summary"][
        "physical_current_size"
    ] == 96
    assert contract["image_batch"]["profiles"]["halfset_0_profile_summary"][
        "physical_image_batch_sizes"
    ] == [200]
    assert contract["fused_pair_fine"]["enabled"] is False
    assert contract["row_totals"]["profiles"]["halfset_0_profile_summary"][
        "totals"
    ]["sum_padded_rows"] == 61_440

    broken = _complete_q32_profile_meta()
    broken["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"][
        "compact_posterior_enabled"
    ] = False
    with pytest.raises(RuntimeError, match="compact_posterior_enabled"):
        _validate_profile_execution_contract(
            broken,
            contract_mode="all_optimized_q32",
            image_shape=(128, 128),
            environment=environment,
        )


def test_late_profile_contract_environment_rejects_mislabeled_runs():
    candidate = _all_optimized_q32_environment()
    assert candidate["RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"] == "1"
    assert candidate["RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE"] == "1"
    assert candidate["RECOVAR_K1_RELION_EXACT_COARSE_SKIP_GENERIC_OPERANDS"] == "1"
    assert candidate["RECOVAR_K1_RELION_EXACT_COMPACT_PREPROCESS"] == "1"
    assert candidate["RECOVAR_EXACT_LOCAL_FUSED_PAIR_FINE_SCORE"] == "0"
    assert candidate["RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM"] == "32"
    assert candidate["RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE"] == "200"

    missing_compact = dict(candidate)
    del missing_compact["RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR"]
    with pytest.raises(RuntimeError, match="qualified stack"):
        _validate_profile_contract_environment(
            "all_optimized_q32",
            missing_compact,
        )
    with pytest.raises(RuntimeError, match="requires candidate selectors to be absent"):
        _validate_profile_contract_environment("default", candidate)


def test_late_profile_slurm_gate_is_one_iteration_and_fail_closed():
    launcher = (ROOT / "scripts" / "run_vdam_late_iteration_profile.sbatch").read_text()
    gdb_commands = (ROOT / "scripts" / "vdam_relion_one_iteration.gdb").read_text()

    assert "#SBATCH --constraint=h100" in launcher
    assert "--capture-range=cudaProfilerApi" in launcher
    assert "--capture-range-end=stop" in launcher
    assert "EXPECTED_RELION_SHA256" in launcher
    assert "EXPECTED_RELION_BIND_SHA256" in launcher
    assert "profile_environment_preflight.json" in launcher
    assert launcher.index("profile_environment_preflight.json") < launcher.index(
        "native_plain_started=",
    )
    assert "VDAM_LATE_PROFILE_REUSE_NATIVE_ROOT" in launcher
    assert "EXPECTED_REUSED_NATIVE_NSYS_SHA256" in launcher
    assert 'AUDIT_RAW_IMAGE_CACHE=${AUDIT_RAW_IMAGE_CACHE:-0}' in launcher
    assert 'RECOVAR_COMMAND+=(--audit-raw-image-cache)' in launcher
    assert "VDAM_LATE_PROFILE_CONTRACT" in launcher
    assert "_all_optimized_q32_environment" in launcher
    assert '--execution-contract "${EXECUTION_CONTRACT}"' in launcher
    assert "recovar_execution_contract.json" in launcher
    assert "status --porcelain=v1 --untracked-files=no" in launcher
    assert "test ! -e" in launcher
    assert 'test ! -e "${NATIVE_PROFILE}/run_it' in launcher
    assert "--diagnostic_continue_optimiser" in (ROOT / "scripts" / "run_vdam_late_iteration_profile.py").read_text()
    assert "VDAM_GDB_FIRST_EXPECTATION" in gdb_commands
    assert "VDAM_GDB_SECOND_EXPECTATION" in gdb_commands
    assert "cudaProfilerStart" in gdb_commands
    assert "cudaProfilerStop" in gdb_commands
    assert "call (void) exit" not in gdb_commands
    assert "process_resources" in (ROOT / "scripts" / "run_vdam_late_iteration_profile.py").read_text()
