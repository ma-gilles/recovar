from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume import iteration_loop, k_class, local_em_engine
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed
from recovar.em.dense_single_volume.local_backprojection import compute_local_mstep_sums

pytestmark = pytest.mark.unit


def _capture_environment() -> dict[str, str]:
    return {
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR": "/tmp/device",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION": "5",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF": "1",
    }


def test_device_signature_scope_activates_only_target_numbered_half():
    env = _capture_environment()

    for iteration in range(1, 5):
        for half in (1, 2):
            assert not iteration_loop._bpref_device_signature_active_for_numbered_half(
                iteration=iteration,
                half=half,
                environ=env,
            )
    assert iteration_loop._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=1,
        environ=env,
    )
    assert not iteration_loop._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        environ=env,
    )
    assert not iteration_loop._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=1,
        final_all_data=True,
        environ=env,
    )


def test_device_signature_scope_rejects_missing_or_invalid_target():
    env = _capture_environment()
    for missing in (
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF",
    ):
        invalid = dict(env)
        invalid.pop(missing)
        with pytest.raises(RuntimeError, match="requires explicit positive"):
            iteration_loop._bpref_device_signature_active_for_numbered_half(
                iteration=1,
                half=1,
                environ=invalid,
            )

    for name, value in (
        ("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "0"),
        ("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "3"),
        ("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "not-an-int"),
    ):
        invalid = dict(env)
        invalid[name] = value
        with pytest.raises((ValueError, RuntimeError)):
            iteration_loop._bpref_device_signature_active_for_numbered_half(
                iteration=1,
                half=1,
                environ=invalid,
            )


def test_scoped_capture_ignores_all_process_flags_off_target(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    for name in (
        "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
        "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY",
        "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.setenv(name, "1")

    inactive = sparse_pass2_bucketed._scoped_bpref_diagnostic_flags(active=False)
    assert inactive == {
        "device_signature_configured": True,
        "sequential_translation_reduction": False,
        "per_particle_launches": False,
        "fused_atomics": False,
        "high_precision_operand_bundle": False,
    }
    assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
    with cuda_backproject.bpref_device_signature_scope(False):
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()

    active = sparse_pass2_bucketed._scoped_bpref_diagnostic_flags(active=True)
    assert all(value for name, value in active.items() if name != "device_signature_configured")
    with cuda_backproject.bpref_device_signature_scope(True):
        assert cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_scoped_device_capture_keeps_live_reduction_and_adjoint_modes_ordinary(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH", "1")
    flags = sparse_pass2_bucketed._scoped_bpref_diagnostic_flags(active=True)

    modes = sparse_pass2_bucketed._resolve_bpref_execution_modes(
        flags,
        device_signature_requested=True,
    )

    assert modes["shadow_only"]
    assert modes["diagnostic_sequential_translation_reduction"]
    assert modes["diagnostic_per_particle_launches"]
    assert not modes["live_sequential_translation_reduction"]
    assert not modes["live_per_particle_launches"]


def test_scoped_device_capture_disables_shadows_for_empty_target_bucket():
    modes = sparse_pass2_bucketed._resolve_bpref_bucket_diagnostic_modes(
        device_signature_requested=True,
        contribution_diagnostics_active=True,
        target_particle_rows=np.empty((0,), dtype=np.int64),
        high_precision_operand_bundle_requested=True,
    )

    assert modes == {
        "device_signature_requested": False,
        "contribution_diagnostics_active": False,
        "shadow_only": False,
        "high_precision_operand_bundle": False,
    }


def test_scoped_device_capture_activates_only_bucket_with_target_rows():
    target_modes = sparse_pass2_bucketed._resolve_bpref_bucket_diagnostic_modes(
        device_signature_requested=True,
        contribution_diagnostics_active=True,
        target_particle_rows=np.asarray([2], dtype=np.int64),
        high_precision_operand_bundle_requested=True,
    )
    assert all(target_modes.values())

    legacy_modes = sparse_pass2_bucketed._resolve_bpref_bucket_diagnostic_modes(
        device_signature_requested=False,
        contribution_diagnostics_active=True,
        target_particle_rows=np.empty((0,), dtype=np.int64),
        high_precision_operand_bundle_requested=True,
    )
    assert legacy_modes == {
        "device_signature_requested": False,
        "contribution_diagnostics_active": True,
        "shadow_only": False,
        "high_precision_operand_bundle": True,
    }


def test_device_panel_flush_writes_separate_class_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "class-aware")
    prefix = (1, 2, "class-aware")
    common = {
        "current_size": 4,
        "max_r": 2.0,
        "image_shape": (4, 4),
        "volume_shape": (5, 5, 5),
        "reconstruction_padding_factor": 2,
        "source_stack_sha256": "a" * 64,
        "rank": 0,
        "causal_arm": "winner-take-all-per-particle-fused-xhalf",
        "winner_take_all": True,
    }
    for class_index in (0, 3):
        key = (*prefix, class_index)
        sparse_pass2_bucketed._bpref_device_panel_accumulators[key] = (
            np.zeros(75, dtype=np.complex64),
            np.zeros(75, dtype=np.float32),
        )
        sparse_pass2_bucketed._bpref_device_panel_launch_counters[key] = class_index + 1
        sparse_pass2_bucketed._bpref_device_panel_metadata[key] = {
            **common,
            "class_index": class_index,
        }

    sparse_pass2_bucketed.flush_bpref_device_panel_accumulator(iteration=1, half=2)

    outputs = sorted(Path(tmp_path).glob("recovar_device_panel_native_*.npz"))
    assert [path.name for path in outputs] == [
        "recovar_device_panel_native_it001_h2_class001_rank000.npz",
        "recovar_device_panel_native_it001_h2_class004_rank000.npz",
    ]
    assert [int(np.load(path)["class_index"]) for path in outputs] == [0, 3]


def test_legacy_native_half_dump_remains_independent_of_class_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_RUN_ID", "native-control")
    monkeypatch.setitem(sparse_pass2_bucketed._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(sparse_pass2_bucketed._bpref_contribution_context, "half", 1)

    sparse_pass2_bucketed._maybe_dump_native_half_mstep(
        np.zeros(4, dtype=np.complex64),
        np.zeros(4, dtype=np.float32),
        current_size=2,
        n_images=1,
        recon_volume_shape=(2, 2, 2),
        stage="pre_x0",
    )

    outputs = list(Path(tmp_path).glob("native_half_mstep_*.npz"))
    assert len(outputs) == 1
    with np.load(outputs[0]) as artifact:
        assert artifact["run_id"] == "native-control"


def test_scoped_soft_row_gate_ignores_unrelated_rows_and_accepts_single_row_bucket():
    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 2, 1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=False,
    )

    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=False,
    )

    with pytest.raises(RuntimeError, match="at least one positive row"):
        sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
            np.asarray([0, 2], dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            device_signature_requested=False,
            winner_take_all=False,
        )


def test_scoped_kclass_row_gate_accepts_empty_class_slice_but_not_invalid_wta():
    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=False,
        posterior_partitioned_across_classes=True,
    )
    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=True,
        posterior_partitioned_across_classes=True,
    )

    with pytest.raises(RuntimeError, match="at most one positive"):
        sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            device_signature_requested=True,
            winner_take_all=True,
            posterior_partitioned_across_classes=True,
        )


def test_empty_kclass_device_signature_payload_preserves_zero_row_topology():
    payload = sparse_pass2_bucketed._empty_bpref_device_signature_arrays(
        2812,
        image_identity_dtype=np.dtype("<U120"),
    )

    assert payload["rotation_keys"].shape == (0, 2812)
    assert payload["source_values"].shape == (0, 2812, 6)
    assert payload["neighbor_indices"].shape == (0, 2812, 8)
    assert payload["neighbor_coefficients"].dtype == np.float32
    assert payload["launch_ordinals"].dtype == np.int64
    assert payload["image_identities"].dtype == np.dtype("<U120")
    assert payload["contributor_rotation_keys"].shape == (0,)


def test_zero_contributor_class_capture_writes_manifest_only_signature(
    tmp_path,
    monkeypatch,
):
    contribution_dir = tmp_path / "contributions"
    signature_dir = tmp_path / "signatures"
    image_names_path = tmp_path / "image_names.npy"
    np.save(
        image_names_path,
        np.asarray(["1@/tmp/frozen.mrcs"]),
        allow_pickle=False,
    )
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", str(contribution_dir))
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", str(signature_dir))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "10")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "1")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY", str(image_names_path))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256", "a" * 64)
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "zero-class")
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_require_bpref_device_soft_particle_arm",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_fused_x_half_backproject_indexed",
        lambda *args, **_kwargs: args[:2],
    )
    sparse_pass2_bucketed._bpref_device_panel_accumulators.clear()
    sparse_pass2_bucketed._bpref_device_panel_launch_counters.clear()
    sparse_pass2_bucketed._bpref_device_panel_metadata.clear()
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=10, half=1)
    scores = np.asarray([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32)
    probs = np.exp(scores).astype(np.float32)
    probs /= probs.sum(axis=(1, 2), keepdims=True)
    zeros = np.zeros((1, 2, 6), dtype=np.float32)
    try:
        sparse_pass2_bucketed._maybe_dump_bpref_contribution_rows(
            experiment_dataset=object(),
            image_indices=np.asarray([0]),
            current_size=4,
            summed=zeros.astype(np.complex64),
            ctf_probs=zeros,
            rotations=np.broadcast_to(np.eye(3), (1, 2, 3, 3)),
            actual_counts=np.asarray([2]),
            rotation_indices=np.asarray([[10, 11]]),
            fine_translations=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            scores=scores,
            preprior_scores=scores,
            probs=probs,
            rotation_log_prior=np.zeros((1, 2)),
            translation_log_prior=np.zeros((1, 2)),
            log_z=np.zeros((1,)),
            best_log_score=np.zeros((1,)),
            reconstruction_probs=np.zeros_like(probs),
            reconstruction_mask=np.zeros_like(probs, dtype=bool),
            reconstruction_sum_weight=np.zeros((1,)),
            reconstruction_threshold=np.zeros((1,)),
            candidate_mask=np.ones_like(probs, dtype=bool),
            high_precision_operand_bundle=False,
            raw_batch_data=None,
            ctf_params=None,
            noise_variance_half=None,
            integer_pre_shifts=None,
            batch_image_corrections=None,
            batch_scale_corrections=None,
            relion_preprocess_normalization_factors=None,
            relion_cuda_preprocess=False,
            score_with_masked_images=False,
            image_mask=None,
            image_mask_mode="not-captured",
            voxel_size=1.0,
            ctf_mode="not-captured",
            ctf_dose_per_tilt=0.0,
            ctf_angle_per_tilt=0.0,
            disc_type="linear_interp",
            projection_padding_factor=2,
            reconstruction_padding_factor=2,
            use_relion_x_half_mstep=True,
            winner_take_all=False,
            max_r=2,
            window_indices=np.arange(6),
            image_shape=(4, 4),
            volume_shape=(4, 4, 4),
            shadow_only_mode=True,
            shadow_score_bitwise_equal=True,
            shadow_reduction_agreement={
                "data_rel_l1": 0.0,
                "data_normalized_max": 0.0,
                "weight_rel_l1": 0.0,
                "weight_normalized_max": 0.0,
                "rel_l1_bound": 1e-3,
                "normalized_max_bound": 1e-3,
            },
            device_signature_active=True,
            class_index=1,
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()
        sparse_pass2_bucketed._bpref_device_panel_accumulators.clear()
        sparse_pass2_bucketed._bpref_device_panel_launch_counters.clear()
        sparse_pass2_bucketed._bpref_device_panel_metadata.clear()

    signature_path = next(signature_dir.glob("*.device.npz"))
    with np.load(signature_path, allow_pickle=False) as signature:
        assert signature["class_index"].item() == 1
        assert signature["particle_launch_ordinals"].tolist() == [0]
        assert signature["particle_contributor_row_counts"].tolist() == [0]
        assert signature["particle_noncontributor_row_counts"].tolist() == [2]
        assert signature["contributor_canonical_rotation_keys"].shape == (0,)
        assert signature["source_values"].shape == (0, 12, 6)
        assert signature["program_axis_sizes"].tolist() == [0, 12, 8]


def test_scoped_wta_row_gate_checks_target_and_rejects_invalid_target_row():
    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=True,
    )

    with pytest.raises(RuntimeError, match="exactly one positive rotation row"):
        sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            device_signature_requested=True,
            winner_take_all=True,
        )

    with pytest.raises(RuntimeError, match="outside the sparse bucket"):
        sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
            device_signature_requested=True,
            winner_take_all=True,
        )


def test_target_dense_half_keeps_block_topology_inactive_for_live_work(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_dense(**kwargs):
        assert kwargs.pop("bpref_device_signature_active") is True
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-live"

    monkeypatch.setattr(iteration_loop, "_score_half_dense", fake_dense)
    assert iteration_loop._score_half_dense_in_bpref_scope(
        bpref_device_signature_active=True,
    ) == "ordinary-live"
    assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_target_capture_preserves_rotation_chunk_plan_and_fails_if_target_is_chunked():
    planned_chunk_size = 809
    assert sparse_pass2_bucketed._guard_bpref_target_rotation_chunking(
        planned_chunk_size,
        bucket_size=128,
        target_particle_rows=np.asarray([3], dtype=np.int64),
    ) == planned_chunk_size
    assert sparse_pass2_bucketed._guard_bpref_target_rotation_chunking(
        64,
        bucket_size=128,
        target_particle_rows=np.empty((0,), dtype=np.int64),
    ) == 64

    with pytest.raises(RuntimeError, match="refuses to change that plan"):
        sparse_pass2_bucketed._guard_bpref_target_rotation_chunking(
            64,
            bucket_size=128,
            target_particle_rows=np.asarray([3], dtype=np.int64),
        )

    source = inspect.getsource(sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed)
    assert "Scoped BPref device capture disables rotation-chunked pass 2" not in source
    assert "rotation_chunk_size = _guard_bpref_target_rotation_chunking(" in source


def test_posterior_mask_and_reduced_operand_diagnostic_branches_agree_on_cpu():
    rng = np.random.default_rng(20260715)
    batch, rotations, translations, pixels = 2, 4, 3, 11
    shifted = (
        rng.normal(size=(batch, translations, pixels))
        + 1j * rng.normal(size=(batch, translations, pixels))
    ).astype(np.complex64)
    projection = (
        rng.normal(size=(batch, rotations, pixels))
        + 1j * rng.normal(size=(batch, rotations, pixels))
    ).astype(np.complex64)
    corr = rng.uniform(0.2, 2.0, size=(batch, pixels)).astype(np.float32)
    half_weights = rng.uniform(0.5, 2.0, size=pixels).astype(np.float32)
    rotation_prior = rng.normal(scale=0.1, size=(batch, rotations)).astype(np.float32)
    translation_prior = rng.normal(scale=0.1, size=(batch, translations)).astype(np.float32)
    candidate_mask = np.ones((batch, rotations, translations), dtype=bool)
    candidate_mask[0, -1, -1] = False

    authoritative_scores = sparse_pass2_bucketed._score_pass2_bucket_relion_gpu_diff2(
        shifted,
        corr,
        projection,
        half_weights,
        rotation_prior,
        translation_prior,
        candidate_mask,
    )
    authoritative_normalized = sparse_pass2_bucketed._normalize_pass2_bucket(
        authoritative_scores
    )
    probs = authoritative_normalized[1]
    authoritative_reconstruction = (
        sparse_pass2_bucketed._relion_pass2_reconstruction_probs_for_mstep(
            authoritative_scores,
            probs,
            adaptive_fraction=0.999,
            use_relion_x_half_mstep=True,
            winner_take_all=False,
        )
    )
    shadow_reconstruction = (
        sparse_pass2_bucketed._relion_pass2_reconstruction_probs_for_mstep(
            authoritative_scores,
            probs,
            adaptive_fraction=0.999,
            use_relion_x_half_mstep=True,
            winner_take_all=False,
            return_diagnostics=True,
        )
    )
    for label, authoritative, shadow in zip(
        ("probabilities", "mask", "counts"),
        authoritative_reconstruction,
        shadow_reconstruction[:3],
        strict=True,
    ):
        sparse_pass2_bucketed._require_bpref_shadow_exact(
            f"CPU test reconstruction {label}", authoritative, shadow
        )

    reconstruction_probs = authoritative_reconstruction[0]
    shifted_reconstruction = (
        rng.normal(size=(batch, translations, pixels))
        + 1j * rng.normal(size=(batch, translations, pixels))
    ).astype(np.complex64)
    ctf2 = rng.uniform(0.1, 1.5, size=(batch, pixels)).astype(np.float32)
    ordinary_summed, ordinary_weights = compute_local_mstep_sums(
        reconstruction_probs,
        shifted_reconstruction,
        ctf2,
        relion_x_half=True,
        sequential_translation_reduction=False,
    )
    shadow_summed, shadow_weights = compute_local_mstep_sums(
        reconstruction_probs,
        shifted_reconstruction,
        ctf2,
        relion_x_half=True,
        sequential_translation_reduction=True,
    )
    metrics = sparse_pass2_bucketed._require_bpref_reduction_shadow_agreement(
        ordinary_summed,
        ordinary_weights,
        shadow_summed,
        shadow_weights,
    )
    assert set(metrics) == {
        "data_rel_l1",
        "data_normalized_max",
        "weight_rel_l1",
        "weight_normalized_max",
        "rel_l1_bound",
        "normalized_max_bound",
    }


def test_standalone_diagnostics_keep_legacy_flags_without_device_capture(monkeypatch):
    monkeypatch.delenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", raising=False)
    for name in (
        "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
        "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY",
        "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.setenv(name, "1")

    flags = sparse_pass2_bucketed._scoped_bpref_diagnostic_flags(active=False)
    assert not flags["device_signature_configured"]
    assert all(value for name, value in flags.items() if name != "device_signature_configured")
    assert cuda_backproject.relion_x_half_bp_block_topology_enabled()


def test_target_half2_cannot_leak_into_final_all_data_or_local_search(monkeypatch):
    env = _capture_environment()
    env["RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF"] = "2"
    assert iteration_loop._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        environ=env,
    )
    assert not iteration_loop._bpref_device_signature_active_for_numbered_half(
        iteration=5,
        half=2,
        final_all_data=True,
        environ=env,
    )

    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    monkeypatch.setenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", "1")

    def fake_local(**kwargs):
        del kwargs
        assert not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        return "ordinary-local"

    monkeypatch.setattr(iteration_loop, "_score_half_local", fake_local)
    assert iteration_loop._score_half_local_in_bpref_scope(
        bpref_device_signature_active=False
    ) == "ordinary-local"
    with pytest.raises(RuntimeError, match="sparse adaptive pass 2"):
        iteration_loop._score_half_local_in_bpref_scope(
            bpref_device_signature_active=True
        )


def test_exact_local_contribution_adapter_is_explicit_and_rejects_device_claims(
    monkeypatch,
):
    forwarded = []
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_maybe_dump_bpref_contribution_rows",
        lambda **kwargs: forwarded.append(kwargs),
    )

    local_em_engine._maybe_dump_exact_local_bpref_contribution_rows(marker="off")
    assert forwarded == []

    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "/tmp/contributions")
    local_em_engine._maybe_dump_exact_local_bpref_contribution_rows(marker="local")
    assert forwarded == [{"marker": "local"}]

    monkeypatch.setenv("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "/tmp/device")
    with pytest.raises(RuntimeError, match="does not yet support device signatures"):
        local_em_engine._maybe_dump_exact_local_bpref_contribution_rows(marker="device")
    assert forwarded == [{"marker": "local"}]


def test_exact_local_contribution_adapter_writes_versioned_pre_scatter_fixture(
    monkeypatch,
    tmp_path,
):
    dump_dir = tmp_path / "contributions"
    image_names_path = tmp_path / "image_names.npy"
    np.save(
        image_names_path,
        np.asarray(["1@/tmp/frozen.mrcs", "2@/tmp/frozen.mrcs"]),
        allow_pickle=False,
    )
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "7")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "2")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE", "4")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY", str(image_names_path))
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256", "a" * 64)
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=7, half=2)
    try:
        scores = np.asarray(
            [[[0.0, -1.0], [-2.0, -3.0]], [[-0.5, -1.5], [-2.5, -3.5]]],
            dtype=np.float32,
        )
        probs = np.exp(scores).astype(np.float32)
        probs /= probs.sum(axis=(1, 2), keepdims=True)
        local_em_engine._maybe_dump_exact_local_bpref_contribution_rows(
            experiment_dataset=object(),
            image_indices=np.asarray([0, 1]),
            current_size=4,
            summed=np.ones((2, 2, 6), dtype=np.complex64),
            ctf_probs=np.ones((2, 2, 6), dtype=np.float32),
            rotations=np.broadcast_to(np.eye(3), (2, 2, 3, 3)),
            actual_counts=np.asarray([2, 1]),
            rotation_indices=np.asarray([[10, 11], [20, 21]]),
            fine_translations=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            scores=scores,
            preprior_scores=scores,
            probs=probs,
            rotation_log_prior=np.zeros((2, 2)),
            translation_log_prior=np.zeros((2, 2)),
            log_z=np.zeros((2,)),
            best_log_score=np.zeros((2,)),
            reconstruction_probs=probs,
            reconstruction_mask=np.ones((2, 2, 2), dtype=bool),
            reconstruction_sum_weight=probs.sum(axis=(1, 2)),
            reconstruction_threshold=np.zeros((2,)),
            candidate_mask=np.ones((2, 2, 2), dtype=bool),
            high_precision_operand_bundle=False,
            raw_batch_data=None,
            ctf_params=None,
            noise_variance_half=None,
            integer_pre_shifts=None,
            batch_image_corrections=None,
            batch_scale_corrections=None,
            relion_preprocess_normalization_factors=None,
            relion_cuda_preprocess=False,
            score_with_masked_images=False,
            image_mask=None,
            image_mask_mode="not-captured",
            voxel_size=1.0,
            ctf_mode="not-captured",
            ctf_dose_per_tilt=0.0,
            ctf_angle_per_tilt=0.0,
            disc_type="linear_interp",
            projection_padding_factor=2,
            reconstruction_padding_factor=2,
            use_relion_x_half_mstep=True,
            winner_take_all=False,
            max_r=2,
            window_indices=np.arange(6),
            image_shape=(4, 4),
            volume_shape=(4, 4, 4),
            shadow_only_mode=False,
            shadow_score_bitwise_equal=True,
            shadow_reduction_agreement=None,
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()

    artifact = next(dump_dir.glob("bpref_contribution_rows_*.npz"))
    with np.load(artifact, allow_pickle=False) as capture:
        assert capture["schema"].item() == "recovar-bpref-contribution-rows-v3"
        assert capture["iteration"].item() == 7
        assert capture["half"].item() == 2
        assert capture["current_size"].item() == 4
        assert capture["relion_native_lane_reduction"].item() is False
        assert capture["active_summed"].dtype == np.complex64
        assert capture["active_ctf_probs"].dtype == np.float32
        assert capture["active_original_indices"].tolist() == [0, 0, 1]


def test_exact_local_contribution_capture_routes_only_the_target_boundary(monkeypatch):
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR", "/tmp/contributions")
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=7, half=2)
    try:
        assert not local_em_engine._exact_local_bpref_contribution_capture_active(
            current_size=50,
            debug_iteration=7,
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()

    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "7")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "2")
    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE", "50")
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=7, half=2)
    try:
        assert local_em_engine._exact_local_bpref_contribution_capture_active(
            current_size=50,
            debug_iteration=7,
        )
        assert not local_em_engine._exact_local_bpref_contribution_capture_for_call(
            current_size=50,
            debug_iteration=7,
            score_only=True,
            mstep_relion_x_half=False,
        )
        assert local_em_engine._exact_local_bpref_contribution_capture_for_call(
            current_size=50,
            debug_iteration=7,
            score_only=False,
            mstep_relion_x_half=True,
        )
        with pytest.raises(RuntimeError, match="requires RELION x-half M-step geometry"):
            local_em_engine._exact_local_bpref_contribution_capture_for_call(
                current_size=50,
                debug_iteration=7,
                score_only=False,
                mstep_relion_x_half=False,
            )
        assert not local_em_engine._exact_local_bpref_contribution_capture_active(
            current_size=52,
            debug_iteration=7,
        )
        assert not local_em_engine._exact_local_bpref_contribution_capture_active(
            current_size=50,
            debug_iteration=8,
        )
        sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=7, half=1)
        assert not local_em_engine._exact_local_bpref_contribution_capture_active(
            current_size=50,
            debug_iteration=7,
        )
    finally:
        sparse_pass2_bucketed.clear_bpref_contribution_dump_context()

    source = inspect.getsource(local_em_engine.run_local_em_exact)
    helper_source = inspect.getsource(
        local_em_engine._exact_local_bpref_contribution_capture_for_call
    )
    assert "_exact_local_bpref_contribution_capture_for_call" in source
    assert "and not bpref_contribution_capture_active" in source
    assert "if bpref_contribution_capture_active and not score_only:" in source
    assert "requires RELION x-half M-step geometry" in helper_source


def test_clear_dump_context_marks_contribution_and_native_dumps_inactive():
    sparse_pass2_bucketed.set_bpref_contribution_dump_context(iteration=5, half=2)
    assert sparse_pass2_bucketed._bpref_contribution_context == {
        "iteration": 5,
        "half": 2,
    }

    sparse_pass2_bucketed.clear_bpref_contribution_dump_context()
    assert sparse_pass2_bucketed._bpref_contribution_context == {
        "iteration": -1,
        "half": -1,
    }


def test_iteration_loop_clears_dump_context_before_every_final_exit_or_half():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    final_decision = source.index("should_run_final_iteration =")
    final_loop = source.index("for k in range(2):", source.index("final_outs = PerHalfOutputs.empty()"))

    assert source.rfind("clear_bpref_contribution_dump_context()", 0, final_decision) >= 0
    assert "clear_bpref_contribution_dump_context()" in source[
        final_loop : source.index("final_half_t0", final_loop)
    ]


def test_bucketed_source_has_no_unscoped_capture_branches():
    source = inspect.getsource(sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed)
    assert 'if os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR")' not in source
    assert "relion_x_half_bp_per_particle_launch_enabled()" not in source
    assert "device_signature_active=bucket_device_signature_requested" in source


def test_active_capture_accepts_fused_kclass_route(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    k_class._validate_bpref_device_signature_sparse_route(
        active=True,
        n_classes=4,
    )

    k_class._validate_bpref_device_signature_sparse_route(
        active=False,
        n_classes=1,
    )


def test_bpref_contribution_class_filter_uses_relion_one_based_numbers(monkeypatch):
    monkeypatch.delenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS", raising=False)
    assert sparse_pass2_bucketed._bpref_contribution_class_enabled(0)
    assert sparse_pass2_bucketed._bpref_contribution_class_enabled(3)

    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS", "2")
    assert not sparse_pass2_bucketed._bpref_contribution_class_enabled(0)
    assert sparse_pass2_bucketed._bpref_contribution_class_enabled(1)
    assert not sparse_pass2_bucketed._bpref_contribution_class_enabled(2)

    monkeypatch.setenv("RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        sparse_pass2_bucketed._bpref_contribution_class_enabled(0)


def test_fused_kclass_compact_capture_materializes_only_target_rows():
    image_indices = np.asarray([10, 11, 12], dtype=np.int64)
    log_priors = [np.asarray([], dtype=np.float32) for _ in range(13)]
    log_priors[10] = np.asarray([0.1, 0.2])
    log_priors[11] = np.asarray([0.3, 0.4, 0.5])
    log_priors[12] = np.asarray([0.6])
    per_image_inputs = {"log_prior": log_priors}
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 4, 3, 3)).copy()
    class_bucket_arrays = {
        "bucket_size": 4,
        "mstep_rotations": rotations,
        "rotation_indices": np.asarray(
            [[1, 2, -1, -1], [3, 4, 5, -1], [6, -1, -1, -1]],
            dtype=np.int64,
        ),
        "actual_counts": np.asarray([2, 3, 1], dtype=np.int64),
    }
    compact_pair_arrays = {
        "local_rotation_row": np.asarray(
            [[0, 1, -1], [2, 0, 1], [0, -1, -1]],
            dtype=np.int32,
        ),
        "translation_idx": np.asarray(
            [[1, 0, -1], [0, 1, 1], [0, -1, -1]],
            dtype=np.int32,
        ),
        "pair_mask": np.asarray(
            [[True, True, False], [True, True, True], [True, False, False]],
            dtype=bool,
        ),
    }
    scores = np.asarray(
        [[1.0, 2.0, -np.inf], [3.0, 4.0, 5.0], [6.0, -np.inf, -np.inf]],
        dtype=np.float32,
    )
    probs = np.asarray(
        [[0.1, 0.2, 0.0], [0.3, 0.4, 0.5], [0.6, 0.0, 0.0]],
        dtype=np.float32,
    )
    capture = sparse_pass2_bucketed._materialize_k_class_capture_rows(
        image_indices=image_indices,
        target_particle_rows=np.asarray([1], dtype=np.int64),
        per_image_inputs=per_image_inputs,
        class_bucket_arrays=class_bucket_arrays,
        compact_pair_arrays=compact_pair_arrays,
        scores=scores,
        probs=probs,
        reconstruction_mask=None,
        reconstruction_probs=None,
        bucket_translation_prior=np.asarray(
            [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
            dtype=np.float32,
        ),
        n_fine_trans=2,
    )

    np.testing.assert_array_equal(capture["image_indices"], np.asarray([11]))
    assert capture["scores"].shape == (1, 4, 2)
    assert capture["scores"][0, 2, 0] == 3.0
    assert capture["scores"][0, 0, 1] == 4.0
    assert capture["scores"][0, 1, 1] == 5.0
    assert np.count_nonzero(capture["candidate_mask"]) == 3
    np.testing.assert_allclose(capture["rotation_log_prior"][0], [0.3, 0.4, 0.5, 0.0])
    np.testing.assert_allclose(capture["translation_log_prior"][0], [-0.3, -0.4])


def test_later_capture_support_excludes_dense_full_support_fallback():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    support_start = source.index("later_soft_particle_fused_supported =")
    support_end = source.index("fused_atomic_diagnostic_supported =", support_start)
    support_block = source[support_start:support_end]
    assert "and not skip_significance_pruning" in support_block
