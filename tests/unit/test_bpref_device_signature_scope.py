from __future__ import annotations

import inspect

import numpy as np
import pytest

from recovar import cuda_backproject
from recovar.em.dense_single_volume import iteration_loop
from recovar.em.dense_single_volume import k_class
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


def test_scoped_soft_row_gate_ignores_unrelated_zero_and_multiple_rows():
    sparse_pass2_bucketed._validate_bpref_positive_rotation_rows(
        np.asarray([0, 2, 1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        device_signature_requested=True,
        winner_take_all=False,
    )

    with pytest.raises(RuntimeError, match="multiple positive rows"):
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


def test_score_posterior_mask_and_reduced_operand_shadow_branches_agree_on_cpu():
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
    shadow_scores, _ = sparse_pass2_bucketed._score_pass2_bucket_relion_gpu_diff2_components(
        shifted,
        corr,
        projection,
        half_weights,
        rotation_prior,
        translation_prior,
        candidate_mask,
    )
    sparse_pass2_bucketed._require_bpref_shadow_exact(
        "CPU test scores", authoritative_scores, shadow_scores
    )

    authoritative_normalized = sparse_pass2_bucketed._normalize_pass2_bucket(
        authoritative_scores
    )
    shadow_normalized = sparse_pass2_bucketed._normalize_pass2_bucket(shadow_scores)
    for authoritative, shadow in zip(authoritative_normalized, shadow_normalized, strict=True):
        sparse_pass2_bucketed._require_bpref_shadow_exact(
            "CPU test posterior", authoritative, shadow
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
            shadow_scores,
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


def test_active_capture_rejects_fused_kclass_route(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    with pytest.raises(RuntimeError, match="single-class sparse pass-2 route"):
        k_class._validate_bpref_device_signature_sparse_route(
            active=True,
            n_classes=1,
        )

    k_class._validate_bpref_device_signature_sparse_route(
        active=False,
        n_classes=1,
    )


def test_later_capture_support_excludes_dense_full_support_fallback():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    support_start = source.index("later_soft_particle_fused_supported =")
    support_end = source.index("fused_atomic_diagnostic_supported =", support_start)
    support_block = source[support_start:support_end]
    assert "and not skip_significance_pruning" in support_block
