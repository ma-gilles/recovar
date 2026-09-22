"""Qualified-lifetime memory estimates retained from Q, not enabled by default."""
from types import SimpleNamespace
from recovar.em.sparse_pass2 import firstiter_bpref
import numpy as np
import pytest
from recovar.em.helpers.batch_planning import _estimate_relion_em_batch_sizes
pytestmark = pytest.mark.unit

def test_box800_k1_score_tile_plan_uses_resolved_precision():
    historical = _box800_plan()
    float64_plan = _box800_plan(use_float64_scoring=True)
    float32_plan = _box800_plan(use_float64_scoring=False)

    assert historical == float64_plan
    assert float64_plan.rotation_block_size == 194
    assert float64_plan.pose_pixel_tile_gb == pytest.approx(0.49930944)
    assert float32_plan.rotation_block_size == 388
    assert float32_plan.pose_pixel_tile_gb == pytest.approx(float64_plan.pose_pixel_tile_gb)
    assert float32_plan.active_score_tile_gb == pytest.approx(
        float64_plan.active_score_tile_gb / 2.0
    )
    assert float32_plan.image_batch_size == float64_plan.image_batch_size == 1
    assert float32_plan.persistent_estimate_gb == float64_plan.persistent_estimate_gb == 81.92


def test_box800_compact_k1_persistent_estimate_uses_model_current_size_phase_max():
    compact = dict(
        gpu_memory_gb=85,
        use_float64_scoring=False,
        compact_k1_relion_layout=True,
    )
    current_size_62 = _box800_plan(**compact, model_current_size=62)
    full_box_model = _box800_plan(**compact, model_current_size=800)
    small_voxels = 127 * 127 * 64
    full_voxels = 1603 * 1603 * 802
    assert current_size_62.persistent_estimate_mode == "compact_k1_relion_phase_max"
    assert current_size_62.persistent_estimate_gb == pytest.approx(
        4.0 + small_voxels * 12 / 1e9
    )
    assert current_size_62.pending_score_persistent_gb == pytest.approx(
        small_voxels * 8 / 1e9
    )
    assert full_box_model.persistent_estimate_gb == pytest.approx(
        4.0 + full_voxels * 12 / 1e9
    )
    assert full_box_model.pending_score_persistent_gb == pytest.approx(
        full_voxels * 8 / 1e9
    )
    assert current_size_62.score_pixel_count == full_box_model.score_pixel_count == 1532


def test_box800_soft_k1_compact_plan_adds_overlapping_projector_and_bpref():
    plan = _box800_plan(
        n_rot=576,
        n_trans=21,
        current_size=294,
        gpu_memory_gb=80,
        runtime_free_memory_gb=80,
        use_float64_scoring=False,
        compact_k1_relion_layout=True,
        compact_k1_relion_score_bpref_overlap=True,
        model_current_size=294,
    )
    compact_voxels = 591 * 591 * 296
    overlapping_bytes = compact_voxels * (8 + 8 + 4)

    assert (
        plan.persistent_estimate_mode
        == "compact_k1_relion_score_bpref_overlap"
    )
    assert plan.persistent_estimate_gb == pytest.approx(
        4.0 + overlapping_bytes / 1e9
    )
    assert plan.pending_score_persistent_gb == pytest.approx(
        overlapping_bytes / 1e9
    )
    assert plan.usable_estimate_gb == pytest.approx(
        (80.0 - overlapping_bytes / 1e9) * 0.8
    )
    assert plan.image_batch_size == 64
    assert plan.rotation_block_size == 576


def test_box800_soft_k1_compact_plan_preserves_headroom_at_10202_size564_boundary():
    common = dict(
        n_rot=416,
        n_trans=84,
        current_size=564,
        gpu_memory_gb=80,
        runtime_free_memory_gb=72.72,
        use_float64_scoring=False,
    )
    historical = _box800_plan(**common)
    compact = _box800_plan(
        **common,
        compact_k1_relion_layout=True,
        compact_k1_relion_score_bpref_overlap=True,
        model_current_size=564,
    )

    # Full-particle job 13356985 failed at this exact boundary after the
    # historical estimate collapsed the tunable plan to 1/4, while five
    # 512-wide tail buckets still had to execute.  Pin the fail-closed compact
    # envelope used by its successor so local wrapper changes cannot silently
    # reintroduce that planner mismatch.
    assert historical.persistent_estimate_mode == "historical_full_cube"
    assert historical.persistent_estimate_gb == pytest.approx(81.92)
    assert (historical.image_batch_size, historical.rotation_block_size) == (1, 4)
    assert historical.usable_estimate_gb == pytest.approx(1.0)

    assert compact.persistent_estimate_mode == "compact_k1_relion_score_bpref_overlap"
    assert compact.persistent_estimate_gb == pytest.approx(18.48010252)
    assert compact.pending_score_persistent_gb == pytest.approx(14.48010252)
    assert compact.usable_estimate_gb == pytest.approx(46.591917984)
    assert (compact.image_batch_size, compact.rotation_block_size) == (13, 26)


@pytest.mark.parametrize(
    ("score_size", "model_size", "n_rot", "n_trans", "live_free_gb", "image_batch", "rotation_block"),
    [
        (62, 62, 576, 21, 79.0, 64, 576),
        (62, 62, 4608, 84, 75.0, 23, 4608),
        (78, 800, 576, 21, 47_724 * 1024**2 / 1e9, 42, 576),
        (None, 800, 4608, 84, 47_724 * 1024**2 / 1e9, 3, 11),
    ],
)
def test_box800_compact_k1_observed_live_plans(
    score_size,
    model_size,
    n_rot,
    n_trans,
    live_free_gb,
    image_batch,
    rotation_block,
):
    plan = _box800_plan(
        n_rot=n_rot,
        n_trans=n_trans,
        gpu_memory_gb=85,
        current_size=score_size,
        use_float64_scoring=False,
        compact_k1_relion_layout=True,
        model_current_size=model_size,
        runtime_free_memory_gb=live_free_gb,
    )
    expected_usable = (live_free_gb - plan.pending_score_persistent_gb) * 0.8
    assert plan.usable_estimate_gb == pytest.approx(expected_usable)
    assert plan.image_batch_size == image_batch
    assert plan.rotation_block_size == rotation_block


def test_compact_k1_plan_rejects_genuine_float64_and_kclass_routes():
    compact = dict(
        compact_k1_relion_layout=True,
        model_current_size=62,
    )
    with pytest.raises(ValueError, match="float32/complex64"):
        _box800_plan(
            **compact,
            n_classes=1,
            use_float64_scoring=True,
        )
    with pytest.raises(ValueError, match="K=1-only"):
        _box800_plan(
            **compact,
            n_classes=2,
            use_float64_scoring=False,
        )
    with pytest.raises(
        ValueError,
        match="compact_k1_relion_score_bpref_overlap requires",
    ):
        _box800_plan(
            compact_k1_relion_layout=False,
            compact_k1_relion_score_bpref_overlap=True,
            model_current_size=62,
            use_float64_scoring=False,
        )



def _box800_plan(**overrides):
    kwargs = dict(
        requested_image_batch_size=64,
        requested_rotation_block_size=8192,
        n_rot=4608,
        n_trans=84,
        image_shape=(800, 800),
        volume_shape=(800, 800, 800),
        padding_factor=2,
        n_classes=1,
        gpu_memory_gb=80,
        current_size=62,
    )
    kwargs.update(overrides)
    return _estimate_relion_em_batch_sizes(**kwargs)

def _compact_route_decision(*, projector_half, score_complex_dtype=np.complex64):
    sparse = firstiter_bpref
    return sparse._relion_firstiter_compact_batch_planning_decision(
        source_faithful_spectrum_norm=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        use_relion_x_half_mstep=True,
        projector_half=projector_half,
        score_complex_dtype=score_complex_dtype,
        recon_volume_size=1603 * 1603 * 802,
        bpref_device_signature_active=False,
        fixed_base_bytes=4_000_000_000,
    )

def test_compact_k1_route_gate_requires_host_c64_and_no_diagnostics(monkeypatch):
    sparse = firstiter_bpref
    host_projector = SimpleNamespace(
        shape=(1603, 1603, 802),
        dtype=np.dtype(np.complex64),
    )
    monkeypatch.setattr(sparse.sparse_pass2_budget, "_device_memory_limit_bytes", lambda: 85 * 1024**3)
    monkeypatch.setattr(sparse.sparse_pass2_budget, "_jax_allocator_free_memory_bytes", lambda: 40 * 1024**3)
    for name in (
        "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS",
        "RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF",
        "RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF",
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
        "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR",
        "RECOVAR_BPREF_ACCUMULATOR_DELTA_DUMP_DIR",
        "RECOVAR_PASS2_DUMP_DIR",
        "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)

    sparse.bpref_diagnostics.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        enabled = _compact_route_decision(projector_half=host_projector)
        assert enabled.enabled is True
        assert enabled.deferred_firstiter_bpref is True

        float64 = _compact_route_decision(
            projector_half=host_projector,
            score_complex_dtype=np.complex128,
        )
        assert float64.enabled is False

        class DeviceProjector:
            shape = (2, 2, 2)
            dtype = np.dtype(np.complex64)

        monkeypatch.setattr(sparse.jax, "Array", DeviceProjector)
        device_owned = _compact_route_decision(projector_half=DeviceProjector())
        assert device_owned.enabled is False

        monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", "/tmp/diagnostic")
        diagnostic = _compact_route_decision(projector_half=host_projector)
        assert diagnostic.enabled is False
        assert diagnostic.deferred_firstiter_bpref is False
    finally:
        sparse.bpref_diagnostics.clear_bpref_contribution_dump_context()

def test_compact_k1_route_gate_honors_explicit_route_env(monkeypatch):
    sparse = firstiter_bpref
    host_projector = SimpleNamespace(
        shape=(1603, 1603, 802),
        dtype=np.dtype(np.complex64),
    )
    monkeypatch.setattr(sparse.sparse_pass2_budget, "_device_memory_limit_bytes", lambda: None)
    monkeypatch.setattr(sparse.sparse_pass2_budget, "_jax_allocator_free_memory_bytes", lambda: None)
    sparse.bpref_diagnostics.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF", "1")
        forced_deferred = _compact_route_decision(projector_half=host_projector)
        assert forced_deferred.enabled is True
        assert forced_deferred.deferred_firstiter_bpref is True

        monkeypatch.setenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF", "0")
        disabled_deferred = _compact_route_decision(projector_half=host_projector)
        assert disabled_deferred.enabled is False
        assert disabled_deferred.deferred_firstiter_bpref is False

        monkeypatch.delenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF")
        monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
        disabled_fused = _compact_route_decision(projector_half=host_projector)
        assert disabled_fused.enabled is False
    finally:
        sparse.bpref_diagnostics.clear_bpref_contribution_dump_context()

def test_soft_compact_k1_route_gate_is_windowed_exact_and_diagnostic_free(
    monkeypatch,
):
    sparse = firstiter_bpref
    host_projector = SimpleNamespace(
        shape=(591, 591, 296),
        dtype=np.dtype(np.complex64),
    )
    for name in (
        "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
        "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR",
        "RECOVAR_BPREF_ACCUMULATOR_DELTA_DUMP_DIR",
        "RECOVAR_PASS2_DUMP_DIR",
        "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
        "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
        "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)

    common = dict(
        source_faithful_spectrum_norm=True,
        preserve_bpref_particle_order=True,
        use_relion_x_half_mstep=True,
        relion_cuda_images=True,
        projector_half=host_projector,
        score_complex_dtype=np.complex64,
        model_current_size=294,
        image_size=800,
        bpref_device_signature_active=False,
    )
    assert sparse._relion_soft_compact_batch_planning_safe(**common)

    for override in (
        {"source_faithful_spectrum_norm": False},
        {"preserve_bpref_particle_order": False},
        {"use_relion_x_half_mstep": False},
        {"relion_cuda_images": False},
        {"score_complex_dtype": np.complex128},
        {"model_current_size": 800},
    ):
        assert not sparse._relion_soft_compact_batch_planning_safe(
            **(common | override)
        )

    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", "/tmp/diagnostic")
    assert not sparse._relion_soft_compact_batch_planning_safe(**common)


@pytest.mark.parametrize("diagnostic_env", ["RECOVAR_EM_FINITE_CHECK", "RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR"])
def test_compact_planning_excludes_current_runtime_diagnostics(monkeypatch, diagnostic_env):
    monkeypatch.setenv(diagnostic_env, "1")
    host = SimpleNamespace(shape=(1603, 1603, 802), dtype=np.dtype(np.complex64))
    monkeypatch.delenv("RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF", raising=False)
    firstiter_bpref.bpref_diagnostics.set_bpref_contribution_dump_context(iteration=1, half=1)
    try:
        decision = _compact_route_decision(projector_half=host)
        assert not decision.enabled
        assert not decision.deferred_firstiter_bpref
        assert not firstiter_bpref._relion_soft_compact_batch_planning_safe(
            source_faithful_spectrum_norm=True,
            preserve_bpref_particle_order=True,
            use_relion_x_half_mstep=True,
            relion_cuda_images=True,
            projector_half=host,
            score_complex_dtype=np.complex64,
            model_current_size=294,
            image_size=800,
            bpref_device_signature_active=False,
        )
    finally:
        firstiter_bpref.bpref_diagnostics.clear_bpref_contribution_dump_context()


@pytest.mark.parametrize("overlap", [False, True])
def test_compact_plan_accounts_for_transient_device_projector_slab(overlap):
    voxels = 1603 * 1603 * 802
    plan = _box800_plan(
        gpu_memory_gb=80, runtime_free_memory_gb=80,
        use_float64_scoring=False, compact_k1_relion_layout=True,
        compact_k1_relion_score_bpref_overlap=overlap,
        model_current_size=800, score_projector_staging_bytes=voxels * 8,
    )
    # Device input and CUDA texture coexist; BPref overlaps only in soft scoring.
    pending_bytes = voxels * (28 if overlap else 16)
    assert plan.pending_score_persistent_gb == pytest.approx(pending_bytes / 1e9)
    assert plan.persistent_estimate_gb == pytest.approx(4 + pending_bytes / 1e9)


@pytest.mark.parametrize("staging", [-1, 0.5])
def test_compact_plan_rejects_invalid_staging_byte_counts(staging):
    with pytest.raises(ValueError, match="non-negative integer"):
        _box800_plan(score_projector_staging_bytes=staging)


def test_staging_estimate_requires_explicit_compact_layout():
    with pytest.raises(ValueError, match="requires compact_k1_relion_layout"):
        _box800_plan(score_projector_staging_bytes=1)


def test_adaptive_planner_distinguishes_allocation_phases_with_equal_windows():
    from recovar.em.helpers.batch_planning import _plan_adaptive_dense_batch_sizes
    calls = []
    def fine(n_rot, n_trans, **kwargs):
        calls.append(("fine", n_rot, n_trans, kwargs["current_size_for_batch"]))
        return 7, 11
    def coarse(n_rot, n_trans, **kwargs):
        calls.append(("coarse", n_rot, n_trans, kwargs["current_size_for_batch"]))
        return 3, 5
    plan = _plan_adaptive_dense_batch_sizes(
        n_rot=40, n_trans=9, n_classes=1, image_shape=(800, 800),
        cs_for_engine=62, coarse_cs=62, safe_batch_sizes=fine,
        significance_safe_batch_sizes=coarse,
    )
    assert calls == [("fine", 40, 9, 62), ("coarse", 40, 9, 62)]
    assert (plan.pass2_image_batch_size, plan.pass2_rotation_block_size) == (7, 11)
    assert (plan.significance_image_batch_size, plan.significance_rotation_block_size) == (3, 5)


@pytest.mark.parametrize("override", [None, "dense", "fused", "local"])
def test_firstiter_compact_budget_requires_single_class_bucketed_route(monkeypatch, override):
    from recovar.em.refinement.firstiter_cc import single_class_bucketed_pass2_selected
    from recovar.em.classification.k1_local_pass2 import K1_PASS2_ENGINE_ENV
    for name in ["RECOVAR_K_CLASS_DENSE_PASS2", "RECOVAR_SPARSE_KCLASS_FUSED", K1_PASS2_ENGINE_ENV]:
        monkeypatch.delenv(name, raising=False)
    if override == "dense": monkeypatch.setenv("RECOVAR_K_CLASS_DENSE_PASS2", "1")
    if override == "fused": monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
    if override == "local": monkeypatch.setenv(K1_PASS2_ENGINE_ENV, "local")
    assert single_class_bucketed_pass2_selected(firstiter=True) is (override is None)


@pytest.mark.parametrize("override", [None, "dense", "fused", "local", "resident"])
def test_soft_compact_budget_requires_bucketed_route(monkeypatch, override):
    from recovar.em.refinement.firstiter_cc import single_class_bucketed_pass2_selected
    from recovar.em.classification.k1_local_pass2 import K1_PASS2_ENGINE_ENV
    names = {"dense": "RECOVAR_K1_DENSE_PASS2", "fused": "RECOVAR_SPARSE_KCLASS_FUSED",
             "local": K1_PASS2_ENGINE_ENV, "resident": "RECOVAR_SPARSE_PASS2_RESIDENT"}
    for name in names.values(): monkeypatch.delenv(name, raising=False)
    if override is not None: monkeypatch.setenv(names[override], "local" if override == "local" else "1")
    assert single_class_bucketed_pass2_selected(firstiter=False) is (override is None)
