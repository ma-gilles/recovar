"""Merge guards for the 174b4c09 K-class firstiter coarse-scoring work.

Companion to ``test_em_parity_lowpass_and_tau2_fudge.py`` (which locks
down the e767ec50 LP-filter + tau2_fudge fixes and the completion
baselines). This file is scoped to the K-class scoring path and the
significance-dump operand-recording schema added by 174b4c09 "Speed up
K-class firstiter coarse scoring":

  * ``_compute_k_class_significance_batched`` API additions
    (``relion_projector_half``, ``score_mode``, ``collect_significance``,
    ``return_class_best``) and the inner ``_score_block(class_index, ...)``
    first-argument convention.
  * ``use_fused_pass1`` guard set: fused env, gaussian score, no
    relion projector, no dump targets.
  * ``_maybe_dump_k_class_significance_batch`` operand kwargs
    (``shifted_data``, ``ctf2_data``, ``window_indices``,
    ``half_weights_used``) and the resulting npz schema.
  * ``RECOVAR_PASS1_FUSED`` env-var contract.
  * ``iteration_loop`` plumbing of ``RELION_WIDTH_FMASK_EDGE`` through
    to ``_reconstruct_and_postprocess_means`` (value=2 is asserted by
    the sibling lowpass-and-tau2 guard file; here we only assert it is
    actually threaded through).

Run on CPU in seconds. Their job is to fail loudly if a future EM /
VDAM / PPCA branch merge silently drops a load-bearing K-class kwarg,
swaps the fused-pass1 guard set, or breaks the dump-operand schema.

Quality of the underlying numerics is covered by the integration tests
in ``tests/integration/test_em_parity_fast.py`` and the 3 completion
baselines locked down by the sibling guard file. Don't duplicate
behavioral coverage here — these are structural merge guards.
"""

from __future__ import annotations

import inspect
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.dense_single_volume.helpers.oversampling as oversampling_mod
import recovar.em.dense_single_volume.helpers.score_constraints as score_constraints_mod
import recovar.em.dense_single_volume.helpers.significance as sig_mod
import recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed as sparse_pass2_mod
import recovar.em.dense_single_volume.iteration_loop as iteration_loop
import recovar.em.dense_single_volume.k_class as k_class_mod

pytestmark = pytest.mark.unit


def test_kclass_mstep_defaults_to_relion_x_half_with_full_and_native_escape_hatches(monkeypatch):
    """K-class quality parity should use RELION x-half BPref accumulators by default."""

    monkeypatch.delenv("RECOVAR_K_CLASS_RELION_X_HALF_MSTEP", raising=False)
    monkeypatch.delenv("RECOVAR_K_CLASS_FULL_VOLUME_MSTEP", raising=False)
    monkeypatch.delenv("RECOVAR_K_CLASS_HALF_VOLUME_MSTEP", raising=False)
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is True
    assert iteration_loop._k_class_relion_half_volume_mstep_enabled() is False

    monkeypatch.setenv("RECOVAR_K_CLASS_RELION_X_HALF_MSTEP", "0")
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is False
    assert iteration_loop._k_class_relion_half_volume_mstep_enabled() is False

    monkeypatch.setenv("RECOVAR_K_CLASS_RELION_X_HALF_MSTEP", "1")
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is True

    monkeypatch.delenv("RECOVAR_K_CLASS_RELION_X_HALF_MSTEP", raising=False)
    monkeypatch.setenv("RECOVAR_K_CLASS_FULL_VOLUME_MSTEP", "1")
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is False
    assert iteration_loop._k_class_relion_half_volume_mstep_enabled() is False

    monkeypatch.setenv("RECOVAR_K_CLASS_FULL_VOLUME_MSTEP", "0")
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is False
    assert iteration_loop._k_class_relion_half_volume_mstep_enabled() is True

    monkeypatch.setenv("RECOVAR_K_CLASS_FULL_VOLUME_MSTEP", "1")
    monkeypatch.setenv("RECOVAR_K_CLASS_HALF_VOLUME_MSTEP", "1")
    assert iteration_loop._k_class_relion_x_half_mstep_enabled() is False
    assert iteration_loop._k_class_relion_half_volume_mstep_enabled() is True


def test_k1_relion_x_half_mstep_defaults_on_with_escape_hatch(monkeypatch):
    """K=1 adaptive RELION mode should use x-half BPref layout by default."""

    monkeypatch.delenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(iteration_loop, "_k1_relion_x_half_mstep_default_available", lambda: True)
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is True

    monkeypatch.setenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, "0")
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is False

    monkeypatch.setenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is True

    monkeypatch.setenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, "invalid")
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is True


def test_k1_relion_x_half_mstep_default_disables_when_cuda_unavailable(monkeypatch):
    """The default must not request CUDA-only x-half adjoints on CPU tests."""

    monkeypatch.delenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, raising=False)
    monkeypatch.setattr(iteration_loop, "_k1_relion_x_half_mstep_default_available", lambda: False)
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is False

    monkeypatch.setenv(iteration_loop._K1_RELION_X_HALF_MSTEP_ENV, "1")
    assert iteration_loop._k1_relion_x_half_mstep_enabled() is True


def test_kclass_pass2_dump_completion_waits_for_full_target_set(tmp_path):
    """Multi-particle diagnostics must not stop after the first matching bucket."""

    kwargs = {
        "dump_dir": tmp_path,
        "target_original_indices": {17, 42},
        "target_classes_one_based": range(1, 5),
        "current_size": 74,
    }
    first_target_paths = [
        tmp_path / f"pass2_orig000017_class{class_id:03d}_cs074.npz"
        for class_id in range(1, 5)
    ]
    for path in first_target_paths:
        path.touch()
    assert sparse_pass2_mod._k_class_pass2_dump_progress(**kwargs) == (4, 8)

    for class_id in range(1, 5):
        (tmp_path / f"pass2_orig000042_class{class_id:03d}_cs074.npz").touch()
    assert sparse_pass2_mod._k_class_pass2_dump_progress(**kwargs) == (8, 8)


def test_kclass_pass2_dump_completion_honors_class_filter(tmp_path):
    """A one-class diagnostic should require one file per selected particle."""

    kwargs = {
        "dump_dir": tmp_path,
        "target_original_indices": {5, 9},
        "target_classes_one_based": {2},
        "current_size": None,
    }
    (tmp_path / "pass2_orig000005_class002_cs-01.npz").touch()
    assert sparse_pass2_mod._k_class_pass2_dump_progress(**kwargs) == (1, 2)
    (tmp_path / "pass2_orig000009_class002_cs-01.npz").touch()
    assert sparse_pass2_mod._k_class_pass2_dump_progress(**kwargs) == (2, 2)


def test_kclass_adaptive_wires_relion_x_half_without_mislabeling_dense_branch():
    source = inspect.getsource(iteration_loop._score_half_dense)
    assert "k_class_relion_x_half_mstep = _k_class_relion_x_half_mstep_enabled()" in source
    assert 'em_kwargs["mstep_relion_x_half"] = bool(k_class_relion_x_half_mstep)' in source
    assert "k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis" in source
    assert 'dense_em_kwargs.pop("mstep_relion_x_half", None)' in source
    assert "mstep_full_half_axis=k_class_mstep_full_half_axis_this_score" in source
    assert "mstep_full_half_axis=k1_adaptive_result.mstep_full_half_axis" in source
    assert "mstep_full_half_axis: int | None = None" in inspect.getsource(k_class_mod.KClassEMResult)
    assert "mstep_accumulator_shape: tuple[int, int, int] | None = None" in inspect.getsource(
        k_class_mod.KClassEMResult
    )
    assert "mstep_accumulator_shape=mstep_accumulator_shape" in inspect.getsource(k_class_mod)


def test_kclass_scatter_uses_mstep_class_mass_for_relion_priors():
    """RELION Class3D occupancies come from StoreWeightedSums, not full evidence sums."""

    stats = [
        SimpleNamespace(rotation_posterior_sums=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        SimpleNamespace(rotation_posterior_sums=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    result = SimpleNamespace(
        pose_assignments=np.array([0, 1, 2], dtype=np.int32),
        noise_stats=("class0", "class1"),
        class_assignments=np.array([0, 1, 1], dtype=np.int32),
        class_posterior_sums=np.array([1.7, 1.3], dtype=np.float32),
        class_mstep_posterior_sums=np.array([1.2, 1.8], dtype=np.float32),
        per_class_stats=stats,
        Ft_y="ft_y",
        Ft_ctf="ft_ctf",
        stats="stats",
        aggregate_noise_stats="aggregate_noise",
        best_pose_rotations=None,
        best_pose_translations=None,
    )
    class_posterior_per_half = [None]
    class_full_posterior_per_half = [None]

    iteration_loop._scatter_dense_k_class_result(
        result,
        k=0,
        effective_rotations=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        rot_pmap_for_collapse=None,
        relion_firstiter_cc_this_iter=False,
        adaptive_os_local=0,
        noise_stats_per_half_per_class=[None],
        class_assignments=[None],
        class_posterior_per_half=class_posterior_per_half,
        class_full_posterior_per_half=class_full_posterior_per_half,
        class_rotation_posterior_per_half=[None],
        best_pose_rotations=[None],
        best_pose_rotation_eulers=[None],
        best_pose_translations=[None],
        require_best_pose_details=False,
    )

    np.testing.assert_allclose(class_posterior_per_half[0], [1.2, 1.8])
    np.testing.assert_allclose(class_full_posterior_per_half[0], [1.7, 1.3])


def test_kclass_weight_trajectories_record_mstep_and_full_posterior_provenance():
    """Full-chain NPZ output must expose the class-mass split used in parity debugging."""

    from recovar.em.dense_single_volume.helpers import iteration_history

    history_source = inspect.getsource(iteration_history.RefinementHistory.record_class_weights)
    assert "self.class_mstep_weight_trajectory.append(mstep_weights)" in history_source
    assert "self.class_full_posterior_weight_trajectory.append(posterior_weights)" in history_source

    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "history.record_class_weights(" in source

    import scripts.run_full_refinement as run_full_refinement

    save_source = inspect.getsource(run_full_refinement)
    assert '"class_mstep_weight_trajectory"' in save_source
    assert '"class_full_posterior_weight_trajectory"' in save_source


def test_kclass_adaptive_dense_fallback_strips_sparse_only_x_half_flag():
    """Adaptive dense fallback must not tag full-volume dense M-steps as x-half."""

    source = inspect.getsource(k_class_mod.run_dense_k_class_em_adaptive)
    assert 'pass2_kwargs.pop("mstep_relion_x_half", False)' in source
    assert "dense backend returns full-volume accumulators" in source


def test_dense_kclass_mstep_layout_is_logged_for_parity_runs():
    """Dense pass-2 fallback must leave an audit trail for the M-step layout."""

    source = inspect.getsource(k_class_mod.run_dense_k_class_em)
    assert "Dense K-class EM M-step: using %s accumulator layout" in source
    assert '"native half-volume" if keep_half_accumulators else "full-volume"' in source


# ----------------------------------------------------------------------
# 174b4c09: K-class firstiter coarse-scoring API additions
# ----------------------------------------------------------------------


def test_kclass_significance_batched_keeps_174b4c09_api():
    """``_compute_k_class_significance_batched`` must keep the
    relion-projector / score-mode / collect-significance / return-class-best
    parameters added by 174b4c09. A merge that drops any of these would
    silently disable the K-class firstiter speedup or the parity hooks.
    """
    sig = inspect.signature(sig_mod._compute_k_class_significance_batched)
    required = {
        "relion_projector_half",
        "relion_projector_r_max",
        "score_mode",
        "collect_significance",
        "return_class_best",
    }
    missing = required - set(sig.parameters)
    assert not missing, (
        f"_compute_k_class_significance_batched is missing 174b4c09 params: {sorted(missing)}"
    )


def test_kclass_score_block_takes_class_index_first():
    """The inner ``_score_block`` closure in
    ``_compute_k_class_significance_batched`` must accept ``class_index``
    as its first positional argument (174b4c09). Without it, the
    relion-projector path indexes into the wrong volume.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    match = re.search(
        r"def _score_block\(\s*([^,)]+)\s*,",
        source,
    )
    assert match is not None, "Could not locate _score_block definition in K-class function"
    first_arg = match.group(1).strip()
    assert first_arg == "class_index", (
        f"_score_block first arg must be 'class_index' (174b4c09), got {first_arg!r}"
    )


def test_kclass_use_fused_pass1_gates_remain_in_place():
    """The K-class ``use_fused_pass1`` gate must keep all four guards:
    fused env, gaussian score_mode, no relion projector, no dump
    targets. Drop any of them and 174b4c09's speedup either fires under
    wrong conditions or silently disables itself.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    fused_idx = source.find("use_fused_pass1 = (")
    assert fused_idx >= 0, "use_fused_pass1 gate is missing from K-class function"
    # Look at the next ~400 chars for the guard clauses.
    window = source[fused_idx : fused_idx + 400]
    for needle in (
        "_pass1_fused_enabled()",
        'score_mode == "gaussian"',
        "not use_relion_projector",
        "dump_target_pre_prior_blocks_per_class is None",
        "dump_target_with_prior_blocks_per_class is None",
    ):
        assert needle in window, f"K-class use_fused_pass1 lost guard: {needle!r}"


def test_adaptive_kclass_pass1_forwards_relion_projector_kwargs():
    """Adaptive K-class pass-1 must preserve exact RELION projector inputs.

    InitialModel and projector-frame diagnostics pass ``relion_projector_half``
    through ``engine_kwargs``.  Dropping it only at the adaptive pass-1
    significance site silently reverts support selection to the JAX projector
    while pass-2 can still use RELION projector tables.
    """

    source = inspect.getsource(k_class_mod.run_dense_k_class_em_adaptive)
    sig_kwargs_idx = source.find("sig_kwargs = dict(")
    assert sig_kwargs_idx >= 0, "adaptive K-class significance kwargs block is missing"
    window = source[sig_kwargs_idx : sig_kwargs_idx + 2000]
    for needle in (
        "relion_projector_half=relion_projector_half",
        "relion_projector_r_max=relion_projector_r_max",
    ):
        assert needle in window, f"adaptive K-class pass-1 lost projector kwarg: {needle!r}"


def test_sparse_pass2_preserves_relion_projector_api_and_forwarding():
    """Adaptive sparse pass-2 must keep exact RELION projector support."""

    for func in (
        oversampling_mod.compute_pass2_stats_sparse,
        sparse_pass2_mod.compute_pass2_stats_sparse_bucketed,
        sparse_pass2_mod.compute_k_class_pass2_stats_sparse_fused,
    ):
        sig = inspect.signature(func)
        for name in ("relion_projector_half", "relion_projector_r_max"):
            assert name in sig.parameters, f"{func.__name__} lost projector parameter {name!r}"

    source = inspect.getsource(k_class_mod._run_sparse_k_class_adaptive_pass2)
    for needle in (
        'fused_common["relion_projector_half"] = relion_projector_half_by_class',
        "relion_projector_half=_select_projector_half_for_class(",
        "relion_projector_r_max=relion_projector_r_max",
    ):
        assert needle in source, f"adaptive sparse pass-2 lost projector forwarding: {needle!r}"


# ----------------------------------------------------------------------
# K-class significance dump operand-recording schema
# ----------------------------------------------------------------------


def test_kclass_dump_helper_accepts_operand_kwargs():
    """``_maybe_dump_k_class_significance_batch`` must accept the
    operand-recording kwargs (``shifted_data``, ``ctf2_data``,
    ``window_indices``, ``half_weights_used``). These let the
    significance-dump diagnostic compare RECOVAR's pass-0 operands
    against RELION's pass-0 ``Fimg`` / ``corr_img`` byte-for-byte.
    Removing them would silently collapse the dump back to the
    pre-instrumentation schema and break the RELION-parity diagnostic.
    """
    sig = inspect.signature(sig_mod._maybe_dump_k_class_significance_batch)
    required = {
        "shifted_data",
        "ctf2_data",
        "window_indices",
        "half_weights_used",
        "projected_reference_rotation_ids",
        "projected_reference_per_class",
        "projected_reference_norm_score_per_class",
        "projected_cross_score_per_class",
    }
    missing = required - set(sig.parameters)
    assert not missing, (
        f"_maybe_dump_k_class_significance_batch is missing operand kwargs: {sorted(missing)}"
    )
    for name in required:
        assert sig.parameters[name].default is None, (
            f"{name} default must stay None so callers without operands still work"
        )


def test_kclass_dump_call_site_passes_operand_kwargs():
    """The K-class dump-emission call site must keep passing the operand
    kwargs. AST-level safety net against merges that strip the kwargs at
    the call site while keeping them on the helper signature.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    call_idx = source.find("_maybe_dump_k_class_significance_batch(")
    assert call_idx >= 0, "K-class function lost its dump-emission call"
    window = source[call_idx : call_idx + 4000]
    for needle in (
        "shifted_data=shifted_data",
        "ctf2_data=ctf2_data",
        "window_indices=window_indices",
        "half_weights_used=",
        "projected_reference_rotation_ids=projected_reference_rotation_ids",
        "projected_reference_per_class=projected_reference_per_class",
        "projected_reference_norm_score_per_class=",
        "projected_cross_score_per_class=projected_cross_score_per_class",
    ):
        assert needle in window, f"K-class dump call site lost kwarg: {needle!r}"
    # The half_weights_used branch must distinguish windowed vs
    # unwindowed weights — that's how the dump records what the score
    # actually used.
    assert "half_weights_windowed if use_window else half_weights" in window, (
        "Dump call site lost the windowed/unwindowed half_weights selection"
    )


def test_kclass_significance_dump_threads_one_based_iteration():
    assert "debug_iteration" in inspect.signature(iteration_loop._score_half_dense).parameters
    assert "debug_iteration" in inspect.signature(
        k_class_mod.run_dense_k_class_em_adaptive
    ).parameters
    assert "debug_iteration" in inspect.signature(
        sig_mod._compute_k_class_significance_batched
    ).parameters
    loop_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    score_source = inspect.getsource(iteration_loop._score_half_dense)
    adaptive_source = inspect.getsource(k_class_mod.run_dense_k_class_em_adaptive)
    significance_source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    assert iteration_loop._numbered_relion_iteration(0, 0) == 1
    assert iteration_loop._numbered_relion_iteration(1, 0) == 2
    assert iteration_loop._numbered_relion_iteration(11, 2) == 14
    assert "numbered_relion_iteration = _numbered_relion_iteration(" in loop_source
    assert loop_source.count("debug_iteration=numbered_relion_iteration") >= 3
    assert score_source.count("debug_iteration=debug_iteration") >= 3
    assert adaptive_source.count("debug_iteration=debug_iteration") >= 1
    assert "debug_iteration=debug_iteration" in significance_source
    firstiter_probe_source = inspect.getsource(
        k_class_mod._run_dense_k_class_joint_firstiter_score_probe
    )
    assert "collect_significance=_significance_debug_dump_matches(" in firstiter_probe_source


def test_significance_dump_work_is_gated_before_scoring(monkeypatch, tmp_path):
    """A future-only dump request must not activate diagnostic scoring work."""

    for name in (
        "RECOVAR_SIGNIFICANCE_DUMP_DIR",
        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES",
        "RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE",
        "RECOVAR_SIGNIFICANCE_DUMP_ITERATION",
    ):
        monkeypatch.delenv(name, raising=False)

    matches = sig_mod._significance_debug_dump_matches
    assert not matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(tmp_path))
    assert not matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    assert matches(current_size=32, debug_iteration=1)

    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE", "64")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "11")
    assert not matches(current_size=32, debug_iteration=1)
    assert not matches(current_size=64, debug_iteration=1)
    assert not matches(current_size=32, debug_iteration=11)
    assert matches(current_size=64, debug_iteration=11)


def test_kclass_dump_writes_operand_arrays_to_npz(monkeypatch, tmp_path):
    """End-to-end behavioral test: a one-particle invocation of
    ``_maybe_dump_k_class_significance_batch`` with operands set writes
    them to the npz with sensible shapes/dtypes.
    """

    n_images = 1
    n_classes = 2
    n_rot = 3
    n_trans = 4
    n_pix = 5

    indices = np.array([0], dtype=np.int64)
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.array([42], dtype=np.int64),
    )

    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    translations = np.zeros((n_trans, 2), dtype=np.float32)
    class_weight_mats = [
        np.ones((n_images, n_rot * n_trans), dtype=np.float64) / (n_rot * n_trans)
        for _ in range(n_classes)
    ]
    batch_sig_mask = np.ones(
        (n_images, n_classes * n_rot * n_trans), dtype=bool
    )
    batch_n_sig = np.array([n_classes * n_rot * n_trans], dtype=np.int64)
    hard_assignment_batch = np.array([0], dtype=np.int64)
    class_assignment_batch = np.array([0], dtype=np.int64)
    global_log_z = np.array([0.0], dtype=np.float64)
    class_log_z_values = [np.array([-0.69], dtype=np.float64) for _ in range(n_classes)]
    best_score = np.array([0.0], dtype=np.float64)
    max_posterior = np.array([0.5], dtype=np.float64)
    class_log_priors = np.zeros(n_classes, dtype=np.float64)

    shifted_data = np.zeros(
        (n_images * n_trans, n_pix), dtype=np.complex128
    )
    ctf2_data = np.zeros((n_images, n_pix), dtype=np.float64)
    window_indices = np.arange(n_pix, dtype=np.int32)
    half_weights_used = np.ones(n_pix, dtype=np.float64)
    projected_reference_rotation_ids = np.asarray([0, 2], dtype=np.int32)
    projected_reference_per_class = np.arange(
        n_classes * projected_reference_rotation_ids.size * n_pix,
        dtype=np.float32,
    ).reshape(n_classes, projected_reference_rotation_ids.size, n_pix).astype(np.complex64)
    component_shape = (
        n_classes,
        n_images,
        projected_reference_rotation_ids.size,
        n_trans,
    )
    projected_reference_norm_score_per_class = np.arange(
        np.prod(component_shape), dtype=np.float64
    ).reshape(component_shape)
    projected_cross_score_per_class = (
        -projected_reference_norm_score_per_class - 1.0
    )

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "2")
    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=experiment_dataset,
        indices=indices,
        n_classes=n_classes,
        rotations=rotations,
        translations=translations,
        class_weight_mats=class_weight_mats,
        batch_sig_mask=batch_sig_mask,
        batch_n_sig=batch_n_sig,
        hard_assignment_batch=hard_assignment_batch,
        class_assignment_batch=class_assignment_batch,
        global_log_z=global_log_z,
        class_log_z_values=class_log_z_values,
        best_score=best_score,
        max_posterior=max_posterior,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=class_log_priors,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=1_000_000,
        shifted_data=shifted_data,
        ctf2_data=ctf2_data,
        window_indices=window_indices,
        half_weights_used=half_weights_used,
        projected_reference_rotation_ids=projected_reference_rotation_ids,
        projected_reference_per_class=projected_reference_per_class,
        projected_reference_norm_score_per_class=(
            projected_reference_norm_score_per_class
        ),
        projected_cross_score_per_class=projected_cross_score_per_class,
        debug_iteration=2,
    )
    files = sorted(os.listdir(dump_dir))
    assert files == ["significance_orig000042_it002_cs014.npz"]
    payload = np.load(dump_dir / files[0])
    for name in ("shifted_data", "ctf2_data", "window_indices", "half_weights"):
        assert name in payload.files, f"Dump npz is missing schema field {name!r}"
    assert payload["shifted_data"].dtype == np.complex128
    assert payload["ctf2_data"].dtype == np.float64
    assert payload["window_indices"].dtype == np.int32
    assert payload["half_weights"].dtype == np.float64
    assert payload["projected_reference_rotation_ids"].dtype == np.int32
    assert payload["projected_reference_per_class"].dtype == np.complex128
    assert payload["projected_reference_norm_score_per_class"].dtype == np.float64
    assert payload["projected_cross_score_per_class"].dtype == np.float64
    assert payload["window_indices"].shape == (n_pix,)
    assert payload["half_weights"].shape == (n_pix,)
    assert payload["projected_reference_rotation_ids"].shape == (2,)
    assert payload["projected_reference_per_class"].shape == (n_classes, 2, n_pix)
    assert payload["projected_reference_norm_score_per_class"].shape == (
        n_classes,
        2,
        n_trans,
    )
    assert payload["projected_cross_score_per_class"].shape == (
        n_classes,
        2,
        n_trans,
    )
    np.testing.assert_array_equal(
        payload["projected_reference_rotation_ids"], projected_reference_rotation_ids,
    )
    np.testing.assert_array_equal(
        payload["projected_reference_per_class"], projected_reference_per_class,
    )
    np.testing.assert_array_equal(
        payload["projected_reference_norm_score_per_class"],
        projected_reference_norm_score_per_class[:, 0],
    )
    np.testing.assert_array_equal(
        payload["projected_cross_score_per_class"],
        projected_cross_score_per_class[:, 0],
    )
    assert int(payload["n_classes"]) == n_classes
    assert int(payload["n_rot"]) == n_rot
    assert int(payload["n_trans"]) == n_trans
    assert int(payload["debug_iteration"]) == 2
    assert int(payload["one_based_iteration"]) == 2


def test_kclass_significance_dump_iteration_gate_suppresses_other_iterations(monkeypatch, tmp_path):
    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "2")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=None,
        indices=None,
        n_classes=1,
        rotations=None,
        translations=None,
        class_weight_mats=None,
        batch_sig_mask=None,
        batch_n_sig=None,
        hard_assignment_batch=None,
        class_assignment_batch=None,
        global_log_z=None,
        class_log_z_values=None,
        best_score=None,
        max_posterior=None,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=None,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=-1,
        debug_iteration=1,
    )

    assert not dump_dir.exists()


def test_kclass_significance_dump_can_stop_after_durable_target(monkeypatch, tmp_path):
    """The opt-in short-run diagnostic stops only after writing its target."""

    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "2")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")

    with pytest.raises(sig_mod.SignificanceDumpComplete) as exc_info:
        sig_mod._maybe_dump_k_class_significance_batch(
            experiment_dataset=SimpleNamespace(
                dataset_indices=np.asarray([42], dtype=np.int64),
            ),
            indices=np.asarray([0], dtype=np.int64),
            n_classes=1,
            rotations=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            translations=np.zeros((3, 2), dtype=np.float32),
            class_weight_mats=[np.full((1, 6), 1.0 / 6.0, dtype=np.float64)],
            batch_sig_mask=np.ones((1, 6), dtype=bool),
            batch_n_sig=np.asarray([6], dtype=np.int64),
            hard_assignment_batch=np.asarray([0], dtype=np.int64),
            class_assignment_batch=np.asarray([0], dtype=np.int64),
            global_log_z=np.asarray([0.0], dtype=np.float64),
            class_log_z_values=[np.asarray([0.0], dtype=np.float64)],
            best_score=np.asarray([0.0], dtype=np.float64),
            max_posterior=np.asarray([1.0], dtype=np.float64),
            rotation_log_prior_padded=None,
            batch_translation_log_prior=None,
            class_log_priors=np.zeros(1, dtype=np.float64),
            current_size=14,
            adaptive_fraction=0.999,
            max_significants=100,
            debug_iteration=2,
        )

    dump_path = dump_dir / "significance_orig000042_it002_cs014.npz"
    assert dump_path.is_file()
    assert exc_info.value.dump_path == str(dump_path)


def test_kclass_significance_stop_respects_iteration_gate(monkeypatch, tmp_path):
    """A future target must not stop the current scoring boundary."""

    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "3")
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=None,
        indices=None,
        n_classes=1,
        rotations=None,
        translations=None,
        class_weight_mats=None,
        batch_sig_mask=None,
        batch_n_sig=None,
        hard_assignment_batch=None,
        class_assignment_batch=None,
        global_log_z=None,
        class_log_z_values=None,
        best_score=None,
        max_posterior=None,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=None,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=-1,
        debug_iteration=2,
    )

    assert not dump_dir.exists()


def test_significance_stop_waits_for_complete_target_set(monkeypatch, tmp_path):
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "1")
    first_path = dump_dir / "significance_orig000042_it002_cs014.npz"
    second_path = dump_dir / "significance_orig000043_it002_cs014.npz"
    first_path.touch()

    sig_mod._maybe_stop_after_significance_dump(
        str(first_path),
        dump_dir=str(dump_dir),
        target_original_indices={42, 43},
        current_size=14,
        debug_iteration=2,
    )

    second_path.touch()
    with pytest.raises(sig_mod.SignificanceDumpComplete):
        sig_mod._maybe_stop_after_significance_dump(
            str(second_path),
            dump_dir=str(dump_dir),
            target_original_indices={42, 43},
            current_size=14,
            debug_iteration=2,
        )


def test_significance_dump_half_selector_is_scoped_to_target_iteration(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    environ = {
        "RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF": "2",
        "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET": "1",
        "RECOVAR_SIGNIFICANCE_DUMP_DIR": str(tmp_path),
        "RECOVAR_SIGNIFICANCE_DUMP_ITERATION": "2",
        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1,3",
    }

    assert iteration_loop._significance_dump_half_indices(
        numbered_iteration=1,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (0, 1)
    assert iteration_loop._significance_dump_half_indices(
        numbered_iteration=2,
        n_classes=1,
        experiment_datasets=datasets,
        environ=environ,
    ) == (1,)


def test_significance_dump_half_selector_fails_closed(tmp_path):
    datasets = [
        SimpleNamespace(dataset_indices=np.asarray([2, 4], dtype=np.int64)),
        SimpleNamespace(dataset_indices=np.asarray([1, 3], dtype=np.int64)),
    ]
    base_environ = {
        "RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF": "2",
        "RECOVAR_SIGNIFICANCE_DUMP_DIR": str(tmp_path),
        "RECOVAR_SIGNIFICANCE_DUMP_ITERATION": "2",
        "RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES": "1",
    }
    with pytest.raises(RuntimeError, match="STOP_AFTER_TARGET"):
        iteration_loop._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ=base_environ,
        )

    target_missing = dict(base_environ)
    target_missing["RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET"] = "1"
    target_missing["RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"] = "2"
    with pytest.raises(RuntimeError, match="not all present"):
        iteration_loop._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=1,
            experiment_datasets=datasets,
            environ=target_missing,
        )
    with pytest.raises(RuntimeError, match="K=1 diagnostic-only"):
        iteration_loop._significance_dump_half_indices(
            numbered_iteration=2,
            n_classes=4,
            experiment_datasets=datasets,
            environ={**base_environ, "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET": "1"},
        )


def test_relion_adaptive_fraction_preserves_text_to_float_boundary():
    expected = float(np.float32("0.999"))
    assert iteration_loop.RELION_ADAPTIVE_FRACTION == expected
    assert iteration_loop.RELION_ADAPTIVE_FRACTION != 0.999
    assert "adaptive_fraction=0.999" not in inspect.getsource(iteration_loop)

    # This two-weight boundary is intentionally between Python's binary64
    # literal and RELION's textToFloat value.  It locks down the observed
    # one-sample support effect without depending on a bulky parity fixture.
    weights = np.asarray([[0.999000006, 0.000999994]], dtype=np.float64)
    _, binary64_count = oversampling_mod._find_significant_mask_full_sort(
        weights,
        adaptive_fraction=0.999,
        max_significants=-1,
    )
    _, relion_count = oversampling_mod._find_significant_mask_full_sort(
        weights,
        adaptive_fraction=iteration_loop.RELION_ADAPTIVE_FRACTION,
        max_significants=-1,
    )
    assert int(np.asarray(binary64_count)[0]) == 1
    assert int(np.asarray(relion_count)[0]) == 2


def test_kclass_significance_dump_uses_original_index_mapper(monkeypatch, tmp_path):
    """Subset datasets must target dumps by original RELION image id.

    Directly indexing ``dataset_indices[local_index]`` is wrong for local
    image ids in subset/pass2 debug runs. Prefer the explicit mapper when the
    dataset provides one.
    """

    n_classes = 1
    n_rot = 2
    n_trans = 3
    local_index = 7

    def original_image_indices_from_local(local_indices):
        assert np.array_equal(np.asarray(local_indices), np.asarray([local_index]))
        return np.asarray([42], dtype=np.int64)

    experiment_dataset = SimpleNamespace(
        dataset_indices=np.arange(100, 200, dtype=np.int64),
        original_image_indices_from_local=original_image_indices_from_local,
    )
    dump_dir = tmp_path / "dump"
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")

    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=experiment_dataset,
        indices=np.asarray([local_index], dtype=np.int64),
        n_classes=n_classes,
        rotations=np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1)),
        translations=np.zeros((n_trans, 2), dtype=np.float32),
        class_weight_mats=[np.ones((1, n_rot * n_trans), dtype=np.float64) / (n_rot * n_trans)],
        batch_sig_mask=np.ones((1, n_classes * n_rot * n_trans), dtype=bool),
        batch_n_sig=np.asarray([n_classes * n_rot * n_trans], dtype=np.int64),
        hard_assignment_batch=np.asarray([0], dtype=np.int64),
        class_assignment_batch=np.asarray([0], dtype=np.int64),
        global_log_z=np.asarray([0.0], dtype=np.float64),
        class_log_z_values=[np.asarray([0.0], dtype=np.float64)],
        best_score=np.asarray([0.0], dtype=np.float64),
        max_posterior=np.asarray([1.0], dtype=np.float64),
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=np.zeros(n_classes, dtype=np.float64),
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=100,
    )

    payload = np.load(dump_dir / "significance_orig000042_cs014.npz")
    assert int(payload["original_index"]) == 42
    assert int(payload["local_index"]) == local_index


def test_sparse_pass2_dump_writes_score_and_recon_operand_arrays(monkeypatch, tmp_path):
    """Sparse pass-2 dumps must include the actual M-step reconstruction window.

    Score-window operands are insufficient for BPref parity once the pass uses
    separate score/reconstruction masks.
    """

    n_rot = 2
    n_trans = 3
    n_score_pix = 5
    n_recon_pix = 4
    experiment_dataset = SimpleNamespace(dataset_indices=np.array([42], dtype=np.int64))
    per_image_inputs = {
        "oversampled_rots": [np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32)],
    }
    scores = np.arange(n_rot * n_trans, dtype=np.float64).reshape(1, n_rot, n_trans)
    probs = np.full((1, n_rot, n_trans), 1.0 / (n_rot * n_trans), dtype=np.float64)
    candidate_mask = np.ones((1, n_rot, n_trans), dtype=bool)
    reconstruction_mask = candidate_mask.copy()
    shifted_score = np.ones((1, n_trans, n_score_pix), dtype=np.complex64)
    direct_score_input = np.ones((1, n_score_pix), dtype=np.complex64) * (5.0 + 6.0j)
    direct_preprocessed = np.ones((1, n_score_pix), dtype=np.complex64) * (2.0 + 1.0j)
    direct_pixel_correction = np.linspace(0.5, 1.5, n_score_pix, dtype=np.float32)[None, :]
    direct_inverse_noise = np.linspace(2.0, 3.0, n_score_pix, dtype=np.float32)
    direct_ctf_rfloat = np.linspace(0.1, 0.9, n_score_pix, dtype=np.float64)[None, :]
    shifted_recon = np.ones((1, n_trans, n_recon_pix), dtype=np.complex64) * (2.0 + 3.0j)
    ctf_score = np.ones((1, n_score_pix), dtype=np.float64)
    ctf_recon = np.ones((1, n_recon_pix), dtype=np.float64) * 4.0

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ITERATION", "2")
    monkeypatch.setitem(sparse_pass2_mod._bpref_contribution_context, "iteration", 2)
    monkeypatch.setitem(sparse_pass2_mod._bpref_contribution_context, "half", 1)
    sparse_pass2_mod._maybe_dump_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([0], dtype=np.int64),
        per_image_inputs=per_image_inputs,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=probs,
        rotation_log_prior=np.zeros((1, n_rot), dtype=np.float64),
        translation_log_prior=np.zeros((1, n_trans), dtype=np.float64),
        candidate_mask=candidate_mask,
        ctf2_over_nv_score=ctf_score,
        proj_half=np.ones((1, n_rot, n_score_pix), dtype=np.complex64),
        half_weights_used=np.ones(n_score_pix, dtype=np.float64),
        window_indices=np.arange(n_score_pix, dtype=np.int32),
        shifted_corrected_score_split=shifted_score,
        direct_score_input=direct_score_input,
        direct_preprocessed_score_input=direct_preprocessed,
        direct_pixel_correction=direct_pixel_correction,
        direct_inverse_noise_score=direct_inverse_noise,
        direct_ctf_rfloat_score=direct_ctf_rfloat,
        direct_preprocess_normalization_factors=np.asarray([0.75], dtype=np.float32),
        direct_integer_pre_shifts=np.asarray([[2, -1]], dtype=np.int32),
        direct_batch_image_corrections=np.asarray([1.5], dtype=np.float32),
        direct_batch_scale_corrections=np.asarray([2.0], dtype=np.float32),
        shifted_recon_split=shifted_recon,
        ctf2_over_nv_recon=ctf_recon,
        recon_window_indices=np.asarray([0, 2, 3, 4], dtype=np.int32),
        reconstruction_mask=reconstruction_mask,
        reconstruction_probs=probs,
        reconstruction_n_significant=np.asarray([n_rot * n_trans], dtype=np.int64),
    )

    files = sorted(os.listdir(dump_dir))
    assert files, "Sparse pass-2 dump helper failed to write any npz files"
    payload = np.load(dump_dir / files[0])
    for name in (
        "shifted_corrected",
        "direct_score_input",
        "direct_preprocessed_score_input",
        "direct_pixel_correction",
        "direct_inverse_noise_score",
        "direct_ctf_rfloat_score",
        "relion_preprocess_normalization_factor",
        "relion_integer_pre_shift",
        "batch_image_correction",
        "batch_scale_correction",
        "ctf2_over_nv_score",
        "window_indices",
        "shifted_recon",
        "ctf2_over_nv_recon",
        "recon_window_indices",
    ):
        assert name in payload.files, f"Sparse pass-2 dump npz is missing schema field {name!r}"
    assert payload["shifted_corrected"].shape == (n_trans, n_score_pix)
    np.testing.assert_array_equal(payload["direct_score_input"], direct_score_input[0])
    np.testing.assert_array_equal(
        payload["direct_preprocessed_score_input"], direct_preprocessed[0]
    )
    np.testing.assert_array_equal(
        payload["direct_pixel_correction"], direct_pixel_correction[0]
    )
    np.testing.assert_array_equal(payload["direct_inverse_noise_score"], direct_inverse_noise)
    np.testing.assert_array_equal(payload["direct_ctf_rfloat_score"], direct_ctf_rfloat[0])
    assert float(payload["relion_preprocess_normalization_factor"]) == 0.75
    np.testing.assert_array_equal(payload["relion_integer_pre_shift"], [2, -1])
    assert float(payload["batch_image_correction"]) == 1.5
    assert float(payload["batch_scale_correction"]) == 2.0
    assert payload["ctf2_over_nv_score"].shape == (n_score_pix,)
    assert payload["window_indices"].shape == (n_score_pix,)
    assert payload["shifted_recon"].shape == (n_trans, n_recon_pix)
    assert payload["ctf2_over_nv_recon"].shape == (n_recon_pix,)
    assert payload["recon_window_indices"].shape == (n_recon_pix,)
    assert payload["shifted_recon"].dtype == np.complex64
    assert payload["direct_score_input"].dtype == np.complex64
    assert payload["ctf2_over_nv_recon"].dtype == np.float64
    assert int(payload["iteration"]) == 2
    assert int(payload["half"]) == 1


def test_sparse_pass2_dump_can_retain_only_selected_rotation_rows(monkeypatch, tmp_path):
    n_rot = 4
    n_trans = 3
    n_pix = 5
    experiment_dataset = SimpleNamespace(dataset_indices=np.array([42], dtype=np.int64))
    rotations = np.arange(n_rot * 9, dtype=np.float32).reshape(n_rot, 3, 3)
    per_image_inputs = {
        "oversampled_rots": [rotations],
        "oversampled_rot_indices": [np.arange(10, 10 + n_rot, dtype=np.int64)],
        "parent_map": [np.arange(n_rot, dtype=np.int32)],
    }
    scores = np.arange(n_rot * n_trans, dtype=np.float64).reshape(1, n_rot, n_trans)
    probs = np.full((1, n_rot, n_trans), 1.0 / (n_rot * n_trans), dtype=np.float64)
    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ROTATION_ROWS", "1,3")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_RAW_OPERANDS", "1")
    raw_diff2 = np.arange(n_rot * n_trans, dtype=np.float32).reshape(
        1, n_rot, n_trans
    ) + np.float32(100)
    full_to_compact = np.asarray([-1, 0, 1, 2, 3, 4], dtype=np.int32)

    sparse_pass2_mod._maybe_dump_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([0], dtype=np.int64),
        per_image_inputs=per_image_inputs,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=probs,
        rotation_log_prior=np.zeros((1, n_rot), dtype=np.float64),
        translation_log_prior=np.zeros((1, n_trans), dtype=np.float64),
        candidate_mask=np.ones((1, n_rot, n_trans), dtype=bool),
        ctf2_over_nv_score=np.ones((1, n_pix), dtype=np.float32),
        proj_half=np.arange(n_rot * n_pix, dtype=np.float32).reshape(1, n_rot, n_pix).astype(np.complex64),
        half_weights_used=np.ones(n_pix, dtype=np.float32),
        window_indices=np.arange(n_pix, dtype=np.int32),
        shifted_corrected_score_split=np.ones((1, n_trans, n_pix), dtype=np.complex64),
        direct_score_input=np.arange(n_pix, dtype=np.float32)[None, :].astype(np.complex64),
        direct_preprocessed_score_input=(
            np.arange(n_pix, dtype=np.float32)[None, :].astype(np.complex64) + 2j
        ),
        direct_pixel_correction=np.ones((1, n_pix), dtype=np.float32) * 3,
        direct_preprocess_normalization_factors=np.asarray([0.25], dtype=np.float32),
        direct_integer_pre_shifts=np.asarray([[1, 2]], dtype=np.int32),
        direct_batch_image_corrections=np.asarray([0.5], dtype=np.float32),
        direct_batch_scale_corrections=np.asarray([2.0], dtype=np.float32),
        relion_highres_xi2_half=np.asarray([17.5], dtype=np.float32),
        relion_raw_diff2=raw_diff2,
        relion_full_to_compact=full_to_compact,
    )

    with np.load(dump_dir / "pass2_orig000042_cs014.npz", allow_pickle=False) as payload:
        assert str(payload["schema"]) == "recovar.em.k1_pass2_selected_rotations.v1"
        np.testing.assert_array_equal(payload["rotation_rows_global"], np.asarray([1, 3]))
        np.testing.assert_array_equal(payload["scores_with_prior"], scores[0, [1, 3]])
        np.testing.assert_array_equal(payload["rotations"], rotations[[1, 3]])
        assert int(payload["candidate_rotation_count"]) == n_rot
        assert int(payload["candidate_mask_total_count"]) == n_rot * n_trans
        assert float(payload["score_max"]) == float(np.max(scores))
        assert int(payload["score_argmax_rotation"]) == 3
        assert int(payload["score_argmax_translation"]) == 2
        assert float(payload["posterior_sum"]) == pytest.approx(1.0)
        assert float(payload["posterior_max"]) == pytest.approx(1.0 / (n_rot * n_trans))
        assert int(payload["posterior_argmax_rotation"]) == 0
        assert int(payload["posterior_argmax_translation"]) == 0
        assert payload["proj_half"].shape == (2, n_pix)
        assert payload["shifted_corrected"].shape == (n_trans, n_pix)
        np.testing.assert_array_equal(
            payload["direct_score_input"],
            np.arange(n_pix, dtype=np.float32).astype(np.complex64),
        )
        np.testing.assert_array_equal(
            payload["direct_preprocessed_score_input"],
            np.arange(n_pix, dtype=np.float32).astype(np.complex64) + 2j,
        )
        np.testing.assert_array_equal(payload["direct_pixel_correction"], 3)
        assert float(payload["relion_preprocess_normalization_factor"]) == 0.25
        np.testing.assert_array_equal(payload["relion_integer_pre_shift"], [1, 2])
        assert (
            str(payload["raw_operand_schema"])
            == "recovar-k1-pass2-selected-raw-operands-v1"
        )
        assert int(payload["raw_operand_actual_rotation_count"]) == 2
        np.testing.assert_array_equal(
            payload["relion_raw_diff2"], raw_diff2[0, [1, 3]]
        )
        np.testing.assert_array_equal(
            payload["raw_operand_raw_diff2"], raw_diff2[0, [1, 3]]
        )
        np.testing.assert_array_equal(
            payload["raw_operand_shifted_corrected"],
            np.ones((n_trans, n_pix), dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            payload["raw_operand_corr_img_score"],
            np.ones(n_pix, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            payload["raw_operand_proj_half"],
            np.arange(n_rot * n_pix, dtype=np.float32)
            .reshape(n_rot, n_pix)[[1, 3]]
            .astype(np.complex64),
        )
        np.testing.assert_array_equal(
            payload["raw_operand_half_weights"],
            np.ones(n_pix, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            payload["raw_operand_relion_full_to_compact"], full_to_compact
        )
        assert float(payload["raw_operand_highres_xi2_half"]) == 17.5


def test_sparse_pass2_raw_operand_dump_fails_closed_without_raw_diff2(
    monkeypatch,
    tmp_path,
):
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.asarray([42], dtype=np.int64)
    )
    per_image_inputs = {
        "oversampled_rots": [np.eye(3, dtype=np.float32)[None]],
        "oversampled_rot_indices": [np.asarray([7], dtype=np.int64)],
        "parent_map": [np.asarray([0], dtype=np.int32)],
    }
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ROTATION_ROWS", "0")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_RAW_OPERANDS", "1")

    with pytest.raises(ValueError, match="requires the production K=1 RELION raw-diff2"):
        sparse_pass2_mod._maybe_dump_pass2_bucket(
            experiment_dataset=experiment_dataset,
            image_indices=np.asarray([0], dtype=np.int64),
            per_image_inputs=per_image_inputs,
            current_size=14,
            n_fine_trans=1,
            fine_translations=np.zeros((1, 2), dtype=np.float32),
            scores=np.zeros((1, 1, 1), dtype=np.float32),
            probs=np.ones((1, 1, 1), dtype=np.float32),
            rotation_log_prior=np.zeros((1, 1), dtype=np.float32),
            translation_log_prior=np.zeros((1, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 1, 1), dtype=bool),
            ctf2_over_nv_score=np.ones((1, 2), dtype=np.float32),
            proj_half=np.ones((1, 1, 2), dtype=np.complex64),
            half_weights_used=np.ones(2, dtype=np.float32),
            window_indices=np.arange(2, dtype=np.int32),
            shifted_corrected_score_split=np.ones(
                (1, 1, 2), dtype=np.complex64
            ),
        )


def test_sparse_pass2_dump_uses_original_index_mapper(monkeypatch, tmp_path):
    """Sparse pass-2 dumps use the same original-id targeting as pass1."""

    n_rot = 2
    n_trans = 3
    n_pix = 5
    local_index = 7

    def original_image_indices_from_local(local_indices):
        assert np.array_equal(np.asarray(local_indices), np.asarray([local_index]))
        return np.asarray([42], dtype=np.int64)

    experiment_dataset = SimpleNamespace(
        dataset_indices=np.arange(100, 200, dtype=np.int64),
        original_image_indices_from_local=original_image_indices_from_local,
    )
    rotations_for_image = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    per_image_inputs = {
        "oversampled_rots": [rotations_for_image.copy() for _ in range(local_index + 1)],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64) for _ in range(local_index + 1)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32) for _ in range(local_index + 1)],
    }

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    sparse_pass2_mod._maybe_dump_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([local_index], dtype=np.int64),
        per_image_inputs=per_image_inputs,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=np.zeros((1, n_rot, n_trans), dtype=np.float64),
        probs=np.full((1, n_rot, n_trans), 1.0 / (n_rot * n_trans), dtype=np.float64),
        rotation_log_prior=np.zeros((1, n_rot), dtype=np.float64),
        translation_log_prior=np.zeros((1, n_trans), dtype=np.float64),
        candidate_mask=np.ones((1, n_rot, n_trans), dtype=bool),
        ctf2_over_nv_score=np.ones((1, n_pix), dtype=np.float64),
        proj_half=np.ones((1, n_rot, n_pix), dtype=np.complex64),
        half_weights_used=np.ones(n_pix, dtype=np.float64),
        window_indices=np.arange(n_pix, dtype=np.int32),
    )

    payload = np.load(dump_dir / "pass2_orig000042_cs014.npz")
    assert int(payload["original_index"]) == 42
    assert int(payload["local_index"]) == local_index
    assert payload["recon_window_indices"].dtype == np.int32


def test_kclass_compact_pass2_dump_uses_original_index_mapper(monkeypatch, tmp_path):
    """K-class compact-pair pass-2 diagnostics must target original image ids."""

    n_rot = 2
    n_trans = 3
    local_index = 7

    def original_image_indices_from_local(local_indices):
        assert np.array_equal(np.asarray(local_indices), np.asarray([local_index]))
        return np.asarray([42], dtype=np.int64)

    experiment_dataset = SimpleNamespace(
        dataset_indices=np.arange(100, 200, dtype=np.int64),
        original_image_indices_from_local=original_image_indices_from_local,
    )
    rotations_for_image = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    per_image_inputs = {
        "oversampled_rots": [rotations_for_image.copy() for _ in range(local_index + 1)],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64) for _ in range(local_index + 1)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32) for _ in range(local_index + 1)],
        "log_prior": [np.asarray([0.1, -0.2], dtype=np.float32) for _ in range(local_index + 1)],
    }
    compact_pair_arrays = {
        "pair_mask": np.asarray([[True, True, False]], dtype=bool),
        "local_rotation_row": np.asarray([[0, 1, 0]], dtype=np.int64),
        "translation_idx": np.asarray([[1, 2, 0]], dtype=np.int64),
    }
    translation_prior = np.asarray([[0.25, -0.5, 0.75]], dtype=np.float32)
    scores = np.asarray([[3.5, 5.5, -np.inf]], dtype=np.float64)
    probs = np.asarray([[0.2, 0.8, 0.0]], dtype=np.float64)
    raw_diff2 = np.asarray([101.0, 102.0, 999.0], dtype=np.float32)
    min_diff2 = np.asarray([100.0], dtype=np.float32)

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CLASS", "2")
    sparse_pass2_mod._maybe_dump_k_class_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([local_index], dtype=np.int64),
        class_index=0,
        per_image_inputs=per_image_inputs,
        class_bucket_arrays={},
        compact_pair_arrays=compact_pair_arrays,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=probs,
        bucket_translation_prior=translation_prior,
        compact_pairs=True,
    )
    assert not list(dump_dir.glob("*.npz"))

    sparse_pass2_mod._maybe_dump_k_class_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([local_index], dtype=np.int64),
        class_index=1,
        per_image_inputs=per_image_inputs,
        class_bucket_arrays={},
        compact_pair_arrays=compact_pair_arrays,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=probs,
        bucket_translation_prior=translation_prior,
        compact_pairs=True,
        raw_diff2_by_batch_row={0: raw_diff2},
        relion_min_diff2=min_diff2,
    )

    payload = np.load(dump_dir / "pass2_orig000042_class002_cs014.npz")
    assert int(payload["original_index"]) == 42
    assert int(payload["local_index"]) == local_index
    assert int(payload["class_index"]) == 1
    np.testing.assert_array_equal(payload["oversampled_rot_indices"], np.asarray([10, 11]))
    assert payload["candidate_mask"].tolist() == [[False, True, False], [False, False, True]]
    assert payload["probs"][0, 1] == pytest.approx(0.2)
    assert payload["probs"][1, 2] == pytest.approx(0.8)
    assert payload["scores_pre_prior"][0, 1] == pytest.approx(3.5 - 0.1 - (-0.5))
    assert payload["scores_pre_prior"][1, 2] == pytest.approx(5.5 - (-0.2) - 0.75)
    assert payload["relion_raw_diff2"].dtype == np.float32
    assert payload["relion_raw_diff2"][0, 1] == np.float32(101.0)
    assert payload["relion_raw_diff2"][1, 2] == np.float32(102.0)
    assert np.isnan(payload["relion_raw_diff2"][0, 0])
    assert payload["relion_min_diff2"] == np.float32(100.0)


def test_kclass_dense_pass2_dump_preserves_selected_raw_diff2(monkeypatch, tmp_path):
    n_rot = 2
    n_trans = 3
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.asarray([42], dtype=np.int64),
    )
    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    per_image_inputs = {
        "oversampled_rots": [rotations],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32)],
        "log_prior": [np.asarray([0.1, -0.2], dtype=np.float32)],
    }
    candidate_mask = np.asarray(
        [[[True, False, True], [False, True, False]]],
        dtype=bool,
    )
    scores = np.arange(
        n_rot * n_trans,
        dtype=np.float32,
    ).reshape(1, n_rot, n_trans)
    raw_diff2 = (
        np.arange(n_rot * n_trans, dtype=np.float32)
        .reshape(n_rot, n_trans)
        + np.float32(500.0)
    )

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    sparse_pass2_mod._maybe_dump_k_class_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([0], dtype=np.int64),
        class_index=0,
        per_image_inputs=per_image_inputs,
        class_bucket_arrays={"candidate_mask": candidate_mask},
        compact_pair_arrays=None,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=scores,
        probs=np.full_like(scores, 1.0 / scores.size),
        bucket_translation_prior=np.zeros((1, n_trans), dtype=np.float32),
        compact_pairs=False,
        raw_diff2_by_batch_row={0: raw_diff2},
        relion_min_diff2=np.asarray([499.0], dtype=np.float32),
    )

    payload = np.load(dump_dir / "pass2_orig000042_class001_cs014.npz")
    np.testing.assert_array_equal(payload["relion_raw_diff2"], raw_diff2)
    assert payload["relion_min_diff2"] == np.float32(499.0)


def test_kclass_pass2_dump_preserves_effective_raw_operands(monkeypatch, tmp_path):
    n_rot = 2
    n_trans = 3
    n_pix = 4
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.asarray([42], dtype=np.int64),
    )
    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    per_image_inputs = {
        "oversampled_rots": [rotations],
        "oversampled_rot_indices": [np.asarray([10, 11], dtype=np.int64)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32)],
        "log_prior": [np.asarray([0.1, -0.2], dtype=np.float32)],
    }
    candidate_mask = np.ones((1, n_rot, n_trans), dtype=bool)
    shifted_corrected = (
        np.arange(n_trans * n_pix, dtype=np.float32).reshape(n_trans, n_pix)
        + 1j
    ).astype(np.complex64)
    proj_half = (
        np.arange(n_rot * n_pix, dtype=np.float32).reshape(n_rot, n_pix)
        - 2j
    ).astype(np.complex64)
    corr_img_score = np.arange(n_pix, dtype=np.float32) + 0.5
    half_weights = np.arange(n_pix, dtype=np.float32) + 1.0
    full_to_compact = np.asarray([0, -1, 1, 2, 3], dtype=np.int32)
    pair_mask = np.asarray([[True, True, False, False]], dtype=bool)
    pair_rotation_row = np.asarray([[0, 1, 0, 0]], dtype=np.int32)
    pair_translation_idx = np.asarray([[1, 2, 0, 0]], dtype=np.int32)
    raw_operands = sparse_pass2_mod._capture_k_class_pass2_raw_operands(
        raw_diff2=np.zeros((1, pair_mask.shape[1]), dtype=np.float32),
        target_rows=np.asarray([0], dtype=np.int64),
        actual_counts=np.asarray([n_rot], dtype=np.int64),
        shifted_corrected=shifted_corrected[None, ...],
        corr_img_score=corr_img_score[None, ...],
        proj_half=proj_half[None, ...],
        half_weights=half_weights,
        relion_full_to_compact=full_to_compact,
        highres_xi2_half=np.asarray([7.25], dtype=np.float32),
        pair_mask=pair_mask,
        pair_rotation_row=pair_rotation_row,
        pair_translation_idx=pair_translation_idx,
    )

    dump_dir = tmp_path / "pass2"
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    sparse_pass2_mod._maybe_dump_k_class_pass2_bucket(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([0], dtype=np.int64),
        class_index=0,
        per_image_inputs=per_image_inputs,
        class_bucket_arrays={"candidate_mask": candidate_mask},
        compact_pair_arrays=None,
        current_size=14,
        n_fine_trans=n_trans,
        fine_translations=np.zeros((n_trans, 2), dtype=np.float32),
        scores=np.zeros((1, n_rot, n_trans), dtype=np.float32),
        probs=np.full((1, n_rot, n_trans), 1.0 / (n_rot * n_trans)),
        bucket_translation_prior=np.zeros((1, n_trans), dtype=np.float32),
        compact_pairs=False,
        raw_operands_by_batch_row=raw_operands,
    )

    payload = np.load(dump_dir / "pass2_orig000042_class001_cs014.npz")
    assert str(payload["raw_operand_schema"]) == (
        "recovar-kclass-pass2-effective-raw-operands-v2"
    )
    assert int(payload["raw_operand_actual_rotation_count"]) == n_rot
    np.testing.assert_array_equal(
        payload["raw_operand_raw_diff2"],
        np.zeros(pair_mask.shape[1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        payload["raw_operand_shifted_corrected"],
        shifted_corrected,
    )
    np.testing.assert_array_equal(payload["raw_operand_proj_half"], proj_half)
    np.testing.assert_array_equal(
        payload["raw_operand_corr_img_score"],
        corr_img_score,
    )
    np.testing.assert_array_equal(
        payload["raw_operand_half_weights"],
        half_weights,
    )
    np.testing.assert_array_equal(
        payload["raw_operand_relion_full_to_compact"],
        full_to_compact,
    )
    assert payload["raw_operand_highres_xi2_half"] == np.float32(7.25)
    np.testing.assert_array_equal(payload["raw_operand_pair_mask"], pair_mask[0])
    np.testing.assert_array_equal(
        payload["raw_operand_pair_rotation_row"],
        pair_rotation_row[0],
    )
    np.testing.assert_array_equal(
        payload["raw_operand_pair_translation_idx"],
        pair_translation_idx[0],
    )


def test_pass2_dump_target_rows_use_original_index_mapping(monkeypatch, tmp_path):
    experiment_dataset = SimpleNamespace(
        original_image_indices_from_local=lambda indices: np.asarray(
            [100, 42, 300],
            dtype=np.int64,
        )
    )
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42,300")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_CURRENT_SIZE", "14")

    rows = sparse_pass2_mod._pass2_dump_target_rows(
        experiment_dataset=experiment_dataset,
        image_indices=np.asarray([7, 8, 9], dtype=np.int64),
        current_size=14,
    )

    np.testing.assert_array_equal(rows, np.asarray([1, 2], dtype=np.int64))


# ----------------------------------------------------------------------
# Pass1 fused gate (env-var contract)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),         # unset → off (174b4c09 ships with default off while validated)
        ("", False),
        ("0", False),
        ("no", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),        # case-insensitive
        ("yes", True),
        ("YES", True),
        ("on", True),
        ("On", True),
    ],
)
def test_pass1_fused_enabled_env_var_contract(monkeypatch, value, expected):
    """``RECOVAR_PASS1_FUSED`` is the public opt-in for the fused pass1
    path; the K-class call site at ``use_fused_pass1 = ...`` reads it.
    The string contract is stable: 1/true/yes/on are truthy
    (case-insensitive), anything else is off, unset is off.
    """
    if value is None:
        monkeypatch.delenv("RECOVAR_PASS1_FUSED", raising=False)
    else:
        monkeypatch.setenv("RECOVAR_PASS1_FUSED", value)
    assert sig_mod._pass1_fused_enabled() is expected


def test_normalized_cc_firstiter_ignores_log_priors():
    """RELION firstiter normalized-CC WTA uses raw scores, not priors."""

    score_constraints_source = inspect.getsource(sig_mod)
    assert 'if score_mode == "normalized_cc":\n            return scores' in score_constraints_source

    dense_constraints_source = inspect.getsource(score_constraints_mod.apply_dense_score_constraints)
    assert 'if score_mode != "normalized_cc":' in dense_constraints_source
    assert "scores = scores + rotation_prior[:, :, None]" in dense_constraints_source
    assert "scores = scores + translation_prior[:, None, :]" in dense_constraints_source

    dense_source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    assert "scores = _add_priors(scores, class_index, r0, r1, batch_translation_log_prior)" in dense_source
    assert "scores_pre_prior" in dense_source
    assert "scores_with_prior" in dense_source


def test_adaptive_significance_forwards_firstiter_score_mode():
    """No-shortcut firstiter diagnostics must still use normalized-CC pass-1 scoring."""

    source = inspect.getsource(k_class_mod.run_dense_k_class_em_adaptive)
    assert 'score_mode=engine_kwargs.get("relion_firstiter_score_mode", "gaussian")' in source


# ----------------------------------------------------------------------
# WIDTH_FMASK_EDGE plumbing through iteration_loop
# (Constant-value assertion lives in test_em_parity_lowpass_and_tau2_fudge.py;
#  here we only assert the constant is actually threaded through to the
#  postprocess call. Plumbing is what breaks in merges.)
# ----------------------------------------------------------------------


def test_iteration_loop_threads_fmask_edge_through_to_postprocess():
    """``_run_relion_iteration_loop`` must forward
    ``RELION_WIDTH_FMASK_EDGE`` to ``_reconstruct_and_postprocess_means``
    via the ``relion_fmask_edge`` kwarg. A merge that defines the
    constant but stops threading it leaves the LP filter using the
    real-space mask edge (RELION ``WIDTH_FMASK_EDGE`` vs
    ``--maskedge`` are different units).
    """
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "relion_fmask_edge=RELION_WIDTH_FMASK_EDGE" in source, (
        "iteration_loop must forward RELION_WIDTH_FMASK_EDGE to _reconstruct_and_postprocess_means"
    )
