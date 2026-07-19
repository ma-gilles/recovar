"""Unit tests for ``scripts/run_full_refinement.py::_build_replay_iteration_overrides``.

Locks down the parity-critical contract that the per-iter replay override
dict always carries ``translation_sigma_angstrom`` sourced from RELION's
``rlnSigmaOffsetsAngst``. Without that, recovar's iter-1 leaves
``current_sigma_offset_angstrom`` at the 10 Å default and iter-2's
translation prior is ~6× too wide → iter-2 ave_Pmax is depressed by ~22 %
relative to RELION (cf. iteration_loop.py:4667-4703).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
    FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
)
from recovar.em.dense_single_volume.iteration_loop import (
    _assert_frozen_scoring_state_unchanged,
    _frozen_scoring_state_arrays,
    _mean_variance_for_scoring_half,
    _updated_mean_variance_per_half,
)
from scripts import run_full_refinement
from scripts.run_full_refinement import (
    _assert_frozen_replay_slots_projector_only,
    _attach_relion_projector_capture,
    _build_frozen_replay_slots,
    _build_replay_iteration_overrides,
    _default_refinement_subsets,
    _fixed_diagnostic_source_paths,
    _format_replay_mean_for_log,
    _load_init_noise_radial_npz,
    _load_init_previous_best_poses_npz,
    _load_initial_noise_cache,
    _load_native_group_ids_per_half,
    _load_relion_it000_model_stars,
    _load_replay_group_particles,
    _make_frozen_boundary_noise_variance,
    _maybe_apply_relion_image_mask,
    _parse_relion_cli_ini_high,
    _parse_relion_tau2_fudge,
    _read_relion_single_optics_sigma2_noise,
    _relion_halfset_and_accuracy_layout,
    _relion_mpi_process_start_scoring_noise_pair,
    _relion_optimiser_star_for_runtime,
    _replay_complete_initial_particle_state,
    _resolve_native_group_layout,
    _resolve_replay_normcorr,
    _resolve_tau2_fudge,
    _save_initial_noise_cache,
    _select_authoritative_group_particles,
    _validate_fixed_diagnostic_arm_cli,
    _validate_fixed_diagnostic_math_environment,
    _verify_fixed_diagnostic_provenance_manifests,
    _verify_frozen_boundary_source_hashes,
)

FIXTURE = Path("/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0")
RUN_FULL_REFINEMENT = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"
ITERATION_LOOP = (
    Path(__file__).resolve().parents[2] / "recovar" / "em" / "dense_single_volume" / "iteration_loop.py"
)


def test_complete_initial_particle_state_is_autorefine_only():
    assert _replay_complete_initial_particle_state(1, 0)
    assert not _replay_complete_initial_particle_state(4, 0)
    assert not _replay_complete_initial_particle_state(1, 1)


def test_frozen_replay_is_exactly_projector_only():
    slots = _build_frozen_replay_slots(1)
    projector = object()
    slots[0]["relion_projector_state"] = projector

    _assert_frozen_replay_slots_projector_only(slots, projector_slot=0)

    assert slots == [{"relion_projector_state": projector}, {}]


@pytest.mark.parametrize("field", ["noise_variance", "class_tau2", "future_override"])
def test_frozen_replay_rejects_any_nonprojector_override(field):
    slots = _build_frozen_replay_slots(1)
    slots[0]["relion_projector_state"] = object()
    slots[0][field] = object()

    with pytest.raises(ValueError, match="not projector-only"):
        _assert_frozen_replay_slots_projector_only(slots, projector_slot=0)


def test_frozen_replay_rejects_projector_in_final_slot():
    slots = _build_frozen_replay_slots(1)
    slots[1]["relion_projector_state"] = object()

    with pytest.raises(ValueError, match="exactly one projector"):
        _assert_frozen_replay_slots_projector_only(slots, projector_slot=0)


def test_frozen_scoring_state_negative_overwrite_regression():
    half_inputs = SimpleNamespace(
        previous_best_rotation_eulers=[
            np.zeros((2, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ],
        previous_best_translations=[
            np.zeros((2, 2), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
        ],
        image_corrections=[
            np.ones(2, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ],
        scale_corrections=[
            np.ones(2, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ],
    )
    expected = _frozen_scoring_state_arrays(
        means=[
            np.zeros(8, dtype=np.complex64),
            np.ones(8, dtype=np.complex64),
        ],
        mean_variance=np.ones(8, dtype=np.float32),
        relion_half_inputs=half_inputs,
        noise_variance_per_half=[
            np.ones(4, dtype=np.float32),
            np.full(4, 2.0, dtype=np.float32),
        ],
        current_sigma_offset_angstrom_per_half=[2.0, 3.0],
        global_direction_prior_per_half=[
            np.full(12, 1.0 / 12.0, dtype=np.float32),
            np.full(12, 1.0 / 12.0, dtype=np.float32),
        ],
        sealed_scoring_context={"slot": 0, "mode": "fixed"},
    )
    assert "mean_variance" in expected
    assert "mean_variance.half1" not in expected
    assert "mean_variance.half2" not in expected
    assert expected["sealed_scoring_context_json"].tobytes().decode("utf-8") == (
        '{"mode":"fixed","slot":0}'
    )
    actual = {name: value.copy() for name, value in expected.items()}
    actual["noise_variance.half2"][0] = actual["noise_variance.half1"][0]

    with pytest.raises(RuntimeError, match="noise_variance.half2 was overwritten"):
        _assert_frozen_scoring_state_unchanged(expected, actual)

    changed_map = {name: value.copy() for name, value in expected.items()}
    changed_map["mean.half1"][0] = np.complex64(1.0 + 0.0j)
    with pytest.raises(RuntimeError, match="mean.half1 was overwritten"):
        _assert_frozen_scoring_state_unchanged(expected, changed_map)

    changed_tau2 = {name: value.copy() for name, value in expected.items()}
    changed_tau2["mean_variance"][0] = np.float32(2.0)
    with pytest.raises(RuntimeError, match="mean_variance was overwritten"):
        _assert_frozen_scoring_state_unchanged(expected, changed_tau2)


def test_unequal_half_tau2_is_dispatched_without_collapsing():
    half1 = np.asarray([1.0, 2.0], dtype=np.float32)
    half2 = np.asarray([3.0, 4.0], dtype=np.float32)

    selected1 = _mean_variance_for_scoring_half([half1, half2], 0)
    selected2 = _mean_variance_for_scoring_half([half1, half2], 1)

    assert selected1 is half1
    assert selected2 is half2
    assert not np.array_equal(selected1, selected2)


def test_ordinary_k1_keeps_shared_tau2_across_multiple_updates():
    """Diagnostic per-half tau2 must not alter the historical K=1 scorer."""

    for iteration in range(2):
        shared = np.asarray([10.0 + iteration, 20.0 + iteration], dtype=np.float32)
        candidate_per_half = [
            np.asarray([1.0 + iteration, 2.0], dtype=np.float32),
            np.asarray([3.0 + iteration, 4.0], dtype=np.float32),
        ]
        scoring_tau2 = _updated_mean_variance_per_half(
            shared,
            candidate_per_half,
            use_per_half_mean_variance=False,
        )

        assert _mean_variance_for_scoring_half(scoring_tau2, 0) is shared
        assert _mean_variance_for_scoring_half(scoring_tau2, 1) is shared


def test_fixed_arm_can_keep_per_half_tau2_across_updates():
    shared = np.asarray([10.0, 20.0], dtype=np.float32)
    candidate_per_half = [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0, 4.0], dtype=np.float32),
    ]

    scoring_tau2 = _updated_mean_variance_per_half(
        shared,
        candidate_per_half,
        use_per_half_mean_variance=True,
    )

    np.testing.assert_array_equal(
        _mean_variance_for_scoring_half(scoring_tau2, 0), candidate_per_half[0]
    )
    np.testing.assert_array_equal(
        _mean_variance_for_scoring_half(scoring_tau2, 1), candidate_per_half[1]
    )


def test_default_state_swap_resyncs_both_scorer_halves_to_substituted_shared_tau2():
    stale_per_half = [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0, 4.0], dtype=np.float32),
    ]
    substituted_shared = np.asarray([30.0, 40.0], dtype=np.float32)

    scoring_tau2 = _updated_mean_variance_per_half(
        substituted_shared,
        stale_per_half,
        use_per_half_mean_variance=False,
    )

    assert _mean_variance_for_scoring_half(scoring_tau2, 0) is substituted_shared
    assert _mean_variance_for_scoring_half(scoring_tau2, 1) is substituted_shared


def test_fixed_arm_provenance_manifests_fail_closed_on_environment_tamper(
    monkeypatch, tmp_path
):
    commit = "1" * 40
    boundary = SimpleNamespace(
        runtime_config={
            "diagnostic_arm_id": "real10076.k1.physical_it2.reconstructed_projector.v1",
            "jax_enable_x64": True,
            "provenance_verification_scope": FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
            "numerical_classification_scope": FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
            "recovar_git_commit": commit,
            "declared_relion_base_git_commit": "2" * 40,
            "declared_relion_command_line": "relion_refine --continue run_it001_optimiser.star",
            "declared_relion_build_id": "relion-test-build",
            "projector_boundary_kind": "reconstructed-projector boundary",
        }
    )
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        '{"recovar_git_commit":"' + commit + '","schema":"recovar.em.source_manifest.v1","worktree_clean":true}',
        encoding="utf-8",
    )
    environment_manifest = tmp_path / "environment.json"
    expected_environment = {
        "schema": "recovar.em.runtime_environment.v1",
        "diagnostic_arm_id": boundary.runtime_config["diagnostic_arm_id"],
        "math_environment_contract": "no_unsealed_recovar_jax_xla_overrides.v1",
        "jax_enable_x64": True,
        "provenance_verification_scope": FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
        "numerical_classification_scope": FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
        "declared_relion_command_line": boundary.runtime_config["declared_relion_command_line"],
        "declared_relion_base_git_commit": boundary.runtime_config["declared_relion_base_git_commit"],
        "declared_relion_build_id": boundary.runtime_config["declared_relion_build_id"],
        "recovar_git_commit": commit,
        "projector_boundary_kind": "reconstructed-projector boundary",
    }
    import json

    environment_manifest.write_text(json.dumps(expected_environment), encoding="utf-8")
    monkeypatch.setattr(
        run_full_refinement,
        "git_worktree_provenance",
        lambda: {"head": commit, "dirty_count": 0},
    )
    paths = {
        "recovar_source_manifest": source_manifest,
        "runtime_environment_manifest": environment_manifest,
    }

    _verify_fixed_diagnostic_provenance_manifests(boundary, paths)

    expected_environment["declared_relion_build_id"] = "tampered"
    environment_manifest.write_text(json.dumps(expected_environment), encoding="utf-8")
    with pytest.raises(ValueError, match="command/build/environment"):
        _verify_fixed_diagnostic_provenance_manifests(boundary, paths)


def _fixed_diagnostic_args():
    return SimpleNamespace(
        max_iter=1,
        skip_final_iteration=True,
        init_resolution=30.0,
        offset_range=3.0,
        offset_step=1.0,
        perturb_factor=0.5,
        adaptive_oversampling=1,
        n_classes=1,
        firstiter_cc=True,
        apply_initial_lowpass=False,
        image_fourier_backend="relion_cuda",
        final_replay_relion_dir=None,
        relion_projector_capture_dir=None,
        relion_projector_capture_manifest=None,
        relion_projector_capture_iteration=None,
        perturb_replay_restart_provenance=None,
        relion_dispatch_schedule=None,
        relion_follower_scale_replay=None,
        init_class_volumes=None,
        init_volume=None,
        init_previous_best_poses_npz=None,
        init_noise_from_npz=None,
        initial_noise_cache_dir=None,
        relion_init_dir=None,
        relion_optimiser=None,
        relion_current_sizes=None,
        relion_healpix_orders=None,
        stop_after_local_search_profile=False,
        stop_after_local_search=False,
        stop_after_local_search_score_only=False,
        diagnostic_single_half=False,
        perturb_replay_restart_state_iterations="",
        replay_relion_normcorr=None,
        relion_scale_followers=None,
    )


def test_fixed_diagnostic_arm_rejects_alternate_projector_and_float_mode():
    args = _fixed_diagnostic_args()
    _validate_fixed_diagnostic_arm_cli(args)
    _validate_fixed_diagnostic_math_environment(
        {
            "RECOVAR_EXPECTED_REPO_ROOT": "/repo",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        }
    )

    args.relion_projector_capture_dir = "/substitute"
    with pytest.raises(ValueError, match="alternate state/projector/oracle"):
        _validate_fixed_diagnostic_arm_cli(args)
    with pytest.raises(ValueError, match="unsealed RECOVAR environment"):
        _validate_fixed_diagnostic_math_environment(
            {"RECOVAR_USE_FLOAT64_SCORING": "1"}
        )


@pytest.mark.parametrize(
    "xla_flags",
    [
        "--xla_gpu_enable_triton_gemm=true",
        "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    ],
)
def test_fixed_diagnostic_arm_rejects_nondefault_xla_flags(xla_flags):
    with pytest.raises(ValueError, match="unsealed compiler/precision environment: XLA_FLAGS"):
        _validate_fixed_diagnostic_math_environment({"XLA_FLAGS": xla_flags})


def test_fixed_arm_rejects_explicit_optimiser_and_internal_resolver_uses_sealed_source(
    tmp_path,
):
    unsealed = tmp_path / "unsealed_optimiser.star"
    sealed = tmp_path / "completed_optimiser.star"
    unsealed.write_text("unsealed", encoding="utf-8")
    sealed.write_text("sealed", encoding="utf-8")
    args = _fixed_diagnostic_args()
    args.relion_optimiser = str(unsealed)

    with pytest.raises(ValueError, match="alternate state/projector/oracle"):
        _validate_fixed_diagnostic_arm_cli(args)

    boundary = SimpleNamespace(fixed_diagnostic_arm=True)
    assert _relion_optimiser_star_for_runtime(
        args,
        frozen_boundary=boundary,
        fixed_diagnostic_source_paths={"completed_optimiser": sealed},
    ) == sealed.resolve()


def test_fixed_arm_rejects_mask_cli_values_that_differ_from_sealed_optimiser(tmp_path):
    sealed = tmp_path / "completed_optimiser.star"
    sealed.write_text(
        "_rlnParticleDiameter 280\n_rlnWidthMaskEdge 5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="particle diameter differs"):
        _maybe_apply_relion_image_mask(
            None,
            SimpleNamespace(particle_diameter_ang=279.0, width_mask_edge_px=5.0),
            sealed_optimiser_star=sealed,
        )
    with pytest.raises(ValueError, match="mask-edge width differs"):
        _maybe_apply_relion_image_mask(
            None,
            SimpleNamespace(particle_diameter_ang=None, width_mask_edge_px=4.0),
            sealed_optimiser_star=sealed,
        )


def test_fixed_diagnostic_source_paths_reject_runtime_prefix_or_consumer_substitution(tmp_path):
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        perturb_replay_relion_dir=str(tmp_path),
        relion_half_sets=str(tmp_path / "half.star"),
        frozen_boundary_live_capture_manifest=str(tmp_path / "capture.sha256"),
        frozen_boundary_runtime_environment_manifest=str(tmp_path / "environment.json"),
        frozen_boundary_recovar_source_manifest=str(tmp_path / "source.json"),
        frozen_boundary_replay_prefix="substituted",
    )
    boundary = SimpleNamespace(
        completed_relion_iteration=7,
        consumer_relion_iteration=8,
        sampling_state={"consumer_relion_iteration": 8},
        runtime_config={"replay_prefix": "sealed"},
    )

    with pytest.raises(ValueError, match="replay prefix mismatch"):
        _fixed_diagnostic_source_paths(args, boundary)

    args.frozen_boundary_replay_prefix = "sealed"
    boundary.sampling_state["consumer_relion_iteration"] = 9
    with pytest.raises(ValueError, match="consumer iteration ownership"):
        _fixed_diagnostic_source_paths(args, boundary)


def test_frozen_boundary_source_hashes_bind_live_stars(tmp_path):
    source_star = tmp_path / "particles.star"
    half_star = tmp_path / "run_it002_data.star"
    source_star.write_text("source\n", encoding="utf-8")
    half_star.write_text("halves\n", encoding="utf-8")
    boundary = SimpleNamespace(
        source_star_sha256=run_full_refinement._sha256_file(source_star),
        relion_half_star_sha256=run_full_refinement._sha256_file(half_star),
    )

    _verify_frozen_boundary_source_hashes(
        boundary,
        source_star=source_star,
        relion_half_star=half_star,
    )

    half_star.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RELION half-set STAR SHA-256 mismatch"):
        _verify_frozen_boundary_source_hashes(
            boundary,
            source_star=source_star,
            relion_half_star=half_star,
        )


def test_frozen_boundary_schedule_is_threaded_exactly_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["init_current_size"], ast.Name)
    assert keywords["init_current_size"].id == "init_current_size"
    relion_incr = keywords["init_relion_incr_size"]
    assert isinstance(relion_incr, ast.IfExp)
    assert isinstance(relion_incr.orelse, ast.Attribute)
    assert relion_incr.orelse.attr == "relion_incr_size"

    assignments = {
        target.id: value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        for value in [node.value]
    }
    current_size = assignments["init_current_size"]
    assert isinstance(current_size, ast.IfExp)
    boundary_current_size = current_size.body
    assert isinstance(boundary_current_size, ast.Call)
    assert isinstance(boundary_current_size.args[0], ast.Attribute)
    assert boundary_current_size.args[0].attr == "current_size"


def test_frozen_boundary_noise_expands_in_float32_scoring_dtype():
    radial_per_half = [
        np.linspace(1.0, 2.0, 5, dtype=np.float64),
        np.linspace(3.0, 4.0, 5, dtype=np.float64),
    ]

    noise_per_half = _make_frozen_boundary_noise_variance(radial_per_half, (8, 8))

    assert len(noise_per_half) == 2
    for noise in noise_per_half:
        assert noise.shape == (64,)
        assert noise.dtype == np.dtype(np.float32)


def test_attach_relion_projector_capture_targets_exact_replay_slot(tmp_path, monkeypatch):
    capture_dir = tmp_path / "score_dump"
    capture_dir.mkdir()
    manifest = capture_dir / "iter3_VALIDATED_SHA256SUMS"
    manifest.write_text("sealed\n")
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    model = relion_dir / "run_it003_half1_model.star"
    model.write_text("model\n")
    expected_state = {
        "projector_half_by_half": [
            np.zeros((1, 87, 87, 44), dtype=np.complex64),
            np.zeros((1, 87, 87, 44), dtype=np.complex64),
        ],
        "projector_r_max_by_half": [21, 21],
        "current_size": 42,
        "padding_factor": 2,
        "volume_shape": [256, 256, 256],
        "n_classes": 1,
        "source_manifest_sha256": "a" * 64,
    }
    observed = {}

    def fake_model_metadata(path):
        observed["model_path"] = Path(path)
        return {"current_image_size": 42}

    def fake_build(capture_root, **kwargs):
        observed["capture_root"] = Path(capture_root)
        observed.update(kwargs)
        return expected_state

    monkeypatch.setattr("recovar.em.sampling.read_relion_model_metadata", fake_model_metadata)
    monkeypatch.setattr(run_full_refinement, "build_relion_projector_replay_state", fake_build)
    overrides = [{"slot": index} for index in range(4)]

    slot, state = _attach_relion_projector_capture(
        overrides,
        capture_dir=capture_dir,
        manifest_path=manifest,
        capture_iteration=3,
        init_relion_iteration=0,
        relion_replay_dir=relion_dir,
        volume_shape=(256, 256, 256),
        n_classes=1,
    )

    assert slot == 2
    assert state is expected_state
    assert overrides[2]["relion_projector_state"] is expected_state
    assert "relion_projector_state" not in overrides[1]
    assert observed == {
        "model_path": model,
        "capture_root": capture_dir.resolve(),
        "manifest_path": manifest.resolve(),
        "iteration": 3,
        "current_size": 42,
        "volume_shape": (256, 256, 256),
        "n_classes": 1,
    }


def test_attach_relion_projector_capture_rejects_unrepresented_iteration(tmp_path):
    with pytest.raises(ValueError, match="outside the configured replay trajectory"):
        _attach_relion_projector_capture(
            [{}],
            capture_dir=tmp_path,
            manifest_path=tmp_path / "manifest",
            capture_iteration=3,
            init_relion_iteration=0,
            relion_replay_dir=tmp_path,
            volume_shape=(8, 8, 8),
            n_classes=1,
        )


def test_attach_relion_projector_capture_rejects_late_restart(tmp_path):
    with pytest.raises(ValueError, match="uninterrupted cold-start trajectory"):
        _attach_relion_projector_capture(
            [{}],
            capture_dir=tmp_path,
            manifest_path=tmp_path / "manifest",
            capture_iteration=3,
            init_relion_iteration=2,
            relion_replay_dir=tmp_path,
            volume_shape=(8, 8, 8),
            n_classes=1,
        )


def test_attach_relion_projector_capture_accepts_immediate_validated_frozen_restart(
    tmp_path, monkeypatch
):
    capture_dir = tmp_path / "score_dump"
    capture_dir.mkdir()
    manifest = capture_dir / "iter3_VALIDATED_SHA256SUMS"
    manifest.write_text("sealed\n")
    relion_dir = tmp_path / "relion"
    relion_dir.mkdir()
    (relion_dir / "run_it003_half1_model.star").write_text("model\n")
    expected_state = {"source_manifest_sha256": "a" * 64}
    monkeypatch.setattr(
        "recovar.em.sampling.read_relion_model_metadata",
        lambda path: {"current_image_size": 42},
    )
    monkeypatch.setattr(
        run_full_refinement,
        "build_relion_projector_replay_state",
        lambda *args, **kwargs: expected_state,
    )
    overrides = [{"state": "it3"}, {"state": "final"}]

    slot, state = _attach_relion_projector_capture(
        overrides,
        capture_dir=capture_dir,
        manifest_path=manifest,
        capture_iteration=3,
        init_relion_iteration=2,
        relion_replay_dir=relion_dir,
        volume_shape=(256, 256, 256),
        n_classes=1,
        validated_frozen_boundary_iteration=2,
    )

    assert slot == 0
    assert state is expected_state
    assert overrides[0]["relion_projector_state"] is expected_state


def test_attach_relion_projector_capture_rejects_nonadjacent_frozen_restart(tmp_path):
    with pytest.raises(ValueError, match="immediately following"):
        _attach_relion_projector_capture(
            [{}, {}, {}],
            capture_dir=tmp_path,
            manifest_path=tmp_path / "manifest",
            capture_iteration=4,
            init_relion_iteration=2,
            relion_replay_dir=tmp_path,
            volume_shape=(8, 8, 8),
            n_classes=1,
            validated_frozen_boundary_iteration=2,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ([{}, None, {}], "has no state override"),
        (
            [{}, {"relion_projector_state": object()}, {}],
            "is already populated",
        ),
    ],
)
def test_attach_relion_projector_capture_rejects_nonatomic_slot(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _attach_relion_projector_capture(
            overrides,
            capture_dir=tmp_path,
            manifest_path=tmp_path / "manifest",
            capture_iteration=2,
            init_relion_iteration=0,
            relion_replay_dir=tmp_path,
            volume_shape=(8, 8, 8),
            n_classes=1,
        )


def test_relion_mpi_autorefine_scoring_noise_uses_rank1_broadcast():
    half1 = np.asarray([1.0, 2.0], dtype=np.float32)
    half2 = np.asarray([3.0, 4.0], dtype=np.float32)

    got = _relion_mpi_process_start_scoring_noise_pair(half1, half2, split_random_halves=True)

    np.testing.assert_array_equal(got[0], half1)
    np.testing.assert_array_equal(got[1], half1)
    assert got[0] is not got[1]


def test_relion_mpi_shared_model_scoring_noise_preserves_second_input():
    half1 = np.asarray([1.0, 2.0], dtype=np.float32)
    half2 = np.asarray([3.0, 4.0], dtype=np.float32)

    got = _relion_mpi_process_start_scoring_noise_pair(half1, half2, split_random_halves=False)

    np.testing.assert_array_equal(got[0], half1)
    np.testing.assert_array_equal(got[1], half2)


def test_relion_strict_replay_rejects_multiple_optics_noise_tables():
    import pandas as pd

    model = {
        "model_optics_group_1": pd.DataFrame({"rlnSigma2Noise": [1.0, 2.0]}),
        "model_optics_group_2": pd.DataFrame({"rlnSigma2Noise": [3.0, 4.0]}),
    }

    with pytest.raises(NotImplementedError, match="2 optics-group sigma2_noise tables"):
        _read_relion_single_optics_sigma2_noise(model, context="unit-test model")


def _read_relion_sigma(model_star: Path) -> float:
    import starfile

    m = starfile.read(str(model_star))
    mg = m["model_general"]
    val = mg["rlnSigmaOffsetsAngst"]
    if hasattr(val, "iloc"):
        val = val.iloc[0]
    elif hasattr(val, "__len__") and not isinstance(val, str):
        val = val[0]
    return float(val)


def test_parse_relion_cli_ini_high_reads_positive_cli_value():
    text = "# --i particles.star --firstiter_cc --ini_high 30 --ctf\n"
    assert _parse_relion_cli_ini_high(text) == pytest.approx(30.0)


def test_parse_relion_cli_ini_high_is_none_when_absent_or_disabled():
    assert _parse_relion_cli_ini_high("# --i particles.star --firstiter_cc --ctf\n") is None
    assert _parse_relion_cli_ini_high("# --i particles.star --firstiter_cc --ini_high -1 --ctf\n") is None


def test_firstiter_cc_passes_relion_cli_ini_high_to_refinement_loop():
    """RELION ``--firstiter_cc`` and ``--ini_high`` are distinct knobs.

    ``--firstiter_cc`` enables normalized-CC scoring in iter 1. RELION only
    reapplies the post-iter1 low-pass when the optimiser command has a
    positive ``--ini_high``. Do not substitute RECOVAR's ``--init_resolution``.
    """

    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "relion_firstiter_ini_high_angstrom" in keywords
    value = keywords["relion_firstiter_ini_high_angstrom"]
    assert isinstance(value, ast.IfExp)
    assert isinstance(value.test, ast.Attribute)
    assert value.test.attr == "firstiter_cc"
    assert isinstance(value.body, ast.Name)
    assert value.body.id == "relion_firstiter_ini_high_angstrom"


def test_refinement_results_persist_final_tau2_weight_combination():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"tau2_weight_combination_final_all_data"' in source
    assert 'save_dict["tau2_weight_combination_final_all_data"]' in source


def test_final_all_data_writes_matched_unfiltered_half_products():
    source = RUN_FULL_REFINEMENT.read_text()

    assert 'unfiltered_means = result.get("unfiltered_means")' in source
    assert 'f"final_half{k + 1}_unfil.mrc"' in source


def test_refinement_results_persist_class_assignment_history():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"class_assignment_history"' in source
    assert "class_assignments_iter_" in source
    assert "class_assignments_by_image_iter_" in source


def test_refinement_results_persist_numbered_follower_scale_boundaries():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '("relion_scale_follower_scales_numbered_pre_score_trajectory", np.float64)' in source
    assert '("relion_scale_follower_scales_numbered_post_mstep_trajectory", np.float64)' in source


def test_runner_threads_fail_closed_sparse_follower_scale_replay():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"--relion-follower-scale-replay"' in source
    assert "load_relion_follower_scale_replay(" in source
    assert "validate_relion_follower_scale_replay(" in source
    assert "schedule_oracle_id=relion_dispatch_schedule.oracle_id" in source
    assert "verify_relion_dispatch_schedule_oracle(" in source
    assert "numbered_iterations=range(" in source
    assert "first_numbered_iteration=int(args.init_relion_iteration) + 1" in source
    assert "relion_follower_scale_replay=relion_follower_scale_replay" in source
    assert 'save_dict["relion_follower_scale_replay_iterations"]' in source
    assert 'save_dict["relion_follower_scale_replay_source"]' in source
    assert 'save_dict["relion_follower_scale_replay_oracle_id"]' in source
    assert 'save_dict["relion_dispatch_oracle_id"]' in source
    assert '("relion_follower_scale_replay_requested_iterations", np.int64)' in source
    assert '("relion_follower_scale_replay_applied_iterations", np.int64)' in source


def test_runner_requires_and_persists_perturbation_restart_provenance():
    source = RUN_FULL_REFINEMENT.read_text()

    assert '"--perturb-replay-restart-provenance"' in source
    assert '"--perturb-replay-restart-state-iterations requires "' in source
    assert '"--perturb-replay-restart-provenance"' in source
    assert '"perturb_replay_restart_state_iterations"' in source
    assert '"perturb_replay_restart_provenance_path"' in source
    assert '"perturb_replay_restart_provenance_sha256"' in source


def test_save_intermediates_skip_unregularized_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["save_intermediates_skip_unregularized"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "save_intermediates_skip_unregularized"


def test_stop_after_local_search_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    assert "--stop_after_local_search" in RUN_FULL_REFINEMENT.read_text()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["stop_after_local_search"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "stop_after_local_search"


def test_stop_after_local_search_score_only_passes_to_refinement_loop():
    tree = ast.parse(RUN_FULL_REFINEMENT.read_text())
    assert "--stop_after_local_search_score_only" in RUN_FULL_REFINEMENT.read_text()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    value = keywords["stop_after_local_search_score_only"]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "bool"
    assert isinstance(value.args[0], ast.Attribute)
    assert value.args[0].attr == "stop_after_local_search_score_only"


def test_stop_after_local_search_score_only_is_diagnostic_score_only_path():
    source = ITERATION_LOOP.read_text()
    assert "if stop_after_local_search_score_only:\n        stop_after_local_search = True" in source
    assert "diagnostic_score_only=bool(stop_after_local_search_score_only)" in source
    assert "score_only=diagnostic_score_only" in source
    assert "accumulate_noise=local_accumulate_noise" in source
    assert '"stop_after_local_search_score_only": bool(stop_after_local_search_score_only)' in source


def test_diagnostic_single_half_is_guarded_to_local_search_stops():
    source = RUN_FULL_REFINEMENT.read_text()

    assert "--diagnostic_single_half" in source
    assert "local_stop_requested = (" in source
    assert "--diagnostic_single_half is only valid with --stop_after_local_search" in source
    assert "--diagnostic_single_half is K=1-only" in source
    assert "half2_idx = np.empty(0, dtype=np.int64)" in source
    assert '"diagnostic_single_half": bool(args.diagnostic_single_half)' in source
    assert "skipping Projector::data build for empty half-%d dataset" in ITERATION_LOOP.read_text()


def test_init_noise_from_npz_loader_uses_latest_numbered_spectrum(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        noise_radial_iter_000=np.asarray([1.0, 2.0], dtype=np.float64),
        noise_radial_iter_003=np.asarray([3.0, 4.0], dtype=np.float64),
    )

    loaded = _load_init_noise_radial_npz(path, "last")

    assert loaded["iteration"] == "003"
    np.testing.assert_allclose(loaded["noise_radial"], np.asarray([3.0, 4.0], dtype=np.float64))


def test_init_noise_from_npz_is_diagnostic_cli_path():
    source = RUN_FULL_REFINEMENT.read_text()
    assert "--init_noise_from_npz" in source
    assert "--init_noise_iter" in source
    assert "_load_init_noise_radial_npz(args.init_noise_from_npz, args.init_noise_iter)" in source
    assert "estimate_initial_noise_spectrum_from_unaligned_images" in source


def test_initial_noise_cache_round_trips_and_marks_safe_to_delete(tmp_path):
    cache_path = _save_initial_noise_cache(
        tmp_path,
        "abc123",
        (8, 8),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
    )

    loaded, loaded_path = _load_initial_noise_cache(tmp_path, "abc123", (8, 8))

    assert loaded_path == cache_path
    assert (tmp_path / "SAFE_TO_DELETE").exists()
    np.testing.assert_allclose(loaded, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))


def test_initial_noise_cache_is_exact_bootstrap_cli_path():
    source = RUN_FULL_REFINEMENT.read_text()
    assert "--initial_noise_cache_dir" in source
    assert "_initial_noise_cache_key(" in source
    assert "_load_initial_noise_cache(" in source
    assert "_save_initial_noise_cache(" in source
    assert "estimate_initial_noise_spectrum_from_unaligned_images" in source
    assert "--init_noise_from_npz" in source


def test_relion_tau2_fudge_parser_accepts_class3d_arg_label():
    text = """
data_optimiser_general

_rlnDoSplitRandomHalves                                  0
_rlnTau2FudgeArg                                          4.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(4.0)


def test_load_native_group_ids_per_half_reads_particles_star(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    starfile.write(
        {
            "particles": pd.DataFrame(
                {
                    "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs", "4@x.mrcs"],
                    "rlnGroupNumber": [1, 2, 3, 2],
                },
            ),
        },
        tmp_path / "particles.star",
    )

    got = _load_native_group_ids_per_half(
        tmp_path / "particles.star",
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
    )

    assert got is not None
    np.testing.assert_array_equal(got[0], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(got[1], np.asarray([1, 1], dtype=np.int64))


def test_native_group_layout_prefers_supplied_relion_groups_and_maps_exact_identities():
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame(
        {
            "rlnImageName": [
                "3@stack_a.mrcs",
                "1@stack_a.mrcs",
                "4@stack_a.mrcs",
                "2@stack_a.mrcs",
            ],
        },
    )
    relion_particles = pd.DataFrame(
        {
            "rlnImageName": [
                "1@stack_a.mrcs",
                "2@stack_a.mrcs",
                "3@stack_a.mrcs",
                "4@stack_a.mrcs",
            ],
            "rlnGroupNumber": [2, 4, 1, 7],
            "rlnOpticsGroup": [1, 2, 1, 3],
        },
    )

    layout = _resolve_native_group_layout(
        our_particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
        relion_particles=relion_particles,
    )

    assert layout is not None
    assert layout.source == "supplied RELION data STAR"
    assert layout.n_groups == 7
    assert layout.n_optics_groups == 3
    np.testing.assert_array_equal(layout.group_ids_per_half[0], [0, 1])
    np.testing.assert_array_equal(layout.group_ids_per_half[1], [6, 3])
    # Internal IDs are authoritative RELION data-STAR row numbers, mapped by
    # full rlnImageName identity rather than RECOVAR row position.
    np.testing.assert_array_equal(layout.particle_ids_per_half[0], [2, 0])
    np.testing.assert_array_equal(layout.particle_ids_per_half[1], [3, 1])
    np.testing.assert_array_equal(layout.optics_group_ids_per_half[0], [0, 0])
    np.testing.assert_array_equal(layout.optics_group_ids_per_half[1], [2, 1])


def test_replay_group_loader_uses_iter0_without_relion_half_sets(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["2@x.mrcs", "1@x.mrcs"],
            "rlnGroupNumber": [7, 3],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it000_data.star")

    loaded, source = _load_replay_group_particles(tmp_path)

    assert source == tmp_path / "run_it000_data.star"
    np.testing.assert_array_equal(loaded["rlnGroupNumber"], [7, 3])


def test_subset_only_halfset_does_not_block_permuted_replay_group_layout(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")
    our_particles = pd.DataFrame(
        {"rlnImageName": ["3@x.mrcs", "1@x.mrcs", "2@x.mrcs"]}
    )
    subset_only = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs"],
            "rlnRandomSubset": [1, 2, 1],
        }
    )
    replay_particles = pd.DataFrame(
        {
            "rlnImageName": ["2@x.mrcs", "3@x.mrcs", "1@x.mrcs"],
            "rlnGroupNumber": [7, 4, 2],
        }
    )
    starfile.write({"particles": replay_particles}, tmp_path / "run_it000_data.star")

    selected, source = _select_authoritative_group_particles(
        halfset_particles=subset_only,
        halfset_source=tmp_path / "halfsets.star",
        replay_dirs=(tmp_path,),
    )
    layout = _resolve_native_group_layout(
        our_particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2], dtype=np.int64),
        relion_particles=selected,
    )

    assert source == tmp_path / "run_it000_data.star"
    assert layout is not None
    assert layout.n_groups == 7
    np.testing.assert_array_equal(layout.group_ids_per_half[0], [3, 1])
    np.testing.assert_array_equal(layout.group_ids_per_half[1], [6])


def test_native_group_layout_preserves_full_group_axis_when_half_max_is_absent():
    pd = pytest.importorskip("pandas")
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@x.mrcs", "2@x.mrcs", "3@x.mrcs", "4@x.mrcs"],
            "rlnGroupNumber": [1, 2, 7, 4],
        },
    )

    layout = _resolve_native_group_layout(
        particles,
        half1_idx=np.asarray([0, 1], dtype=np.int64),
        half2_idx=np.asarray([2, 3], dtype=np.int64),
    )

    assert layout is not None
    assert layout.n_groups == 7
    assert int(np.max(layout.group_ids_per_half[0])) == 1
    np.testing.assert_array_equal(layout.group_ids_per_half[1], [6, 3])


@pytest.mark.parametrize(
    ("relion_names", "message"),
    [
        (["1@x.mrcs", "1@x.mrcs"], "duplicate rlnImageName/stack identities"),
        (["1@x.mrcs", "2@other.mrcs"], "do not contain the same rlnImageName/stack identities"),
    ],
)
def test_native_group_layout_rejects_duplicate_or_missing_relion_identities(relion_names, message):
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame({"rlnImageName": ["1@x.mrcs", "2@x.mrcs"]})
    relion_particles = pd.DataFrame(
        {"rlnImageName": relion_names, "rlnGroupNumber": [1, 2]},
    )

    with pytest.raises(ValueError, match=message):
        _resolve_native_group_layout(
            our_particles,
            half1_idx=np.asarray([0], dtype=np.int64),
            half2_idx=np.asarray([1], dtype=np.int64),
            relion_particles=relion_particles,
        )


def test_relion_expected_accuracy_layout_preserves_relion_particle_rows():
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame(
        {"rlnImageName": ["30@x.mrcs", "10@x.mrcs", "40@x.mrcs", "20@x.mrcs"]},
    )
    relion_particles = pd.DataFrame(
        {
            "rlnImageName": ["10@x.mrcs", "20@x.mrcs", "30@x.mrcs", "40@x.mrcs"],
            "rlnRandomSubset": [1, 2, 1, 1],
            "rlnOpticsGroup": [2, 1, 1, 2],
        },
    )

    half1, half2, base_order, optics, particle_ids = _relion_halfset_and_accuracy_layout(
        our_particles,
        relion_particles,
    )

    np.testing.assert_array_equal(half1, [0, 1, 2])
    np.testing.assert_array_equal(half2, [3])
    np.testing.assert_array_equal(base_order, [1, 0, 2])
    np.testing.assert_array_equal(optics, [1, 2, 2])
    np.testing.assert_array_equal(particle_ids, [2, 0, 3])


def test_relion_expected_accuracy_layout_supports_repeated_indices_across_stacks():
    pd = pytest.importorskip("pandas")
    our_particles = pd.DataFrame(
        {"rlnImageName": ["1@b.mrcs", "1@a.mrcs", "2@b.mrcs", "2@a.mrcs"]},
    )
    relion_particles = pd.DataFrame(
        {
            "rlnImageName": ["1@a.mrcs", "2@a.mrcs", "1@b.mrcs", "2@b.mrcs"],
            "rlnRandomSubset": [1, 2, 1, 1],
            "rlnOpticsGroup": [2, 1, 1, 2],
        },
    )

    half1, half2, base_order, optics, particle_ids = _relion_halfset_and_accuracy_layout(
        our_particles,
        relion_particles,
    )

    np.testing.assert_array_equal(half1, [0, 1, 2])
    np.testing.assert_array_equal(half2, [3])
    np.testing.assert_array_equal(base_order, [1, 0, 2])
    np.testing.assert_array_equal(optics, [1, 2, 2])
    np.testing.assert_array_equal(particle_ids, [2, 0, 3])


def test_runner_keeps_input_particle_names_for_replay_mapping():
    source = RUN_FULL_REFINEMENT.read_text()
    bind = 'our_names = np.asarray(our_particles["rlnImageName"])'
    first_replay_use = "particle_names=our_names"

    assert bind in source
    assert source.index(bind) < source.index(first_replay_use)
    assert "max(int(args.max_iter) - 1, 0)" not in source


def test_replay_mapping_distinguishes_repeated_indices_across_stacks(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@a.mrcs", "1@b.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [11.0, 21.0],
            "rlnAnglePsi": [12.0, 22.0],
            "rlnOriginXAngst": [2.0, 4.0],
            "rlnOriginYAngst": [-2.0, -4.0],
            "rlnNormCorrection": [2.0, 4.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": pd.DataFrame(
                {"rlnNormCorrectionAverage": [8.0], "rlnSigmaOffsetsAngst": [3.0]}
            ),
            "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
        },
        tmp_path / "run_it001_model.star",
    )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
        particle_names=["1@b.mrcs", "1@a.mrcs"],
    )

    h1_corr, h2_corr = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1_corr, [2.0])
    np.testing.assert_allclose(h2_corr, [4.0])
    h1_eulers, h2_eulers = overrides[1]["previous_best_rotation_eulers"]
    np.testing.assert_allclose(h1_eulers[:, 0], [20.0])
    np.testing.assert_allclose(h2_eulers[:, 0], [10.0])


def test_native_group_ids_are_available_to_k_class_refinement():
    source = RUN_FULL_REFINEMENT.read_text()
    group_start = source.index("native_group_layout = _resolve_native_group_layout")
    group_end = source.index("optimiser_star = _relion_optimiser_star_for_runtime(", group_start)
    group_block = source[group_start:group_end]

    assert "args.n_classes == 1" not in group_block
    assert "Native group-scale updates remain disabled for K-class refinement" not in source
    assert "relion_particles=relion_group_particles" in group_block
    replay_group_load = source.index("_select_authoritative_group_particles(", source.index("def main"))
    assert replay_group_load < group_start
    assert "native_group_count =" in group_block
    assert "init_group_ids=native_group_ids_per_half" in source
    assert "init_group_count=native_group_count" in source


def test_load_init_previous_best_poses_npz_selects_latest_numbered_iter(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_iter_000_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_iter_000_half1=np.ones((1, 3), dtype=np.float32) * 2,
        best_translations_iter_000_half0=np.ones((2, 2), dtype=np.float32),
        best_translations_iter_000_half1=np.ones((1, 2), dtype=np.float32) * 2,
        best_rotation_eulers_iter_003_half0=np.ones((2, 3), dtype=np.float32) * 3,
        best_rotation_eulers_iter_003_half1=np.ones((1, 3), dtype=np.float32) * 4,
        best_translations_iter_003_half0=np.ones((2, 2), dtype=np.float32) * 5,
        best_translations_iter_003_half1=np.ones((1, 2), dtype=np.float32) * 6,
    )

    got = _load_init_previous_best_poses_npz(path, "last")

    assert got["iteration"] == "003"
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][0], np.ones((2, 3)) * 3)
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][1], np.ones((1, 3)) * 4)
    np.testing.assert_allclose(got["previous_best_translations"][0], np.ones((2, 2)) * 5)
    np.testing.assert_allclose(got["previous_best_translations"][1], np.ones((1, 2)) * 6)


def test_load_init_previous_best_poses_npz_accepts_final_all_data(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_final_all_data_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_final_all_data_half1=np.ones((1, 3), dtype=np.float32) * 2,
        best_translations_final_all_data_half0=np.ones((2, 2), dtype=np.float32) * 3,
        best_translations_final_all_data_half1=np.ones((1, 2), dtype=np.float32) * 4,
    )

    got = _load_init_previous_best_poses_npz(path, "final_all_data")

    assert got["iteration"] == "final_all_data"
    np.testing.assert_allclose(got["previous_best_rotation_eulers"][0], np.ones((2, 3)))
    np.testing.assert_allclose(got["previous_best_translations"][1], np.ones((1, 2)) * 4)


def test_load_init_previous_best_poses_npz_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "refinement_results.npz"
    np.savez(
        path,
        best_rotation_eulers_iter_000_half0=np.ones((2, 3), dtype=np.float32),
        best_rotation_eulers_iter_000_half1=np.ones((1, 3), dtype=np.float32),
        best_translations_iter_000_half0=np.ones((3, 2), dtype=np.float32),
        best_translations_iter_000_half1=np.ones((1, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="row mismatch"):
        _load_init_previous_best_poses_npz(path, 0)


def _write_minimal_relion_model_star(path: Path, *, sigma2_noise_start: float) -> None:
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    starfile.write(
        {
            "model_general": pd.DataFrame({"rlnTau2FudgeFactor": [1.0]}),
            "model_optics_group_1": pd.DataFrame(
                {"rlnSigma2Noise": np.asarray([sigma2_noise_start, sigma2_noise_start + 1.0])},
            ),
            "model_class_1": pd.DataFrame(
                {
                    "rlnSpectralIndex": [0, 1],
                    "rlnReferenceTau2": [0.1, 0.2],
                },
            ),
        },
        path,
    )


def test_load_relion_it000_model_stars_prefers_shared_model(tmp_path):
    shared = tmp_path / "run_it000_model.star"
    _write_minimal_relion_model_star(shared, sigma2_noise_start=1.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half1_model.star", sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half2_model.star", sigma2_noise_start=3.0)

    bundle = _load_relion_it000_model_stars(tmp_path, n_classes=1)

    assert bundle["source"] == "shared"
    assert bundle["model_paths"] == [shared]
    assert bundle["reference_model_path"] == shared
    assert len(bundle["models"]) == 1


def test_load_relion_it000_model_stars_accepts_autorefine_half_models(tmp_path):
    half1 = tmp_path / "run_it000_half1_model.star"
    half2 = tmp_path / "run_it000_half2_model.star"
    _write_minimal_relion_model_star(half1, sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(half2, sigma2_noise_start=3.0)

    bundle = _load_relion_it000_model_stars(tmp_path, n_classes=1)

    assert bundle["source"] == "half-specific"
    assert bundle["model_paths"] == [half1, half2]
    assert bundle["reference_model_path"] == half1
    assert len(bundle["models"]) == 2
    np.testing.assert_allclose(bundle["models"][0]["model_optics_group_1"]["rlnSigma2Noise"], [2.0, 3.0])
    np.testing.assert_allclose(bundle["models"][1]["model_optics_group_1"]["rlnSigma2Noise"], [3.0, 4.0])


def test_load_relion_it000_model_stars_requires_shared_model_for_kclass(tmp_path):
    _write_minimal_relion_model_star(tmp_path / "run_it000_half1_model.star", sigma2_noise_start=2.0)
    _write_minimal_relion_model_star(tmp_path / "run_it000_half2_model.star", sigma2_noise_start=3.0)

    with pytest.raises(SystemExit, match="no compatible iter-0 model STAR"):
        _load_relion_it000_model_stars(tmp_path, n_classes=4)


def test_replay_mean_log_formatter_handles_empty_half_without_warning():
    assert _format_replay_mean_for_log(np.asarray([], dtype=np.float32)) == "empty"
    assert _format_replay_mean_for_log(np.asarray([1.0, 3.0], dtype=np.float32)) == "2.0000"


def test_relion_tau2_fudge_parser_accepts_factor_label():
    text = "_rlnTau2FudgeFactor 1.000000\n"
    assert _parse_relion_tau2_fudge(text) == pytest.approx(1.0)


def test_relion_tau2_fudge_parser_maps_arg_negative_one_to_none():
    """RELION's ``_rlnTau2FudgeArg=-1`` (in optimiser.star) is the sentinel
    for "user did not pass --tau2_fudge". RELION's ml_optimiser.cpp:881-882
    resolves it as ``tau2_fudge_factor = tau2_fudge_arg > 0 ? arg : 1``,
    i.e. -1 → 1.0 (the auto-refine binary default; Class3D's 4.0 comes from
    the GUI which always passes --tau2_fudge 4.0 — see
    pipeline_jobs.cpp::initialiseClass3DJob). The recovar parser must
    return None so _resolve_tau2_fudge falls back to the K-class default.
    Passing -1 downstream inverts the Wiener regularization
    (``inv_tau = 1 / (pf^3 * tau2_fudge * tau)`` becomes negative) which
    corrupts iter-1's reconstruction and collapses iter-2+ ave_Pmax —
    diagnosed on K=1 100k/256 replay job 8255968 (iter1 Pmax=0.94 at
    RELION parity, then iter2=0.32 vs RELION 0.98)."""
    text = """
data_optimiser_general

_rlnDoSplitRandomHalves                                  1
_rlnTau2FudgeArg                                          -1.000000
"""
    assert _parse_relion_tau2_fudge(text) is None


def test_relion_tau2_fudge_parser_prefers_factor_over_arg():
    """When both labels appear in the same text (combined parse), the
    Factor field from model.star is authoritative (actual value used)
    while Arg from optimiser.star is just the CLI input. RELION never
    writes both into the same file, but the parser must still prefer
    Factor to be robust."""
    text = """
_rlnTau2FudgeFactor 1.000000
_rlnTau2FudgeArg    -1.000000
"""
    assert _parse_relion_tau2_fudge(text) == pytest.approx(1.0)


def test_tau2_fudge_resolver_matches_relion_mode_defaults():
    assert _resolve_tau2_fudge(1, None, None) == (1.0, "RELION auto-refine default")
    assert _resolve_tau2_fudge(4, None, None) == (4.0, "RELION Class3D default")
    assert _resolve_tau2_fudge(4, 2.5, None) == (2.5, "explicit CLI")
    assert _resolve_tau2_fudge(4, 2.5, 4.0) == (4.0, "RELION it000 optimiser")


def test_replay_normcorr_defaults_to_strict_replay_only():
    assert _resolve_replay_normcorr(None, None) is False
    assert _resolve_replay_normcorr("/relion/run", None) is True
    assert _resolve_replay_normcorr("/relion/run", False) is False
    assert _resolve_replay_normcorr(None, True) is True


def test_default_refinement_subsets_keep_gold_standard_for_k1():
    half1, half2 = _default_refinement_subsets(9, seed=3, n_classes=1)

    assert half1.shape == (4,)
    assert half2.shape == (5,)
    np.testing.assert_array_equal(np.sort(np.concatenate([half1, half2])), np.arange(9))


def test_default_refinement_subsets_use_all_data_once_for_class3d():
    half1, half2 = _default_refinement_subsets(9, seed=3, n_classes=4)

    np.testing.assert_array_equal(half1, np.arange(9))
    assert half2.size == 0


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_inject_per_iter_sigma_offset():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=8,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=False,
    )

    assert overrides[0] is None, "iter 0 (recovar iter 1) has no upstream RELION state"
    for recovar_iter in range(1, 8):
        assert overrides[recovar_iter] is not None, f"iter {recovar_iter} override missing"
        assert "translation_sigma_angstrom" in overrides[recovar_iter]
        # No normcorr/scale corrections when include_normcorr=False — only
        # sigma_offset, which is parity-critical regardless of normcorr replay.
        assert "image_corrections" not in overrides[recovar_iter]
        assert "serialized_scale_corrections" not in overrides[recovar_iter]

        m1 = FIXTURE / f"run_it{recovar_iter:03d}_half1_model.star"
        m2 = FIXTURE / f"run_it{recovar_iter:03d}_half2_model.star"
        relion_sigma = 0.5 * (_read_relion_sigma(m1) + _read_relion_sigma(m2))
        recovar_sigma = float(overrides[recovar_iter]["translation_sigma_angstrom"])
        assert recovar_sigma == pytest.approx(relion_sigma, abs=1e-6), (
            f"iter {recovar_iter}: recovar override sigma_offset {recovar_sigma:.6f} != "
            f"RELION rlnSigmaOffsetsAngst mean {relion_sigma:.6f}"
        )
        np.testing.assert_allclose(
            overrides[recovar_iter]["translation_sigma_angstrom_per_half"],
            np.asarray([_read_relion_sigma(m1), _read_relion_sigma(m2)], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_can_load_complete_iter0_cold_start():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=0,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=True,
        include_initial_state=True,
    )

    cold_start = overrides[0]
    assert cold_start is not None
    assert cold_start["previous_best_translations"][0].shape == (2515, 2)
    assert cold_start["previous_best_translations"][1].shape == (2485, 2)
    assert cold_start["previous_best_rotation_eulers"][0].shape == (2515, 3)
    assert cold_start["previous_best_rotation_eulers"][1].shape == (2485, 3)
    assert cold_start["image_corrections"][0].shape == (2515,)
    assert cold_start["serialized_scale_corrections"][1].shape == (2485,)
    assert cold_start["direction_prior"][0].ndim == 1
    assert cold_start["translation_sigma_angstrom"] == pytest.approx(
        0.5
        * (
            _read_relion_sigma(FIXTURE / "run_it000_half1_model.star")
            + _read_relion_sigma(FIXTURE / "run_it000_half2_model.star")
        ),
        abs=1e-6,
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_iter2_sigma_matches_relion_iter1():
    """Specifically lock down the iter-2 cliff fix.

    recovar iter 2 (i.e. ``overrides[1]``) must use RELION iter-1's
    ``rlnSigmaOffsetsAngst``, since RELION iter-2 loads the prior from
    iter-1's model.star at E-step entry.
    """
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=3,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=False,
    )

    iter1_h1 = FIXTURE / "run_it001_half1_model.star"
    iter1_h2 = FIXTURE / "run_it001_half2_model.star"
    relion_iter1_sigma = 0.5 * (_read_relion_sigma(iter1_h1) + _read_relion_sigma(iter1_h2))

    recovar_iter2_sigma = float(overrides[1]["translation_sigma_angstrom"])
    assert recovar_iter2_sigma == pytest.approx(relion_iter1_sigma, abs=1e-6)
    np.testing.assert_allclose(
        overrides[1]["translation_sigma_angstrom_per_half"],
        np.asarray([_read_relion_sigma(iter1_h1), _read_relion_sigma(iter1_h2)], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    # Sanity: this is the data-driven RELION sigma, not the 10 Å init default.
    assert recovar_iter2_sigma < 5.0, (
        "recovar iter-2 should use RELION's iter-1 data-driven sigma (~2 Å), "
        f"not the init default (10 Å); got {recovar_iter2_sigma:.4f} Å"
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"fixture missing: {FIXTURE}")
def test_replay_overrides_include_normcorr_adds_image_corrections():
    half1_idx = np.arange(2515, dtype=np.int64)
    half2_idx = np.arange(2515, 5000, dtype=np.int64)

    overrides = _build_replay_iteration_overrides(
        FIXTURE,
        half1_idx,
        half2_idx,
        max_iter=2,
        ds_voxel=4.25,
        ds_grid=128,
        include_normcorr=True,
    )

    assert "image_corrections" in overrides[1]
    assert "serialized_scale_corrections" in overrides[1]
    assert "scoring_scale_corrections" not in overrides[1]
    assert "translation_sigma_angstrom" in overrides[1]
    h1, h2 = overrides[1]["image_corrections"]
    assert h1.shape == (2515,)
    assert h2.shape == (2485,)


def test_replay_overrides_use_shared_class3d_model_star(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": [
                "1@particles.mrcs",
                "2@particles.mrcs",
                "3@particles.mrcs",
                "4@particles.mrcs",
            ],
            "rlnAngleRot": [10.0, 20.0, 30.0, 40.0],
            "rlnAngleTilt": [11.0, 21.0, 31.0, 41.0],
            "rlnAnglePsi": [12.0, 22.0, 32.0, 42.0],
            "rlnOriginXAngst": [2.0, -4.0, 6.0, 8.0],
            "rlnOriginYAngst": [1.0, -3.0, 5.0, 7.0],
            "rlnNormCorrection": [1.0, 2.0, 4.0, 5.0],
            "rlnGroupNumber": [1, 2, 1, 2],
        }
    )
    model_general = pd.DataFrame(
        {
            "rlnNormCorrectionAverage": [3.0],
            "rlnSigmaOffsetsAngst": [6.5],
        }
    )
    model_groups = pd.DataFrame({"rlnGroupScaleCorrection": [10.0, 20.0]})
    model_class_1 = pd.DataFrame({"rlnReferenceTau2": [0.1, 0.2]})
    model_class_2 = pd.DataFrame({"rlnReferenceTau2": [0.3, 0.4]})
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": model_general,
            "model_groups": model_groups,
            "model_class_1": model_class_1,
            "model_class_2": model_class_2,
        },
        tmp_path / "run_it001_model.star",
    )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=2,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
    )

    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[1]["translation_sigma_angstrom_per_half"] == pytest.approx([6.5, 6.5])
    h1, h2 = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([30.0, 7.5], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([30.0, 12.0], dtype=np.float32))
    s1, s2 = overrides[1]["serialized_scale_corrections"]
    np.testing.assert_allclose(s1, np.asarray([10.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(s2, np.asarray([20.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(
        overrides[1]["class_tau2"],
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64) * 8**4,
    )
    assert "mean_variance" not in overrides[1]

    t1, t2 = overrides[1]["previous_best_translations"]
    np.testing.assert_allclose(t1, np.asarray([[1.0, 0.5], [3.0, 2.5]], dtype=np.float32))
    np.testing.assert_allclose(t2, np.asarray([[-2.0, -1.5], [4.0, 3.5]], dtype=np.float32))

    e1, e2 = overrides[1]["previous_best_rotation_eulers"]
    expected_e1 = np.asarray([[10.0, 11.0, 12.0], [30.0, 31.0, 32.0]], dtype=np.float32)
    expected_e2 = np.asarray([[20.0, 21.0, 22.0], [40.0, 41.0, 42.0]], dtype=np.float32)
    np.testing.assert_allclose(e1, expected_e1)
    np.testing.assert_allclose(e2, expected_e2)

    from recovar import utils

    r1, r2 = overrides[1]["previous_best_rotations"]
    np.testing.assert_allclose(r1, utils.R_from_relion(expected_e1, degrees=True).astype(np.float32))
    np.testing.assert_allclose(r2, utils.R_from_relion(expected_e2, degrees=True).astype(np.float32))


def test_replay_overrides_k1_mean_variance_is_explicit_and_n4_scaled(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnNormCorrection": [1.0, 1.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    raw_tau2 = np.asarray([0.1, 0.2], dtype=np.float64)
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": pd.DataFrame(
                {
                    "rlnNormCorrectionAverage": [1.0],
                    "rlnSigmaOffsetsAngst": [2.0],
                }
            ),
            "model_class_1": pd.DataFrame({"rlnReferenceTau2": raw_tau2}),
        },
        tmp_path / "run_it001_model.star",
    )
    common_kwargs = {
        "relion_dir": tmp_path,
        "half1_idx": np.asarray([0], dtype=np.int64),
        "half2_idx": np.asarray([1], dtype=np.int64),
        "max_iter": 1,
        "ds_voxel": 2.0,
        "ds_grid": 4,
        "include_normcorr": False,
    }

    default_overrides = _build_replay_iteration_overrides(**common_kwargs)
    diagnostic_overrides = _build_replay_iteration_overrides(
        **common_kwargs,
        include_k1_mean_variance=True,
    )

    assert "mean_variance" not in default_overrides[1]
    assert "class_tau2" not in default_overrides[1]
    assert "class_tau2" not in diagnostic_overrides[1]

    from recovar import utils

    expected = np.asarray(
        utils.make_radial_image(
            raw_tau2 * 4**4,
            (4, 4, 4),
            extend_last_frequency=True,
        ),
        dtype=np.float64,
    ).reshape(-1)
    observed = diagnostic_overrides[1]["mean_variance"]
    assert observed.shape == (4**3,)
    assert observed.dtype == np.float64
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_replay_overrides_map_noncontiguous_subset_stack_indices(tmp_path):
    """Real-data subsets retain original stack IDs instead of renumbering rows."""
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    # RELION may reorder rows; values are deliberately keyed to stack ID.
    relion_stack_ids = [126, 16, 200, 51]
    particles = pd.DataFrame(
        {
            "rlnImageName": [f"{i}@particles.mrcs" for i in relion_stack_ids],
            "rlnAngleRot": np.asarray(relion_stack_ids, dtype=float),
            "rlnAngleTilt": np.asarray(relion_stack_ids, dtype=float) + 0.25,
            "rlnAnglePsi": np.asarray(relion_stack_ids, dtype=float) + 0.5,
            "rlnOriginXAngst": np.asarray(relion_stack_ids, dtype=float) * 2.0,
            "rlnOriginYAngst": -np.asarray(relion_stack_ids, dtype=float) * 2.0,
            "rlnNormCorrection": np.asarray(relion_stack_ids, dtype=float),
            "rlnGroupNumber": [1, 1, 1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")
    starfile.write(
        {
            "model_general": pd.DataFrame(
                {"rlnNormCorrectionAverage": [10.0], "rlnSigmaOffsetsAngst": [3.0]}
            ),
            "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
        },
        tmp_path / "run_it001_model.star",
    )

    input_names = [
        "16@particles.mrcs",
        "51@particles.mrcs",
        "126@particles.mrcs",
        "200@particles.mrcs",
    ]
    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0, 2], dtype=np.int64),
        half2_idx=np.asarray([1, 3], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
        particle_names=input_names,
    )

    h1_corr, h2_corr = overrides[1]["image_corrections"]
    np.testing.assert_allclose(h1_corr, np.asarray([10.0 / 16.0, 10.0 / 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_corr, np.asarray([10.0 / 51.0, 10.0 / 200.0], dtype=np.float32))

    h1_eulers, h2_eulers = overrides[1]["previous_best_rotation_eulers"]
    np.testing.assert_allclose(h1_eulers[:, 0], np.asarray([16.0, 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_eulers[:, 0], np.asarray([51.0, 200.0], dtype=np.float32))

    h1_trans, h2_trans = overrides[1]["previous_best_translations"]
    np.testing.assert_allclose(h1_trans[:, 0], np.asarray([16.0, 126.0], dtype=np.float32))
    np.testing.assert_allclose(h2_trans[:, 0], np.asarray([51.0, 200.0], dtype=np.float32))


def test_replay_overrides_include_max_iter_state_for_final_all_data(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [11.0, 21.0],
            "rlnAnglePsi": [12.0, 22.0],
            "rlnOriginXAngst": [2.0, -4.0],
            "rlnOriginYAngst": [1.0, -3.0],
            "rlnNormCorrection": [2.0, 4.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    direction_prior = np.linspace(1.0, 12.0, 12, dtype=np.float32)
    direction_prior /= direction_prior.sum()
    for relion_iter, sigma in ((1, 6.5), (2, 9.5)):
        starfile.write({"particles": particles}, tmp_path / f"run_it{relion_iter:03d}_data.star")
        sigma2_noise = np.linspace(1.0, 5.0, 5, dtype=np.float64) * float(relion_iter)
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {
                        "rlnNormCorrectionAverage": [float(relion_iter)],
                        "rlnSigmaOffsetsAngst": [sigma],
                    }
                ),
                "model_optics_group_1": pd.DataFrame({"rlnSigma2Noise": sigma2_noise}),
                "model_pdf_orient_class_1": pd.DataFrame({"rlnOrientationDistribution": direction_prior}),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{relion_iter:03d}_model.star",
        )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=2,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=True,
    )

    assert len(overrides) == 3
    assert overrides[0] is None
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[2]["translation_sigma_angstrom"] == pytest.approx(9.5)
    assert overrides[1]["translation_sigma_angstrom_per_half"] == pytest.approx([6.5, 6.5])
    assert overrides[2]["translation_sigma_angstrom_per_half"] == pytest.approx([9.5, 9.5])
    h1, h2 = overrides[2]["image_corrections"]
    np.testing.assert_allclose(h1, np.asarray([1.0], dtype=np.float32))
    np.testing.assert_allclose(h2, np.asarray([0.5], dtype=np.float32))

    noise_h1, noise_h2 = overrides[2]["noise_variance"]
    assert noise_h1.shape == (64,)
    assert noise_h2.shape == (64,)
    # RELION sigma2_noise is stored in RELION frame and replay expands it to
    # RECOVAR's image-shaped N^4-scaled variance convention.
    assert float(np.min(noise_h1)) == pytest.approx(2.0 * 8**4)
    assert float(np.max(noise_h1)) == pytest.approx(10.0 * 8**4)
    np.testing.assert_allclose(noise_h1, noise_h2)

    prior_h1, prior_h2 = overrides[2]["direction_prior"]
    np.testing.assert_allclose(prior_h1, direction_prior, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prior_h2, direction_prior, rtol=1e-5, atol=1e-6)


def test_full_refinement_requests_max_iter_replay_state_for_final_all_data():
    source = inspect.getsource(run_full_refinement.main)
    start = source.index("replay_iteration_overrides = _build_replay_iteration_overrides(")
    end = source.index("\n        )", start)
    replay_call = source[start:end]

    assert "int(args.max_iter)," in replay_call
    assert "max(int(args.max_iter) - 1, 0)" not in replay_call


def test_autorefine_continuation_noise_emulates_relion_rank1_broadcast(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnNormCorrection": [1.0, 1.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")

    def write_model(path, sigma2_noise):
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {
                        "rlnNormCorrectionAverage": [1.0],
                        "rlnSigmaOffsetsAngst": [2.0],
                    }
                ),
                "model_optics_group_1": pd.DataFrame(
                    {"rlnSigma2Noise": np.asarray(sigma2_noise, dtype=np.float64)}
                ),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            path,
        )

    write_model(tmp_path / "run_it001_half1_model.star", [1.0, 2.0, 3.0, 4.0, 5.0])
    write_model(tmp_path / "run_it001_half2_model.star", [6.0, 7.0, 8.0, 9.0, 10.0])

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=0,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=1,
    )

    noise_h1, noise_h2 = overrides[0]["noise_variance"]
    np.testing.assert_array_equal(noise_h2, noise_h1)
    assert noise_h1 is not noise_h2
    assert float(np.min(noise_h2)) == pytest.approx(1.0 * 8**4)
    assert float(np.max(noise_h2)) == pytest.approx(5.0 * 8**4)

    uninterrupted_overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=0,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=1,
        process_start_noise_broadcast=False,
    )
    uninterrupted_h1, uninterrupted_h2 = uninterrupted_overrides[0]["noise_variance"]
    assert not np.array_equal(uninterrupted_h2, uninterrupted_h1)
    assert float(np.min(uninterrupted_h1)) == pytest.approx(1.0 * 8**4)
    assert float(np.min(uninterrupted_h2)) == pytest.approx(6.0 * 8**4)


def test_autorefine_later_replay_noise_remains_half_specific(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnNormCorrection": [1.0, 1.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    starfile.write({"particles": particles}, tmp_path / "run_it001_data.star")

    for half, sigma2_noise in ((1, [1.0, 2.0, 3.0, 4.0, 5.0]), (2, [6.0, 7.0, 8.0, 9.0, 10.0])):
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {"rlnNormCorrectionAverage": [1.0], "rlnSigmaOffsetsAngst": [2.0]}
                ),
                "model_optics_group_1": pd.DataFrame(
                    {"rlnSigma2Noise": np.asarray(sigma2_noise, dtype=np.float64)}
                ),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it001_half{half}_model.star",
        )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
    )

    noise_h1, noise_h2 = overrides[1]["noise_variance"]
    assert not np.array_equal(noise_h2, noise_h1)
    assert float(np.min(noise_h1)) == pytest.approx(1.0 * 8**4)
    assert float(np.min(noise_h2)) == pytest.approx(6.0 * 8**4)


def test_replay_overrides_respect_init_relion_iteration_offset(tmp_path):
    pd = pytest.importorskip("pandas")
    starfile = pytest.importorskip("starfile")

    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [11.0, 21.0],
            "rlnAnglePsi": [12.0, 22.0],
            "rlnOriginXAngst": [2.0, -4.0],
            "rlnOriginYAngst": [1.0, -3.0],
            "rlnNormCorrection": [2.0, 4.0],
            "rlnGroupNumber": [1, 1],
        }
    )
    for relion_iter, sigma in ((6, 6.5), (7, 7.5)):
        starfile.write({"particles": particles}, tmp_path / f"run_it{relion_iter:03d}_data.star")
        starfile.write(
            {
                "model_general": pd.DataFrame(
                    {
                        "rlnNormCorrectionAverage": [float(relion_iter)],
                        "rlnSigmaOffsetsAngst": [sigma],
                    }
                ),
                "model_groups": pd.DataFrame({"rlnGroupScaleCorrection": [1.0]}),
            },
            tmp_path / f"run_it{relion_iter:03d}_model.star",
        )

    overrides = _build_replay_iteration_overrides(
        tmp_path,
        half1_idx=np.asarray([0], dtype=np.int64),
        half2_idx=np.asarray([1], dtype=np.int64),
        max_iter=1,
        ds_voxel=2.0,
        ds_grid=8,
        include_normcorr=False,
        init_relion_iteration=6,
    )

    assert len(overrides) == 2
    assert overrides[0]["translation_sigma_angstrom"] == pytest.approx(6.5)
    assert overrides[1]["translation_sigma_angstrom"] == pytest.approx(7.5)
    np.testing.assert_allclose(
        overrides[0]["previous_best_rotation_eulers"][0],
        np.asarray([[10.0, 11.0, 12.0]], dtype=np.float32),
    )
