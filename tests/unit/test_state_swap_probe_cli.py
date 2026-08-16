"""CLI and ordering contract for the resident-state swap diagnostic."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.state_swap_probe import (
    REQUIRED_STATE_SWAP_REPLAY_KEYS,
    add_state_swap_probe_arguments,
    build_state_swap_probe,
    state_swap_probe_loop_index,
    state_swap_variant_choices,
    validate_state_swap_probe_application,
)
from recovar.em.dense_single_volume.iteration_loop import (
    _STATE_SWAP_VARIANT_COMPONENTS,
    _apply_state_swap_probe,
    _scale_state_swap_reference_maps,
    _snapshot_state_swap_inputs,
)

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_state_swap_args(*tokens):
    parser = argparse.ArgumentParser()
    add_state_swap_probe_arguments(parser)
    return parser.parse_args(tokens)


def _loop_index(args, *, init_relion_iteration=0, max_iter=5):
    return state_swap_probe_loop_index(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=init_relion_iteration,
        max_iter=max_iter,
    )


def _complete_replay_override():
    return {key: object() for key in REQUIRED_STATE_SWAP_REPLAY_KEYS}


def test_state_swap_cli_default_is_inert():
    args = _parse_state_swap_args()

    assert _loop_index(args) is None
    assert build_state_swap_probe(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=0,
        max_iter=5,
        replay_iteration_overrides=None,
    ) is None
    validate_state_swap_probe_application(None, [])


def test_state_swap_cli_builds_case20_iteration5_maps_noise_probe():
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        "5",
        "--state-swap-variant",
        "recovar_maps_tau2_noise",
        "--state-swap-replay-relion-references",
    )
    replay_overrides = [None, None, None, None, _complete_replay_override()]

    probe = build_state_swap_probe(
        target_relion_iteration=args.state_swap_target_relion_iteration,
        variant=args.state_swap_variant,
        replay_relion_references=args.state_swap_replay_relion_references,
        init_relion_iteration=0,
        max_iter=5,
        replay_iteration_overrides=replay_overrides,
    )

    assert probe == {
        "iteration": 4,
        "target_relion_iteration": 5,
        "variant": "recovar_maps_tau2_noise",
        "replay_relion_references": True,
        "replay_override_keys": tuple(sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS)),
        "required_replay_override_keys": tuple(sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS)),
    }
    validate_state_swap_probe_application(probe, [5])


@pytest.mark.parametrize(
    ("physical_target", "expected_loop_index"),
    [(4, 0), (5, 1), (6, 2)],
)
def test_state_swap_physical_iteration_mapping_handles_nonzero_start(
    physical_target,
    expected_loop_index,
):
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        str(physical_target),
        "--state-swap-variant",
        "all_relion",
        "--state-swap-replay-relion-references",
    )

    assert _loop_index(args, init_relion_iteration=3, max_iter=3) == expected_loop_index


@pytest.mark.parametrize(
    ("tokens", "match"),
    [
        (("--state-swap-target-relion-iteration", "5"), "must be provided together"),
        (("--state-swap-variant", "all_relion"), "must be provided together"),
        (
            (
                "--state-swap-target-relion-iteration",
                "5",
                "--state-swap-variant",
                "all_relion",
            ),
            "require --state-swap-replay-relion-references",
        ),
    ],
)
def test_state_swap_cli_rejects_incomplete_boundary_contract(tokens, match):
    args = _parse_state_swap_args(*tokens)

    with pytest.raises(ValueError, match=match):
        _loop_index(args)


@pytest.mark.parametrize("target", [3, 6])
def test_state_swap_cli_rejects_target_outside_emitted_iterations(target):
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        str(target),
        "--state-swap-variant",
        "recovar_tau2_only",
        "--state-swap-replay-relion-references",
    )

    with pytest.raises(ValueError, match="expected 4..5"):
        _loop_index(args, init_relion_iteration=3, max_iter=2)


def test_state_swap_cli_choices_match_iteration_loop_variants():
    assert state_swap_variant_choices() == tuple(sorted(_STATE_SWAP_VARIANT_COMPONENTS))
    args = _parse_state_swap_args(
        "--state-swap-target-relion-iteration",
        "2",
        "--state-swap-variant",
        "not_a_variant",
        "--state-swap-replay-relion-references",
    )

    with pytest.raises(ValueError, match="unknown state-swap variant"):
        _loop_index(args, max_iter=2)


@pytest.mark.parametrize(
    "replay_overrides",
    [None, [None, None], [None, {}]],
)
def test_state_swap_cli_requires_nonempty_replay_context_at_target(replay_overrides):
    with pytest.raises(ValueError, match="replay"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="recovar_tau2_only",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=replay_overrides,
        )


def test_state_swap_cli_requires_complete_target_replay_context():
    incomplete = _complete_replay_override()
    del incomplete["mean_variance"]

    with pytest.raises(ValueError, match="missing.*mean_variance"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="recovar_tau2_only",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=[None, incomplete],
        )


def test_state_swap_cli_requires_exact_scoring_scale_oracle():
    incomplete = _complete_replay_override()
    del incomplete["scoring_scale_corrections"]

    with pytest.raises(ValueError, match="missing.*scoring_scale_corrections"):
        build_state_swap_probe(
            target_relion_iteration=2,
            variant="all_relion",
            replay_relion_references=True,
            init_relion_iteration=0,
            max_iter=2,
            replay_iteration_overrides=[None, incomplete],
        )


@pytest.mark.parametrize("applied", [None, [], [4], [5, 5]])
def test_state_swap_application_telemetry_fails_closed(applied):
    probe = {
        "iteration": 4,
        "target_relion_iteration": 5,
        "variant": "all_relion",
        "replay_relion_references": True,
    }

    with pytest.raises(ValueError, match="application mismatch"):
        validate_state_swap_probe_application(probe, applied)


def test_full_runner_propagates_and_serializes_state_swap_probe():
    tree = ast.parse((REPO_ROOT / "scripts/run_full_refinement.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {
        node.func.id
        for node in calls
        if isinstance(node.func, ast.Name)
    }
    refinement_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "refine_single_volume"
    ]
    # state_swap_probe is forwarded via the EngineDebugOptions group of
    # refine_single_volume's options= bundle (commit cd6661f2), not as a
    # top-level refine_single_volume keyword.
    debug_option_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "EngineDebugOptions"
    ]

    assert {
        "add_state_swap_probe_arguments",
        "state_swap_probe_loop_index",
        "build_state_swap_probe",
        "validate_state_swap_probe_application",
    } <= called_names
    assert len(refinement_calls) == 1
    assert len(debug_option_calls) == 1
    state_swap_keywords = [
        keyword for keyword in debug_option_calls[0].keywords if keyword.arg == "state_swap_probe"
    ]
    assert len(state_swap_keywords) == 1
    assert isinstance(state_swap_keywords[0].value, ast.Name)
    assert state_swap_keywords[0].value.id == "state_swap_probe"

    source = (REPO_ROOT / "scripts/run_full_refinement.py").read_text()
    for field in (
        "state_swap_probe_target_relion_iteration",
        "state_swap_probe_loop_index",
        "state_swap_probe_variant",
        "state_swap_probe_replay_relion_references",
        "state_swap_probe_applied_relion_iterations",
        "state_swap_probe_replay_override_keys",
        "state_swap_probe_required_replay_override_keys",
    ):
        assert f'"{field}"' in source


def test_relion_references_are_applied_before_state_restoration():
    tree = ast.parse(
        (REPO_ROOT / "recovar/em/dense_single_volume/iteration_loop.py").read_text()
    )
    loop_function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_relion_iteration_loop"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(loop_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {
            "_maybe_debug_replay_relion_references",
            "_apply_state_swap_probe",
        }
    }

    assert call_lines["_maybe_debug_replay_relion_references"] < call_lines[
        "_apply_state_swap_probe"
    ]


def test_state_swap_snapshot_is_bounded_to_target_iteration():
    source = (REPO_ROOT / "recovar/em/dense_single_volume/iteration_loop.py").read_text()
    snapshot_block = source.split("recovar_state_swap_snapshot = None", 1)[1].split(
        "replay_result = apply_iter_replay_overrides", 1
    )[0]

    assert "if state_swap_target_this_iteration:" in snapshot_block
    assert "if state_swap_probe is not None:" not in snapshot_block


def test_sigma_offset_state_swap_preserves_asymmetric_half_values():
    state = SimpleNamespace(marker="resident")
    half_inputs = SimpleNamespace(
        image_corrections=[np.array([1.0]), np.array([2.0])],
        scale_corrections=[np.array([3.0]), np.array([4.0])],
        previous_best_translations=[np.zeros((1, 2)), np.ones((1, 2))],
        previous_best_rotation_eulers=[np.zeros((1, 3)), np.ones((1, 3))],
    )
    snapshot = _snapshot_state_swap_inputs(
        state=state,
        cs=52,
        means=[np.array([1.0]), np.array([2.0])],
        mean_variance=np.array([5.0]),
        noise_variance_per_half=[np.array([6.0]), np.array([7.0])],
        noise_variance=np.array([6.5]),
        previous_noise_radial_per_half=[np.array([8.0]), np.array([9.0])],
        previous_noise_radial=np.array([8.5]),
        relion_half_inputs=half_inputs,
        previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]],
        current_sigma_offset_angstrom=3.0,
        current_sigma_offset_angstrom_per_half=[2.0, 4.0],
        class_direction_prior_per_half=[np.array([0.4]), np.array([0.6])],
        class_direction_prior_order_per_half=[3, 3],
        global_direction_prior_per_half=[np.array([0.4]), np.array([0.6])],
        global_direction_prior_order_per_half=[3, 3],
    )

    restored = _apply_state_swap_probe(
        probe={"iteration": 6, "variant": "recovar_sigma_offset"},
        iteration=6,
        recovar_snapshot=snapshot,
        state=state,
        cs=52,
        volume_shape=(1, 1, 1),
        means=[np.array([10.0]), np.array([20.0])],
        mean_variance=np.array([50.0]),
        noise_variance_per_half=[np.array([60.0]), np.array([70.0])],
        noise_variance=np.array([65.0]),
        previous_noise_radial_per_half=[np.array([80.0]), np.array([90.0])],
        previous_noise_radial=np.array([85.0]),
        relion_half_inputs=half_inputs,
        previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]],
        current_sigma_offset_angstrom=3.1,
        current_sigma_offset_angstrom_per_half=[2.9, 3.3],
        class_direction_prior_per_half=[np.array([0.5]), np.array([0.5])],
        class_direction_prior_order_per_half=[4, 4],
        global_direction_prior_per_half=[np.array([0.5]), np.array([0.5])],
        global_direction_prior_order_per_half=[4, 4],
    )

    restored_sigma = restored[8]
    restored_sigma_per_half = restored[9]
    assert restored_sigma == pytest.approx(3.0)
    assert restored_sigma_per_half == pytest.approx([2.0, 4.0])
    restored_sigma_per_half[0] = 99.0
    assert snapshot["current_sigma_offset_angstrom_per_half"] == pytest.approx([2.0, 4.0])


def test_state_swap_global_map_scaling_recovers_scalar_target():
    source = [
        np.arange(1, 9, dtype=np.float64).astype(np.complex128),
        (np.arange(1, 9, dtype=np.float64) * (1.0 + 2.0j)).astype(np.complex128),
    ]
    target = [1.25 * source[0], 0.75 * source[1]]

    scaled, summaries = _scale_state_swap_reference_maps(
        source,
        target,
        mode="global",
        volume_shape=(2, 2, 2),
    )

    np.testing.assert_allclose(scaled[0], target[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaled[1], target[1], rtol=1e-12, atol=1e-12)
    assert [summary["scale_min"] for summary in summaries] == pytest.approx(
        [1.25, 0.75]
    )
    assert [summary["relative_l2_after"] for summary in summaries] == pytest.approx(
        [0.0, 0.0],
        abs=1e-15,
    )


def test_state_swap_shell_map_scaling_recovers_shell_amplitudes_and_preserves_phase():
    volume_shape = (4, 4, 4)
    frequencies = np.fft.fftfreq(4) * 4
    grids = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    shell_labels = np.rint(np.sqrt(sum(grid * grid for grid in grids))).astype(
        np.int32
    ).reshape(-1)
    source = np.exp(1j * np.linspace(0.1, 2.5, shell_labels.size))
    shell_scales = 0.8 + 0.15 * shell_labels
    target = source * shell_scales

    scaled, summaries = _scale_state_swap_reference_maps(
        [source],
        [target],
        mode="shell",
        volume_shape=volume_shape,
    )

    np.testing.assert_allclose(scaled[0], target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.angle(scaled[0]),
        np.angle(source),
        rtol=0.0,
        atol=1e-12,
    )
    assert summaries[0]["scale_min"] == pytest.approx(float(np.min(shell_scales)))
    assert summaries[0]["scale_max"] == pytest.approx(float(np.max(shell_scales)))
