"""Fail-closed CLI helpers for resident-state swap diagnostics."""

from collections.abc import Mapping, Sequence

REQUIRED_STATE_SWAP_REPLAY_KEYS = frozenset(
    {
        "direction_prior",
        "image_corrections",
        "mean_variance",
        "noise_variance",
        "previous_best_rotation_eulers",
        "previous_best_rotations",
        "previous_best_translations",
        "serialized_scale_corrections",
        "translation_sigma_angstrom",
        "translation_sigma_angstrom_per_half",
    }
)


def state_swap_variant_choices() -> tuple[str, ...]:
    """Return the variants implemented by the iteration-loop diagnostic."""
    from recovar.em.dense_single_volume.iteration_loop import _STATE_SWAP_VARIANT_COMPONENTS

    return tuple(sorted(_STATE_SWAP_VARIANT_COMPONENTS))


def add_state_swap_probe_arguments(parser) -> None:
    """Add the physical-iteration state-swap CLI contract."""
    parser.add_argument(
        "--state-swap-target-relion-iteration",
        type=int,
        default=None,
        help=(
            "Diagnostic only: physical numbered RELION expectation iteration at "
            "which to restore the selected RECOVAR-produced resident state after "
            "the RELION replay override. Requires --state-swap-variant, "
            "--state-swap-replay-relion-references, and a complete replay override "
            "at the target."
        ),
    )
    parser.add_argument(
        "--state-swap-variant",
        default=None,
        metavar="VARIANT",
        help=(
            "Diagnostic resident-state arm to apply at "
            "--state-swap-target-relion-iteration. The value is validated against "
            "the iteration loop's resident-state variant table."
        ),
    )
    parser.add_argument(
        "--state-swap-replay-relion-references",
        action="store_true",
        help=(
            "Diagnostic only: use the target boundary's RELION half maps as the "
            "all-RELION scoring-reference control before restoring the selected "
            "RECOVAR state. Required for every CLI state-swap arm so arm labels "
            "have identical, complete boundary semantics."
        ),
    )


def state_swap_probe_loop_index(
    *,
    target_relion_iteration: int | None,
    variant: str | None,
    replay_relion_references: bool,
    init_relion_iteration: int,
    max_iter: int,
) -> int | None:
    """Validate a request and map its physical target to a zero-based loop index."""
    requested = (
        target_relion_iteration is not None
        or variant is not None
        or bool(replay_relion_references)
    )
    if not requested:
        return None
    if target_relion_iteration is None or variant is None:
        raise ValueError(
            "--state-swap-target-relion-iteration and --state-swap-variant "
            "must be provided together"
        )
    if not replay_relion_references:
        raise ValueError(
            "state-swap CLI diagnostics require "
            "--state-swap-replay-relion-references so all arms use the same "
            "complete all-RELION boundary control"
        )
    if variant not in state_swap_variant_choices():
        raise ValueError(
            f"unknown state-swap variant {variant!r}; expected one of "
            f"{list(state_swap_variant_choices())}"
        )

    start_iteration = int(init_relion_iteration)
    numbered_iterations = int(max_iter)
    target_iteration = int(target_relion_iteration)
    first_target = start_iteration + 1
    last_target = start_iteration + numbered_iterations
    if target_iteration < first_target or target_iteration > last_target:
        raise ValueError(
            "--state-swap-target-relion-iteration must name a numbered "
            f"expectation emitted by this run: expected {first_target}..{last_target}, "
            f"got {target_iteration}"
        )
    return target_iteration - start_iteration - 1


def build_state_swap_probe(
    *,
    target_relion_iteration: int | None,
    variant: str | None,
    replay_relion_references: bool,
    init_relion_iteration: int,
    max_iter: int,
    replay_iteration_overrides: Sequence[Mapping | None] | None,
) -> dict[str, object] | None:
    """Build the internal probe after proving target replay state exists."""
    loop_index = state_swap_probe_loop_index(
        target_relion_iteration=target_relion_iteration,
        variant=variant,
        replay_relion_references=replay_relion_references,
        init_relion_iteration=init_relion_iteration,
        max_iter=max_iter,
    )
    if loop_index is None:
        return None
    if replay_iteration_overrides is None or loop_index >= len(replay_iteration_overrides):
        raise ValueError(
            "state-swap diagnostics require replay_iteration_overrides covering "
            f"physical RELION iteration {int(target_relion_iteration)}"
        )
    target_override = replay_iteration_overrides[loop_index]
    if not isinstance(target_override, Mapping) or not target_override:
        raise ValueError(
            "state-swap diagnostics require a non-empty replay override at "
            f"physical RELION iteration {int(target_relion_iteration)} "
            f"(zero-based RECOVAR loop index {loop_index})"
        )
    missing_keys = sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS - set(target_override))
    if missing_keys:
        raise ValueError(
            "state-swap diagnostics require a complete target replay override; "
            f"physical RELION iteration {int(target_relion_iteration)} is missing {missing_keys}"
        )
    return {
        "iteration": loop_index,
        "target_relion_iteration": int(target_relion_iteration),
        "variant": str(variant),
        "replay_relion_references": True,
        "replay_override_keys": tuple(sorted(target_override)),
        "required_replay_override_keys": tuple(sorted(REQUIRED_STATE_SWAP_REPLAY_KEYS)),
    }


def validate_state_swap_probe_application(
    probe: Mapping | None,
    applied_relion_iterations: Sequence[int] | None,
) -> None:
    """Fail if an enabled diagnostic did not apply exactly once at its target."""
    if probe is None:
        if applied_relion_iterations:
            raise ValueError(
                "state-swap application telemetry is non-empty without a requested probe"
            )
        return
    expected = [int(probe["target_relion_iteration"])]
    applied = [int(iteration) for iteration in (applied_relion_iterations or [])]
    if applied != expected:
        raise ValueError(
            "state-swap diagnostic application mismatch: "
            f"expected physical RELION iterations {expected}, got {applied}"
        )
