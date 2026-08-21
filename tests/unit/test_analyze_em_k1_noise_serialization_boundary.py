from __future__ import annotations

from scripts import analyze_em_k1_noise_serialization_boundary as analyzer


def _passing_shell(
    shell: int,
    *,
    control: bool = False,
) -> dict[str, float | int]:
    return {
        "shell": shell,
        "evaluated_particles": analyzer.EXPECTED_PARTICLES,
        "recovar_vs_token_max_abs": 1.0e-10,
        "live_relion_vs_token_min_abs": 1.0e-7 if not control else 1.0e-10,
        "live_relion_vs_token_max_abs": 2.0e-7 if not control else 2.0e-10,
        "serialized_closeness_ratio_min": 1_000.0,
        "recovar_within_shell_ptp_max": 1.0e-9,
        "live_relion_within_shell_ptp_max": 2.0e-9,
    }


def _passing_metrics() -> dict[int, dict[str, float | int]]:
    return {
        shell: _passing_shell(shell)
        for shell in analyzer.FIXED_DECIMAL_SHELLS
    } | {
        analyzer.SCIENTIFIC_CONTROL_SHELL: _passing_shell(
            analyzer.SCIENTIFIC_CONTROL_SHELL,
            control=True,
        )
    }


def test_classifies_live_versus_serialized_noise_boundary() -> None:
    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=_passing_metrics(),
        )
        == analyzer.CLASSIFICATION
    )


def test_rejects_unqualified_parent() -> None:
    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=False,
            shell_metrics=_passing_metrics(),
        )
        == "noise_serialization_parent_not_qualified"
    )


def test_rejects_changed_shell_denominator() -> None:
    metrics = _passing_metrics()
    del metrics[4]

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "noise_serialization_shell_denominator_changed"
    )


def test_rejects_fixed_shell_that_matches_live_token() -> None:
    metrics = _passing_metrics()
    metrics[2]["live_relion_vs_token_min_abs"] = 1.0e-10

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "live_and_serialized_noise_state_boundary_did_not_close"
    )


def test_rejects_recovar_fixed_shell_away_from_serialized_token() -> None:
    metrics = _passing_metrics()
    metrics[3]["recovar_vs_token_max_abs"] = 1.0e-7

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "live_and_serialized_noise_state_boundary_did_not_close"
    )


def test_rejects_insufficient_serialized_closeness_ratio() -> None:
    metrics = _passing_metrics()
    metrics[1]["serialized_closeness_ratio_min"] = 99.0

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "live_and_serialized_noise_state_boundary_did_not_close"
    )


def test_rejects_scientific_control_shell_that_does_not_close() -> None:
    metrics = _passing_metrics()
    metrics[analyzer.SCIENTIFIC_CONTROL_SHELL][
        "live_relion_vs_token_max_abs"
    ] = 1.0e-7

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "live_and_serialized_noise_state_boundary_did_not_close"
    )


def test_rejects_within_shell_variation_above_gate() -> None:
    metrics = _passing_metrics()
    metrics[4]["live_relion_within_shell_ptp_max"] = 1.0e-7

    assert (
        analyzer.classify_serialization_boundary(
            parent_qualified=True,
            shell_metrics=metrics,
        )
        == "live_and_serialized_noise_state_boundary_did_not_close"
    )
