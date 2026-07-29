from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_em_k1_map_amplitude_trajectory import (
    CaseSpec,
    analyze,
    parse_case_spec,
    parse_reference_iterations,
    summarize_fourier_pair,
    unshifted_shell_labels,
)


def test_shellwise_fit_recovers_known_positive_amplitude_profile():
    rng = np.random.default_rng(7)
    source = (
        rng.standard_normal((8, 8, 8))
        + 1j * rng.standard_normal((8, 8, 8))
    )
    labels = unshifted_shell_labels(source.shape)
    expected_scales = 0.98 + 0.002 * labels
    target = source * expected_scales

    summary = summarize_fourier_pair(source, target)

    assert summary["relative_l2_before"] > 0.0
    assert summary["relative_l2_after_shell_scale"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert summary["shell_scale_explained_fraction"] == pytest.approx(1.0)
    for row in summary["shells"]:
        shell = row["shell"]
        assert row["scale_recovar_to_relion"] == pytest.approx(
            0.98 + 0.002 * shell
        )


def test_shellwise_fit_rejects_phase_inverted_shell():
    source = np.ones((4, 4, 4), dtype=np.complex64)
    target = source.copy()
    target[unshifted_shell_labels(source.shape) == 1] *= -1

    with pytest.raises(ValueError, match="amplitude scale is not positive"):
        summarize_fourier_pair(source, target)


def test_case_and_iteration_parsers_are_fail_closed():
    spec = parse_case_spec("case22=/tmp/recovar,/tmp/relion")
    assert spec == CaseSpec(
        label="case22",
        recovar_intermediates=Path("/tmp/recovar"),
        relion_reference=Path("/tmp/relion"),
    )
    assert spec.label == "case22"
    assert str(spec.recovar_intermediates) == "/tmp/recovar"
    assert str(spec.relion_reference) == "/tmp/relion"
    assert parse_reference_iterations("1, 2;3") == (1, 2, 3)
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_reference_iterations("2,1")
    with pytest.raises(ValueError, match="LABEL="):
        parse_case_spec("case22")


def test_analyze_rejects_duplicate_labels_before_file_io():
    spec = parse_case_spec("duplicate=/missing/recovar,/missing/relion")
    with pytest.raises(ValueError, match="labels must be unique"):
        analyze([spec, spec], reference_iterations=(1,))
    with pytest.raises(ValueError, match="unique and increasing"):
        analyze([spec], reference_iterations=(2, 1))
