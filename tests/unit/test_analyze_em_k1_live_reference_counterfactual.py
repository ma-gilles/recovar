from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import analyze_em_k1_live_reference_counterfactual as live_factorial
from scripts.analyze_em_k1_live_reference_counterfactual import (
    build_report,
    classify_live_operands,
    classify_live_reference,
    recovar_score_components,
    reference_swap_counterfactual,
    relion_reference_on_recovar_window,
)


def test_pass1_v1_operand_capture_replay_is_validated(tmp_path, monkeypatch) -> None:
    (tmp_path / "part1_stack7.p1-v1.bin").touch()
    operand_path = tmp_path / "part1_stack7.p1-op-v2.bin"
    operand_path.touch()
    component_header = [0] * 40
    component_header[12] = 2
    component_header[27] = 100
    component = SimpleNamespace(
        part_id=1,
        stack_index=7,
        mpi_rank=1,
        header=tuple(component_header),
        raw_diff2=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    operand_header = [0] * 40
    operand_header[12] = 100
    operand_header[14] = 2
    operand = SimpleNamespace(
        part_id=1,
        stack_index=7,
        mpi_rank=1,
        header=tuple(operand_header),
        rotation_keys=np.asarray([0, 2], dtype=np.uint64),
        path=operand_path,
    )
    monkeypatch.setattr(
        live_factorial,
        "validate_pass1",
        lambda *args, **kwargs: ((component,), {"status": "pass"}),
    )
    monkeypatch.setattr(live_factorial, "load_operand_artifact", lambda path: operand)
    monkeypatch.setattr(
        live_factorial,
        "replay_production_diff2",
        lambda item: component.raw_diff2[item.rotation_keys] + np.float32(8.0),
    )

    operands, components, validation = live_factorial._validate_operand_capture(
        tmp_path,
        expected_particles=1,
        expected_stacks=np.asarray([7]),
    )

    assert operands == (operand,)
    assert components == (component,)
    assert validation["status"] == "pass"
    assert validation["classification_ready"]
    assert validation["metrics"][operand_path.name][
        "production_diff2_additive_constant_median"
    ] == 8.0


def test_single_particle_diagnostic_denominator_is_supported(tmp_path) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text('{"selected_particle_count": 2, "rows": []}\n')
    try:
        build_report(
            cohort_json=cohort,
            capture_directory=tmp_path / "relion",
            recovar_directory=tmp_path / "recovar",
            full_image_size=256,
            expected_particles=1,
        )
    except ValueError as exc:
        assert "cohort denominator differs" in str(exc)
    else:
        raise AssertionError("mismatched diagnostic denominator must fail closed")


def test_maps_relion_fftw_rows_to_recovar_centered_window() -> None:
    current_size = 6
    full_size = 8
    current_half = 4
    reference = np.arange(current_size * current_half).reshape(1, -1)
    # Centered full rows ky=-2, 0, +3 and columns 1, 2, 0.
    window = np.asarray(
        [
            (full_size // 2 - 2) * (full_size // 2 + 1) + 1,
            (full_size // 2) * (full_size // 2 + 1) + 2,
            (full_size // 2 + 3) * (full_size // 2 + 1),
        ]
    )
    selected = relion_reference_on_recovar_window(
        reference,
        window,
        full_image_size=full_size,
        current_size=current_size,
    )
    expected_indices = np.asarray([4 * current_half + 1, 2, 3 * current_half])
    assert np.array_equal(
        selected[0],
        -(full_size**2) * reference[0, expected_indices],
    )


def test_recomputes_recovar_norm_and_cross_components() -> None:
    references = np.asarray([[1 + 2j, 3 - 1j], [2 - 1j, -1 + 0.5j]])
    shifted = np.asarray([[0.5 + 1j, 2 - 0.5j], [-1 + 0j, 0.25 + 2j]])
    ctf2 = np.asarray([2.0, 0.5])
    half_weights = np.asarray([1.0, 2.0])
    norm, cross = recovar_score_components(
        references,
        shifted,
        ctf2,
        half_weights,
    )
    expected_norm = -0.5 * np.sum(
        ctf2[None] * np.abs(references) ** 2 * half_weights[None],
        axis=1,
    )
    expected_cross = np.real(np.einsum("tp,rp,p->rt", np.conj(shifted), references, half_weights))
    assert np.allclose(norm, expected_norm[:, None])
    assert np.allclose(cross, expected_cross)


def test_reference_swap_reports_causal_energy_removal() -> None:
    rows = np.arange(4, dtype=np.float64)[:, None]
    columns = np.arange(3, dtype=np.float64)[None, :]
    baseline = 4.0 * rows + columns
    swapped = 0.1 * rows + 0.05 * columns
    report = reference_swap_counterfactual(baseline, swapped)
    assert report["live_reference_dominated"]
    assert report["counterfactual_energy_removal_fraction"] > 0.99


def test_classifies_fixed_cohort_live_reference_outcomes() -> None:
    assert (
        classify_live_reference(capture_qualified=False, dominated=14, expected=14)
        == "operand_capture_not_qualified"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=14, expected=14)
        == "raw_coarse_residual_is_live_projected_reference_dominated"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=0, expected=14)
        == "live_projected_reference_rejected_as_raw_coarse_residual_cause"
    )
    assert (
        classify_live_reference(capture_qualified=True, dominated=3, expected=14)
        == "raw_coarse_residual_has_mixed_live_projected_reference_effect"
    )


def test_classifies_live_base_image_factorial() -> None:
    dominated = {
        "reference": 0,
        "shifted_image": 14,
        "correction": 0,
        "reference_and_shifted_image": 14,
        "reference_and_correction": 0,
        "shifted_image_and_correction": 14,
        "all_live": 14,
        "base_corrected_image": 14,
        "translation_phase": 0,
    }
    assert (
        classify_live_operands(
            capture_qualified=True,
            dominated=dominated,
            expected=14,
        )
        == "raw_coarse_residual_is_live_base_corrected_image_dominated_"
        "not_reference_correction_or_translation_phase"
    )
    assert (
        classify_live_operands(
            capture_qualified=False,
            dominated=dominated,
            expected=14,
        )
        == "operand_capture_not_qualified"
    )
