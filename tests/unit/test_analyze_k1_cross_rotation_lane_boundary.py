from types import SimpleNamespace

import numpy as np

from scripts import analyze_k1_cross_rotation_lane_boundary as subject
from scripts.analyze_em_k1_coarse_pass1_boundary import _map_relion_table
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
)


def test_outcome_report_checks_exact_float32_membership():
    values = np.asarray([1.0, 1.5], dtype=np.float32)

    assert subject._outcome_report(values, 1.5)["production_is_legal"]
    assert not subject._outcome_report(values, 1.25)["production_is_legal"]


def test_analyze_compares_captured_projection_operands(monkeypatch, tmp_path):
    component_header = [0] * 40
    component_header[10:13] = [2, 2, 2]
    component_header[27] = 2
    native_raw = np.ones((4, 2), dtype=np.float32)
    native_raw[3, 0] = 2.0
    components = SimpleNamespace(
        part_id=7,
        stack_index=1,
        header=tuple(component_header),
        translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        raw_diff2=native_raw,
        reference_norms=np.zeros((4, 2), dtype=np.float32),
        cross_terms=np.zeros((4, 2), dtype=np.float32),
    )
    native_reference = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    operands = SimpleNamespace(
        part_id=7,
        stack_index=1,
        rotation_keys=np.asarray([0, 3], dtype=np.uint64),
        reference_real=native_reference,
        reference_imag=np.zeros_like(native_reference),
        shifted_real=np.ones((2, 4), dtype=np.float32),
        shifted_imag=np.zeros((2, 4), dtype=np.float32),
        correction=np.full(4, 16.0, dtype=np.float32),
    )
    lane_header = [0] * 21
    lane_header[20] = 0
    lanes = SimpleNamespace(
        part_id=7,
        stack_index=1,
        header=tuple(lane_header),
        rotation_keys=np.asarray([0, 3], dtype=np.uint64),
        lane_partials=np.asarray(
            [[0.5, 0.5, 0.5, 0.5], [1.0, 0.5, 1.0, 0.5]],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(subject, "load_components", lambda _: components)
    monkeypatch.setattr(subject, "load_operands", lambda _: operands)
    monkeypatch.setattr(subject, "load_lanes", lambda _: lanes)
    monkeypatch.setattr(
        subject,
        "validate_lanes",
        lambda *_: {"status": "pass", "classification": "synthetic"},
    )

    window_indices = np.arange(4, dtype=np.int32)
    projected_reference = relion_reference_on_recovar_window(
        native_reference.astype(np.complex64),
        window_indices,
        full_image_size=2,
        current_size=2,
    )
    mapped_raw = _map_relion_table(
        native_raw,
        n_directions=2,
        n_psi=2,
        relion_to_recovar_translation=np.arange(2),
    )
    recovar_scores = -mapped_raw
    component_residual = np.asarray([[0.0, 0.25], [-0.125, 0.0]], dtype=np.float32)
    recovar_scores[[0, 3]] += component_residual
    recovar_path = tmp_path / "recovar.npz"
    np.savez(
        recovar_path,
        original_index=np.asarray(0),
        translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        scores_pre_prior_per_class=recovar_scores[np.newaxis],
        projected_reference_rotation_ids=np.asarray([0, 3], dtype=np.int32),
        projected_reference_per_class=projected_reference[np.newaxis],
        projected_reference_norm_score_per_class=component_residual[np.newaxis],
        projected_cross_score_per_class=np.zeros((1, 2, 2), dtype=np.float64),
        window_indices=window_indices,
        shifted_data=np.full((2, 4), -4.0, dtype=np.complex128),
        ctf2_data=np.ones((1, 4), dtype=np.float64),
        half_weights=np.ones(4, dtype=np.float64),
    )
    component_path = tmp_path / "components.bin"
    operand_path = tmp_path / "operands.bin"
    lane_path = tmp_path / "lanes.bin"
    for path in (component_path, operand_path, lane_path):
        path.write_bytes(path.name.encode())

    report = subject.analyze(
        components_path=component_path,
        operands_path=operand_path,
        lanes_path=lane_path,
        recovar_path=recovar_path,
        winner_native_rotation=0,
        winner_recovar=(0, 0),
        target_native_rotation=3,
        target_recovar=(3, 0),
        physical_image_size=2,
    )

    assert report["operand_boundary"]["status"] == "complete"
    assert report["operand_boundary"]["operand_relative_l2"] == {
        "projected_reference": 0.0,
        "weighted_shifted_image": 0.0,
        "correction": 0.0,
    }
    assert (
        report["operand_boundary"]["component_decomposition"][
            "counterfactual_energy_removal_fraction"
        ]["reference_norm"]
        == 1.0
    )
