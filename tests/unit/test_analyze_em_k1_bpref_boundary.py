import numpy as np
import pytest

from scripts.analyze_em_k1_bpref_boundary import (
    classify_boundary,
    compare_fsc_curves,
    complex_pair_metrics,
)


def test_complex_pair_metrics_recovers_positive_scale_and_fsc():
    target = np.asarray([1 + 2j, 3 - 1j, 2 + 0.5j, -1 + 1j])
    source = target / 2.0
    shell = np.asarray([0, 1, 1, 2])

    result = complex_pair_metrics(source, target, shell=shell, max_shell=2)

    assert result["fsc_auc"] == pytest.approx(1.0)
    assert result["global_scale_recovar_to_relion"] == pytest.approx(2.0)
    assert result["relative_l2"] == pytest.approx(0.5)
    assert result["relative_l2_after_global_scale"] == pytest.approx(0.0)
    assert result["global_scale_explained_fraction"] == pytest.approx(1.0)


def test_complex_pair_metrics_tracks_phase_residual():
    target = np.asarray([1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j])
    source = np.asarray([1 + 0j, 1j, -1j, 1 + 0j])
    shell = np.asarray([0, 1, 1, 2])

    result = complex_pair_metrics(source, target, shell=shell, max_shell=2)

    assert result["fsc_auc"] < 1.0
    assert result["relative_l2_after_global_scale"] > 0.0


def test_compare_fsc_curves_excludes_dc_from_auc():
    result = compare_fsc_curves(
        np.asarray([-1.0, 0.9, 0.8]),
        np.asarray([1.0, 0.8, 0.8]),
        max_shell=2,
    )

    assert result["recovar_fsc_auc"] == pytest.approx(0.85)
    assert result["relion_fsc_auc"] == pytest.approx(0.8)
    assert result["fsc_auc_delta_recovar_minus_relion"] == pytest.approx(0.05)
    assert result["maximum_delta_shell"] == 1


def test_classify_boundary_requires_repeat_gate():
    result = classify_boundary(
        repeat_fsc_aucs=[0.99, 1.0],
        accumulator_fsc_aucs=[0.9, 0.9],
        accumulator_relative_l2=[0.1, 0.1],
        repeat_fsc_auc_gate=0.99999,
        accumulator_fsc_auc_gate=0.99999,
        accumulator_relative_l2_gate=1.0e-3,
    )

    assert result["classification"] == (
        "inconclusive_relion_diagnostic_repeat_gate_failed"
    )
    assert not result["relion_repeat_gate_pass"]


@pytest.mark.parametrize(
    ("fsc_aucs", "relative_l2", "expected"),
    [
        (
            [0.999999, 0.999999],
            [1.0e-4, 2.0e-4],
            "physical_iteration2_postjoin_bpref_has_no_material_cross_engine_residual",
        ),
        (
            [0.999, 0.999999],
            [1.0e-4, 2.0e-4],
            "physical_iteration2_postjoin_bpref_contains_cross_engine_residual",
        ),
        (
            [0.999999, 0.999999],
            [1.0e-4, 2.0e-3],
            "physical_iteration2_postjoin_bpref_contains_cross_engine_residual",
        ),
    ],
)
def test_classify_boundary_material_gate(fsc_aucs, relative_l2, expected):
    result = classify_boundary(
        repeat_fsc_aucs=[1.0, 1.0],
        accumulator_fsc_aucs=fsc_aucs,
        accumulator_relative_l2=relative_l2,
        repeat_fsc_auc_gate=0.99999,
        accumulator_fsc_auc_gate=0.99999,
        accumulator_relative_l2_gate=1.0e-3,
    )

    assert result["classification"] == expected
    assert result["relion_repeat_gate_pass"]
