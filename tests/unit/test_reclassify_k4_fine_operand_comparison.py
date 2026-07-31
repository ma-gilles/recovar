from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reclassify_k4_fine_operand_comparison import reclassify


def _counterfactual(
    *,
    centered: bool,
    target_l2: float,
    fractions: dict[str, float],
) -> dict[str, object]:
    return {
        "deltas_centered": centered,
        "single_component_substitution": {
            name: {
                "target_all_recovar_delta_l2": target_l2,
                "after_single_component_substitution_l2": 0.0,
                "target_delta_energy_removed_fraction": fractions[name],
            }
            for name in ("reference", "shifted_image", "corr")
        },
        "strongest_single_component": max(fractions, key=fractions.get),
        "strongest_target_delta_energy_removed_fraction": max(
            fractions.values()
        ),
    }


def _write_comparison(
    path: Path,
    *,
    candidate_count: int,
    raw_l2: float,
    centered_l2: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "k4_relion_recovar_fine_operand_comparison_v8",
                "status": "complete",
                "classification": (
                    "reference_dominates_centered_fine_operand_residual"
                ),
                "candidates": [{} for _ in range(candidate_count)],
                "raw_diff2_component_counterfactual": _counterfactual(
                    centered=False,
                    target_l2=raw_l2,
                    fractions={
                        "reference": 0.0,
                        "shifted_image": 0.5,
                        "corr": 1.0,
                    },
                ),
                "centered_raw_diff2_component_counterfactual": (
                    _counterfactual(
                        centered=True,
                        target_l2=centered_l2,
                        fractions={
                            "reference": 0.0,
                            "shifted_image": 1.0,
                            "corr": 0.0,
                        },
                    )
                ),
            }
        )
    )


@pytest.mark.unit
def test_reclassifies_one_candidate_from_raw_counterfactual(
    tmp_path: Path,
) -> None:
    path = tmp_path / "comparison.json"
    _write_comparison(
        path,
        candidate_count=1,
        raw_l2=2.0,
        centered_l2=0.0,
    )

    report = reclassify(path)

    assert report["classification_basis"] == "raw_diff2"
    assert report["selected_component"] == "corr"
    assert (
        report["classification"]
        == "corr_has_largest_raw_fine_operand_single_substitution_effect"
    )
    assert report["classification_changed"] is True
    assert report["scorecard_change_admissible"] is False


@pytest.mark.unit
def test_reclassifies_multiple_candidates_from_centered_counterfactual(
    tmp_path: Path,
) -> None:
    path = tmp_path / "comparison.json"
    _write_comparison(
        path,
        candidate_count=3,
        raw_l2=20.0,
        centered_l2=2.0,
    )

    report = reclassify(path)

    assert report["classification_basis"] == "centered_raw_diff2"
    assert report["selected_component"] == "shifted_image"
    assert (
        report["classification"]
        == (
            "shifted_image_has_largest_centered_fine_operand_"
            "single_substitution_effect"
        )
    )


@pytest.mark.unit
def test_rejects_inconsistent_component_target_energy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "comparison.json"
    _write_comparison(
        path,
        candidate_count=1,
        raw_l2=2.0,
        centered_l2=0.0,
    )
    value = json.loads(path.read_text())
    value["raw_diff2_component_counterfactual"][
        "single_component_substitution"
    ]["reference"]["target_all_recovar_delta_l2"] = 3.0
    path.write_text(json.dumps(value))

    with pytest.raises(
        ValueError,
        match="component records disagree on target delta L2",
    ):
        reclassify(path)
