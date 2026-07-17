from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recovar.data_io.starfile import write_star
from scripts import analyze_em_hidden_change_distribution as analyzer


def _write_star(path: Path, rows: dict[str, object]) -> Path:
    write_star(str(path), pd.DataFrame(rows))
    return path


@pytest.mark.unit
def test_hidden_change_analysis_aligns_rows_and_exposes_tail_subgroup(tmp_path):
    names = np.asarray([f"{index + 1:06d}@particles.mrcs" for index in range(6)])
    source = _write_star(
        tmp_path / "particles.star",
        {"_rlnImageName": names, "_rlnRandomSubset": [1, 2, 1, 2, 1, 2]},
    )
    rec_previous = np.zeros((6, 3), dtype=np.float32)
    rec_current = rec_previous.copy()
    rec_current[:, 0] = [1, 2, 3, 4, 5, 90]
    rec_pmax_previous = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    rec_pmax_current = rec_pmax_previous - 0.01
    results = tmp_path / "results.npz"
    np.savez(
        results,
        half1_indices=np.asarray([0, 2, 4]),
        half2_indices=np.asarray([1, 3, 5]),
        best_rotation_eulers_by_image_iter_000=rec_previous,
        best_rotation_eulers_by_image_iter_001=rec_current,
        pmax_per_image_by_image_iter_000=rec_pmax_previous,
        pmax_per_image_by_image_iter_001=rec_pmax_current,
    )

    permutation = np.asarray([5, 1, 4, 0, 3, 2])
    rel_previous = rec_previous.copy()
    rel_current = rec_current.copy()
    rel_current[5, 0] = 10.0
    previous_star = _write_star(
        tmp_path / "run_it001_data.star",
        {
            "_rlnImageName": names[permutation],
            "_rlnAngleRot": rel_previous[permutation, 0],
            "_rlnAngleTilt": rel_previous[permutation, 1],
            "_rlnAnglePsi": rel_previous[permutation, 2],
            "_rlnMaxValueProbDistribution": rec_pmax_previous[permutation],
        },
    )
    current_star = _write_star(
        tmp_path / "run_it002_data.star",
        {
            "_rlnImageName": names[permutation],
            "_rlnAngleRot": rel_current[permutation, 0],
            "_rlnAngleTilt": rel_current[permutation, 1],
            "_rlnAnglePsi": rel_current[permutation, 2],
            "_rlnMaxValueProbDistribution": rec_pmax_current[permutation],
        },
    )

    report, arrays = analyzer.analyze(
        recovar_results=results,
        recovar_particles_star=source,
        relion_previous_star=previous_star,
        relion_current_star=current_star,
        previous_iteration=1,
        current_iteration=2,
        subgroup_threshold_deg=0.1,
    )

    assert report["schema"] == analyzer.SCHEMA
    assert report["subgroup"]["count"] == 1
    assert report["subgroup"]["half1_fraction"] == pytest.approx(0.0)
    assert report["subgroup"]["cross_pose_union_contingency"] == {
        "subgroup_and_cross_pose": 1,
        "subgroup_only": 0,
        "cross_pose_only": 0,
        "neither": 5,
    }
    assert report["classification"]["localized_to_cross_engine_pose_mismatch"] is True
    # RELION's C1 metric averages the three matrix-row angles. A pure 80-degree
    # Z rotation contributes (80 + 80 + 0) / 3 degrees.
    assert report["absolute_change_difference_deg"]["max"] == pytest.approx(160.0 / 3.0)
    assert np.flatnonzero(arrays["subgroup_mask"]).tolist() == [5]
    assert "no correlation" in report["quality_metric_policy"].lower()


@pytest.mark.unit
def test_hidden_change_analysis_requires_consecutive_iterations(tmp_path):
    with pytest.raises(Exception, match="consecutive boundary"):
        analyzer.analyze(
            recovar_results=tmp_path / "missing.npz",
            recovar_particles_star=tmp_path / "missing.star",
            relion_previous_star=tmp_path / "a.star",
            relion_current_star=tmp_path / "b.star",
            previous_iteration=1,
            current_iteration=3,
        )
