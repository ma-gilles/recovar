from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import audit_relion_k4_coarse_score_capture_inertness as audit


def test_particle_table_comparison_aligns_by_image_identity():
    control = pd.DataFrame(
        {
            "rlnImageName": ["1@stack.mrcs", "2@stack.mrcs"],
            "rlnAngleRot": [1.0, 2.0],
            "rlnMaxValueProbDistribution": [0.9, 0.8],
        }
    )
    instrumented = pd.DataFrame(
        {
            "rlnImageName": ["2@stack.mrcs", "1@stack.mrcs"],
            "rlnAngleRot": [2.0, 1.0],
            "rlnMaxValueProbDistribution": [0.8, 0.7],
        }
    )

    report = audit._compare_particle_tables(control, instrumented)

    assert report["raw_row_order_exact"] is False
    assert report["fields"]["rlnAngleRot"]["exact"] is True
    assert report["fields"]["rlnMaxValueProbDistribution"]["mismatch_count"] == 1
    assert report["fields"]["rlnMaxValueProbDistribution"]["max_abs"] == pytest.approx(0.2)


def test_particle_table_comparison_reports_exact_reordered_table():
    control = pd.DataFrame(
        {"rlnImageName": ["1@a", "2@a"], "rlnClassNumber": [1, 2]}
    )
    instrumented = control.iloc[::-1].reset_index(drop=True)

    report = audit._compare_particle_tables(control, instrumented)

    assert report["raw_row_order_exact"] is False
    assert report["exact_field_count"] == report["field_count"]
    assert np.all([field["exact"] for field in report["fields"].values()])
