import copy

import pandas as pd
import pytest

from scripts import analyze_relion_control_capture_inertness as analyzer


def _particles():
    return pd.DataFrame(
        {
            "rlnImageName": ["3@particles.mrcs", "1@particles.mrcs", "2@particles.mrcs"],
            "rlnAngleRot": [3.0, 1.0, 2.0],
            "rlnAngleTilt": [13.0, 11.0, 12.0],
            "rlnAnglePsi": [23.0, 21.0, 22.0],
            "rlnOriginXAngst": [33.0, 31.0, 32.0],
            "rlnOriginYAngst": [43.0, 41.0, 42.0],
            "rlnClassNumber": [1, 1, 1],
            "rlnMaxValueProbDistribution": [0.3, 0.1, 0.2],
            "rlnNrOfSignificantSamples": [3, 1, 2],
        }
    )


def test_target_is_resolved_by_image_identity_not_dataframe_position():
    control = _particles()
    capture = control.copy()
    capture.loc[capture["rlnImageName"] == "2@particles.mrcs", "rlnOriginYAngst"] = 40.937502

    report = analyzer.compare_particle_tables(
        control,
        capture,
        target_original_index=1,
        target_image_identity="2@particles.mrcs",
    )

    assert report["target"]["control_raw_row"] == 2
    assert report["target"]["image_identity"] == "2@particles.mrcs"
    assert report["target"]["mismatch_fields"] == ["rlnOriginYAngst"]
    result = report["fields"]["rlnOriginYAngst"]
    assert result["mismatch_count"] == 1
    assert result["max_abs"] == pytest.approx(1.062498)
    assert result["mismatch_examples"] == [
        {
            "image_identity": "2@particles.mrcs",
            "control_value": 42.0,
            "capture_value": 40.937502,
            "control_raw_row": 2,
            "capture_raw_row": 2,
        }
    ]


def test_capture_row_order_is_ignored_after_exact_identity_alignment():
    control = _particles()
    capture = control.iloc[::-1].reset_index(drop=True)

    report = analyzer.compare_particle_tables(
        control,
        capture,
        target_original_index=1,
        target_image_identity="2@particles.mrcs",
    )

    assert report["raw_row_order_exact"] is False
    assert all(result["exact"] for result in report["fields"].values())
    assert report["target"]["capture_raw_row"] == 0


@pytest.mark.parametrize(
    ("target_original_index", "target_identity", "message"),
    [
        (1, "3@particles.mrcs", "does not equal target original index plus one"),
        (1, "particles.mrcs", "positive one-based stack prefix"),
        (-1, "1@particles.mrcs", "must be nonnegative"),
    ],
)
def test_target_identity_must_encode_original_index(
    target_original_index, target_identity, message
):
    with pytest.raises(ValueError, match=message):
        analyzer.compare_particle_tables(
            _particles(),
            _particles(),
            target_original_index=target_original_index,
            target_image_identity=target_identity,
        )


def test_duplicate_or_different_identity_sets_are_rejected():
    duplicate = _particles()
    duplicate.loc[2, "rlnImageName"] = "1@particles.mrcs"
    with pytest.raises(ValueError, match="duplicate identities"):
        analyzer.compare_particle_tables(
            _particles(),
            duplicate,
            target_original_index=1,
            target_image_identity="2@particles.mrcs",
        )

    different = copy.deepcopy(_particles())
    different.loc[0, "rlnImageName"] = "4@particles.mrcs"
    with pytest.raises(ValueError, match="identity sets differ"):
        analyzer.compare_particle_tables(
            _particles(),
            different,
            target_original_index=1,
            target_image_identity="2@particles.mrcs",
        )


def test_build_report_rejects_nonpositive_expected_particle_count(tmp_path):
    with pytest.raises(ValueError, match="expected particle count must be positive"):
        analyzer.build_report(
            control_root=tmp_path / "control",
            capture_root=tmp_path / "capture",
            target_original_index=1,
            target_image_identity="2@particles.mrcs",
            expected_particle_count=0,
            gpu_uuid="GPU-test",
        )
