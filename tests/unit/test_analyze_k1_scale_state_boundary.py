import numpy as np
import pandas as pd
import pytest
import starfile

from scripts.analyze_k1_scale_state_boundary import analyze


def _write_model(path, *, counts, scales, size=8, general_as_mapping=False):
    groups = pd.DataFrame(
        {
            "rlnGroupNumber": np.arange(1, len(counts) + 1),
            "rlnGroupName": [f"group_{index}" for index in range(len(counts))],
            "rlnGroupNrParticles": counts,
            "rlnGroupScaleCorrection": scales,
        }
    )
    general = (
        {"rlnOriginalImageSize": size}
        if general_as_mapping
        else pd.DataFrame({"rlnOriginalImageSize": [size]})
    )
    starfile.write({"model_general": general, "model_groups": groups}, path)


def test_scale_state_boundary_ranks_active_group_residuals(tmp_path):
    counts1 = np.asarray([1, 0, 1, 0])
    counts2 = np.asarray([0, 1, 0, 1])
    input_scales = np.ones(4)
    relion1 = np.asarray([0.5, 1.0, 1.5, 1.0])
    relion2 = np.asarray([1.0, 0.75, 1.0, 1.25])
    for half, counts, output_scales in ((1, counts1, relion1), (2, counts2, relion2)):
        _write_model(
            tmp_path / f"in{half}.star",
            counts=counts,
            scales=input_scales,
            general_as_mapping=half == 1,
        )
        _write_model(tmp_path / f"out{half}.star", counts=counts, scales=output_scales)

    parity = tmp_path / "iter_002.npz"
    np.savez(
        parity,
        half1_wsum_scale_correction_xa=np.asarray([1.0, 0.0, 3.0, 0.0]),
        half1_wsum_scale_correction_aa=np.asarray([2.0, 0.0, 2.0, 0.0]),
        half1_group_particle_counts=counts1,
        half1_group_scale_corrections=np.asarray([0.51, 1.0, 1.49, 1.0]),
        half2_wsum_scale_correction_xa=np.asarray([0.0, 0.75, 0.0, 1.25]),
        half2_wsum_scale_correction_aa=np.asarray([0.0, 1.0, 0.0, 1.0]),
        half2_group_particle_counts=counts2,
        half2_group_scale_corrections=relion2,
    )

    report = analyze(
        parity,
        [tmp_path / "in1.star", tmp_path / "in2.star"],
        [tmp_path / "out1.star", tmp_path / "out2.star"],
        top_count=1,
    )

    assert report["halves"][0]["largest_scale_residuals"][0]["part_and_group_id_zero_based"] in (0, 2)
    assert report["halves"][0]["comparisons"]["recovar_dump_vs_relion_output"]["max_abs"] == pytest.approx(0.01)
    assert report["halves"][1]["comparisons"]["recovar_dump_vs_relion_output"]["max_abs"] == 0.0
    targets = {int(value) for value in report["native_capture_part_ids_csv"].split(",")}
    assert len(targets) == 2
    assert targets & {0, 2}
    assert targets & {1, 3}
    assert report["halves"][0]["native_unit_divisor"] == float(8**4)
