from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from scripts.analyze_k1_scale_native_targets import analyze


def _write_model(path: Path, *, counts, scales, size: int) -> None:
    starfile.write(
        {
            "model_general": {"rlnOriginalImageSize": size},
            "model_groups": pd.DataFrame(
                {
                    "rlnGroupNumber": np.arange(1, len(counts) + 1),
                    "rlnGroupName": [f"group_{index}" for index in range(len(counts))],
                    "rlnGroupNrParticles": counts,
                    "rlnGroupScaleCorrection": scales,
                }
            ),
        },
        path,
    )


def _write_native(path: Path, *, half: int, part_id: int, xa: float, aa: float) -> None:
    path.write_text(
        "scale_totals"
        f"\titer=1\tpart_id={part_id}\thalfset={half}\tgroup_id={part_id}"
        f"\told_scale=1\txa={xa:.17g}\taa={aa:.17g}\traw={xa / aa:.17g}\n"
        "scale_totals"
        f"\titer=2\tpart_id={part_id}\thalfset={half}\tgroup_id={part_id}"
        f"\told_scale=1\txa={xa:.17g}\taa={aa:.17g}\traw={xa / aa:.17g}\n"
    )


def test_native_target_comparison_separates_xa_and_aa(tmp_path):
    counts1 = np.asarray([1, 0, 0, 0])
    counts2 = np.asarray([0, 1, 0, 0])
    for half, counts in ((1, counts1), (2, counts2)):
        _write_model(tmp_path / f"in{half}.star", counts=counts, scales=np.ones(4), size=2)
        _write_model(tmp_path / f"out{half}.star", counts=counts, scales=np.ones(4), size=2)
    np.savez(
        tmp_path / "dump.npz",
        half1_wsum_scale_correction_xa=np.asarray([32.0, 0.0, 0.0, 0.0]),
        half1_wsum_scale_correction_aa=np.asarray([16.0, 0.0, 0.0, 0.0]),
        half1_group_scale_corrections=np.ones(4),
        half2_wsum_scale_correction_xa=np.asarray([0.0, 48.0, 0.0, 0.0]),
        half2_wsum_scale_correction_aa=np.asarray([0.0, 32.0, 0.0, 0.0]),
        half2_group_scale_corrections=np.ones(4),
    )
    _write_native(tmp_path / "half1.tsv", half=1, part_id=0, xa=2.0, aa=1.0)
    _write_native(tmp_path / "half2.tsv", half=2, part_id=1, xa=2.0, aa=2.0)

    report = analyze(
        tmp_path / "dump.npz",
        [tmp_path / "half1.tsv", tmp_path / "half2.tsv"],
        [tmp_path / "in1.star", tmp_path / "in2.star"],
        [tmp_path / "out1.star", tmp_path / "out2.star"],
    )

    assert report["combined_comparisons"]["xa"]["max_abs"] == 1.0
    assert report["combined_comparisons"]["aa"]["max_abs"] == 0.0
    assert report["combined_comparisons"]["raw_with_relion_xa"]["max_abs"] == 0.0
    assert report["combined_comparisons"]["raw_with_relion_aa"]["max_abs"] == 0.5

    iteration1 = analyze(
        tmp_path / "dump.npz",
        [tmp_path / "half1.tsv", tmp_path / "half2.tsv"],
        [tmp_path / "in1.star", tmp_path / "in2.star"],
        [tmp_path / "out1.star", tmp_path / "out2.star"],
        iteration=1,
    )
    assert iteration1["iteration"] == 1
    assert iteration1["combined_comparisons"] == report["combined_comparisons"]
