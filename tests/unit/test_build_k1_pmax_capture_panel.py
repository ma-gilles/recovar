from pathlib import Path

import numpy as np
import pytest

from scripts.build_k1_pmax_capture_panel import build_panel


def _write_star(path: Path, rows: list[tuple[str, int, float | None]]) -> None:
    labels = ["_rlnImageName #1", "_rlnRandomSubset #2"]
    if rows[0][2] is not None:
        labels.append("_rlnMaxValueProbDistribution #3")
    lines = ["data_particles", "", "loop_", *labels]
    for name, subset, pmax in rows:
        fields = [name, str(subset)]
        if pmax is not None:
            fields.append(str(pmax))
        lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_build_panel_aligns_by_identity_and_freezes_largest_errors(tmp_path):
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    parity = tmp_path / "iter_002.npz"
    _write_star(
        input_star,
        [("1@stack.mrcs", 1, None), ("2@stack.mrcs", 2, None), ("3@stack.mrcs", 1, None)],
    )
    _write_star(
        relion_star,
        [("3@stack.mrcs", 1, 0.7), ("1@stack.mrcs", 1, 0.2), ("2@stack.mrcs", 2, 0.9)],
    )
    np.savez(
        parity,
        relion_iteration=np.int32(2),
        half1_original_image_indices=np.asarray([2, 0]),
        half1_max_posterior=np.asarray([0.6, 0.25]),
        half2_original_image_indices=np.asarray([1]),
        half2_max_posterior=np.asarray([0.4]),
    )

    report = build_panel(
        parity_dump=parity,
        input_star=input_star,
        relion_star=relion_star,
        top_n=2,
    )

    assert report["relion_iteration"] == 2
    assert report["population"]["n_particles"] == 3
    assert [row["source_row_zero_based"] for row in report["rows"]] == [1, 2]
    assert [row["rank_by_absolute_pmax_error"] for row in report["rows"]] == [1, 2]
    assert report["rows"][0]["signed_pmax_delta_recovar_minus_relion"] == pytest.approx(-0.5)


@pytest.mark.unit
def test_build_panel_rejects_incomplete_half_coverage(tmp_path):
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    parity = tmp_path / "iter_002.npz"
    _write_star(input_star, [("1@stack.mrcs", 1, None), ("2@stack.mrcs", 2, None)])
    _write_star(relion_star, [("1@stack.mrcs", 1, 0.2), ("2@stack.mrcs", 2, 0.9)])
    np.savez(
        parity,
        relion_iteration=np.int32(2),
        half1_original_image_indices=np.asarray([0]),
        half1_max_posterior=np.asarray([0.2]),
        half2_original_image_indices=np.asarray([], dtype=np.int64),
        half2_max_posterior=np.asarray([], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="cover every input source row"):
        build_panel(parity_dump=parity, input_star=input_star, relion_star=relion_star, top_n=2)
