from pathlib import Path

import pytest

from scripts.materialize_vdam_relion_continuation import materialize


@pytest.mark.unit
def test_materialize_replaces_only_sampling_path(tmp_path: Path) -> None:
    source = tmp_path / "source_optimiser.star"
    sampling = tmp_path / "sampling.star"
    output = tmp_path / "continuation" / "optimiser.star"
    source.write_text(
        "# header\n"
        "_rlnModelStarFile /sealed/model.star\n"
        "_rlnOrientSamplingStarFile /cleaned/run_it000_sampling.star\n"
        "_rlnCurrentIteration 0\n"
    )
    sampling.write_text("data_sampling_general\n")

    materialize(source, sampling, output)

    materialized = output.read_text()
    assert "_rlnModelStarFile /sealed/model.star" in materialized
    assert "_rlnOrientSamplingStarFile" in materialized
    assert str(sampling.resolve()) in materialized
    assert "cleaned/run_it000_sampling.star" not in materialized
    assert "_rlnCurrentIteration 0" in materialized


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampling_rows",
    [[], ["_rlnOrientSamplingStarFile first.star", "_rlnOrientSamplingStarFile second.star"]],
)
def test_materialize_rejects_missing_or_ambiguous_sampling_row(
    tmp_path: Path, sampling_rows: list[str]
) -> None:
    source = tmp_path / "source_optimiser.star"
    sampling = tmp_path / "sampling.star"
    source.write_text("\n".join(["data_optimiser_general", *sampling_rows]) + "\n")
    sampling.write_text("data_sampling_general\n")

    with pytest.raises(ValueError, match="exactly one sampling-file row"):
        materialize(source, sampling, tmp_path / "output.star")
