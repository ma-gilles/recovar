import json
from pathlib import Path

import pytest


PANEL_PATH = (
    Path(__file__).parents[2]
    / "docs"
    / "math"
    / "em_k1_case10_it2_capture_panel_v1.json"
)


def _load_rows():
    return json.loads(PANEL_PATH.read_text())["rows"]


@pytest.mark.unit
def test_case10_it2_capture_panel_freezes_top_24_pmax_errors() -> None:
    rows = _load_rows()

    assert len(rows) == 24
    assert [row["rank_by_absolute_pmax_error"] for row in rows] == list(range(1, 25))
    assert len({row["original_index_zero_based"] for row in rows}) == 24
    assert [row["absolute_pmax_error"] for row in rows] == sorted(
        (row["absolute_pmax_error"] for row in rows),
        reverse=True,
    )
    for row in rows:
        assert row["absolute_pmax_error"] == abs(
            row["recovar_pmax"] - row["relion_pmax"]
        )


@pytest.mark.unit
def test_case10_it2_capture_panel_has_unambiguous_cross_engine_identities() -> None:
    rows = _load_rows()

    assert len({row["stack_index_one_based"] for row in rows}) == 24
    assert len({row["relion_part_id"] for row in rows}) == 24
    assert {row["random_subset"] for row in rows} == {1, 2}
    for row in rows:
        assert row["stack_index_one_based"] == row["original_index_zero_based"] + 1
