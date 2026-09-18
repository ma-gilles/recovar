import pytest

from scripts import run_vdam_schedule_capacity_sweep as sweep


def test_capacity_sweep_rejects_duplicate_capacities() -> None:
    with pytest.raises(SystemExit):
        sweep._parse_args(
            [
                "--input-star",
                "input.star",
                "--checkpoint-optimiser",
                "checkpoint.star",
                "--data-dir",
                "data",
                "--output-root",
                "output",
                "--checkpoint-iteration",
                "25",
                "--capacity",
                "128",
                "--capacity",
                "128",
            ]
        )


def test_capacity_sweep_preserves_requested_order() -> None:
    args = sweep._parse_args(
        [
            "--input-star",
            "input.star",
            "--checkpoint-optimiser",
            "checkpoint.star",
            "--data-dir",
            "data",
            "--output-root",
            "output",
            "--checkpoint-iteration",
            "25",
            "--capacity",
            "96",
            "--capacity",
            "128",
            "--capacity",
            "192",
        ]
    )

    assert args.capacity == [96, 128, 192]
