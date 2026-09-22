"""A mid-trajectory replay must enter with RELION's saved sampling order.

RELION computes pass-1 ``image_coarse_size`` before ``updateAngularSampling``
switches order (ml_optimiser_mpi.cpp expectation steps A then D), so the first
iteration after an order switch sizes its coarse images from the previous order.
A replay that starts after RELION iteration N takes that order from
``--healpix_order``; on EMPIAR-10097 it17 a mismatched order (2 instead of 3)
sized pass-1 images at 28 instead of 56 pixels.
"""

import pytest

from scripts import run_full_refinement

pytestmark = pytest.mark.unit


def _write_sampling(tmp_path, iteration, healpix_order):
    path = tmp_path / f"run_it{iteration:03d}_sampling.star"
    path.write_text(
        "\n".join(
            [
                "data_sampling_general",
                "",
                f"_rlnHealpixOrder {healpix_order}",
                "_rlnPsiStep 7.5",
                "_rlnOffsetRange 6.55",
                "_rlnOffsetStep 2.62",
                "_rlnSamplingPerturbInstance 0.1",
                "_rlnSamplingPerturbFactor 0.5",
                "",
            ]
        )
    )


def test_replay_rejects_healpix_order_that_differs_from_saved_sampling(tmp_path):
    _write_sampling(tmp_path, 16, 3)

    with pytest.raises(ValueError, match="does not match RELION's saved sampling order 3"):
        run_full_refinement._validate_replay_saved_healpix_order(tmp_path, 16, 2)


def test_replay_accepts_saved_sampling_healpix_order(tmp_path):
    _write_sampling(tmp_path, 16, 3)

    run_full_refinement._validate_replay_saved_healpix_order(tmp_path, 16, 3)


@pytest.mark.parametrize(("replay_dir", "init_relion_iteration"), [(None, 16), ("unused", 0)])
def test_cold_start_and_autonomous_runs_skip_saved_order_check(replay_dir, init_relion_iteration):
    run_full_refinement._validate_replay_saved_healpix_order(replay_dir, init_relion_iteration, 2)


def test_cli_validates_saved_order_before_refinement():
    import inspect

    source = inspect.getsource(run_full_refinement.main)
    start = source.index("_validate_replay_saved_healpix_order(")
    assert start < source.index("frozen_boundary = None")
    assert "args.healpix_order" in source[start : start + 240]
