"""Pin the RELION volume convention helpers so they cannot be silently removed.

Background: in commit 7df73fa (2026-04-01) the helpers
``relion_volume_to_recovar`` and ``recovar_volume_to_relion`` were added
to ``recovar/utils/helpers.py`` along with documentation in
``recovar/em/CLAUDE.md``.

In commit 4703c634 (one hour later) the helpers were silently removed
"to revert helpers.py to clean origin/dev state", but the documentation
in ``recovar/em/CLAUDE.md`` was left behind.

The result was a year of intermittent confusion: every maintainer who
tried to follow the EM CLAUDE.md got an ``ImportError`` and had to
rediscover the convention by hand. This wasted ~25 commits in the
RELION-parity work (see ``docs/relion_parity_commit_audit.md``).

These tests pin the helpers + their behavior so any future "revert
helpers.py" PR breaks loudly instead of silently.
"""
import numpy as np
import pytest


def test_helpers_exist():
    """The canonical RELION-MRC helpers must exist in recovar.utils.helpers."""
    from recovar.utils.helpers import (
        relion_volume_to_recovar,
        recovar_volume_to_relion,
        load_relion_volume,
        write_relion_mrc,
        load_mrc,
        write_mrc,
    )
    # If any of these is missing this import will fail. The test name in
    # the failure message will tell future-you exactly what was removed.
    assert callable(relion_volume_to_recovar)
    assert callable(recovar_volume_to_relion)
    assert callable(load_relion_volume)
    assert callable(write_relion_mrc)
    assert callable(load_mrc)
    assert callable(write_mrc)


def test_relion_to_recovar_is_negate_and_transpose():
    """The conversion is exactly ``-np.transpose(vol, (2, 1, 0))``.

    See ``recovar/em/CLAUDE.md`` and issue #86. The convention difference
    was empirically verified to give CC=0.998 vs the recovar
    reconstruction on EMPIAR challenge1.
    """
    from recovar.utils.helpers import relion_volume_to_recovar

    rng = np.random.default_rng(0)
    vol = rng.standard_normal((8, 8, 8)).astype(np.float32)

    expected = -np.transpose(vol, (2, 1, 0))
    got = relion_volume_to_recovar(vol)

    assert got.shape == vol.shape
    np.testing.assert_array_equal(got, expected)


def test_recovar_to_relion_is_inverse():
    """``recovar_volume_to_relion`` is the inverse of
    ``relion_volume_to_recovar``.

    Both operations are the same function (negate + transpose(2,1,0))
    because that operation is an involution. The two names exist for
    readability at call sites.
    """
    from recovar.utils.helpers import (
        relion_volume_to_recovar,
        recovar_volume_to_relion,
    )

    rng = np.random.default_rng(1)
    vol = rng.standard_normal((8, 8, 8)).astype(np.float32)

    round_tripped = recovar_volume_to_relion(relion_volume_to_recovar(vol))
    np.testing.assert_array_equal(round_tripped, vol)

    round_tripped_other = relion_volume_to_recovar(recovar_volume_to_relion(vol))
    np.testing.assert_array_equal(round_tripped_other, vol)


def test_recovar_load_mrc_round_trip(tmp_path):
    """``load_mrc`` and ``write_mrc`` round-trip cleanly.

    This pins the cryosparc/cryoDRGN axis-flip pair so that no future
    refactor can desync them silently.
    """
    from recovar.utils.helpers import load_mrc, write_mrc

    rng = np.random.default_rng(2)
    vol = rng.standard_normal((8, 8, 8)).astype(np.float32)

    path = tmp_path / "round_trip.mrc"
    write_mrc(str(path), vol, voxel_size=1.0)

    loaded = load_mrc(str(path))
    np.testing.assert_array_equal(loaded, vol)


def test_load_relion_volume_round_trip(tmp_path):
    """``load_relion_volume`` correctly inverts a RELION-frame MRC write.

    Simulates the situation we care about: a file that was written in
    RELION's frame (e.g. by RELION itself) needs to be loaded in
    recovar's frame for direct comparison.
    """
    import mrcfile
    from recovar.utils.helpers import (
        load_relion_volume,
        recovar_volume_to_relion,
    )

    rng = np.random.default_rng(3)
    vol_recovar = rng.standard_normal((8, 8, 8)).astype(np.float32)

    # Convert into RELION's frame and write with raw mrcfile (the way
    # RELION would write it).
    vol_relion = recovar_volume_to_relion(vol_recovar)
    path = tmp_path / "relion_style.mrc"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol_relion.astype(np.float32))

    loaded = load_relion_volume(str(path))
    np.testing.assert_allclose(loaded, vol_recovar, atol=1e-5)


def test_write_relion_mrc_round_trips_with_load_relion_volume(tmp_path):
    """``write_relion_mrc`` and ``load_relion_volume`` round-trip cleanly.

    This pins the helper that writes a recovar-frame volume into a
    RELION-readable MRC file. Without this helper, ``prepare_relion_parity_benchmark.py``
    used the wrong write path (``save_volume`` -> ``write_mrc``), which
    produced a cryosparc-frame MRC. Feeding that file to RELION as
    ``--ref reference_init.mrc`` made RELION refine into the antipode
    basin (median pose error ~133°), invalidating the entire RELION-parity
    comparison.
    """
    from recovar.utils.helpers import (
        write_relion_mrc,
        load_relion_volume,
    )

    rng = np.random.default_rng(7)
    vol_recovar = rng.standard_normal((8, 8, 8)).astype(np.float32)

    path = tmp_path / "for_relion.mrc"
    write_relion_mrc(str(path), vol_recovar, voxel_size=1.0)

    loaded = load_relion_volume(str(path))
    np.testing.assert_allclose(loaded, vol_recovar, atol=1e-5)


def test_write_relion_mrc_disk_bytes_match_recovar_to_relion(tmp_path):
    """The on-disk bytes of a write_relion_mrc file must equal recovar_volume_to_relion(vol).

    Belt-and-suspenders test: even if some future refactor reshuffles the
    helpers, the disk-side invariant guarantees RELION reads the volume
    in its expected frame. Reading the file back with raw mrcfile and
    comparing against the explicit recovar->relion conversion catches any
    drift in either direction.
    """
    import mrcfile
    from recovar.utils.helpers import (
        write_relion_mrc,
        recovar_volume_to_relion,
    )

    rng = np.random.default_rng(11)
    vol_recovar = rng.standard_normal((8, 8, 8)).astype(np.float32)

    path = tmp_path / "raw_disk.mrc"
    write_relion_mrc(str(path), vol_recovar, voxel_size=2.5)

    with mrcfile.open(str(path)) as m:
        raw_disk = np.array(m.data, dtype=np.float32)

    expected_disk = recovar_volume_to_relion(vol_recovar)
    np.testing.assert_allclose(raw_disk, expected_disk, atol=1e-5)


def test_em_claude_md_helper_reference_is_valid():
    """The helper that ``recovar/em/CLAUDE.md`` references must exist.

    This is a meta-test: the EM developer guide tells maintainers to
    use ``relion_volume_to_recovar``. If that helper ever gets removed
    while the docs still reference it, this test fails so the
    documentation drift is caught at CI time, not after a week of
    debugging the wrong thing.
    """
    import importlib
    helpers = importlib.import_module("recovar.utils.helpers")
    assert hasattr(helpers, "relion_volume_to_recovar"), (
        "recovar/em/CLAUDE.md references relion_volume_to_recovar but the "
        "function is missing from recovar/utils/helpers.py. This was the "
        "exact bug that wasted a week of RELION-parity work in 2026-04. "
        "DO NOT remove the helper without first updating recovar/em/CLAUDE.md "
        "and recovar/CLAUDE.md to point to the replacement."
    )
