"""Regression test for parse_relion5_tomo using real RELION5 data subset.

Tests that convert() produces identical output on a small fixture extracted
from a real ribosome tilt-series dataset (Ryan Feathers, Princeton).

Fixture: tests/fixtures/relion5_tomo_subset/
  - tomograms.star (2 tilt series)
  - particles.star (10 particles, 5 per tilt series)
  - Polish/job185/tilt_series/*.star (per-tilt geometry)
  - reference_output.star (expected conversion output)
  - reference_values.npy (defocus + Euler angles for regression check)
"""

import os

import numpy as np
import pytest

from recovar.commands.parse_relion5_tomo import convert
from recovar.data_io.starfile import read_star

pytestmark = pytest.mark.unit

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'fixtures', 'relion5_tomo_subset'
)


@pytest.fixture
def fixture_paths():
    return {
        'tomograms': os.path.join(FIXTURE_DIR, 'tomograms.star'),
        'particles': os.path.join(FIXTURE_DIR, 'particles.star'),
        'reference_output': os.path.join(FIXTURE_DIR, 'reference_output.star'),
        'reference_values': os.path.join(FIXTURE_DIR, 'reference_values.npy'),
    }


def test_fixture_exists(fixture_paths):
    """Sanity check that fixture files exist."""
    for name, path in fixture_paths.items():
        assert os.path.isfile(path), f"Missing fixture: {name} at {path}"


def test_conversion_row_count(fixture_paths, tmp_path):
    """Conversion should produce the same number of rows as reference."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    ref_df, _ = read_star(fixture_paths['reference_output'])
    assert len(out_df) == len(ref_df), (
        f"Row count mismatch: got {len(out_df)}, expected {len(ref_df)}"
    )


def test_conversion_columns_match(fixture_paths, tmp_path):
    """Output should have the same columns as reference."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    ref_df, _ = read_star(fixture_paths['reference_output'])
    assert set(out_df.columns) == set(ref_df.columns)


def test_conversion_string_columns_exact(fixture_paths, tmp_path):
    """GroupName, ImageName, MicrographName should match exactly."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    ref_df, _ = read_star(fixture_paths['reference_output'])

    # Sort both by GroupName + dose for fair comparison
    for df in (out_df, ref_df):
        df['_rlnMicrographPreExposure'] = df['_rlnMicrographPreExposure'].astype(float)
    out_df = out_df.sort_values(
        ['_rlnGroupName', '_rlnMicrographPreExposure']
    ).reset_index(drop=True)
    ref_df = ref_df.sort_values(
        ['_rlnGroupName', '_rlnMicrographPreExposure']
    ).reset_index(drop=True)

    for col in ['_rlnGroupName', '_rlnImageName', '_rlnMicrographName']:
        assert (out_df[col].values == ref_df[col].values).all(), (
            f"Column {col} mismatch"
        )


def test_conversion_numerical_regression(fixture_paths, tmp_path):
    """Defocus and Euler angles should match reference to machine precision."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    out_df['_rlnMicrographPreExposure'] = out_df['_rlnMicrographPreExposure'].astype(float)
    out_df = out_df.sort_values(
        ['_rlnGroupName', '_rlnMicrographPreExposure']
    ).reset_index(drop=True)

    ref_values = np.load(fixture_paths['reference_values'])

    key_cols = ['_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle',
                '_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']
    out_values = out_df[key_cols].values.astype(float)

    np.testing.assert_allclose(
        out_values, ref_values,
        atol=1e-10, rtol=1e-12,
        err_msg="Numerical regression: defocus/Euler values differ from reference",
    )


def test_conversion_dose_sorted(fixture_paths, tmp_path):
    """Within each group, rows should be sorted by dose."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    for group, sub in out_df.groupby('_rlnGroupName'):
        doses = sub['_rlnMicrographPreExposure'].values.astype(float)
        assert (np.diff(doses) >= 0).all(), (
            f"Group {group} not sorted by dose"
        )


def test_conversion_random_subset_preserved(fixture_paths, tmp_path):
    """Each group should have a consistent _rlnRandomSubset."""
    output = str(tmp_path / 'output.star')
    convert(fixture_paths['tomograms'], fixture_paths['particles'], output)

    out_df, _ = read_star(output)
    for group, sub in out_df.groupby('_rlnGroupName'):
        subsets = sub['_rlnRandomSubset'].values.astype(int)
        assert len(set(subsets)) == 1, (
            f"Group {group} has mixed random subsets: {set(subsets)}"
        )
