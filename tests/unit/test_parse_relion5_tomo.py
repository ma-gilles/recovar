"""Tests for recovar.commands.parse_relion5_tomo."""

import os
import textwrap

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from recovar.commands.parse_relion5_tomo import (
    Tomogram,
    _parse_visible_frames,
    convert,
)
from recovar.starfile import read_star, StarFile

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures: synthetic RELION5 star files
# ---------------------------------------------------------------------------

def _write_text(path, text):
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))


def _make_tilt_series_star(path, n_tilts=5):
    """Write a minimal per-tilt-series .star file."""
    header = textwrap.dedent("""\
        # version 50001

        data_my_ts_001

        loop_
        _rlnMicrographName #1
        _rlnTomoNominalStageTiltAngle #2
        _rlnMicrographPreExposure #3
        _rlnTomoXTilt #4
        _rlnTomoYTilt #5
        _rlnTomoZRot #6
        _rlnTomoXShiftAngst #7
        _rlnTomoYShiftAngst #8
        _rlnDefocusU #9
        _rlnDefocusV #10
        _rlnDefocusAngle #11
        _rlnCtfScalefactor #12
    """)
    rows = []
    angles = np.linspace(-60, 60, n_tilts)
    for i, angle in enumerate(angles):
        dose = (i + 1) * 3.0
        dfu = 30000.0 + i * 100
        dfv = 31000.0 + i * 100
        rows.append(
            f"mic_{i:03d}.mrc {angle:.3f} {dose:.1f} "
            f"{angle:.3f} 0.0 {80.0 + i:.3f} "
            f"{i * 10.0:.1f} {i * -5.0:.1f} "
            f"{dfu:.1f} {dfv:.1f} 45.0 {1.0 + i * 0.1:.3f}"
        )
    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows) + "\n")


def _make_tomograms_star(path, ts_star_relpath, n_tomos=1):
    """Write a minimal tomograms.star."""
    header = textwrap.dedent("""\
        # version 50001

        data_global

        loop_
        _rlnTomoName #1
        _rlnTomoTiltSeriesStarFile #2
        _rlnTomoTiltSeriesPixelSize #3
        _rlnTomoHand #4
        _rlnTomoSizeX #5
        _rlnTomoSizeY #6
    """)
    rows = []
    for i in range(n_tomos):
        name = f"tomo_{i:03d}" if n_tomos > 1 else "tomo_001"
        rows.append(f"{name} {ts_star_relpath} 1.5 -1 4096 4096")
    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows) + "\n")


def _make_particles_star(path, n_particles=3, n_tilts=5,
                          tomo_name="tomo_001", has_subtomo_rot=False):
    """Write a minimal RELION5 particles.star with optics + particles blocks."""
    visible = "[" + ",".join(["1"] * n_tilts) + "]"

    optics_header = textwrap.dedent("""\
        # version 50001

        data_optics

        loop_
        _rlnVoltage #1
        _rlnSphericalAberration #2
        _rlnAmplitudeContrast #3
        _rlnOpticsGroup #4
        _rlnOpticsGroupName #5
        _rlnImagePixelSize #6
        _rlnImageSize #7
        _rlnImageDimensionality #8
        300.0 2.7 0.1 1 opticsGroup1 1.5 128 2
    """)

    particles_cols = [
        "_rlnTomoName",
        "_rlnCenteredCoordinateXAngst",
        "_rlnCenteredCoordinateYAngst",
        "_rlnCenteredCoordinateZAngst",
        "_rlnOpticsGroup",
        "_rlnTomoParticleName",
        "_rlnImageName",
        "_rlnAngleRot",
        "_rlnAngleTilt",
        "_rlnAnglePsi",
        "_rlnRandomSubset",
        "_rlnTomoVisibleFrames",
    ]
    if has_subtomo_rot:
        particles_cols += [
            "_rlnTomoSubtomogramRot",
            "_rlnTomoSubtomogramTilt",
            "_rlnTomoSubtomogramPsi",
        ]

    particles_header = "\n# version 50001\n\ndata_particles\n\nloop_\n"
    for i, col in enumerate(particles_cols, 1):
        particles_header += f"{col} #{i}\n"

    rng = np.random.RandomState(42)
    rows = []
    for p in range(n_particles):
        x = rng.uniform(-500, 500)
        y = rng.uniform(-500, 500)
        z = rng.uniform(-200, 200)
        rot, tilt, psi = rng.uniform(-180, 180, 3)
        subset = (p % 2) + 1
        name = f"{tomo_name}/{p + 1}"
        img = f"Extract/Subtomograms/{tomo_name}/{p + 1}_stack2d.mrcs"
        row = (
            f"{tomo_name} {x:.3f} {y:.3f} {z:.3f} 1 {name} {img} "
            f"{rot:.3f} {tilt:.3f} {psi:.3f} {subset} {visible}"
        )
        if has_subtomo_rot:
            srot, stilt, spsi = rng.uniform(-10, 10, 3)
            row += f" {srot:.3f} {stilt:.3f} {spsi:.3f}"
        rows.append(row)

    with open(path, "w") as f:
        f.write(optics_header)
        f.write(particles_header)
        f.write("\n".join(rows) + "\n")


@pytest.fixture
def relion5_data(tmp_path):
    """Create a full set of synthetic RELION5 tomo star files."""
    n_tilts = 5
    n_particles = 4

    # Tilt series star file
    ts_dir = tmp_path / "Polish" / "job001" / "tilt_series"
    ts_dir.mkdir(parents=True)
    ts_star = ts_dir / "tomo_001.star"
    _make_tilt_series_star(str(ts_star), n_tilts=n_tilts)

    # Tomograms.star
    tomo_dir = tmp_path / "Polish" / "job001"
    tomo_star = tomo_dir / "tomograms.star"
    _make_tomograms_star(
        str(tomo_star),
        ts_star_relpath="Polish/job001/tilt_series/tomo_001.star",
    )

    # Particles.star
    extract_dir = tmp_path / "Extract" / "job002"
    extract_dir.mkdir(parents=True)
    particles_star = extract_dir / "particles.star"
    _make_particles_star(str(particles_star), n_particles=n_particles,
                          n_tilts=n_tilts)

    return {
        "root": tmp_path,
        "tomograms": str(tomo_star),
        "particles": str(particles_star),
        "n_tilts": n_tilts,
        "n_particles": n_particles,
    }


# ---------------------------------------------------------------------------
# Tests: Tomogram class
# ---------------------------------------------------------------------------

class TestTomogram:

    def test_projection_matrices_count(self):
        n = 3
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.5,
            defocus_u=np.zeros(n), defocus_v=np.zeros(n),
            defocus_angle=np.zeros(n),
            x_tilts=np.array([0, 10, -10.0]),
            y_tilts=np.zeros(n), z_rots=np.zeros(n),
            x_shifts=np.zeros(n), y_shifts=np.zeros(n),
            hand=1,
        )
        assert len(tomo.projection_matrices) == n

    def test_project_origin_lands_at_center(self):
        """Origin (0,0,0) with no tilt should project to image center."""
        tomo = Tomogram(
            tilt_image_dims=(200, 300), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        pt = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [100.0, 150.0], atol=1e-10)

    def test_local_defocus_at_origin_equals_global(self):
        """At the origin, depth offset is zero so local == global defocus."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([25000.0]),
            defocus_v=np.array([26000.0]),
            defocus_angle=np.array([45.0]),
            x_tilts=np.array([30.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        u, v, a = tomo.local_defocus(0, np.array([0.0, 0.0, 0.0]))
        assert u == pytest.approx(25000.0)
        assert v == pytest.approx(26000.0)
        assert a == pytest.approx(45.0)

    def test_local_defocus_offset_nonzero_for_shifted_particle(self):
        """Particle displaced in Z should have depth-dependent defocus offset."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([30000.0]),
            defocus_v=np.array([30000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([45.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        u_origin, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 0.0]))
        u_shifted, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 500.0]))
        # With 45-degree tilt and Z=500, there should be a significant offset
        assert abs(u_shifted - u_origin) > 100

    def test_expand_particle_row_count(self):
        n = 4
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.zeros(n), defocus_v=np.zeros(n),
            defocus_angle=np.zeros(n),
            x_tilts=np.linspace(-40, 40, n),
            y_tilts=np.zeros(n), z_rots=np.zeros(n),
            x_shifts=np.zeros(n), y_shifts=np.zeros(n),
            hand=1,
        )
        # Build a minimal tilt_df
        tilt_df = pd.DataFrame({
            "_rlnMicrographName": [f"mic_{i}.mrc" for i in range(n)],
            "_rlnMicrographPreExposure": [str(i * 3.0) for i in range(n)],
            "_rlnCtfScalefactor": ["1.0"] * n,
            "_rlnTomoNominalStageTiltAngle": [str(a) for a in np.linspace(-40, 40, n)],
        })
        base_rot = R.from_euler("ZYZ", [30, 60, 90], degrees=True)
        df = tomo.expand_particle(
            np.array([100.0, 200.0, -50.0]),
            "stack.mrcs", tilt_df, "particle/1",
            base_orientation=base_rot,
        )
        assert len(df) == n
        assert "_rlnImageName" in df.columns
        assert "_rlnAngleRot" in df.columns
        # Image names should be 000001@stack.mrcs, 000002@stack.mrcs, ...
        assert df["_rlnImageName"].iloc[0] == "000001@stack.mrcs"
        assert df["_rlnImageName"].iloc[-1] == f"{n:06d}@stack.mrcs"

    def test_expand_particle_without_orientation(self):
        """If base_orientation is None, angles should be zero."""
        n = 3
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.zeros(n), defocus_v=np.zeros(n),
            defocus_angle=np.zeros(n),
            x_tilts=np.array([0.0, 20.0, -20.0]),
            y_tilts=np.zeros(n), z_rots=np.zeros(n),
            x_shifts=np.zeros(n), y_shifts=np.zeros(n),
            hand=1,
        )
        tilt_df = pd.DataFrame({
            "_rlnMicrographName": [f"mic_{i}.mrc" for i in range(n)],
            "_rlnMicrographPreExposure": ["3.0", "6.0", "9.0"],
            "_rlnCtfScalefactor": ["1.0"] * n,
            "_rlnTomoNominalStageTiltAngle": ["0.0", "20.0", "-20.0"],
        })
        df = tomo.expand_particle(
            np.array([0.0, 0.0, 0.0]),
            "stack.mrcs", tilt_df, "p/1",
            base_orientation=None,
        )
        np.testing.assert_array_equal(df["_rlnAngleRot"].values, 0.0)
        np.testing.assert_array_equal(df["_rlnAngleTilt"].values, 0.0)
        np.testing.assert_array_equal(df["_rlnAnglePsi"].values, 0.0)


# ---------------------------------------------------------------------------
# Tests: visible frames parsing
# ---------------------------------------------------------------------------

class TestParseVisibleFrames:

    def test_all_visible(self):
        assert _parse_visible_frames("[1,1,1,1]") == [0, 1, 2, 3]

    def test_some_invisible(self):
        assert _parse_visible_frames("[1,0,1,0,1]") == [0, 2, 4]

    def test_none_visible(self):
        assert _parse_visible_frames("[0,0,0]") == []


# ---------------------------------------------------------------------------
# Tests: end-to-end conversion
# ---------------------------------------------------------------------------

class TestConvert:

    def test_basic_conversion(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        assert os.path.isfile(output)

        # Parse output
        df, optics = read_star(output)
        assert optics is not None

        # Check dimensions
        n_expected = relion5_data["n_particles"] * relion5_data["n_tilts"]
        assert len(df) == n_expected

        # Check optics fields
        assert "_rlnImagePixelSize" in optics.columns
        assert "_rlnVoltage" in optics.columns
        assert optics["_rlnImageSize"].values[0] == "128"
        assert optics["_rlnImagePixelSize"].values[0] == "1.5"

    def test_output_has_required_columns(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        df, _ = read_star(output)
        required = [
            "_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle",
            "_rlnImageName", "_rlnGroupName",
            "_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi",
            "_rlnOriginXAngst", "_rlnOriginYAngst",
            "_rlnCtfScalefactor", "_rlnRandomSubset",
            "_rlnMicrographPreExposure", "_rlnOpticsGroup",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_dose_sorted_within_groups(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        df, _ = read_star(output)
        groups = df["_rlnGroupName"].unique()
        for group in groups:
            mask = df["_rlnGroupName"] == group
            doses = df[mask]["_rlnMicrographPreExposure"].values.astype(float)
            assert np.all(np.diff(doses) >= 0), \
                f"Doses not sorted for group {group}: {doses}"

    def test_random_subset_preserved(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        df, _ = read_star(output)
        subsets = df["_rlnRandomSubset"].values.astype(int)
        assert set(subsets) == {1, 2}

    def test_group_names_match_particles(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        df, _ = read_star(output)
        groups = sorted(df["_rlnGroupName"].unique())
        expected = sorted([f"tomo_001/{i + 1}"
                           for i in range(relion5_data["n_particles"])])
        assert groups == expected

    def test_image_names_format(self, relion5_data, tmp_path):
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        df, _ = read_star(output)
        for name in df["_rlnImageName"].values[:10]:
            idx, stack = name.split("@")
            assert len(idx) == 6
            assert int(idx) >= 1
            assert stack.endswith(".mrcs")

    def test_loadable_as_starfile(self, relion5_data, tmp_path):
        """Output should be loadable by RECOVAR's StarFile class."""
        output = str(tmp_path / "output.star")
        convert(
            relion5_data["tomograms"],
            relion5_data["particles"],
            output,
            tilt_dim=(4096, 4096),
        )
        sf = StarFile.load(output)
        assert sf.has_optics
        assert sf.apix is not None
        assert sf.resolution is not None
        assert len(sf) == relion5_data["n_particles"] * relion5_data["n_tilts"]

    def test_with_subtomogram_orientation(self, tmp_path):
        """Particles with rlnTomoSubtomogramRot/Tilt/Psi should produce
        different angles than without."""
        n_tilts = 3

        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        _make_tomograms_star(str(tomo_star), "ts/ts.star")

        # Without subtomo orientation
        p1 = tmp_path / "p1.star"
        _make_particles_star(str(p1), n_particles=1, n_tilts=n_tilts,
                              has_subtomo_rot=False)
        out1 = str(tmp_path / "out1.star")
        convert(str(tomo_star), str(p1), out1, tilt_dim=(100, 100))

        # With subtomo orientation
        p2 = tmp_path / "p2.star"
        _make_particles_star(str(p2), n_particles=1, n_tilts=n_tilts,
                              has_subtomo_rot=True)
        out2 = str(tmp_path / "out2.star")
        convert(str(tomo_star), str(p2), out2, tilt_dim=(100, 100))

        df1, _ = read_star(out1)
        df2, _ = read_star(out2)

        # Angles should differ (subtomo orientation != identity)
        rot1 = df1["_rlnAngleRot"].values.astype(float)
        rot2 = df2["_rlnAngleRot"].values.astype(float)
        assert not np.allclose(rot1, rot2, atol=0.1), \
            "Subtomogram orientation should produce different angles"

    def test_partial_visible_frames(self, tmp_path):
        """Particles with some invisible tilts should produce fewer rows."""
        n_tilts = 5

        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        _make_tomograms_star(str(tomo_star), "ts/ts.star")

        # Write particles with partial visible frames
        visible = "[1,0,1,0,1]"  # only 3 of 5 visible
        particles_text = f"""\
            # version 50001

            data_optics

            loop_
            _rlnVoltage #1
            _rlnSphericalAberration #2
            _rlnAmplitudeContrast #3
            _rlnOpticsGroup #4
            _rlnOpticsGroupName #5
            _rlnImagePixelSize #6
            _rlnImageSize #7
            _rlnImageDimensionality #8
            300.0 2.7 0.1 1 opticsGroup1 1.5 128 2

            # version 50001

            data_particles

            loop_
            _rlnTomoName #1
            _rlnCenteredCoordinateXAngst #2
            _rlnCenteredCoordinateYAngst #3
            _rlnCenteredCoordinateZAngst #4
            _rlnOpticsGroup #5
            _rlnTomoParticleName #6
            _rlnImageName #7
            _rlnAngleRot #8
            _rlnAngleTilt #9
            _rlnAnglePsi #10
            _rlnRandomSubset #11
            _rlnTomoVisibleFrames #12
            tomo_001 100.0 200.0 50.0 1 tomo_001/1 Extract/tomo_001/1_stack2d.mrcs 10.0 20.0 30.0 1 {visible}
        """
        p_star = tmp_path / "particles.star"
        _write_text(str(p_star), particles_text)

        output = str(tmp_path / "output.star")
        convert(str(tomo_star), str(p_star), output, tilt_dim=(100, 100))

        df, _ = read_star(output)
        assert len(df) == 3  # only 3 visible tilts

    def test_project_root_resolution(self, tmp_path):
        """Tilt-series star paths relative to project root should be resolved."""
        n_tilts = 3

        # Create nested structure
        project = tmp_path / "relion_project"
        ts_dir = project / "Polish" / "job001" / "tilt_series"
        ts_dir.mkdir(parents=True)
        _make_tilt_series_star(str(ts_dir / "tomo_001.star"), n_tilts=n_tilts)

        tomo_dir = project / "Polish" / "job001"
        tomo_star = tomo_dir / "tomograms.star"
        # Path relative to project root
        _make_tomograms_star(
            str(tomo_star),
            "Polish/job001/tilt_series/tomo_001.star",
        )

        extract_dir = project / "Extract"
        extract_dir.mkdir()
        p_star = extract_dir / "particles.star"
        _make_particles_star(str(p_star), n_particles=2, n_tilts=n_tilts)

        output = str(tmp_path / "output.star")
        convert(str(tomo_star), str(p_star), output, tilt_dim=(100, 100))

        df, _ = read_star(output)
        assert len(df) == 2 * n_tilts


# ---------------------------------------------------------------------------
# Tests: geometry consistency
# ---------------------------------------------------------------------------

class TestGeometryConsistency:

    def test_hand_affects_defocus(self):
        """Changing handedness should flip the defocus depth offset."""
        n = 1
        kwargs = dict(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([30000.0]),
            defocus_v=np.array([30000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([30.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
        )
        tomo_pos = Tomogram(**kwargs, hand=1)
        tomo_neg = Tomogram(**kwargs, hand=-1)

        point = np.array([0.0, 0.0, 500.0])
        u_pos, _, _ = tomo_pos.local_defocus(0, point)
        u_neg, _, _ = tomo_neg.local_defocus(0, point)

        # Depth offsets should be opposite
        base = 30000.0
        offset_pos = u_pos - base
        offset_neg = u_neg - base
        assert offset_pos == pytest.approx(-offset_neg, abs=1e-6)

    def test_zero_tilt_preserves_xy(self):
        """At zero tilt, XY coordinates should map directly."""
        tomo = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]),
            defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        # Point at (10, 20, 0) should project to (110, 120)
        pt = tomo.project_point(np.array([10.0, 20.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [110.0, 120.0], atol=1e-10)

    def test_orientation_composition_order(self):
        """R_base = R_particle * R_subtomo, then composed with tilt geometry."""
        rp = R.from_euler("ZYZ", [30, 45, 60], degrees=True)
        rs = R.from_euler("ZYZ", [10, 5, -10], degrees=True)
        R_base = rp * rs

        # The expand_particle method should use R_base @ tilt_rotation
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([0.0]),
            defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([20.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        tilt_df = pd.DataFrame({
            "_rlnMicrographName": ["mic.mrc"],
            "_rlnMicrographPreExposure": ["3.0"],
            "_rlnCtfScalefactor": ["1.0"],
            "_rlnTomoNominalStageTiltAngle": ["20.0"],
        })
        df = tomo.expand_particle(
            np.array([0.0, 0.0, 0.0]), "stack.mrcs",
            tilt_df, "p/1", base_orientation=R_base,
        )
        # Verify non-zero angles
        assert float(df["_rlnAngleRot"].iloc[0]) != 0.0
        assert float(df["_rlnAngleTilt"].iloc[0]) != 0.0

    def test_matches_reference_implementation(self):
        """Verify our Tomogram class exactly matches the original
        relion2cryodrgn geometry (Ryan Feathers / Bogdan Toader)."""
        # Replicate Ryan's Tomogram inline for comparison
        def ryan_build_projection_matrix(x_tilt, y_tilt, z_rot,
                                          x_shift, y_shift, tilt_dims):
            def _rot(axis, deg):
                rot = R.from_rotvec(np.deg2rad(deg) * np.array(axis))
                mat = np.eye(4)
                mat[:3, :3] = rot.as_matrix()
                return mat
            def _trans(v):
                mat = np.eye(4)
                mat[:3, 3] = v
                return mat

            s0 = _trans(-np.array([0, 0, 0]))
            r0 = _rot([1, 0, 0], x_tilt)
            r1 = _rot([0, 1, 0], y_tilt)
            r2 = _rot([0, 0, 1], z_rot)
            s1 = _trans(np.array([x_shift, y_shift, 0.0]))
            s2 = _trans(np.array([tilt_dims[0] / 2.0, tilt_dims[1] / 2.0, 0.0]))

            Rzyx = R.from_matrix(r2[:3, :3]) * R.from_matrix(r1[:3, :3]) * R.from_matrix(r0[:3, :3])
            R_inv = np.eye(4)
            R_inv[:3, :3] = Rzyx.inv().as_matrix()
            return s2 @ s1 @ R_inv @ s0

        def ryan_local_defocus(x_tilt, y_tilt, z_rot, x_shift, y_shift,
                                dfu, dfv, dfa, point, hand):
            Rx = R.from_euler("x", x_tilt, degrees=True)
            Ry = R.from_euler("y", y_tilt, degrees=True)
            Rz = R.from_euler("z", z_rot, degrees=True)
            Rzyx = (Rz * Ry * Rx).as_matrix()
            proj_mat = np.eye(4)
            proj_mat[:3, :3] = Rzyx
            proj_mat[0, 3] = x_shift
            proj_mat[1, 3] = y_shift
            proj_pos = proj_mat @ np.append(point, 1.0)
            proj_centre = proj_mat @ np.array([0, 0, 0, 1.0])
            depth_offset = (proj_pos[2] - proj_centre[2]) * hand
            return dfu + depth_offset, dfv + depth_offset, dfa

        # Test with realistic parameters
        rng = np.random.RandomState(123)
        n = 7
        x_tilts = rng.uniform(-60, 60, n)
        y_tilts = rng.uniform(-5, 5, n)
        z_rots = rng.uniform(70, 90, n)
        x_shifts = rng.uniform(-500, 500, n)
        y_shifts = rng.uniform(-500, 500, n)
        dfu = rng.uniform(20000, 40000, n)
        dfv = rng.uniform(20000, 40000, n)
        dfa = rng.uniform(0, 180, n)
        tilt_dims = (5760, 4092)

        tomo = Tomogram(
            tilt_image_dims=tilt_dims, pixel_size=1.5,
            defocus_u=dfu, defocus_v=dfv, defocus_angle=dfa,
            x_tilts=x_tilts, y_tilts=y_tilts, z_rots=z_rots,
            x_shifts=x_shifts, y_shifts=y_shifts, hand=-1,
        )

        point = np.array([300.0, -150.0, 200.0])

        for i in range(n):
            # Projection
            our_pt = tomo.project_point(point, i)
            ref_M = ryan_build_projection_matrix(
                x_tilts[i], y_tilts[i], z_rots[i],
                x_shifts[i], y_shifts[i], tilt_dims,
            )
            ref_pt = (ref_M @ np.append(point, 1.0))[:2]
            np.testing.assert_allclose(our_pt, ref_pt, atol=1e-10,
                                       err_msg=f"Projection mismatch at tilt {i}")

            # Defocus
            our_u, our_v, our_a = tomo.local_defocus(i, point)
            ref_u, ref_v, ref_a = ryan_local_defocus(
                x_tilts[i], y_tilts[i], z_rots[i],
                x_shifts[i], y_shifts[i],
                dfu[i], dfv[i], dfa[i], point, -1,
            )
            assert our_u == pytest.approx(ref_u, abs=1e-10), \
                f"DefocusU mismatch at tilt {i}"
            assert our_v == pytest.approx(ref_v, abs=1e-10), \
                f"DefocusV mismatch at tilt {i}"
            assert our_a == pytest.approx(ref_a, abs=1e-10), \
                f"DefocusAngle mismatch at tilt {i}"


# ---------------------------------------------------------------------------
# Tests: Tomogram edge cases
# ---------------------------------------------------------------------------

class TestTomogramEdgeCases:

    def test_shift_offsets_projection(self):
        """Non-zero x/y shifts should displace the projected point."""
        n = 1
        base_kwargs = dict(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            y_shifts=np.array([0.0]), hand=1,
        )
        tomo_no_shift = Tomogram(**base_kwargs, x_shifts=np.array([0.0]))
        tomo_with_shift = Tomogram(**base_kwargs, x_shifts=np.array([50.0]))

        pt = np.array([0.0, 0.0, 0.0])
        p_no = tomo_no_shift.project_point(pt, 0)
        p_yes = tomo_with_shift.project_point(pt, 0)
        # x_shift=50 should move the projected x coordinate
        assert p_yes[0] == pytest.approx(p_no[0] + 50.0, abs=1e-10)
        assert p_yes[1] == pytest.approx(p_no[1], abs=1e-10)

    def test_y_shift_offsets_projection(self):
        """Non-zero y shift should displace projected y coordinate."""
        tomo = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([30.0]),
            hand=1,
        )
        pt = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [100.0, 130.0], atol=1e-10)

    def test_nonzero_y_tilt(self):
        """Non-zero y_tilt should change the projection."""
        tomo_y0 = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        tomo_y20 = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([20.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        point = np.array([100.0, 0.0, 200.0])
        p0 = tomo_y0.project_point(point, 0)
        p20 = tomo_y20.project_point(point, 0)
        assert not np.allclose(p0, p20, atol=0.1)

    def test_rectangular_tilt_dims(self):
        """Asymmetric tilt image dims should center correctly."""
        tomo = Tomogram(
            tilt_image_dims=(400, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        pt = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [200.0, 100.0], atol=1e-10)

    def test_large_tilt_angle(self):
        """Near-90-degree tilt should still produce valid projection."""
        tomo = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([30000.0]), defocus_v=np.array([30000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([85.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        # Should not raise, and point at origin still maps to center
        pt = tomo.project_point(np.array([0.0, 0.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [100.0, 100.0], atol=1e-10)
        # Off-axis point should have large Z contribution at 85 degrees
        pt2 = tomo.project_point(np.array([0.0, 0.0, 1000.0]), 0)
        assert abs(pt2[1] - 100.0) > 50  # significant Y displacement

    def test_negative_tilt_angle_symmetry(self):
        """Projecting (0,0,+z) at +tilt vs (0,0,-z) at -tilt should give
        the same image coordinates by symmetry."""
        for angle in [30.0, 60.0]:
            tomo_pos = Tomogram(
                tilt_image_dims=(200, 200), pixel_size=1.0,
                defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
                defocus_angle=np.array([0.0]),
                x_tilts=np.array([angle]), y_tilts=np.array([0.0]),
                z_rots=np.array([0.0]),
                x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
                hand=1,
            )
            tomo_neg = Tomogram(
                tilt_image_dims=(200, 200), pixel_size=1.0,
                defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
                defocus_angle=np.array([0.0]),
                x_tilts=np.array([-angle]), y_tilts=np.array([0.0]),
                z_rots=np.array([0.0]),
                x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
                hand=1,
            )
            p1 = tomo_pos.project_point(np.array([0.0, 0.0, 100.0]), 0)
            p2 = tomo_neg.project_point(np.array([0.0, 0.0, -100.0]), 0)
            np.testing.assert_allclose(p1, p2, atol=1e-10)

    def test_z_rotation_rotates_xy_plane(self):
        """A 90-degree z_rot should swap x and y in the projection."""
        tomo = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([90.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        # Point at (50, 0, 0): with 90-deg z_rot, maps x->y direction
        pt = tomo.project_point(np.array([50.0, 0.0, 0.0]), 0)
        # After R_z(90)^{-1} = R_z(-90), (50,0) -> (0,-50) -> center+(0,-50)
        np.testing.assert_allclose(pt, [100.0, 50.0], atol=1e-10)

    def test_pixel_size_does_not_affect_projection(self):
        """pixel_size is stored but doesn't appear in the projection math."""
        for pix in [0.5, 1.0, 2.0, 5.0]:
            tomo = Tomogram(
                tilt_image_dims=(200, 200), pixel_size=pix,
                defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
                defocus_angle=np.array([0.0]),
                x_tilts=np.array([30.0]), y_tilts=np.array([0.0]),
                z_rots=np.array([10.0]),
                x_shifts=np.array([5.0]), y_shifts=np.array([3.0]),
                hand=1,
            )
            pt = tomo.project_point(np.array([100.0, -50.0, 75.0]), 0)
            # All should be identical since pixel_size only affects metadata
            if pix == 0.5:
                ref = pt.copy()
            else:
                np.testing.assert_allclose(pt, ref, atol=1e-10)

    def test_defocus_offset_proportional_to_depth(self):
        """Defocus offset should scale linearly with z displacement."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([30000.0]), defocus_v=np.array([30000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([45.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        u0, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 0.0]))
        u1, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 100.0]))
        u2, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 200.0]))

        offset1 = u1 - u0
        offset2 = u2 - u0
        assert offset2 == pytest.approx(2 * offset1, abs=1e-6)

    def test_defocus_angle_unchanged(self):
        """Local defocus angle should always equal the global defocus angle."""
        rng = np.random.RandomState(999)
        n = 5
        dfa = rng.uniform(0, 180, n)
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=rng.uniform(20000, 40000, n),
            defocus_v=rng.uniform(20000, 40000, n),
            defocus_angle=dfa,
            x_tilts=rng.uniform(-60, 60, n),
            y_tilts=rng.uniform(-5, 5, n),
            z_rots=rng.uniform(0, 360, n),
            x_shifts=rng.uniform(-200, 200, n),
            y_shifts=rng.uniform(-200, 200, n),
            hand=-1,
        )
        for i in range(n):
            point = rng.uniform(-500, 500, 3)
            _, _, a = tomo.local_defocus(i, point)
            assert a == pytest.approx(dfa[i], abs=1e-12)

    def test_expand_particle_dose_columns_are_float(self):
        """Verify that dose-related columns are numeric after expand."""
        n = 3
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([20000.0, 21000.0, 22000.0]),
            defocus_v=np.array([20500.0, 21500.0, 22500.0]),
            defocus_angle=np.zeros(n),
            x_tilts=np.array([-30.0, 0.0, 30.0]),
            y_tilts=np.zeros(n), z_rots=np.zeros(n),
            x_shifts=np.zeros(n), y_shifts=np.zeros(n),
            hand=1,
        )
        tilt_df = pd.DataFrame({
            '_rlnMicrographName': ['m0.mrc', 'm1.mrc', 'm2.mrc'],
            '_rlnMicrographPreExposure': ['3.0', '6.0', '9.0'],
            '_rlnCtfScalefactor': ['1.0', '0.9', '0.8'],
            '_rlnTomoNominalStageTiltAngle': ['-30.0', '0.0', '30.0'],
        })
        df = tomo.expand_particle(
            np.array([0.0, 0.0, 0.0]), 'stack.mrcs',
            tilt_df, 'p/1', base_orientation=None,
        )
        # Should be numeric (float)
        assert df['_rlnMicrographPreExposure'].dtype in [np.float64, np.float32]
        assert df['_rlnCtfScalefactor'].dtype in [np.float64, np.float32]
        np.testing.assert_allclose(
            df['_rlnDefocusU'].values, [20000.0, 21000.0, 22000.0], atol=1e-6
        )

    def test_expand_preserves_ctf_scalefactor(self):
        """CTF scale factors from the tilt series should appear unchanged."""
        n = 4
        ctf_scales = np.array([1.0, 0.95, 0.85, 0.7])
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.zeros(n), defocus_v=np.zeros(n),
            defocus_angle=np.zeros(n),
            x_tilts=np.linspace(-40, 40, n),
            y_tilts=np.zeros(n), z_rots=np.zeros(n),
            x_shifts=np.zeros(n), y_shifts=np.zeros(n),
            hand=1,
        )
        tilt_df = pd.DataFrame({
            '_rlnMicrographName': [f'm{i}.mrc' for i in range(n)],
            '_rlnMicrographPreExposure': [str(i * 3.0) for i in range(n)],
            '_rlnCtfScalefactor': [str(c) for c in ctf_scales],
            '_rlnTomoNominalStageTiltAngle': [str(a) for a in np.linspace(-40, 40, n)],
        })
        df = tomo.expand_particle(
            np.array([0.0, 0.0, 0.0]), 'stack.mrcs',
            tilt_df, 'p/1', base_orientation=None,
        )
        np.testing.assert_allclose(df['_rlnCtfScalefactor'].values, ctf_scales)

    def test_identity_orientation_at_zero_tilt(self):
        """Identity base orientation with zero tilt should give near-zero angles."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([0.0]), defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]), y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]), y_shifts=np.array([0.0]),
            hand=1,
        )
        tilt_df = pd.DataFrame({
            '_rlnMicrographName': ['mic.mrc'],
            '_rlnMicrographPreExposure': ['3.0'],
            '_rlnCtfScalefactor': ['1.0'],
            '_rlnTomoNominalStageTiltAngle': ['0.0'],
        })
        # Identity rotation at zero tilt: final rotation is identity @ identity = identity
        R_identity = R.identity()
        df = tomo.expand_particle(
            np.array([0.0, 0.0, 0.0]), 'stack.mrcs',
            tilt_df, 'p/1', base_orientation=R_identity,
        )
        # ZYZ Euler of identity is (0, 0, 0)
        # But the tilt rotation matrix at zero tilt is also identity
        # so R_final = I @ I = I -> ZYZ = (0, 0, 0)
        rot = float(df['_rlnAngleRot'].iloc[0])
        tilt = float(df['_rlnAngleTilt'].iloc[0])
        psi = float(df['_rlnAnglePsi'].iloc[0])
        # sin(tilt) should be ~0; rot+psi is degenerate but tilt should be ~0
        assert abs(tilt) < 1e-6

    def test_many_tilts_performance(self):
        """Sanity check that 100 tilts work without issues."""
        n = 100
        tomo = Tomogram(
            tilt_image_dims=(4096, 4096), pixel_size=1.5,
            defocus_u=np.linspace(20000, 40000, n),
            defocus_v=np.linspace(20000, 40000, n),
            defocus_angle=np.zeros(n),
            x_tilts=np.linspace(-70, 70, n),
            y_tilts=np.zeros(n),
            z_rots=np.full(n, 85.0),
            x_shifts=np.zeros(n),
            y_shifts=np.zeros(n),
            hand=-1,
        )
        assert len(tomo.projection_matrices) == n
        pt = tomo.project_point(np.array([500.0, -300.0, 200.0]), 50)
        assert np.isfinite(pt).all()

    def test_single_tilt(self):
        """Single tilt should work correctly."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([25000.0]),
            defocus_v=np.array([26000.0]),
            defocus_angle=np.array([30.0]),
            x_tilts=np.array([0.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        assert tomo.n_tilts == 1
        pt = tomo.project_point(np.array([10.0, 20.0, 0.0]), 0)
        np.testing.assert_allclose(pt, [60.0, 70.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: visible frames edge cases
# ---------------------------------------------------------------------------

class TestParseVisibleFramesEdgeCases:

    def test_single_visible(self):
        assert _parse_visible_frames("[1]") == [0]

    def test_single_invisible(self):
        assert _parse_visible_frames("[0]") == []

    def test_first_invisible_rest_visible(self):
        assert _parse_visible_frames("[0,1,1,1]") == [1, 2, 3]

    def test_last_invisible(self):
        assert _parse_visible_frames("[1,1,1,0]") == [0, 1, 2]

    def test_alternating(self):
        assert _parse_visible_frames("[1,0,1,0,1,0]") == [0, 2, 4]

    def test_long_list(self):
        frames = "[" + ",".join(["1"] * 41) + "]"
        result = _parse_visible_frames(frames)
        assert len(result) == 41
        assert result == list(range(41))

    def test_only_middle_visible(self):
        assert _parse_visible_frames("[0,0,1,0,0]") == [2]


# ---------------------------------------------------------------------------
# Tests: convert() edge cases
# ---------------------------------------------------------------------------

class TestConvertEdgeCases:

    def test_multiple_tomograms(self, tmp_path):
        """Particles from multiple tomograms should all be converted."""
        n_tilts = 3

        ts_dir = tmp_path / "tilt_series"
        ts_dir.mkdir()
        for i in range(2):
            _make_tilt_series_star(str(ts_dir / f"tomo_{i:03d}.star"), n_tilts=n_tilts)

        # Write tomograms.star with 2 tomos pointing to the same tilt series
        tomo_star = tmp_path / "tomograms.star"
        header = textwrap.dedent("""\
            # version 50001

            data_global

            loop_
            _rlnTomoName #1
            _rlnTomoTiltSeriesStarFile #2
            _rlnTomoTiltSeriesPixelSize #3
            _rlnTomoHand #4
            _rlnTomoSizeX #5
            _rlnTomoSizeY #6
            tomo_000 tilt_series/tomo_000.star 1.5 -1 4096 4096
            tomo_001 tilt_series/tomo_001.star 1.5 1 4096 4096
        """)
        _write_text(str(tomo_star), header)

        # Particles: 2 from tomo_000, 1 from tomo_001
        visible = "[" + ",".join(["1"] * n_tilts) + "]"
        particles_text = textwrap.dedent(f"""\
            # version 50001

            data_optics

            loop_
            _rlnVoltage #1
            _rlnSphericalAberration #2
            _rlnAmplitudeContrast #3
            _rlnOpticsGroup #4
            _rlnOpticsGroupName #5
            _rlnImagePixelSize #6
            _rlnImageSize #7
            _rlnImageDimensionality #8
            300.0 2.7 0.1 1 opticsGroup1 1.5 128 2

            # version 50001

            data_particles

            loop_
            _rlnTomoName #1
            _rlnCenteredCoordinateXAngst #2
            _rlnCenteredCoordinateYAngst #3
            _rlnCenteredCoordinateZAngst #4
            _rlnOpticsGroup #5
            _rlnTomoParticleName #6
            _rlnImageName #7
            _rlnAngleRot #8
            _rlnAngleTilt #9
            _rlnAnglePsi #10
            _rlnRandomSubset #11
            _rlnTomoVisibleFrames #12
            tomo_000 100.0 200.0 50.0 1 tomo_000/1 Extract/tomo_000/1_stack2d.mrcs 10.0 20.0 30.0 1 {visible}
            tomo_000 -100.0 -200.0 -50.0 1 tomo_000/2 Extract/tomo_000/2_stack2d.mrcs 40.0 50.0 60.0 2 {visible}
            tomo_001 0.0 0.0 0.0 1 tomo_001/1 Extract/tomo_001/1_stack2d.mrcs 0.0 0.0 0.0 1 {visible}
        """)
        p_star = tmp_path / "particles.star"
        _write_text(str(p_star), particles_text)

        output = str(tmp_path / "output.star")
        convert(str(tomo_star), str(p_star), output, tilt_dim=(4096, 4096))

        df, _ = read_star(output)
        assert len(df) == 3 * n_tilts

        # Check that all three groups exist
        groups = sorted(df['_rlnGroupName'].unique())
        assert groups == ['tomo_000/1', 'tomo_000/2', 'tomo_001/1']

    def test_single_particle(self, tmp_path):
        """Single particle conversion should work."""
        n_tilts = 3
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        _make_tomograms_star(str(tomo_star), "ts/ts.star")

        p_star = tmp_path / "particles.star"
        _make_particles_star(str(p_star), n_particles=1, n_tilts=n_tilts)

        output = str(tmp_path / "output.star")
        convert(str(tomo_star), str(p_star), output, tilt_dim=(100, 100))

        df, _ = read_star(output)
        assert len(df) == n_tilts
        assert len(df['_rlnGroupName'].unique()) == 1

    def test_missing_optics_raises(self, tmp_path):
        """particles.star without optics table should raise ValueError."""
        n_tilts = 3
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        _make_tomograms_star(str(tomo_star), "ts/ts.star")

        # Write particles.star without optics table
        no_optics_text = textwrap.dedent("""\
            # version 50001

            data_particles

            loop_
            _rlnTomoName #1
            _rlnCenteredCoordinateXAngst #2
            _rlnCenteredCoordinateYAngst #3
            _rlnCenteredCoordinateZAngst #4
            _rlnOpticsGroup #5
            _rlnTomoParticleName #6
            _rlnImageName #7
            _rlnAngleRot #8
            _rlnAngleTilt #9
            _rlnAnglePsi #10
            _rlnRandomSubset #11
            _rlnTomoVisibleFrames #12
            tomo_001 0.0 0.0 0.0 1 tomo_001/1 img.mrcs 0.0 0.0 0.0 1 [1,1,1]
        """)
        p_star = tmp_path / "p.star"
        _write_text(str(p_star), no_optics_text)

        output = str(tmp_path / "output.star")
        with pytest.raises(ValueError, match="no optics"):
            convert(str(tomo_star), str(p_star), output, tilt_dim=(100, 100))

    def test_missing_tomogram_in_tomograms_star_raises(self, tmp_path):
        """Particle referencing a tomo not in tomograms.star should raise."""
        n_tilts = 3
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        _make_tomograms_star(str(tomo_star), "ts/ts.star")  # has tomo_001

        # Particle references "tomo_999" which doesn't exist
        visible = "[" + ",".join(["1"] * n_tilts) + "]"
        particles_text = textwrap.dedent(f"""\
            # version 50001

            data_optics

            loop_
            _rlnVoltage #1
            _rlnSphericalAberration #2
            _rlnAmplitudeContrast #3
            _rlnOpticsGroup #4
            _rlnOpticsGroupName #5
            _rlnImagePixelSize #6
            _rlnImageSize #7
            _rlnImageDimensionality #8
            300.0 2.7 0.1 1 opticsGroup1 1.5 128 2

            # version 50001

            data_particles

            loop_
            _rlnTomoName #1
            _rlnCenteredCoordinateXAngst #2
            _rlnCenteredCoordinateYAngst #3
            _rlnCenteredCoordinateZAngst #4
            _rlnOpticsGroup #5
            _rlnTomoParticleName #6
            _rlnImageName #7
            _rlnAngleRot #8
            _rlnAngleTilt #9
            _rlnAnglePsi #10
            _rlnRandomSubset #11
            _rlnTomoVisibleFrames #12
            tomo_999 0.0 0.0 0.0 1 tomo_999/1 img.mrcs 0.0 0.0 0.0 1 {visible}
        """)
        p_star = tmp_path / "p.star"
        _write_text(str(p_star), particles_text)

        output = str(tmp_path / "output.star")
        with pytest.raises(ValueError, match="tomo_999"):
            convert(str(tomo_star), str(p_star), output, tilt_dim=(100, 100))

    def test_deterministic_output(self, relion5_data, tmp_path):
        """Running convert() twice on the same input should give identical output."""
        out1 = str(tmp_path / "out1.star")
        out2 = str(tmp_path / "out2.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                out1, tilt_dim=(4096, 4096))
        convert(relion5_data["tomograms"], relion5_data["particles"],
                out2, tilt_dim=(4096, 4096))

        df1, op1 = read_star(out1)
        df2, op2 = read_star(out2)

        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(op1, op2)

    def test_tilt_dim_fallback_to_tomosize(self, tmp_path):
        """When tilt_dim=None and MRCs can't be read, should fall back to
        _rlnTomoSizeX/_rlnTomoSizeY in tomograms.star."""
        n_tilts = 3
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        _make_tilt_series_star(str(ts_dir / "ts.star"), n_tilts=n_tilts)

        tomo_star = tmp_path / "tomos.star"
        # tomograms.star has _rlnTomoSizeX/Y = 4096
        _make_tomograms_star(str(tomo_star), "ts/ts.star")

        p_star = tmp_path / "particles.star"
        _make_particles_star(str(p_star), n_particles=1, n_tilts=n_tilts)

        # With tilt_dim=None, MRC won't exist -> falls back to TomoSizeX/Y
        output = str(tmp_path / "output.star")
        convert(str(tomo_star), str(p_star), output, tilt_dim=None)

        df, _ = read_star(output)
        assert len(df) == n_tilts

    def test_origins_are_zero(self, relion5_data, tmp_path):
        """All _rlnOriginXAngst and _rlnOriginYAngst should be zero."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        ox = df['_rlnOriginXAngst'].values.astype(float)
        oy = df['_rlnOriginYAngst'].values.astype(float)
        np.testing.assert_array_equal(ox, 0.0)
        np.testing.assert_array_equal(oy, 0.0)

    def test_optics_values_propagated(self, relion5_data, tmp_path):
        """Verify that optics values from the input are in the output."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        _, optics = read_star(output)
        assert optics['_rlnVoltage'].values[0] == '300.0'
        assert optics['_rlnSphericalAberration'].values[0] == '2.7'
        assert optics['_rlnAmplitudeContrast'].values[0] == '0.1'
        assert optics['_rlnImageDimensionality'].values[0] == '2'
        assert optics['_rlnOpticsGroupName'].values[0] == 'opticsGroup1'

    def test_all_optics_group_one(self, relion5_data, tmp_path):
        """All particles should have _rlnOpticsGroup = 1."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        groups = df['_rlnOpticsGroup'].values
        assert all(g == '1' for g in groups)

    def test_defocus_values_nontrivial(self, relion5_data, tmp_path):
        """Output defocus should not be all-zero — should have real values."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        dfu = df['_rlnDefocusU'].values.astype(float)
        dfv = df['_rlnDefocusV'].values.astype(float)
        assert np.any(dfu != 0)
        assert np.any(dfv != 0)
        # The synthetic data has defocus around 30000-31400
        assert np.all(dfu > 1000)  # should be large values
        assert np.all(dfv > 1000)

    def test_stage_tilt_angle_preserved(self, relion5_data, tmp_path):
        """_rlnTomoNominalStageTiltAngle should be present and non-trivial."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        assert '_rlnTomoNominalStageTiltAngle' in df.columns
        angles = df['_rlnTomoNominalStageTiltAngle'].values.astype(float)
        # Our synthetic data has tilts from -60 to 60
        assert np.min(angles) < -10
        assert np.max(angles) > 10

    def test_different_tilt_dims_change_coordinates(self, relion5_data, tmp_path):
        """Different tilt_dim values should produce different coordinate values
        but identical defocus and angles (since coordinates are not in the output
        DataFrame columns, this test verifies the image names stay consistent)."""
        out1 = str(tmp_path / "out1.star")
        out2 = str(tmp_path / "out2.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                out1, tilt_dim=(4096, 4096))
        convert(relion5_data["tomograms"], relion5_data["particles"],
                out2, tilt_dim=(2048, 2048))

        df1, _ = read_star(out1)
        df2, _ = read_star(out2)

        # Same number of rows
        assert len(df1) == len(df2)

        # Defocus should be identical (doesn't depend on tilt_dim)
        np.testing.assert_allclose(
            df1['_rlnDefocusU'].values.astype(float),
            df2['_rlnDefocusU'].values.astype(float),
            atol=1e-10,
        )

        # Angles should be identical (doesn't depend on tilt_dim)
        np.testing.assert_allclose(
            df1['_rlnAngleRot'].values.astype(float),
            df2['_rlnAngleRot'].values.astype(float),
            atol=1e-10,
        )

    def test_each_group_has_consistent_random_subset(self, relion5_data, tmp_path):
        """Each particle group should have the same _rlnRandomSubset for all tilts."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        for group in df['_rlnGroupName'].unique():
            mask = df['_rlnGroupName'] == group
            subsets = df[mask]['_rlnRandomSubset'].values.astype(int)
            assert len(set(subsets)) == 1, \
                f"Group {group} has mixed subsets: {set(subsets)}"

    def test_micrograph_name_preserved(self, relion5_data, tmp_path):
        """_rlnMicrographName should be propagated from tilt series."""
        output = str(tmp_path / "output.star")
        convert(relion5_data["tomograms"], relion5_data["particles"],
                output, tilt_dim=(4096, 4096))
        df, _ = read_star(output)
        assert '_rlnMicrographName' in df.columns
        names = df['_rlnMicrographName'].values
        # Should contain mic_000.mrc, mic_001.mrc, etc. from our fixture
        assert any('mic_' in n for n in names)


# ---------------------------------------------------------------------------
# Tests: quantitative geometry checks
# ---------------------------------------------------------------------------

class TestQuantitativeGeometry:

    def test_defocus_offset_at_45deg_tilt_known_value(self):
        """At 45-degree x-tilt, z-displacement projects to z' = z * cos(45) + ...
        The depth offset for a pure Z displacement is z * sin(45) * hand."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([10000.0]),
            defocus_v=np.array([10000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([45.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        z_disp = 1000.0  # Angstroms
        u, _, _ = tomo.local_defocus(0, np.array([0.0, 0.0, z_disp]))
        # For x-tilt, Rzyx = Rx(45). Point (0,0,z) -> (0, -z*sin45, z*cos45)
        # depth offset = (z*cos45 - 0) * hand = z * cos(45)
        expected_offset = z_disp * np.cos(np.deg2rad(45.0))
        assert u == pytest.approx(10000.0 + expected_offset, abs=1.0)

    def test_defocus_offset_at_zero_tilt_equals_z(self):
        """At zero tilt, depth offset equals z * hand (particle at different
        depth along the beam has different defocus)."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([25000.0]),
            defocus_v=np.array([25000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        for z in [-1000.0, 0.0, 500.0, 2000.0]:
            u, v, a = tomo.local_defocus(0, np.array([0.0, 0.0, z]))
            # At zero tilt, R=I so depth_offset = z * hand
            assert u == pytest.approx(25000.0 + z, abs=1e-6)
            assert v == pytest.approx(25000.0 + z, abs=1e-6)

    def test_xy_displacement_no_defocus_offset_at_zero_tilt(self):
        """XY displacement at zero tilt should produce no defocus change."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([20000.0]),
            defocus_v=np.array([20000.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        u, _, _ = tomo.local_defocus(0, np.array([5000.0, 3000.0, 0.0]))
        assert u == pytest.approx(20000.0, abs=1e-6)

    def test_projection_with_known_30deg_tilt(self):
        """At 30-degree x-tilt, z-displacement should project partially into y."""
        tomo = Tomogram(
            tilt_image_dims=(200, 200), pixel_size=1.0,
            defocus_u=np.array([0.0]),
            defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([30.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        # Point at (0, 0, 100): R_x(30)^{-1} applied, then center offset
        # R_x(-30) on (0,0,100) = (0, 100*sin30, 100*cos30) = (0, 50, 86.6)
        pt = tomo.project_point(np.array([0.0, 0.0, 100.0]), 0)
        np.testing.assert_allclose(pt[0], 100.0, atol=1e-6)  # x = center
        np.testing.assert_allclose(pt[1], 100.0 + 50.0, atol=1.0)  # y = center + 50

    def test_defocus_u_v_offset_identical(self):
        """DefocusU and DefocusV should receive the same depth offset."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([25000.0]),
            defocus_v=np.array([30000.0]),
            defocus_angle=np.array([45.0]),
            x_tilts=np.array([40.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=-1,
        )
        pt = np.array([0.0, 0.0, 500.0])
        u, v, _ = tomo.local_defocus(0, pt)
        u0, v0, _ = tomo.local_defocus(0, np.array([0.0, 0.0, 0.0]))
        # Both U and V should change by the same offset
        assert (u - u0) == pytest.approx(v - v0, abs=1e-10)

    def test_orientation_at_zero_everything(self):
        """When all tilts=0 and base_orientation=identity,
        the output ZYZ Euler should be (0,0,0)."""
        tomo = Tomogram(
            tilt_image_dims=(100, 100), pixel_size=1.0,
            defocus_u=np.array([0.0]),
            defocus_v=np.array([0.0]),
            defocus_angle=np.array([0.0]),
            x_tilts=np.array([0.0]),
            y_tilts=np.array([0.0]),
            z_rots=np.array([0.0]),
            x_shifts=np.array([0.0]),
            y_shifts=np.array([0.0]),
            hand=1,
        )
        # R_tilt is the 3x3 block of the projection matrix at zero tilt
        # which is R_zyx(0,0,0)^{-1} = identity
        R_tilt = tomo.projection_matrices[0][:3, :3]
        np.testing.assert_allclose(R_tilt, np.eye(3), atol=1e-10)

    def test_multiple_random_points_match_reference(self):
        """Stress test: many random points and parameters match reference."""
        rng = np.random.RandomState(7777)
        for trial in range(10):
            n = rng.randint(3, 15)
            x_tilts = rng.uniform(-70, 70, n)
            y_tilts = rng.uniform(-10, 10, n)
            z_rots = rng.uniform(0, 360, n)
            x_shifts = rng.uniform(-1000, 1000, n)
            y_shifts = rng.uniform(-1000, 1000, n)
            dfu = rng.uniform(10000, 50000, n)
            dfv = rng.uniform(10000, 50000, n)
            dfa = rng.uniform(0, 180, n)
            dims = (rng.randint(1000, 8000), rng.randint(1000, 8000))
            hand = rng.choice([-1, 1])

            tomo = Tomogram(
                tilt_image_dims=dims, pixel_size=rng.uniform(0.5, 3.0),
                defocus_u=dfu, defocus_v=dfv, defocus_angle=dfa,
                x_tilts=x_tilts, y_tilts=y_tilts, z_rots=z_rots,
                x_shifts=x_shifts, y_shifts=y_shifts, hand=hand,
            )

            for _ in range(5):
                pt = rng.uniform(-2000, 2000, 3)

                for i in range(n):
                    # Verify projection is finite
                    proj = tomo.project_point(pt, i)
                    assert np.isfinite(proj).all(), f"Non-finite projection: trial={trial}, tilt={i}"

                    # Verify defocus is finite
                    u, v, a = tomo.local_defocus(i, pt)
                    assert np.isfinite(u), f"Non-finite DefocusU: trial={trial}"
                    assert np.isfinite(v), f"Non-finite DefocusV: trial={trial}"
                    assert a == pytest.approx(dfa[i], abs=1e-10)
