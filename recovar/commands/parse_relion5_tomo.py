"""Convert RELION5 tilt-series data to RECOVAR 2D tilt format.

Reads a RELION5 ``tomograms.star`` (from Polish/Tomograms jobs) and
``particles.star`` (from Extract/Refine) and produces a 2D STAR file where
each row is one tilt of one particle, grouped by ``_rlnGroupName``.  The
output is directly compatible with ``recovar pipeline --tilt-series``.

Projection geometry adapted from relion2cryodrgn by Ryan Feathers
(Princeton/Zhong lab), based on code by Bogdan Toader (MRC-LMB/RELION team).

Usage::

    recovar parse_relion5_tomo \\
        -t Polish/job249/tomograms.star \\
        -p Extract/job260/particles.star \\
        -o particles_2d.star
"""

import argparse
import logging
import os
from ast import literal_eval

import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from recovar.data_io import starfile

logger = logging.getLogger(__name__)

## TODO: I would like to have some tests for this related to a real dataset to see if we break something.
## I need to find a real dataset for that to work, though.
##
# ---------------------------------------------------------------------------
# Tomogram geometry
# ---------------------------------------------------------------------------

class Tomogram:
    """Per-tilt-series geometry and defocus data.

    Precomputes 4x4 projection matrices that map 3D tomogram coordinates
    (in Angstroms) to 2D tilt-image coordinates (in pixels).
    """

    def __init__(self, tilt_image_dims, pixel_size,
                 defocus_u, defocus_v, defocus_angle,
                 x_tilts, y_tilts, z_rots, x_shifts, y_shifts, hand):
        self.tilt_image_dims = tilt_image_dims
        self.pixel_size = pixel_size
        self.defocus_u = defocus_u
        self.defocus_v = defocus_v
        self.defocus_angle = defocus_angle
        self.x_tilts = x_tilts
        self.y_tilts = y_tilts
        self.z_rots = z_rots
        self.x_shifts = x_shifts
        self.y_shifts = y_shifts
        self.hand = hand
        self.n_tilts = len(x_tilts)
        self.projection_matrices = self._build_projection_matrices()

    # -- internal helpers --

    @staticmethod
    def _translation_matrix(shift_3d):
        mat = np.eye(4)
        mat[:3, 3] = shift_3d
        return mat

    @staticmethod
    def _rotation_matrix(axis, angle_deg):
        rot = R.from_rotvec(np.deg2rad(angle_deg) * np.array(axis))
        mat = np.eye(4)
        mat[:3, :3] = rot.as_matrix()
        return mat

    def _build_projection_matrices(self):
        matrices = {}
        specimen_center = np.array([0.0, 0.0, 0.0])

        for i in range(self.n_tilts):
            s0 = self._translation_matrix(-specimen_center)
            r0 = self._rotation_matrix([1, 0, 0], self.x_tilts[i])
            r1 = self._rotation_matrix([0, 1, 0], self.y_tilts[i])
            r2 = self._rotation_matrix([0, 0, 1], self.z_rots[i])

            shift = np.array([self.x_shifts[i], self.y_shifts[i], 0.0])
            s1 = self._translation_matrix(shift)

            center = np.array([
                self.tilt_image_dims[0] / 2.0,
                self.tilt_image_dims[1] / 2.0,
                0.0,
            ])
            s2 = self._translation_matrix(center)

            # Combined rotation (inverse of ZYX Euler composition)
            Rzyx = R.from_matrix(r2[:3, :3]) * R.from_matrix(r1[:3, :3]) * R.from_matrix(r0[:3, :3])
            R_inv = np.eye(4)
            R_inv[:3, :3] = Rzyx.inv().as_matrix()

            matrices[i] = s2 @ s1 @ R_inv @ s0

        return matrices

    # -- public API --

    def project_point(self, point_3d, i_tilt):
        """Project a 3D coordinate (Angstroms) to 2D tilt-image pixels."""
        pt = np.append(point_3d, 1.0)
        return (self.projection_matrices[i_tilt] @ pt)[:2]

    def local_defocus(self, i_tilt, point_3d):
        """Compute depth-corrected defocus at a 3D particle position."""
        Rx = R.from_euler("x", self.x_tilts[i_tilt], degrees=True)
        Ry = R.from_euler("y", self.y_tilts[i_tilt], degrees=True)
        Rz = R.from_euler("z", self.z_rots[i_tilt], degrees=True)
        Rzyx = (Rz * Ry * Rx).as_matrix()

        proj_mat = np.eye(4)
        proj_mat[:3, :3] = Rzyx
        proj_mat[0, 3] = self.x_shifts[i_tilt]
        proj_mat[1, 3] = self.y_shifts[i_tilt]

        proj_pos = proj_mat @ np.append(point_3d, 1.0)
        proj_centre = proj_mat @ np.array([0.0, 0.0, 0.0, 1.0])
        depth_offset = (proj_pos[2] - proj_centre[2]) * self.hand

        return (
            self.defocus_u[i_tilt] + depth_offset,
            self.defocus_v[i_tilt] + depth_offset,
            self.defocus_angle[i_tilt],
        )

    def expand_particle(self, point_3d, image_name, tilt_df,
                        group_name, base_orientation=None):
        """Expand one 3D particle into one 2D row per visible tilt.

        Parameters
        ----------
        point_3d : array (3,)
            Particle position in Angstroms.
        image_name : str
            ``_rlnImageName`` from particles.star (stack path).
        tilt_df : DataFrame
            Tilt-series rows for the visible tilts (RECOVAR string-typed).
        group_name : str
            Particle group name (``_rlnTomoParticleName``).
        base_orientation : Rotation or None
            Combined particle + subtomogram orientation (ZYZ).

        Returns
        -------
        DataFrame with one row per tilt.
        """
        n = self.n_tilts
        coords_2d = np.empty((n, 2))
        dfu = np.empty(n)
        dfv = np.empty(n)
        dfa = np.empty(n)
        euler_zyz = np.zeros((n, 3))

        for i in range(n):
            coords_2d[i] = self.project_point(point_3d, i)
            dfu[i], dfv[i], dfa[i] = self.local_defocus(i, point_3d)

            if base_orientation is not None:
                tilt_rot = self.projection_matrices[i][:3, :3]
                R_final = R.from_matrix(base_orientation.as_matrix() @ tilt_rot)
                euler_zyz[i] = R_final.as_euler('ZYZ', degrees=True)

        image_names = [f"{i + 1:06d}@{image_name}" for i in range(n)]

        micrograph_names = tilt_df['_rlnMicrographName'].values
        pre_exposure = tilt_df['_rlnMicrographPreExposure'].values.astype(float)
        ctf_scale = tilt_df['_rlnCtfScalefactor'].values.astype(float)
        stage_tilt = tilt_df['_rlnTomoNominalStageTiltAngle'].values.astype(float)

        return pd.DataFrame({
            '_rlnDefocusU': dfu,
            '_rlnDefocusV': dfv,
            '_rlnDefocusAngle': dfa,
            '_rlnImageName': image_names,
            '_rlnMicrographName': micrograph_names,
            '_rlnCtfScalefactor': ctf_scale,
            '_rlnGroupName': group_name,
            '_rlnAngleRot': euler_zyz[:, 0],
            '_rlnAngleTilt': euler_zyz[:, 1],
            '_rlnAnglePsi': euler_zyz[:, 2],
            '_rlnOriginXAngst': 0.0,
            '_rlnOriginYAngst': 0.0,
            '_rlnRandomSubset': 0,  # placeholder, set per-particle later
            '_rlnMicrographPreExposure': pre_exposure,
            '_rlnTomoNominalStageTiltAngle': stage_tilt,
        })


# ---------------------------------------------------------------------------
# Tilt image dimension auto-detection
# ---------------------------------------------------------------------------

def _detect_tilt_dims(tomograms_dir, tilt_df):
    """Read the first micrograph MRC header to get tilt image dimensions."""
    mic_name = tilt_df['_rlnMicrographName'].values[0]
    mic_path = os.path.join(tomograms_dir, mic_name)
    if not os.path.isfile(mic_path):
        return None
    with mrcfile.open(mic_path, header_only=True) as mrc:
        nx = int(mrc.header.nx)
        ny = int(mrc.header.ny)
    return (nx, ny)


# ---------------------------------------------------------------------------
# Star file reading helpers
# ---------------------------------------------------------------------------

def _read_star_all_blocks(filepath):
    """Read a RELION5 star file returning all named data blocks.

    RECOVAR's read_star returns (main_df, optics_df) picking the largest
    non-optics block. For tomograms.star we just need the single data block.
    For per-tilt star files we also just need the single block.
    """
    return starfile.read_star(filepath)


def _parse_visible_frames(frames_str):
    """Parse rlnTomoVisibleFrames string like '[1,1,0,1,...]' to indices."""
    frames = literal_eval(frames_str)
    return [i for i, v in enumerate(frames) if v == 1]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(tomograms_path, particles_path, output_path, tilt_dim=None):
    """Convert RELION5 tomo data to RECOVAR 2D tilt format."""

    tomo_dir = os.path.dirname(os.path.abspath(tomograms_path))
    # Walk up to find the RELION project root (parent of the job directory)
    # e.g. Polish/job249/tomograms.star -> project root is 2 levels up
    project_root = tomo_dir
    # Try to find the project root by looking for the tilt series star paths
    # The paths in tomograms.star are relative to the RELION project root

    # ---- Load tomograms.star ----
    tomo_df, _ = _read_star_all_blocks(tomograms_path)
    logger.info("Loaded %d tomograms from %s", len(tomo_df), tomograms_path)

    # ---- Load particles.star ----
    particles_df, optics_df = _read_star_all_blocks(particles_path)
    logger.info("Loaded %d particles from %s", len(particles_df), particles_path)

    if optics_df is None:
        raise ValueError(
            f"particles.star has no optics table: {particles_path}\n"
            "Expected RELION5 format with data_optics and data_particles blocks."
        )

    # Determine RELION project root from tilt series star paths
    # The rlnTomoTiltSeriesStarFile is relative to the project root
    sample_ts_path = tomo_df['_rlnTomoTiltSeriesStarFile'].values[0]
    # Try resolving relative to tomograms.star directory first
    if os.path.isfile(os.path.join(tomo_dir, sample_ts_path)):
        project_root = tomo_dir
    else:
        # Walk up directories until we find a valid resolution
        test_dir = tomo_dir
        for _ in range(5):
            test_dir = os.path.dirname(test_dir)
            if os.path.isfile(os.path.join(test_dir, sample_ts_path)):
                project_root = test_dir
                break
        else:
            raise FileNotFoundError(
                f"Cannot find tilt-series star file: {sample_ts_path}\n"
                f"Searched up from: {tomo_dir}\n"
                "Check that tomograms.star paths are correct relative to the "
                "RELION project directory."
            )

    logger.info("RELION project root: %s", project_root)

    # ---- Build tomogram name -> tilt series mapping ----
    tilt_series_cache = {}

    def _get_tilt_series(tomo_name):
        if tomo_name in tilt_series_cache:
            return tilt_series_cache[tomo_name]

        tomo_row_mask = tomo_df['_rlnTomoName'].values == tomo_name
        if not np.any(tomo_row_mask):
            raise ValueError(f"Tomogram '{tomo_name}' not found in tomograms.star")
        tomo_idx = np.where(tomo_row_mask)[0][0]

        ts_star_relpath = tomo_df['_rlnTomoTiltSeriesStarFile'].values[tomo_idx]
        ts_star_path = os.path.join(project_root, ts_star_relpath)

        logger.info("Loading tilt-series: %s -> %s", tomo_name, ts_star_path)
        ts_df, _ = _read_star_all_blocks(ts_star_path)

        # Also cache tomo-level metadata
        hand_val = tomo_df['_rlnTomoHand'].values[tomo_idx]
        hand = -1 if float(hand_val) == -1 else 1
        pixel_size = float(tomo_df['_rlnTomoTiltSeriesPixelSize'].values[tomo_idx])

        tilt_series_cache[tomo_name] = (ts_df, hand, pixel_size)
        return ts_df, hand, pixel_size

    # ---- Auto-detect tilt image dimensions if needed ----
    if tilt_dim is None:
        first_tomo = tomo_df['_rlnTomoName'].values[0]
        ts_df_first, _, _ = _get_tilt_series(first_tomo)
        tilt_dim = _detect_tilt_dims(project_root, ts_df_first)
        if tilt_dim is not None:
            logger.info("Auto-detected tilt image dimensions: %d x %d", *tilt_dim)
        else:
            logger.warning(
                "Could not auto-detect tilt dimensions from MRC headers. "
                "Using tomogram dimensions from tomograms.star."
            )
            # Fall back to TomoSizeX/Y if available
            if '_rlnTomoSizeX' in tomo_df.columns:
                sx = int(tomo_df['_rlnTomoSizeX'].values[0])
                sy = int(tomo_df['_rlnTomoSizeY'].values[0])
                tilt_dim = (sx, sy)
                logger.info("Using tomogram dimensions: %d x %d", sx, sy)
            else:
                raise ValueError(
                    "Cannot determine tilt image dimensions. "
                    "Use --tilt-dim to specify manually."
                )

    logger.info("Tilt image dimensions: %d x %d pixels", *tilt_dim)

    # ---- Get optics info ----
    # Use first optics group for global values
    voltage = float(optics_df['_rlnVoltage'].values[0])
    cs = float(optics_df['_rlnSphericalAberration'].values[0])
    amp_contrast = float(optics_df['_rlnAmplitudeContrast'].values[0])
    angpix = float(optics_df['_rlnImagePixelSize'].values[0])
    image_size = int(optics_df['_rlnImageSize'].values[0])

    # ---- Process each particle ----
    all_rows = []
    n_particles = len(particles_df)

    for idx in range(n_particles):
        if idx % 500 == 0:
            logger.info("Processing particle %d / %d", idx + 1, n_particles)

        tomo_name = particles_df['_rlnTomoName'].values[idx]
        ts_df, hand, pixel_size = _get_tilt_series(tomo_name)

        # Parse visible frames
        frames_str = particles_df['_rlnTomoVisibleFrames'].values[idx]
        visible_indices = _parse_visible_frames(frames_str)
        sub_ts_df = ts_df.iloc[visible_indices].copy()

        # Build Tomogram object
        x_tilts = sub_ts_df['_rlnTomoXTilt'].values.astype(float)
        y_tilts = sub_ts_df['_rlnTomoYTilt'].values.astype(float)
        z_rots = sub_ts_df['_rlnTomoZRot'].values.astype(float)
        x_shifts = sub_ts_df['_rlnTomoXShiftAngst'].values.astype(float)
        y_shifts = sub_ts_df['_rlnTomoYShiftAngst'].values.astype(float)
        defocus_u = sub_ts_df['_rlnDefocusU'].values.astype(float)
        defocus_v = sub_ts_df['_rlnDefocusV'].values.astype(float)
        defocus_angle = sub_ts_df['_rlnDefocusAngle'].values.astype(float)

        tomogram = Tomogram(
            tilt_dim, pixel_size,
            defocus_u, defocus_v, defocus_angle,
            x_tilts, y_tilts, z_rots, x_shifts, y_shifts,
            hand=hand,
        )

        # Particle orientation (ZYZ Euler)
        rot = float(particles_df['_rlnAngleRot'].values[idx])
        tilt = float(particles_df['_rlnAngleTilt'].values[idx])
        psi = float(particles_df['_rlnAnglePsi'].values[idx])
        R_particle = R.from_euler('ZYZ', [rot, tilt, psi], degrees=True)

        # Optional subtomogram orientation
        has_subtomo_rot = '_rlnTomoSubtomogramRot' in particles_df.columns
        if has_subtomo_rot:
            srot = float(particles_df['_rlnTomoSubtomogramRot'].values[idx])
            stilt = float(particles_df['_rlnTomoSubtomogramTilt'].values[idx])
            spsi = float(particles_df['_rlnTomoSubtomogramPsi'].values[idx])
            R_subtomo = R.from_euler('ZYZ', [srot, stilt, spsi], degrees=True)
        else:
            R_subtomo = R.identity()

        R_base = R_particle * R_subtomo

        # 3D position in Angstroms
        x_ang = float(particles_df['_rlnCenteredCoordinateXAngst'].values[idx])
        y_ang = float(particles_df['_rlnCenteredCoordinateYAngst'].values[idx])
        z_ang = float(particles_df['_rlnCenteredCoordinateZAngst'].values[idx])
        point_3d = np.array([x_ang, y_ang, z_ang])

        # Image stack name
        image_name = particles_df['_rlnImageName'].values[idx]
        group_name = particles_df['_rlnTomoParticleName'].values[idx]

        # Expand to 2D rows
        df_2d = tomogram.expand_particle(
            point_3d, image_name, sub_ts_df,
            group_name, base_orientation=R_base,
        )

        # Set per-particle metadata
        random_subset = int(particles_df['_rlnRandomSubset'].values[idx])
        df_2d['_rlnRandomSubset'] = random_subset

        all_rows.append(df_2d)

    # ---- Concatenate and sort by dose within each group ----
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df['_rlnMicrographPreExposure'] = final_df['_rlnMicrographPreExposure'].astype(float)
    final_df = final_df.sort_values(
        ['_rlnGroupName', '_rlnMicrographPreExposure'],
        kind='mergesort',
    ).reset_index(drop=True)

    logger.info("Total 2D rows: %d (%d particles x ~%d tilts)",
                len(final_df), n_particles,
                len(final_df) // max(n_particles, 1))

    # ---- Build output optics table ----
    out_optics = pd.DataFrame({
        '_rlnOpticsGroup': ['1'],
        '_rlnOpticsGroupName': ['opticsGroup1'],
        '_rlnImagePixelSize': [str(angpix)],
        '_rlnImageSize': [str(image_size)],
        '_rlnVoltage': [str(voltage)],
        '_rlnSphericalAberration': [str(cs)],
        '_rlnAmplitudeContrast': [str(amp_contrast)],
        '_rlnImageDimensionality': ['2'],
    })

    # Add _rlnOpticsGroup to particles (all group 1)
    final_df['_rlnOpticsGroup'] = '1'

    # ---- Write output ----
    starfile.write_star(output_path, final_df, data_optics=out_optics)
    logger.info("Wrote %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_args(parser):
    parser.add_argument(
        "-t", "--tomograms", required=True,
        help="RELION5 tomograms.star (from Polish or Tomograms job)",
    )
    parser.add_argument(
        "-p", "--particles", required=True,
        help="RELION5 particles.star (from Extract or Refine job)",
    )
    parser.add_argument(
        "-o", "--output", default="particles_2d.star",
        help="Output 2D STAR file (default: particles_2d.star)",
    )
    parser.add_argument(
        "--tilt-dim", nargs=2, type=int, default=None, metavar=("W", "H"),
        help="Tilt image dimensions in pixels (auto-detected from MRC headers if omitted)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_args(parser)
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    tilt_dim = tuple(args.tilt_dim) if args.tilt_dim else None
    convert(args.tomograms, args.particles, args.output, tilt_dim=tilt_dim)


if __name__ == "__main__":
    main()
