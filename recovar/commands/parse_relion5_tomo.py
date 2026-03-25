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

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from recovar.data_io import starfile

logger = logging.getLogger(__name__)

# Real-data regression tests: tests/unit/test_parse_relion5_tomo_realdata.py
# ---------------------------------------------------------------------------
# Tomogram geometry
# ---------------------------------------------------------------------------


class Tomogram:
    """Per-tilt-series geometry and defocus data.

    Precomputes rotation matrices for mapping 3D particle coordinates to
    per-tilt defocus corrections and orientation angles.
    """

    def __init__(
        self, pixel_size, defocus_u, defocus_v, defocus_angle, x_tilts, y_tilts, z_rots, x_shifts, y_shifts, hand
    ):
        self.pixel_size = pixel_size
        self.defocus_u = np.asarray(defocus_u, dtype=np.float64)
        self.defocus_v = np.asarray(defocus_v, dtype=np.float64)
        self.defocus_angle = np.asarray(defocus_angle, dtype=np.float64)
        self.x_tilts = x_tilts
        self.y_tilts = y_tilts
        self.z_rots = z_rots
        self.x_shifts = x_shifts
        self.y_shifts = y_shifts
        self.hand = hand
        self.n_tilts = len(x_tilts)

        # Build per-tilt rotation matrices
        self._rzyx_matrices, self._tilt_rots = self._build_rotation_matrices()
        self._depth_rows = self._rzyx_matrices[:, 2, :]  # (n, 3)

    def _build_rotation_matrices(self):
        """Build Rzyx and R_inv (tilt rotation) matrices for all tilts."""
        n = self.n_tilts
        rzyx_mats = np.empty((n, 3, 3))
        tilt_rots = np.empty((n, 3, 3))

        for i in range(n):
            Rx = R.from_rotvec(np.deg2rad(self.x_tilts[i]) * np.array([1, 0, 0]))
            Ry = R.from_rotvec(np.deg2rad(self.y_tilts[i]) * np.array([0, 1, 0]))
            Rz = R.from_rotvec(np.deg2rad(self.z_rots[i]) * np.array([0, 0, 1]))

            Rzyx = Rz * Ry * Rx
            rzyx_mats[i] = Rzyx.as_matrix()
            tilt_rots[i] = Rzyx.inv().as_matrix()

        return rzyx_mats, tilt_rots

    # -- public API --

    def local_defocus(self, i_tilt, point_3d):
        """Compute depth-corrected defocus at a 3D particle position."""
        depth_offset = self._rzyx_matrices[i_tilt, 2, :] @ point_3d * self.hand
        return (
            self.defocus_u[i_tilt] + depth_offset,
            self.defocus_v[i_tilt] + depth_offset,
            self.defocus_angle[i_tilt],
        )

    def expand_particles_batch(self, points_3d, image_names, tilt_df, group_names, base_orientations, random_subsets):
        """Expand M particles sharing this Tomogram into 2D rows.

        All particles must share the same visible-frame set (same Tomogram).

        Parameters
        ----------
        points_3d : ndarray (M, 3)
        image_names : list[str] of length M
        tilt_df : DataFrame with n_tilts rows
        group_names : list[str] of length M
        base_orientations : Rotation batch of size M, or None
        random_subsets : ndarray (M,)

        Returns
        -------
        DataFrame with M * n_tilts rows.
        """
        M = len(points_3d)
        n = self.n_tilts

        # --- Defocus: depth = points_3d @ depth_rows.T * hand -> (M, n) ---
        depth_all = (points_3d @ self._depth_rows.T) * self.hand  # (M, n)
        dfu_all = self.defocus_u[None, :] + depth_all  # (M, n)
        dfv_all = self.defocus_v[None, :] + depth_all
        dfa_all = np.broadcast_to(self.defocus_angle[None, :], (M, n))

        # --- Euler angles ---
        if base_orientations is not None:
            base_mats = base_orientations.as_matrix()
            if base_mats.ndim == 2:
                base_mats = base_mats[None, :, :]  # single rotation -> (1, 3, 3)
            # (M, 3, 3) @ (n, 3, 3) -> (M, n, 3, 3)
            final_mats = np.einsum("mij,njk->mnik", base_mats, self._tilt_rots)
            euler_all = R.from_matrix(final_mats.reshape(M * n, 3, 3)).as_euler("ZYZ", degrees=True).reshape(M, n, 3)
        else:
            euler_all = np.zeros((M, n, 3))

        # --- Tilt-level metadata (same for all particles sharing this Tomogram) ---
        mic_names = tilt_df["_rlnMicrographName"].values
        pre_exposure = tilt_df["_rlnMicrographPreExposure"].values.astype(float)
        ctf_scale = tilt_df["_rlnCtfScalefactor"].values.astype(float)
        stage_tilt = tilt_df["_rlnTomoNominalStageTiltAngle"].values.astype(float)

        # --- Build flat arrays for the output DataFrame ---
        total = M * n

        img_name_flat = np.empty(total, dtype=object)
        group_flat = np.empty(total, dtype=object)
        mic_flat = np.tile(mic_names, M)
        pre_exp_flat = np.tile(pre_exposure, M)
        ctf_scale_flat = np.tile(ctf_scale, M)
        stage_tilt_flat = np.tile(stage_tilt, M)
        subset_flat = np.repeat(random_subsets, n)

        for j in range(M):
            start = j * n
            for t in range(n):
                img_name_flat[start + t] = f"{t + 1:06d}@{image_names[j]}"
            group_flat[start : start + n] = group_names[j]

        return pd.DataFrame(
            {
                "_rlnDefocusU": dfu_all.ravel(),
                "_rlnDefocusV": dfv_all.ravel(),
                "_rlnDefocusAngle": dfa_all.ravel(),
                "_rlnImageName": img_name_flat,
                "_rlnMicrographName": mic_flat,
                "_rlnCtfScalefactor": ctf_scale_flat,
                "_rlnGroupName": group_flat,
                "_rlnAngleRot": euler_all[:, :, 0].ravel(),
                "_rlnAngleTilt": euler_all[:, :, 1].ravel(),
                "_rlnAnglePsi": euler_all[:, :, 2].ravel(),
                "_rlnOriginXAngst": 0.0,
                "_rlnOriginYAngst": 0.0,
                "_rlnRandomSubset": subset_flat,
                "_rlnMicrographPreExposure": pre_exp_flat,
                "_rlnTomoNominalStageTiltAngle": stage_tilt_flat,
            }
        )


# ---------------------------------------------------------------------------
# Star file reading helpers
# ---------------------------------------------------------------------------


def _read_star_all_blocks(filepath):
    """Read a RELION5 star file returning all named data blocks."""
    return starfile.read_star(filepath)


def _parse_visible_frames(frames_str):
    """Parse rlnTomoVisibleFrames string like '[1,1,0,1,...]' to indices."""
    frames = literal_eval(frames_str)
    return [i for i, v in enumerate(frames) if v == 1]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(tomograms_path, particles_path, output_path):
    """Convert RELION5 tomo data to RECOVAR 2D tilt format."""

    tomo_dir = os.path.dirname(os.path.abspath(tomograms_path))
    project_root = tomo_dir

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
    sample_ts_path = tomo_df["_rlnTomoTiltSeriesStarFile"].values[0]
    if os.path.isfile(os.path.join(tomo_dir, sample_ts_path)):
        project_root = tomo_dir
    else:
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

        tomo_row_mask = tomo_df["_rlnTomoName"].values == tomo_name
        if not np.any(tomo_row_mask):
            raise ValueError(f"Tomogram '{tomo_name}' not found in tomograms.star")
        tomo_idx = np.where(tomo_row_mask)[0][0]

        ts_star_relpath = tomo_df["_rlnTomoTiltSeriesStarFile"].values[tomo_idx]
        ts_star_path = os.path.join(project_root, ts_star_relpath)

        logger.info("Loading tilt-series: %s -> %s", tomo_name, ts_star_path)
        ts_df, _ = _read_star_all_blocks(ts_star_path)

        hand_val = tomo_df["_rlnTomoHand"].values[tomo_idx]
        hand = -1 if float(hand_val) == -1 else 1
        pixel_size = float(tomo_df["_rlnTomoTiltSeriesPixelSize"].values[tomo_idx])

        tilt_series_cache[tomo_name] = (ts_df, hand, pixel_size)
        return ts_df, hand, pixel_size

    # ---- Get optics info ----
    voltage = float(optics_df["_rlnVoltage"].values[0])
    cs = float(optics_df["_rlnSphericalAberration"].values[0])
    amp_contrast = float(optics_df["_rlnAmplitudeContrast"].values[0])
    angpix = float(optics_df["_rlnImagePixelSize"].values[0])
    image_size = int(optics_df["_rlnImageSize"].values[0])

    # ---- Group particles by (tomo_name, visible_frames) for batching ----
    n_particles = len(particles_df)
    has_subtomo_rot = "_rlnTomoSubtomogramRot" in particles_df.columns

    groups = {}  # (tomo_name, visible_frames_tuple) -> list of particle indices
    for idx in range(n_particles):
        tomo_name = particles_df["_rlnTomoName"].values[idx]
        frames_str = particles_df["_rlnTomoVisibleFrames"].values[idx]
        visible_indices = tuple(_parse_visible_frames(frames_str))
        key = (tomo_name, visible_indices)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    logger.info("Grouped %d particles into %d unique (tomo, visible_frames) groups", n_particles, len(groups))

    # ---- Process each group as a batch ----
    all_rows = []
    particles_processed = 0

    for (tomo_name, visible_indices), particle_indices in groups.items():
        ts_df, hand, pixel_size = _get_tilt_series(tomo_name)
        sub_ts_df = ts_df.iloc[list(visible_indices)].copy()

        # Build Tomogram once for this group
        tomogram = Tomogram(
            pixel_size,
            defocus_u=sub_ts_df["_rlnDefocusU"].values.astype(float),
            defocus_v=sub_ts_df["_rlnDefocusV"].values.astype(float),
            defocus_angle=sub_ts_df["_rlnDefocusAngle"].values.astype(float),
            x_tilts=sub_ts_df["_rlnTomoXTilt"].values.astype(float),
            y_tilts=sub_ts_df["_rlnTomoYTilt"].values.astype(float),
            z_rots=sub_ts_df["_rlnTomoZRot"].values.astype(float),
            x_shifts=sub_ts_df["_rlnTomoXShiftAngst"].values.astype(float),
            y_shifts=sub_ts_df["_rlnTomoYShiftAngst"].values.astype(float),
            hand=hand,
        )

        M = len(particle_indices)
        idxs = np.array(particle_indices)

        # Gather per-particle data
        points_3d = np.column_stack(
            [
                particles_df["_rlnCenteredCoordinateXAngst"].values[idxs].astype(float),
                particles_df["_rlnCenteredCoordinateYAngst"].values[idxs].astype(float),
                particles_df["_rlnCenteredCoordinateZAngst"].values[idxs].astype(float),
            ]
        )

        image_names = particles_df["_rlnImageName"].values[idxs].tolist()
        group_names = particles_df["_rlnTomoParticleName"].values[idxs].tolist()
        random_subsets = particles_df["_rlnRandomSubset"].values[idxs].astype(int)

        # Build batched Rotation objects
        rots = particles_df["_rlnAngleRot"].values[idxs].astype(float)
        tilts = particles_df["_rlnAngleTilt"].values[idxs].astype(float)
        psis = particles_df["_rlnAnglePsi"].values[idxs].astype(float)
        R_particles = R.from_euler("ZYZ", np.column_stack([rots, tilts, psis]), degrees=True)

        if has_subtomo_rot:
            srots = particles_df["_rlnTomoSubtomogramRot"].values[idxs].astype(float)
            stilts = particles_df["_rlnTomoSubtomogramTilt"].values[idxs].astype(float)
            spsis = particles_df["_rlnTomoSubtomogramPsi"].values[idxs].astype(float)
            R_subtomo = R.from_euler("ZYZ", np.column_stack([srots, stilts, spsis]), degrees=True)
        else:
            R_subtomo = R.identity() if M == 1 else R.from_matrix(np.broadcast_to(np.eye(3), (M, 3, 3)).copy())

        R_base = R_particles * R_subtomo

        df_2d = tomogram.expand_particles_batch(
            points_3d,
            image_names,
            sub_ts_df,
            group_names,
            R_base,
            random_subsets,
        )

        all_rows.append(df_2d)

        particles_processed += M
        if particles_processed % 500 < M or particles_processed == n_particles:
            logger.info("Processing particle %d / %d", min(particles_processed, n_particles), n_particles)

    # ---- Concatenate and sort by dose within each group ----
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df["_rlnMicrographPreExposure"] = final_df["_rlnMicrographPreExposure"].astype(float)
    final_df = final_df.sort_values(
        ["_rlnGroupName", "_rlnMicrographPreExposure"],
        kind="mergesort",
    ).reset_index(drop=True)

    logger.info(
        "Total 2D rows: %d (%d particles x ~%d tilts)", len(final_df), n_particles, len(final_df) // max(n_particles, 1)
    )

    # ---- Build output optics table ----
    out_optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": ["1"],
            "_rlnOpticsGroupName": ["opticsGroup1"],
            "_rlnImagePixelSize": [str(angpix)],
            "_rlnImageSize": [str(image_size)],
            "_rlnVoltage": [str(voltage)],
            "_rlnSphericalAberration": [str(cs)],
            "_rlnAmplitudeContrast": [str(amp_contrast)],
            "_rlnImageDimensionality": ["2"],
        }
    )

    final_df["_rlnOpticsGroup"] = "1"

    # ---- Write output ----
    starfile.write_star(output_path, final_df, data_optics=out_optics)
    logger.info("Wrote %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_args(parser):
    parser.add_argument(
        "-t",
        "--tomograms",
        required=True,
        help="RELION5 tomograms.star (from Polish or Tomograms job)",
    )
    parser.add_argument(
        "-p",
        "--particles",
        required=True,
        help="RELION5 particles.star (from Extract or Refine job)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="particles_2d.star",
        help="Output 2D STAR file (default: particles_2d.star)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
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

    convert(args.tomograms, args.particles, args.output)


if __name__ == "__main__":
    main()
