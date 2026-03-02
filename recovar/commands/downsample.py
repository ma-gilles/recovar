"""Downsample particle images to a new box size and save to disk.

This replaces the need for ``cryodrgn downsample``.  Output is a single
``.mrcs`` stack plus an updated ``.star`` file with correct pixel size and
image references.

Usage::

    recovar downsample particles.star -D 128 -o downsampled/
    recovar downsample particles.cs   -D 128 -o downsampled/
    recovar downsample particles.mrcs -D 128 -o downsampled/

The output ``downsampled/particles.D.mrcs`` and ``downsampled/particles.D.star``
can then be passed directly to ``recovar pipeline``.
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import mrcfile
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def downsample_to_disk(
    particles_file: str,
    target_D: int,
    outdir: str,
    datadir: str = "",
    strip_prefix: Optional[str] = None,
    batch_size: int = 1000,
) -> Tuple[str, str]:
    """Pre-downsample particle images to disk via Fourier cropping.

    Reads images from *particles_file*, Fourier-crops each to *target_D*,
    and writes a new ``.mrcs`` stack plus an updated ``.star`` file.

    Parameters
    ----------
    particles_file : str
        Input particles (.mrcs, .star, .cs, or .txt).
    target_D : int
        Target box size in pixels (must be even).
    outdir : str
        Output directory (created if it does not exist).
    datadir : str
        Path prefix for resolving relative image paths.
    strip_prefix : str or None
        Prefix to strip from paths in star/cs file.
    batch_size : int
        Number of images per batch.

    Returns
    -------
    (mrcs_path, star_path) : tuple of str
        Paths to the output ``.mrcs`` stack and ``.star`` metadata file.
    """
    from recovar.data_io.downsample import downsample_images
    from recovar.data_io.image_loader import load_images

    if target_D % 2 != 0:
        raise ValueError(f"Target box size must be even, got {target_D}")

    os.makedirs(outdir, exist_ok=True)

    # ── Load image source ────────────────────────────────────────────────
    loader = load_images(
        particles_file,
        datadir=datadir,
        lazy=True,
        strip_prefix=strip_prefix,
    )

    n_images = loader.num_images
    orig_D = loader.image_size
    logger.info("Loaded %d images at %d\u00d7%d", n_images, orig_D, orig_D)

    if target_D > orig_D:
        raise ValueError(
            f"Target D ({target_D}) > original D ({orig_D})"
        )
    if target_D == orig_D:
        logger.info("Target D == original D (%d), nothing to do.", orig_D)

    # ── Determine pixel size ─────────────────────────────────────────────
    orig_apix = _get_pixel_size(particles_file)
    if orig_apix is None:
        raise ValueError(
            f"Could not determine pixel size from input file: {particles_file}. "
            "Ensure the STAR file has an optics table with rlnImagePixelSize, "
            "the CS file has blob/psize_A, or the MRC file has a valid voxel size."
        )
    new_apix = orig_apix * orig_D / target_D
    logger.info("Pixel size: %.4f \u00c5 \u2192 %.4f \u00c5", orig_apix, new_apix)

    # ── Fourier-crop and write ───────────────────────────────────────────
    out_mrcs = os.path.join(outdir, f"particles.{target_D}.mrcs")

    logger.info("Writing downsampled stack to %s", out_mrcs)
    t0 = time.time()

    with mrcfile.new_mmap(
        out_mrcs,
        shape=(n_images, target_D, target_D),
        mrc_mode=2,  # float32
        overwrite=True,
    ) as mrc:
        if new_apix:
            mrc.voxel_size = new_apix

        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            indices = np.arange(start, end)

            images = loader.get(indices)
            images_ds = downsample_images(images, target_D)
            mrc.data[start:end] = images_ds.astype(np.float32)

            elapsed = time.time() - t0
            rate = end / elapsed
            eta = (n_images - end) / rate if rate > 0 else 0
            logger.info(
                "  %d / %d  (%.1f img/s, ETA %.0fs)",
                end, n_images, rate, eta,
            )

    elapsed = time.time() - t0
    logger.info("Done: %.1f seconds (%.1f images/sec)", elapsed, n_images / elapsed)

    # ── Write updated STAR file ──────────────────────────────────────────
    out_star = os.path.join(outdir, f"particles.{target_D}.star")
    _write_output_star(particles_file, out_mrcs, out_star, target_D, new_apix, n_images)

    logger.info("Output files:")
    logger.info("  Stack: %s", out_mrcs)
    logger.info("  Star:  %s", out_star)

    return out_mrcs, out_star


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Downsample particle images via Fourier cropping and save to disk."
    )
    parser.add_argument("particles", help="Input particles (.mrcs, .star, .cs, or .txt)")
    parser.add_argument("-D", "--target-D", type=int, required=True,
                        help="Target box size in pixels (must be even)")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Output directory")
    parser.add_argument("--datadir", default=None,
                        help="Path prefix for resolving relative image paths")
    parser.add_argument("--strip-prefix", default=None,
                        help="Prefix to strip from paths in star file")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of images per batch (default: 1000)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Split output into chunks of this many images "
                             "(default: single file)")

    args = parser.parse_args()

    # Set up logging for CLI usage
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "downsample.log")
    from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            RobustStreamHandler(),
            RobustFileHandler(log_path),
        ],
    )

    if args.target_D % 2 != 0:
        logger.error("Target box size must be even, got %d", args.target_D)
        sys.exit(1)

    logger.info("Downsampling %s \u2192 D=%d", args.particles, args.target_D)
    logger.info("Output directory: %s", args.outdir)

    mrcs_path, star_path = downsample_to_disk(
        particles_file=args.particles,
        target_D=args.target_D,
        outdir=args.outdir,
        datadir=args.datadir or "",
        strip_prefix=args.strip_prefix,
        batch_size=args.batch_size,
    )

    logger.info("")
    logger.info("To run the pipeline:")
    logger.info("  recovar pipeline %s -o output --mask mask.mrc", star_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pixel_size(filepath: str):
    """Try to extract pixel size from the input file."""
    ext = filepath.rsplit('.', 1)[-1].lower()

    if ext == 'star':
        try:
            from recovar.data_io.starfile import StarFile
            sf = StarFile.load(filepath)
            apix = sf.apix
            if apix is not None and len(apix) > 0:
                return float(apix[0])
        except (ImportError, KeyError, ValueError, OSError):
            pass

    elif ext == 'cs':
        try:
            data = np.load(filepath, allow_pickle=True)
            if 'blob/psize_A' in data.dtype.names:
                return float(data['blob/psize_A'][0])
        except (ValueError, KeyError, OSError, IndexError):
            pass

    elif ext in ('mrcs', 'mrc'):
        try:
            with mrcfile.open(filepath, mode='r', header_only=True) as mrc:
                vsize = float(mrc.voxel_size.x)
                if vsize > 0:
                    return vsize
        except (ValueError, OSError):
            pass

    return None


def _write_output_star(input_path, mrcs_path, star_path, target_D, new_apix, n_images):
    """Write a STAR file referencing the new downsampled stack.

    If the input was a STAR file, copy pose/CTF metadata with updated pixel size.
    If the input was a CS file, convert metadata to RELION STAR format.
    Otherwise, write a minimal file with just image references.
    """
    import pandas as pd

    ext = input_path.rsplit('.', 1)[-1].lower()

    # Relative path to mrcs from star file location
    star_dir = os.path.dirname(os.path.abspath(star_path))
    mrcs_abs = os.path.abspath(mrcs_path)
    mrcs_rel = os.path.relpath(mrcs_abs, star_dir)

    if ext == 'star':
        # Copy metadata from input star, update pixel size and image names.
        # Use our own cached read_star (fast) rather than the external starfile package.
        try:
            from recovar.data_io.starfile import read_star, write_star
            import pandas as pd

            particles_df, optics_df = read_star(input_path)

            if optics_df is not None:
                if '_rlnImagePixelSize' in optics_df.columns:
                    optics_df['_rlnImagePixelSize'] = str(new_apix)
                if '_rlnImageSize' in optics_df.columns:
                    optics_df['_rlnImageSize'] = str(target_D)

            if '_rlnImageName' in particles_df.columns:
                particles_df['_rlnImageName'] = [
                    f"{i+1}@{mrcs_rel}" for i in range(len(particles_df))
                ]

            write_star(star_path, particles_df, optics_df)
            logger.info("Wrote STAR file with full metadata from input")
            return
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Failed to copy STAR metadata: %s. Writing minimal STAR.", e)

    elif ext == 'cs':
        # Convert cryoSPARC metadata to RELION STAR format
        try:
            _write_star_from_cs(input_path, star_path, mrcs_rel, target_D,
                                new_apix, n_images)
            return
        except (ImportError, KeyError, ValueError, OSError) as e:
            logger.warning("Failed to convert CS metadata: %s. Writing minimal STAR.", e)

    # Fallback: write minimal STAR file
    _write_minimal_star(star_path, mrcs_rel, target_D, new_apix, n_images)


def _write_star_from_cs(cs_path, star_path, mrcs_rel, target_D, new_apix, n_images):
    """Convert cryoSPARC .cs metadata to a RELION 3.1 STAR file."""
    from scipy.spatial.transform import Rotation as SciPyRot
    import starfile
    import pandas as pd

    data = np.load(cs_path, allow_pickle=True)

    # --- Optics table ---
    # Get microscope params from first particle
    volt = float(data['ctf/accel_kv'][0]) if 'ctf/accel_kv' in data.dtype.names else 300.0
    cs_mm = float(data['ctf/cs_mm'][0]) if 'ctf/cs_mm' in data.dtype.names else 2.7
    amp = float(data['ctf/amp_contrast'][0]) if 'ctf/amp_contrast' in data.dtype.names else 0.1

    optics = pd.DataFrame({
        'rlnOpticsGroup': [1],
        'rlnOpticsGroupName': ['opticsGroup1'],
        'rlnImagePixelSize': [new_apix or 1.0],
        'rlnImageSize': [target_D],
        'rlnImageDimensionality': [2],
        'rlnVoltage': [volt],
        'rlnSphericalAberration': [cs_mm],
        'rlnAmplitudeContrast': [amp],
    })

    # --- Particles table ---
    particles = pd.DataFrame({
        'rlnImageName': [f"{i+1}@{mrcs_rel}" for i in range(n_images)],
        'rlnOpticsGroup': np.ones(n_images, dtype=int),
    })

    # CTF parameters
    if 'ctf/df1_A' in data.dtype.names:
        particles['rlnDefocusU'] = data['ctf/df1_A'][:n_images].astype(np.float64)
        particles['rlnDefocusV'] = data['ctf/df2_A'][:n_images].astype(np.float64)
        particles['rlnDefocusAngle'] = np.degrees(
            data['ctf/df_angle_rad'][:n_images].astype(np.float64)
        )
    if 'ctf/phase_shift_rad' in data.dtype.names:
        particles['rlnPhaseShift'] = np.degrees(
            data['ctf/phase_shift_rad'][:n_images].astype(np.float64)
        )

    # Poses (Rodrigues → RELION Euler angles)
    pose_key = None
    for key in ('alignments3D/pose', 'alignments_class_0/pose'):
        if key in data.dtype.names:
            pose_key = key
            break

    if pose_key is not None:
        rotvecs = data[pose_key][:n_images].astype(np.float64)
        # CS → internal rotation matrices (transpose to match convention)
        R_int = SciPyRot.from_rotvec(rotvecs).as_matrix().transpose(0, 2, 1)
        # Internal → RELION Euler angles (invert R_from_relion)
        frame_adjust = np.array(
            [[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64
        )
        R_adj = R_int * frame_adjust
        angles = SciPyRot.from_matrix(R_adj).as_euler('zxz', degrees=True)
        angles[:, 0] -= 90.0
        angles[:, 2] += 90.0

        particles['rlnAngleRot'] = angles[:, 0]
        particles['rlnAngleTilt'] = angles[:, 1]
        particles['rlnAnglePsi'] = angles[:, 2]

    # Translations (pixels → Angstroms)
    shift_key = None
    for key in ('alignments3D/shift', 'alignments_class_0/shift'):
        if key in data.dtype.names:
            shift_key = key
            break

    if shift_key is not None:
        orig_apix = float(data['blob/psize_A'][0]) if 'blob/psize_A' in data.dtype.names else 1.0
        shifts_px = data[shift_key][:n_images].astype(np.float64)
        shifts_angst = shifts_px * orig_apix
        particles['rlnOriginXAngst'] = shifts_angst[:, 0]
        particles['rlnOriginYAngst'] = shifts_angst[:, 1]

    # Half-set assignments
    split_key = None
    for key in ('alignments3D/split', 'alignments_class_0/split'):
        if key in data.dtype.names:
            split_key = key
            break

    if split_key is not None:
        splits = data[split_key][:n_images].astype(int)
        # CS uses 0/1, RELION uses 1/2
        particles['rlnRandomSubset'] = splits + 1

    starfile.write({'optics': optics, 'particles': particles}, star_path, overwrite=True)
    logger.info("Wrote STAR file with metadata converted from cryoSPARC .cs")


def _write_minimal_star(star_path, mrcs_rel, target_D, new_apix, n_images):
    """Write a minimal RELION 3.1 STAR file."""
    import starfile
    import pandas as pd

    optics = pd.DataFrame({
        'rlnOpticsGroup': [1],
        'rlnOpticsGroupName': ['opticsGroup1'],
        'rlnImagePixelSize': [new_apix or 1.0],
        'rlnImageSize': [target_D],
        'rlnImageDimensionality': [2],
        'rlnVoltage': [300.0],
        'rlnAmplitudeContrast': [0.1],
        'rlnSphericalAberration': [2.7],
    })

    particles = pd.DataFrame({
        'rlnImageName': [f"{i+1}@{mrcs_rel}" for i in range(n_images)],
        'rlnOpticsGroup': np.ones(n_images, dtype=int),
    })

    starfile.write({'optics': optics, 'particles': particles}, star_path, overwrite=True)
    logger.info("Wrote minimal STAR file (no poses/CTF \u2014 use original file for pipeline)")


if __name__ == "__main__":
    main()
