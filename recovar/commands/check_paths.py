"""Diagnose path resolution for .star and .cs particle files.

Usage::

    recovar check_paths particles.star
    recovar check_paths particles.cs --datadir /path/to/project
    recovar check_paths particles.star --strip-prefix Extract/job193 --datadir /data

Prints a summary of how paths resolve without running the full pipeline.
"""

import argparse
import logging
import os
import sys
import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Check path resolution for .star/.cs particle files")
    parser.add_argument("particles", help="Input particles file (.star or .cs)")
    parser.add_argument("--datadir", default=None, help="Base directory for resolving relative paths")
    parser.add_argument("--strip-prefix", default=None, help="Prefix to strip from paths in metadata")
    parser.add_argument("--show", type=int, default=10, help="Number of sample paths to show (default: 10)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    filepath = args.particles
    if not os.path.isfile(filepath):
        logger.error("File not found: %s", filepath)
        sys.exit(1)

    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "star":
        _check_star(filepath, args.datadir, args.strip_prefix, args.show)
    elif ext == "cs":
        _check_cs(filepath, args.datadir, args.strip_prefix, args.show)
    else:
        logger.error("Unsupported format: .%s (expected .star or .cs)", ext)
        sys.exit(1)


def _check_star(filepath, datadir, strip_prefix, n_show):
    from recovar.data_io.starfile import StarFile

    star = StarFile.load(filepath)
    df = star.df.copy()

    if "_rlnImageName" not in df.columns:
        logger.error("STAR file missing _rlnImageName column")
        return

    image_names = df["_rlnImageName"].astype(str)
    parts = image_names.str.split("@", n=1, expand=True)
    if parts.shape[1] < 2:
        logger.error("_rlnImageName entries not in '<index>@<path>' format")
        return
    raw_paths = parts[1].tolist()

    _check_paths("STAR", filepath, raw_paths, datadir, strip_prefix, n_show)


def _check_cs(filepath, datadir, strip_prefix, n_show):
    cs_data = np.load(filepath)

    if "blob/path" not in cs_data.dtype.names:
        logger.error("CS file missing blob/path field")
        return

    raw_paths = []
    for p in cs_data["blob/path"]:
        if isinstance(p, (bytes, np.bytes_)):
            p = p.decode("utf-8", errors="replace")
        else:
            p = str(p)
        raw_paths.append(p.lstrip(">"))

    _check_paths("CS", filepath, raw_paths, datadir, strip_prefix, n_show)


def _check_paths(fmt, filepath, raw_paths, datadir, strip_prefix, n_show):
    from recovar.data_io.image_loader import _resolve_mrc_path

    unique_raw = sorted(set(raw_paths))
    logger.info("")
    logger.info("=" * 60)
    logger.info("  File: %s", filepath)
    logger.info("  Format: %s", fmt)
    logger.info("  Total image references: %d", len(raw_paths))
    logger.info("  Unique MRC paths in metadata: %d", len(unique_raw))
    logger.info("=" * 60)

    # Show sample raw paths
    logger.info("")
    logger.info("Raw paths from metadata (first %d):", min(n_show, len(unique_raw)))
    for p in unique_raw[:n_show]:
        logger.info("  %s", p)
    if len(unique_raw) > n_show:
        logger.info("  ... and %d more", len(unique_raw) - n_show)

    # Apply strip-prefix
    if strip_prefix:
        logger.info("")
        logger.info("Applying --strip-prefix '%s':", strip_prefix)
        stripped = []
        matched = 0
        for p in unique_raw:
            if p.startswith(strip_prefix):
                s = p[len(strip_prefix) :].lstrip("/")
                stripped.append(s)
                matched += 1
            else:
                stripped.append(p)
        logger.info("  %d/%d paths matched the prefix", matched, len(unique_raw))
        if matched > 0:
            logger.info("  Example: %s -> %s", unique_raw[0], stripped[0])
        unique_raw = stripped

    # Resolve datadir (consistent with both StarLoader and CryoSparcLoader)
    if not datadir:
        datadir = os.path.abspath(os.path.dirname(filepath))
        logger.info("")
        logger.info("No --datadir specified, using metadata file directory:")
        logger.info("  %s", datadir)
    else:
        datadir = os.path.abspath(datadir)
        logger.info("")
        logger.info("Using --datadir: %s", datadir)
        if not os.path.isdir(datadir):
            logger.warning("  directory does not exist!")

    # Resolve all unique paths
    # Resolve all paths once and cache results
    resolved_cache = {}
    for raw_p in unique_raw:
        candidate = os.path.join(datadir, raw_p)
        resolved_cache[raw_p] = (candidate, _resolve_mrc_path(candidate))

    logger.info("")
    logger.info("Path resolution:")
    total_found = 0
    total_fallback = 0
    missing_paths = []

    for raw_p in unique_raw:
        candidate, resolved = resolved_cache[raw_p]
        exists = os.path.isfile(resolved)

        if exists:
            total_found += 1
            if resolved != candidate:
                total_fallback += 1
        else:
            missing_paths.append((raw_p, candidate))

    # Show details for a subset
    for raw_p in unique_raw[:n_show]:
        candidate, resolved = resolved_cache[raw_p]
        exists = os.path.isfile(resolved)

        if exists and resolved == candidate:
            logger.info("  OK     %s", raw_p)
        elif exists:
            logger.info("  OK(*)  %s", raw_p)
            logger.info("         -> %s", resolved)
        else:
            logger.info("  MISS   %s", raw_p)
            logger.info("         tried: %s", candidate)

    if len(unique_raw) > n_show:
        logger.info("  ... (%d more paths not shown)", len(unique_raw) - n_show)

    # Summary
    logger.info("")
    logger.info("%s", "\u2500" * 60)
    logger.info("  Summary:")
    logger.info("    Found:            %d/%d", total_found, len(unique_raw))
    if total_fallback > 0:
        logger.info("    Via fallback (*): %d", total_fallback)
    logger.info("    Missing:          %d/%d", len(missing_paths), len(unique_raw))

    if total_found == len(unique_raw):
        logger.info("")
        logger.info("  All paths resolved successfully!")
        if total_fallback > 0:
            logger.info("  (%d used automatic fallbacks: extension swap or basename match)", total_fallback)
    elif missing_paths:
        logger.info("")
        logger.info("  Some paths could not be resolved. Suggestions:")
        sample_basename = os.path.basename(missing_paths[0][0])
        logger.info("    - Find where '%s' lives on this system:", sample_basename)
        logger.info("      find /path/to/data -name '%s'", sample_basename)
        logger.info("    - Then re-run with --datadir pointing to that directory")
        if "/" in missing_paths[0][0]:
            prefix = missing_paths[0][0].rsplit("/", 1)[0]
            logger.info("    - You may also need: --strip-prefix %s", prefix)
    logger.info("")
