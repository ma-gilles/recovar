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
    parser = argparse.ArgumentParser(
        description="Check path resolution for .star/.cs particle files"
    )
    parser.add_argument("particles", help="Input particles file (.star or .cs)")
    parser.add_argument("--datadir", default=None,
                        help="Base directory for resolving relative paths")
    parser.add_argument("--strip-prefix", default=None,
                        help="Prefix to strip from paths in metadata")
    parser.add_argument("--show", type=int, default=10,
                        help="Number of sample paths to show (default: 10)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    filepath = args.particles
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    ext = filepath.rsplit('.', 1)[-1].lower()

    if ext == 'star':
        _check_star(filepath, args.datadir, args.strip_prefix, args.show)
    elif ext == 'cs':
        _check_cs(filepath, args.datadir, args.strip_prefix, args.show)
    else:
        print(f"Unsupported format: .{ext} (expected .star or .cs)")
        sys.exit(1)


def _check_star(filepath, datadir, strip_prefix, n_show):
    from recovar.data_io.starfile import StarFile

    star = StarFile.load(filepath)
    df = star.df.copy()

    if '_rlnImageName' not in df.columns:
        print("ERROR: STAR file missing _rlnImageName column")
        return

    image_names = df["_rlnImageName"].astype(str)
    parts = image_names.str.split('@', n=1, expand=True)
    if parts.shape[1] < 2:
        print("ERROR: _rlnImageName entries not in '<index>@<path>' format")
        return
    raw_paths = parts[1].tolist()

    _check_paths("STAR", filepath, raw_paths, datadir, strip_prefix, n_show)


def _check_cs(filepath, datadir, strip_prefix, n_show):
    cs_data = np.load(filepath)

    if 'blob/path' not in cs_data.dtype.names:
        print("ERROR: CS file missing blob/path field")
        return

    raw_paths = []
    for p in cs_data['blob/path']:
        if isinstance(p, (bytes, np.bytes_)):
            p = p.decode("utf-8", errors="replace")
        else:
            p = str(p)
        raw_paths.append(p.lstrip('>'))

    _check_paths("CS", filepath, raw_paths, datadir, strip_prefix, n_show)


def _check_paths(fmt, filepath, raw_paths, datadir, strip_prefix, n_show):
    from recovar.data_io.image_loader import _resolve_mrc_path

    unique_raw = sorted(set(raw_paths))
    print(f"\n{'=' * 60}")
    print(f"  File: {filepath}")
    print(f"  Format: {fmt}")
    print(f"  Total image references: {len(raw_paths)}")
    print(f"  Unique MRC paths in metadata: {len(unique_raw)}")
    print(f"{'=' * 60}")

    # Show sample raw paths
    print(f"\nRaw paths from metadata (first {min(n_show, len(unique_raw))}):")
    for p in unique_raw[:n_show]:
        print(f"  {p}")
    if len(unique_raw) > n_show:
        print(f"  ... and {len(unique_raw) - n_show} more")

    # Apply strip-prefix
    if strip_prefix:
        print(f"\nApplying --strip-prefix '{strip_prefix}':")
        stripped = []
        matched = 0
        for p in unique_raw:
            if p.startswith(strip_prefix):
                s = p[len(strip_prefix):].lstrip('/')
                stripped.append(s)
                matched += 1
            else:
                stripped.append(p)
        print(f"  {matched}/{len(unique_raw)} paths matched the prefix")
        if matched > 0:
            print(f"  Example: {unique_raw[0]} -> {stripped[0]}")
        unique_raw = stripped

    # Resolve datadir (consistent with both StarLoader and CryoSparcLoader)
    if not datadir:
        datadir = os.path.abspath(os.path.dirname(filepath))
        print(f"\nNo --datadir specified, using metadata file directory:")
        print(f"  {datadir}")
    else:
        datadir = os.path.abspath(datadir)
        print(f"\nUsing --datadir: {datadir}")
        if not os.path.isdir(datadir):
            print(f"  WARNING: directory does not exist!")

    # Resolve all unique paths
    # Resolve all paths once and cache results
    resolved_cache = {}
    for raw_p in unique_raw:
        candidate = os.path.join(datadir, raw_p)
        resolved_cache[raw_p] = (candidate, _resolve_mrc_path(candidate))

    print(f"\nPath resolution:")
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
            print(f"  OK     {raw_p}")
        elif exists:
            print(f"  OK(*)  {raw_p}")
            print(f"         -> {resolved}")
        else:
            print(f"  MISS   {raw_p}")
            print(f"         tried: {candidate}")

    if len(unique_raw) > n_show:
        print(f"  ... ({len(unique_raw) - n_show} more paths not shown)")

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  Summary:")
    print(f"    Found:            {total_found}/{len(unique_raw)}")
    if total_fallback > 0:
        print(f"    Via fallback (*): {total_fallback}")
    print(f"    Missing:          {len(missing_paths)}/{len(unique_raw)}")

    if total_found == len(unique_raw):
        print(f"\n  All paths resolved successfully!")
        if total_fallback > 0:
            print(f"  ({total_fallback} used automatic fallbacks: extension swap or basename match)")
    elif missing_paths:
        print(f"\n  Some paths could not be resolved. Suggestions:")
        sample_basename = os.path.basename(missing_paths[0][0])
        print(f"    - Find where '{sample_basename}' lives on this system:")
        print(f"      find /path/to/data -name '{sample_basename}'")
        print(f"    - Then re-run with --datadir pointing to that directory")
        if '/' in missing_paths[0][0]:
            prefix = missing_paths[0][0].rsplit('/', 1)[0]
            print(f"    - You may also need: --strip-prefix {prefix}")
    print()
