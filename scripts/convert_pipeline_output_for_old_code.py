#!/usr/bin/env python
"""Convert a NEW-format pipeline output to OLD-format for ~/recovar compatibility.

The NEW code stores embeddings under:
    latent_coords, latent_precision, contrasts,
    latent_coords_noreg, latent_precision_noreg, contrasts_noreg

The OLD code expects:
    zs, cov_zs, contrasts

This script:
  1. Creates a shallow copy of the pipeline output (symlinks for large files).
  2. Patches model/embeddings.pkl to add OLD-format keys.
  3. Optionally verifies the converted output loads with OLD PipelineOutput.

Usage:
    python scripts/convert_pipeline_output_for_old_code.py \\
        /path/to/new_pipeline_output /path/to/converted_output

    # With verification against ~/recovar:
    python scripts/convert_pipeline_output_for_old_code.py \\
        /path/to/new_pipeline_output /path/to/converted_output --verify
"""

import argparse
import os
import pickle
import shutil
import sys


def convert_pipeline_output(src_dir, dst_dir, verify=False, old_recovar_path=None):
    """Convert NEW pipeline output to OLD format.

    Parameters
    ----------
    src_dir : str
        Path to NEW-format pipeline output directory.
    dst_dir : str
        Path to create the converted (OLD-compatible) output directory.
    verify : bool
        If True, verify the converted output loads with OLD PipelineOutput.
    old_recovar_path : str or None
        Path to old recovar repo (default: ~/recovar).
    """
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    # Symlink everything except model/embeddings.pkl (which we patch)
    embeddings_rel = os.path.join("model", "embeddings.pkl")

    for root, dirs, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel_root) if rel_root != "." else dst_dir

        for d in dirs:
            dst_d = os.path.join(dst_root, d)
            os.makedirs(dst_d, exist_ok=True)

        for f in files:
            rel_file = os.path.join(rel_root, f) if rel_root != "." else f
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_root, f)

            if rel_file == embeddings_rel:
                # Will be patched below
                continue

            # Symlink large files, copy small ones
            try:
                os.symlink(src_file, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)

    # Patch embeddings.pkl
    src_emb_path = os.path.join(src_dir, embeddings_rel)
    dst_emb_path = os.path.join(dst_dir, embeddings_rel)

    if not os.path.isfile(src_emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {src_emb_path}")

    with open(src_emb_path, "rb") as f:
        emb = pickle.load(f)

    # Map NEW → OLD keys.
    # CRITICAL: we must REMOVE the NEW keys after creating OLD ones, because
    # OLD PipelineOutput.load_embedding() iterates over ALL entries and applies
    # halfset filtering to each.  If both 'zs' and 'latent_coords' exist and
    # share the same arrays, the filtering is applied twice — corrupting the
    # particle ordering.  Solution: build OLD-only entries and delete NEW ones.
    key_map = {
        "latent_coords": "zs",
        "latent_precision": "cov_zs",
        # "contrasts" stays the same in both OLD and NEW
    }

    patched = False
    for new_key, old_key in key_map.items():
        if new_key in emb:
            if old_key not in emb:
                emb[old_key] = emb.pop(new_key)
                patched = True
            else:
                # OLD key already exists, just remove the NEW one to avoid duplication
                del emb[new_key]

    # Handle _noreg variants: OLD uses "{zdim}_noreg" keys inside zs/cov_zs/contrasts.
    # NEW has separate top-level entries: latent_coords_noreg, etc.
    # OLD code accesses po.get('zs')[f"{zdim}_noreg"]
    noreg_map = {
        "latent_coords_noreg": "zs",
        "latent_precision_noreg": "cov_zs",
        "contrasts_noreg": "contrasts",
    }
    for new_key, old_entry in noreg_map.items():
        if new_key in emb:
            noreg_data = emb.pop(new_key)  # remove NEW key
            if old_entry not in emb:
                emb[old_entry] = {}
            # Merge noreg zdims into old entry with "{zdim}_noreg" key format
            for zdim, data in noreg_data.items():
                noreg_key = f"{zdim}_noreg"
                if noreg_key not in emb[old_entry]:
                    emb[old_entry][noreg_key] = data
                    patched = True

    os.makedirs(os.path.dirname(dst_emb_path), exist_ok=True)
    with open(dst_emb_path, "wb") as f:
        pickle.dump(emb, f)

    if patched:
        print(f"Patched embeddings: added OLD-format keys to {dst_emb_path}")
    else:
        print(f"No patching needed (OLD keys already present or NEW keys absent)")

    print(f"Converted pipeline output: {dst_dir}")

    if verify:
        _verify_with_old_code(dst_dir, old_recovar_path)

    return dst_dir


def _verify_with_old_code(converted_dir, old_recovar_path=None):
    """Verify that the converted output loads with OLD PipelineOutput."""
    if old_recovar_path is None:
        old_recovar_path = os.path.expanduser("~/recovar")

    if not os.path.isdir(old_recovar_path):
        print(f"WARNING: Old recovar not found at {old_recovar_path}, skipping verification")
        return

    # Run verification in subprocess to avoid module conflicts
    import subprocess

    script = f"""
import sys
sys.path.insert(0, {old_recovar_path!r})
from recovar import output as o
po = o.PipelineOutput({converted_dir!r} + '/')
zs = po.get('zs')
print("Available zdims in zs:", list(zs.keys()))
for zdim in zs:
    print(f"  zs[{{zdim}}]: shape={{zs[zdim].shape}}")
cov_zs = po.get('cov_zs')
print("Available zdims in cov_zs:", list(cov_zs.keys()))
contrasts = po.get('contrasts')
print("Available zdims in contrasts:", list(contrasts.keys()))
print("VERIFY_OK")
"""
    python = sys.executable
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode != 0:
        print(f"Verification FAILED:\n{result.stderr}")
        raise RuntimeError("OLD PipelineOutput failed to load converted output")
    if "VERIFY_OK" not in result.stdout:
        print(f"Verification output:\n{result.stdout}")
        raise RuntimeError("OLD PipelineOutput verification did not complete")
    print(f"Verification OK:\n{result.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_dir", help="NEW-format pipeline output directory")
    parser.add_argument("dst_dir", help="Output directory for converted (OLD-compatible) output")
    parser.add_argument("--verify", action="store_true", help="Verify converted output loads with OLD PipelineOutput")
    parser.add_argument("--old-recovar-path", default=None, help="Path to old recovar repo (default: ~/recovar)")
    args = parser.parse_args()
    convert_pipeline_output(args.src_dir, args.dst_dir, args.verify, args.old_recovar_path)


if __name__ == "__main__":
    main()
