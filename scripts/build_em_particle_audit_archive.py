#!/usr/bin/env python3
"""Build a source-order particle-audit archive for every EM iteration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_PMAX_KEY = re.compile(r"^pmax_per_image_iter_(\d{3})$")
_TRAJECTORY_KEYS = (
    "voxel_size",
    "volume_shape",
    "current_sizes",
    "pixel_resolutions",
    "ave_Pmax_trajectory",
    "healpix_order_trajectory",
    "acc_rot_trajectory",
    "smallest_change_angles_trajectory",
    "smallest_change_offsets_trajectory",
    "frac_changed_trajectory",
    "final_all_data_ran",
    "n_images",
)


def _scatter_source_order(values: np.ndarray, half_order: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != half_order.size:
        raise ValueError(
            f"particle array has {values.shape[0]} rows, expected {half_order.size}",
        )
    source_order = np.empty_like(values)
    source_order[half_order] = values
    return source_order


def _load_half_order_scalar(
    source: np.lib.npyio.NpzFile,
    *,
    explicit_key: str,
    fallback_key: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Load a half-order scalar array across old and current result schemas."""

    key = explicit_key if explicit_key in source.files else fallback_key
    if key not in source.files:
        raise ValueError(f"missing half-order particle array: {explicit_key} or {fallback_key}")
    return np.asarray(source[key], dtype=dtype)


def _load_half_order_vectors(
    source: np.lib.npyio.NpzFile,
    *,
    stem: str,
    suffix: str,
    half1_size: int,
    half2_size: int,
) -> np.ndarray:
    """Load vector rows from one-based, zero-based, or combined half schemas."""

    one_based = (f"{stem}_{suffix}_half1", f"{stem}_{suffix}_half2")
    zero_based = (f"{stem}_{suffix}_half0", f"{stem}_{suffix}_half1")
    combined = f"{stem}_{suffix}"
    if all(key in source.files for key in one_based):
        first, second = (np.asarray(source[key], dtype=np.float64) for key in one_based)
    elif all(key in source.files for key in zero_based):
        first, second = (np.asarray(source[key], dtype=np.float64) for key in zero_based)
    elif combined in source.files:
        values = np.asarray(source[combined], dtype=np.float64)
        first, second = values[:half1_size], values[half1_size:]
    else:
        raise ValueError(
            f"missing particle vectors for {stem}_{suffix}; expected one-based halves, "
            "zero-based halves, or one combined array",
        )
    if first.shape[0] != half1_size or second.shape[0] != half2_size:
        raise ValueError(
            f"{stem}_{suffix} half sizes are inconsistent: "
            f"{first.shape[0]}/{second.shape[0]} != {half1_size}/{half2_size}",
        )
    return np.concatenate((first, second), axis=0)


def build_archive(output_dir: Path, destination: Path | None = None) -> Path:
    """Build and atomically write a complete source-order audit archive."""

    output_dir = Path(output_dir).resolve()
    source_path = output_dir / "refinement_results.npz"
    if not source_path.is_file():
        raise ValueError(f"missing refinement results: {source_path}")
    destination = (
        output_dir.parent / "analysis" / "refinement_results_audit_source_order.npz"
        if destination is None
        else Path(destination).resolve()
    )

    arrays: dict[str, np.ndarray] = {}
    with np.load(source_path, allow_pickle=False) as source:
        for key in _TRAJECTORY_KEYS:
            if key in source.files:
                arrays[key] = np.asarray(source[key])

        half1 = np.asarray(source["half1_indices"], dtype=np.int64)
        half2 = np.asarray(source["half2_indices"], dtype=np.int64)
        half_order = np.concatenate((half1, half2))
        n_images = int(half_order.size)
        if not np.array_equal(np.sort(half_order), np.arange(n_images)):
            raise ValueError("half indices are not a complete source-row permutation")
        arrays.update(
            n_images=np.asarray(n_images, dtype=np.int64),
            half1_indices=half1,
            half2_indices=half2,
        )

        iterations = sorted(
            int(match.group(1))
            for key in source.files
            if (match := _PMAX_KEY.match(key)) is not None
        )
        if not iterations or iterations != list(range(len(iterations))):
            raise ValueError(
                f"Pmax iterations are not contiguous zero-based: {iterations}",
            )
        if "current_sizes" in source.files:
            expected_count = int(np.asarray(source["current_sizes"]).size)
            if len(iterations) != expected_count:
                raise ValueError(
                    f"particle/result iteration count mismatch: {len(iterations)} != {expected_count}",
                )

        for iteration in iterations:
            suffix = f"{iteration:03d}"
            pmax = _load_half_order_scalar(
                source,
                explicit_key=f"pmax_per_half_order_iter_{suffix}",
                fallback_key=f"pmax_per_image_iter_{suffix}",
                dtype=np.dtype(np.float64),
            )
            support = _load_half_order_scalar(
                source,
                explicit_key=f"sig_counts_half_order_iter_{suffix}",
                fallback_key=f"sig_counts_by_image_iter_{suffix}",
                dtype=np.dtype(np.int32),
            )
            eulers = _load_half_order_vectors(
                source,
                stem="best_rotation_eulers_iter",
                suffix=suffix,
                half1_size=half1.size,
                half2_size=half2.size,
            )
            translations = _load_half_order_vectors(
                source,
                stem="best_translations_iter",
                suffix=suffix,
                half1_size=half1.size,
                half2_size=half2.size,
            )
            arrays[f"pmax_per_image_by_image_iter_{suffix}"] = _scatter_source_order(
                pmax,
                half_order,
            )
            arrays[f"sig_counts_by_image_iter_{suffix}"] = _scatter_source_order(
                support,
                half_order,
            )
            arrays[f"best_rotation_eulers_by_image_iter_{suffix}"] = _scatter_source_order(
                eulers,
                half_order,
            )
            arrays[f"best_translations_by_image_iter_{suffix}"] = _scatter_source_order(
                translations,
                half_order,
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(build_archive(args.output_dir, args.destination))


if __name__ == "__main__":
    main()
