#!/usr/bin/env python
"""Export RECOVAR EM/PPCA best poses to the pipeline poses.pkl format."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from compare_pose_refinement_runs import _load_em_npz, _load_ppca_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-npz", required=True, type=Path)
    parser.add_argument("--n-images", required=True, type=int)
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help="Image box size used to convert pixel translations to fractional units for .pkl output.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result_path = args.result_npz.expanduser().resolve()
    with np.load(result_path, allow_pickle=False) as npz:
        is_ppca = "best_rotation_matrix" in npz.files
        image_shape = np.asarray(npz["image_shape"], dtype=np.int64) if "image_shape" in npz.files else None
    pose_set = (
        _load_ppca_npz("poses", result_path, int(args.n_images))
        if is_ppca
        else _load_em_npz("poses", result_path, int(args.n_images))
    )
    if pose_set.translations is None:
        translations = np.zeros((int(args.n_images), 2), dtype=np.float32)
    else:
        translations = np.asarray(pose_set.translations, dtype=np.float32)
    rotations = np.asarray(pose_set.rotations, dtype=np.float32)
    if rotations.shape != (int(args.n_images), 3, 3):
        raise ValueError(f"rotations have shape {rotations.shape}, expected {(int(args.n_images), 3, 3)}")
    if translations.shape != (int(args.n_images), 2):
        raise ValueError(f"translations have shape {translations.shape}, expected {(int(args.n_images), 2)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".npz":
        np.savez_compressed(args.output, best_rotation_matrix=rotations, best_translation=translations)
    else:
        box_size = args.box_size
        if box_size is None and image_shape is not None and image_shape.size > 0:
            box_size = int(image_shape.reshape(-1)[0])
        if box_size is None or box_size <= 0:
            raise ValueError("--box-size is required for .pkl output when result NPZ has no image_shape")
        # RECOVAR's legacy poses.pkl loader expects fractional shifts and
        # converts them back to pixels during dataset loading.
        translations = translations / float(box_size)
        if not np.all(np.abs(translations) <= 1.0 + 1e-6):
            raise ValueError(
                f"Converted translations exceed fractional bounds for box size {box_size}; "
                "check the source translation units."
            )
        with args.output.open("wb") as f:
            pickle.dump((rotations, translations), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(args.output)


if __name__ == "__main__":
    main()
