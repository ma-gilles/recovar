#!/usr/bin/env python
"""Compare per-particle poses from K=1, K-class, and PPCA refinement runs."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar.utils import helpers as utils


@dataclass
class PoseSet:
    name: str
    path: Path
    rotations: np.ndarray
    translations: np.ndarray | None = None
    class_assignments: np.ndarray | None = None


def _angular_error_deg(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rel = np.einsum("nij,nij->n", lhs, rhs)
    cos = np.clip((rel - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _latest_key(keys: list[str], prefix: str, *, exact_iter_suffix: bool = False) -> str | None:
    if exact_iter_suffix:
        pattern = re.compile(rf"^{re.escape(prefix)}[0-9]{{3}}$")
        matches = sorted(k for k in keys if pattern.match(k))
    else:
        matches = sorted(k for k in keys if k.startswith(prefix))
    return matches[-1] if matches else None


def _latest_half_stem(keys: list[str], prefix: str) -> str | None:
    pattern = re.compile(rf"^({re.escape(prefix)}[0-9]{{3}})_half[0-9]+$")
    stems = sorted({match.group(1) for key in keys if (match := pattern.match(key))})
    return stems[-1] if stems else None


def _combine_halves(npz, base: str, n_images: int, trailing_shape: tuple[int, ...]) -> np.ndarray | None:
    by_image_key = _latest_key(list(npz.files), f"{base}_by_image_iter_")
    if by_image_key is not None:
        return np.asarray(npz[by_image_key], dtype=np.float32)

    compact_key = _latest_key(list(npz.files), f"{base}_iter_", exact_iter_suffix=True)
    if compact_key is not None and f"{compact_key}_half0" not in npz.files:
        return np.asarray(npz[compact_key], dtype=np.float32)

    half_arrays = []
    half_key = _latest_half_stem(list(npz.files), f"{base}_iter_")
    if half_key is None:
        return None
    for k, half_index_key in enumerate(("half1_indices", "half2_indices")):
        half_pose_key = f"{half_key}_half{k}"
        if half_pose_key not in npz.files or half_index_key not in npz.files:
            continue
        half_arrays.append((np.asarray(npz[half_index_key], dtype=np.int64), np.asarray(npz[half_pose_key])))
    if not half_arrays:
        return None
    out = np.full((n_images, *trailing_shape), np.nan, dtype=np.float32)
    for indices, arr in half_arrays:
        out[indices] = arr
    return out


def _combine_class_assignments(npz, n_images: int) -> np.ndarray | None:
    parts = []
    for k, half_index_key in enumerate(("half1_indices", "half2_indices")):
        class_key = f"class_assignments_half{k}"
        if class_key not in npz.files:
            continue
        classes = np.asarray(npz[class_key], dtype=np.int32)
        if half_index_key in npz.files and classes.shape[0] == np.asarray(npz[half_index_key]).shape[0]:
            parts.append((np.asarray(npz[half_index_key], dtype=np.int64), classes))
        elif classes.shape[0] == n_images:
            return classes
    if not parts:
        return None
    out = np.full((n_images,), -1, dtype=np.int32)
    for indices, classes in parts:
        out[indices] = classes
    return out


def _load_em_npz(name: str, path: Path, n_images: int) -> PoseSet:
    with np.load(path, allow_pickle=False) as npz:
        if "best_rotation_eulers_final_by_image" in npz.files:
            eulers = np.asarray(npz["best_rotation_eulers_final_by_image"], dtype=np.float64)
        else:
            eulers = _combine_halves(npz, "best_rotation_eulers", n_images, (3,))
        if eulers is None:
            raise ValueError(f"{path} has no exported best_rotation_eulers pose history")
        rotations = np.asarray(utils.R_from_relion(eulers, degrees=True), dtype=np.float32)
        if "best_translations_final_by_image" in npz.files:
            translations = np.asarray(npz["best_translations_final_by_image"], dtype=np.float32)
        else:
            translations = _combine_halves(npz, "best_translations", n_images, (2,))
        classes = _combine_class_assignments(npz, n_images)
    return PoseSet(name=name, path=path, rotations=rotations, translations=translations, class_assignments=classes)


def _load_ppca_npz(name: str, path: Path, n_images: int) -> PoseSet:
    with np.load(path, allow_pickle=False) as npz:
        if "best_rotation_matrix" not in npz.files:
            raise ValueError(f"{path} does not look like a PPCA pose result; missing best_rotation_matrix")
        rotations = np.asarray(npz["best_rotation_matrix"], dtype=np.float32)
        translations = np.asarray(npz["best_translation"], dtype=np.float32) if "best_translation" in npz.files else None
    if rotations.shape[0] != n_images:
        raise ValueError(f"{path} has {rotations.shape[0]} poses, expected {n_images}")
    return PoseSet(name=name, path=path, rotations=rotations, translations=translations)


def _star_particles(star_obj):
    if hasattr(star_obj, "columns"):
        return star_obj
    for key in ("particles", "data_particles", "optics"):
        value = star_obj.get(key) if isinstance(star_obj, dict) else None
        if hasattr(value, "columns") and "rlnAngleRot" in value.columns:
            return value
    for value in star_obj.values() if isinstance(star_obj, dict) else ():
        if hasattr(value, "columns") and "rlnAngleRot" in value.columns:
            return value
    raise ValueError("STAR file does not contain a particle table with RELION angles")


def _image_names_from_star(path: Path) -> np.ndarray | None:
    import starfile

    particles = _star_particles(starfile.read(str(path)))
    if "rlnImageName" not in particles.columns:
        return None
    return particles["rlnImageName"].to_numpy(dtype=str)


def _reindex_relion_particles(particles, reference_image_names: np.ndarray | None):
    if reference_image_names is None or "rlnImageName" not in particles.columns:
        return particles
    if len(reference_image_names) != len(particles):
        return particles
    source_names = particles["rlnImageName"].to_numpy(dtype=str)
    if np.array_equal(source_names, reference_image_names):
        return particles
    order_by_name = {name: idx for idx, name in enumerate(source_names)}
    try:
        take = np.asarray([order_by_name[name] for name in reference_image_names], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"RELION STAR image name {exc.args[0]!r} is missing from pose set") from exc
    return particles.iloc[take].reset_index(drop=True)


def _load_relion_star(name: str, path: Path, n_images: int, reference_image_names: np.ndarray | None) -> PoseSet:
    import starfile

    particles = _star_particles(starfile.read(str(path)))
    particles = _reindex_relion_particles(particles, reference_image_names)
    angle_cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    missing = [col for col in angle_cols if col not in particles.columns]
    if missing:
        raise ValueError(f"{path} is missing RELION angle columns: {missing}")
    eulers = particles[angle_cols].to_numpy(dtype=np.float64)
    rotations = np.asarray(utils.R_from_relion(eulers, degrees=True), dtype=np.float32)
    if rotations.shape[0] != n_images:
        raise ValueError(f"{path} has {rotations.shape[0]} poses, expected {n_images}")

    translations = None
    if {"rlnOriginXAngst", "rlnOriginYAngst"}.issubset(particles.columns):
        translations = particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(dtype=np.float32)
    elif {"rlnOriginX", "rlnOriginY"}.issubset(particles.columns):
        translations = particles[["rlnOriginX", "rlnOriginY"]].to_numpy(dtype=np.float32)

    classes = None
    if "rlnClassNumber" in particles.columns:
        classes = particles["rlnClassNumber"].to_numpy(dtype=np.int32)
    return PoseSet(name=name, path=path, rotations=rotations, translations=translations, class_assignments=classes)


def _load_pose_set(spec: str, n_images: int, reference_image_names: np.ndarray | None) -> PoseSet:
    if ":" not in spec:
        raise ValueError("--pose-set must be NAME:/path/to/result.npz")
    name, raw_path = spec.split(":", 1)
    path = Path(raw_path).expanduser().resolve()
    if path.suffix == ".star":
        return _load_relion_star(name, path, n_images, reference_image_names)
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
    if "best_rotation_matrix" in keys:
        return _load_ppca_npz(name, path, n_images)
    return _load_em_npz(name, path, n_images)


def _valid_rotations(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.isfinite(lhs).all(axis=(1, 2)) & np.isfinite(rhs).all(axis=(1, 2))


def _translation_delta(lhs: np.ndarray | None, rhs: np.ndarray | None) -> dict[str, float | int] | None:
    if lhs is None or rhs is None:
        return None
    valid = np.isfinite(lhs).all(axis=1) & np.isfinite(rhs).all(axis=1)
    if not np.any(valid):
        return {"n": 0}
    return _summary(np.linalg.norm(lhs[valid] - rhs[valid], axis=1))


def _best_class_accuracy(pred: np.ndarray, truth: np.ndarray) -> dict[str, object] | None:
    valid = (pred >= 0) & (truth >= 0)
    if not np.any(valid):
        return None
    pred = pred[valid]
    truth = truth[valid]
    pred_classes = np.unique(pred)
    true_classes = np.unique(truth)
    if len(pred_classes) > 8 or len(true_classes) > 8:
        return None
    confusion = np.zeros((len(pred_classes), len(true_classes)), dtype=np.int64)
    for i, pc in enumerate(pred_classes):
        for j, tc in enumerate(true_classes):
            confusion[i, j] = int(np.sum((pred == pc) & (truth == tc)))
    best = (-1, None)
    for cols in itertools.permutations(range(len(true_classes)), min(len(pred_classes), len(true_classes))):
        score = sum(confusion[i, c] for i, c in enumerate(cols))
        if score > best[0]:
            best = (score, cols)
    mapping = {
        int(pred_classes[i]): int(true_classes[c])
        for i, c in enumerate(best[1] or ())
    }
    return {
        "n": int(valid.sum()),
        "accuracy": float(best[0] / valid.sum()),
        "mapping": mapping,
        "confusion_rows_pred_cols_truth": confusion.tolist(),
        "pred_classes": pred_classes.astype(int).tolist(),
        "true_classes": true_classes.astype(int).tolist(),
    }


def compare_pose_sets(simulation_info: Path, pose_specs: list[str]) -> dict[str, object]:
    with open(simulation_info, "rb") as f:
        sim = pickle.load(f)
    gt_rotations = np.asarray(sim["rots"], dtype=np.float32)
    gt_translations = np.asarray(sim["trans"], dtype=np.float32) if "trans" in sim else None
    gt_classes = np.asarray(sim["image_assignment"], dtype=np.int32) if "image_assignment" in sim else None
    reference_star = simulation_info.resolve().parent / "particles.star"
    reference_image_names = _image_names_from_star(reference_star) if reference_star.exists() else None
    pose_sets = [_load_pose_set(spec, gt_rotations.shape[0], reference_image_names) for spec in pose_specs]

    out: dict[str, object] = {
        "simulation_info": str(simulation_info.resolve()),
        "n_images": int(gt_rotations.shape[0]),
        "pose_sets": [ps.name for ps in pose_sets],
        "vs_gt": {},
        "pairwise": {},
    }
    for ps in pose_sets:
        valid = _valid_rotations(ps.rotations, gt_rotations)
        angles = _angular_error_deg(ps.rotations[valid], gt_rotations[valid])
        rec = {
            "path": str(ps.path),
            "rotation_error_deg": _summary(angles),
            "translation_error": _translation_delta(ps.translations, gt_translations),
        }
        if ps.class_assignments is not None and gt_classes is not None:
            rec["class_assignment"] = _best_class_accuracy(ps.class_assignments, gt_classes)
        out["vs_gt"][ps.name] = rec

    for lhs, rhs in itertools.combinations(pose_sets, 2):
        valid = _valid_rotations(lhs.rotations, rhs.rotations)
        angles = _angular_error_deg(lhs.rotations[valid], rhs.rotations[valid])
        out["pairwise"][f"{lhs.name}__vs__{rhs.name}"] = {
            "rotation_delta_deg": _summary(angles),
            "translation_delta": _translation_delta(lhs.translations, rhs.translations),
        }
    return out


def _print_summary(result: dict[str, object]) -> None:
    print(f"n_images={result['n_images']}")
    for name, rec in result["vs_gt"].items():
        rot = rec["rotation_error_deg"]
        print(
            f"{name} vs GT: mean={rot.get('mean', float('nan')):.4f} deg, "
            f"p95={rot.get('p95', float('nan')):.4f}, max={rot.get('max', float('nan')):.4f}"
        )
        cls = rec.get("class_assignment")
        if cls is not None:
            print(f"{name} class vs GT: accuracy={cls['accuracy']:.4f}, mapping={cls['mapping']}")
    for name, rec in result["pairwise"].items():
        rot = rec["rotation_delta_deg"]
        print(
            f"{name}: mean={rot.get('mean', float('nan')):.4f} deg, "
            f"p95={rot.get('p95', float('nan')):.4f}, max={rot.get('max', float('nan')):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-info", required=True, type=Path)
    parser.add_argument("--pose-set", action="append", required=True, help="NAME:/path/to/result.npz")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = compare_pose_sets(args.simulation_info, args.pose_set)
    _print_summary(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
