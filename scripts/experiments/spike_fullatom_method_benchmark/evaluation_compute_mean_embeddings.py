#!/usr/bin/env python3
"""Compute per-GT-label mean embeddings for spike method-benchmark outputs.

The script is deliberately conservative: it writes means for outputs it can
identify and prints clear skip messages for methods whose latent arrays are not
ready or are not exposed in a known format yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/"
    "n00100000/runs/n00100000_seed0000"
)
DEFAULT_BENCH_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517")


@dataclass(frozen=True)
class EmbeddingSource:
    method: str
    run_name: str
    path: Path
    epoch: int | None
    array: np.ndarray


def _load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data)


def _epoch_from_name(path: Path) -> int | None:
    match = re.search(r"\.(\d+)\.", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _select_epoch(paths: Iterable[Path], requested_epoch: int | None) -> list[Path]:
    candidates = sorted(paths)
    if requested_epoch is None:
        by_parent: dict[Path, Path] = {}
        for path in candidates:
            epoch = _epoch_from_name(path)
            old = by_parent.get(path.parent)
            if old is None or (epoch if epoch is not None else -1) > (_epoch_from_name(old) or -1):
                by_parent[path.parent] = path
        return sorted(by_parent.values())
    selected = [p for p in candidates if _epoch_from_name(p) == requested_epoch]
    return selected


def _parse_labels(value: str | None) -> list[int] | None:
    if value is None:
        return None
    labels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one integer label")
    return labels


def _cryodrgn_sources(bench_root: Path, requested_epoch: int | None) -> list[EmbeddingSource]:
    sources: list[EmbeddingSource] = []
    for path in _select_epoch((bench_root / "cryodrgn").glob("zdim*/z.*.pkl"), requested_epoch):
        try:
            arr = _load_pickle_array(path)
        except Exception as exc:  # pragma: no cover - defensive on external artifacts
            print(f"SKIP cryodrgn {path}: could not load pickle array: {exc}")
            continue
        if arr.ndim == 1:
            arr = arr[:, None]
        sources.append(
            EmbeddingSource(
                method="cryodrgn",
                run_name=path.parent.name,
                path=path,
                epoch=_epoch_from_name(path),
                array=np.asarray(arr, dtype=np.float32),
            )
        )
    if not sources:
        print(f"SKIP cryodrgn: no z.*.pkl files found under {bench_root / 'cryodrgn'}")
    return sources


def _discover_array_files(root: Path) -> list[Path]:
    tokens = ("latent", "embedding", "encod", "z.")
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".npy", ".npz", ".pkl"}
        and any(token in path.name.lower() for token in tokens)
    ]


def _load_generic_arrays(path: Path) -> list[np.ndarray]:
    try:
        if path.suffix == ".npy":
            return [np.load(path)]
        if path.suffix == ".npz":
            data = np.load(path)
            return [np.asarray(data[key]) for key in data.files]
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if isinstance(data, dict):
                return [np.asarray(value) for value in data.values()]
            return [np.asarray(data)]
    except Exception as exc:  # pragma: no cover - defensive on external artifacts
        print(f"SKIP {path}: could not load candidate latent file: {exc}")
    return []


def _generic_sources(bench_root: Path, method: str, labels_size: int) -> list[EmbeddingSource]:
    root = bench_root / method
    sources: list[EmbeddingSource] = []
    if not root.exists():
        print(f"SKIP {method}: missing directory {root}")
        return sources
    for path in _discover_array_files(root):
        for idx, arr in enumerate(_load_generic_arrays(path)):
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == labels_size:
                arr = arr[:, None]
            if arr.ndim == 2 and arr.shape[0] == labels_size:
                rel_parent = path.parent.relative_to(root)
                run_name = str(rel_parent).replace("/", "_") or path.stem
                if idx:
                    run_name = f"{run_name}_{idx}"
                sources.append(
                    EmbeddingSource(method=method, run_name=run_name, path=path, epoch=None, array=arr.astype(np.float32))
                )
    if not sources:
        print(f"SKIP {method}: no ready particle-level latent arrays found under {root}")
    return sources


def _write_means(
    source: EmbeddingSource,
    labels: np.ndarray,
    out_root: Path,
    label_subset: list[int] | None = None,
) -> Path:
    labels = np.asarray(labels)
    if label_subset is None:
        unique_labels = np.unique(labels)
    else:
        observed = set(int(x) for x in np.unique(labels))
        missing = [label for label in label_subset if label not in observed]
        if missing:
            raise ValueError(f"Requested labels not present in state assignments: {missing}")
        unique_labels = np.asarray(label_subset, dtype=labels.dtype)
    if source.array.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{source.path} has {source.array.shape[0]} rows, but labels have {labels.shape[0]} rows"
        )

    means = np.zeros((unique_labels.size, source.array.shape[1]), dtype=np.float32)
    counts = np.zeros(unique_labels.size, dtype=np.int64)
    for out_idx, label in enumerate(unique_labels):
        mask = labels == label
        counts[out_idx] = int(mask.sum())
        means[out_idx] = source.array[mask].mean(axis=0)

    out_dir = out_root / source.method / source.run_name / "mean_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "labels_mean_z" if source.epoch is None else f"labels_mean_z_epoch{source.epoch:03d}"
    np.save(out_dir / f"{stem}.npy", means)
    np.savetxt(out_dir / f"{stem}.txt", means, fmt="%.8g")

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gt_label", "count", *[f"z{i}" for i in range(means.shape[1])]])
        for label, count, row in zip(unique_labels, counts, means, strict=True):
            writer.writerow([int(label), int(count), *[float(x) for x in row]])

    manifest = {
        "method": source.method,
        "run_name": source.run_name,
        "source_embedding": str(source.path),
        "epoch": source.epoch,
        "n_particles": int(labels.shape[0]),
        "n_labels": int(unique_labels.size),
        "zdim": int(means.shape[1]),
        "labels": [int(x) for x in unique_labels],
        "counts_min": int(counts.min()),
        "counts_max": int(counts.max()),
        "npy": str(out_dir / f"{stem}.npy"),
        "txt": str(out_dir / f"{stem}.txt"),
        "csv": str(csv_path),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"WROTE {source.method}/{source.run_name}: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--out-root", type=Path, default=None, help="Default: BENCH_ROOT/evaluation")
    parser.add_argument("--cryodrgn-epoch", type=int, default=None, help="Use a specific cryoDRGN z.N.pkl epoch.")
    parser.add_argument("--labels", default=None, help="Comma-separated GT labels to average, e.g. 0,25,50. Default: all labels.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cryodrgn", "cryosparc_3dflex", "dynamight"],
        choices=["cryodrgn", "cryosparc_3dflex", "dynamight"],
    )
    args = parser.parse_args()

    labels_path = args.source_run / "03_dataset" / "state_assignment.npy"
    labels = np.load(labels_path)
    label_subset = _parse_labels(args.labels)
    out_root = args.out_root or (args.bench_root / "evaluation")
    print(f"labels={labels_path} shape={labels.shape} n_unique={np.unique(labels).size}")
    if label_subset is not None:
        print(f"label_subset={label_subset}")

    sources: list[EmbeddingSource] = []
    if "cryodrgn" in args.methods:
        sources.extend(_cryodrgn_sources(args.bench_root, args.cryodrgn_epoch))
    for method in ("cryosparc_3dflex", "dynamight"):
        if method in args.methods:
            sources.extend(_generic_sources(args.bench_root, method, labels.shape[0]))

    if not sources:
        print("No mean embeddings written; no ready method embeddings were found.")
        return
    for source in sources:
        _write_means(source, labels, out_root, label_subset)


if __name__ == "__main__":
    main()
