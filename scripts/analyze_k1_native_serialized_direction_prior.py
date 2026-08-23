#!/usr/bin/env python3
"""Compare RELION's in-memory coarse direction log-prior with its saved model STAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.em.dense_single_volume.helpers.orientation_priors import (
    make_relion_direction_log_prior,
)
from recovar.em.sampling import read_relion_direction_prior
from scripts.parse_relion_dump_dir import _read_flat_real


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "max_abs": float(np.max(np.abs(values))),
        "mean": float(np.mean(values)),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
    }


def analyze(
    *,
    native_log_prior_path: Path,
    model_star_path: Path,
    healpix_order: int,
) -> dict[str, object]:
    serialized_pdf = np.asarray(read_relion_direction_prior(model_star_path), dtype=np.float32)
    serialized_log_psi_major = make_relion_direction_log_prior(
        serialized_pdf,
        int(healpix_order),
    ).astype(np.float64)

    native_log_pixel_major = _read_flat_real(native_log_prior_path)
    n_directions = int(serialized_pdf.size)
    if native_log_pixel_major.size % n_directions:
        raise ValueError(
            "native coarse direction prior is not a direction-by-psi grid: "
            f"{native_log_pixel_major.size} values for {n_directions} directions",
        )
    n_psi = native_log_pixel_major.size // n_directions
    native_log_psi_major = native_log_pixel_major.reshape(n_directions, n_psi).T.reshape(-1)
    if native_log_psi_major.shape != serialized_log_psi_major.shape:
        raise ValueError(
            "native and serialized expanded priors have different shapes: "
            f"{native_log_psi_major.shape} versus {serialized_log_psi_major.shape}",
        )

    serialized_active = np.isfinite(serialized_log_psi_major)
    native_active = native_log_psi_major != 0.0
    common_active = serialized_active & native_active
    if not np.any(common_active):
        raise ValueError("native and serialized direction priors have no common active rotations")
    raw_difference = native_log_psi_major[common_active] - serialized_log_psi_major[common_active]
    centered_difference = raw_difference - np.median(raw_difference)

    return {
        "schema": "recovar.em.k1_native_serialized_direction_prior.v1",
        "status": "complete",
        "layout": {
            "native": "RELION pixel-major (direction, psi)",
            "comparison": "RECOVAR psi-major (psi, direction)",
            "healpix_order": int(healpix_order),
            "n_directions": n_directions,
            "n_psi": n_psi,
        },
        "active_support": {
            "native_count": int(np.count_nonzero(native_active)),
            "serialized_count": int(np.count_nonzero(serialized_active)),
            "common_count": int(np.count_nonzero(common_active)),
            "native_only_count": int(np.count_nonzero(native_active & ~serialized_active)),
            "serialized_only_count": int(np.count_nonzero(serialized_active & ~native_active)),
        },
        "raw_log_prior_difference": _error_stats(raw_difference),
        "centered_log_prior_difference": _error_stats(centered_difference),
        "raw_difference_median": float(np.median(raw_difference)),
        "artifacts": {
            "native_log_prior": str(native_log_prior_path.resolve()),
            "native_log_prior_sha256": _sha256(native_log_prior_path),
            "model_star": str(model_star_path.resolve()),
            "model_star_sha256": _sha256(model_star_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-log-prior", type=Path, required=True)
    parser.add_argument("--model-star", type=Path, required=True)
    parser.add_argument("--healpix-order", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        native_log_prior_path=args.native_log_prior,
        model_star_path=args.model_star,
        healpix_order=args.healpix_order,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
