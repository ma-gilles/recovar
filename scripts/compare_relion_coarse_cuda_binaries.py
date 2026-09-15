#!/usr/bin/env python3
"""Bitwise A/B gate for the shared RELION coarse-projector CUDA kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CASES = (
    (8, 1, 128, 21),
    (8, 2, 129, 29),
    (14, 5, 257, 37),
    (18, 7, 130, 25),
    (10, 3, 130, 97),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _random_rotations(rng, count):
    import numpy as np

    rotations = np.empty((count, 3, 3), dtype=np.float32)
    for index in range(count):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotations[index] = q.astype(np.float32)
    return rotations


def _dump(output: Path) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from recovar import cuda_backproject

    if jax.default_backend() != "gpu":
        raise RuntimeError("the coarse-projector binary gate requires a GPU")
    results = {}
    rng = np.random.default_rng(20260830)
    for case_index, (current_size, model_max_r, rotation_count, translation_count) in enumerate(_CASES):
        projector_size = 2 * model_max_r + 3
        projector = (
            rng.normal(0.0, 0.02, (projector_size,) * 3)
            + 1j * rng.normal(0.0, 0.02, (projector_size,) * 3)
        ).astype(np.complex64)
        rotations = _random_rotations(rng, rotation_count)
        full_pixel_count = current_size * (current_size // 2 + 1)
        retained = np.flatnonzero(np.arange(full_pixel_count) % 5 != 1)
        lookup = np.full(full_pixel_count, -1, dtype=np.int32)
        lookup[retained] = np.arange(retained.size, dtype=np.int32)
        images = (
            rng.normal(0.0, 0.02, (2, retained.size))
            + 1j * rng.normal(0.0, 0.02, (2, retained.size))
        ).astype(np.complex64)
        translations = rng.uniform(-0.3, 0.3, (translation_count, 2)).astype(np.float32)
        weight = rng.uniform(0.1, 3.0, images.shape).astype(np.float32)
        initial_diff2 = rng.uniform(10.0, 20.0, images.shape[0]).astype(np.float32)

        with jax.default_device(jax.devices("gpu")[0]):
            _, lanes = cuda_backproject.relion_coarse_diff2_projector_lanes_f32(
                jnp.asarray(projector),
                jnp.asarray(rotations),
                jnp.asarray(images),
                jnp.asarray(translations),
                jnp.asarray(weight),
                jnp.asarray(initial_diff2),
                jnp.asarray(lookup),
                current_size=current_size,
                physical_image_size=current_size,
                model_max_r=model_max_r,
            )
            canonical = cuda_backproject.relion_coarse_diff2_projector_f32(
                jnp.asarray(projector),
                jnp.asarray(rotations),
                jnp.asarray(images),
                jnp.asarray(translations),
                jnp.asarray(weight),
                jnp.asarray(initial_diff2),
                jnp.asarray(lookup),
                current_size=current_size,
                physical_image_size=current_size,
                model_max_r=model_max_r,
                canonical_reduction=True,
            )
        results[f"case_{case_index}_lanes"] = np.asarray(lanes).view(np.uint32)
        results[f"case_{case_index}_canonical"] = np.asarray(canonical).view(np.uint32)
    np.savez(output, **results)


def _run_dump(script: Path, library: Path, output: Path) -> None:
    env = dict(os.environ)
    env["RECOVAR_CUDA_LIB"] = str(library)
    env["RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS"] = "1"
    env.pop("RECOVAR_DISABLE_CUDA", None)
    subprocess.run(
        [sys.executable, str(script), "--dump", str(output)],
        check=True,
        env=env,
    )


def _compare(baseline_path: Path, candidate_path: Path) -> tuple[bool, dict]:
    import numpy as np

    mismatches = {}
    with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
        if baseline.files != candidate.files:
            raise RuntimeError(
                f"dump keys differ: baseline={baseline.files}, candidate={candidate.files}"
            )
        for key in baseline.files:
            left = baseline[key]
            right = candidate[key]
            unequal = left != right
            count = int(np.count_nonzero(unequal))
            mismatches[key] = {
                "shape": list(left.shape),
                "mismatch_count": count,
                "first_mismatch_flat_index": (
                    int(np.flatnonzero(unequal)[0]) if count else None
                ),
            }
    return all(item["mismatch_count"] == 0 for item in mismatches.values()), mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lib", type=Path)
    parser.add_argument("--candidate-lib", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dump", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dump is not None:
        _dump(args.dump)
        return 0
    if args.baseline_lib is None or args.candidate_lib is None or args.output_json is None:
        parser.error("--baseline-lib, --candidate-lib, and --output-json are required")
    baseline_lib = args.baseline_lib.resolve(strict=True)
    candidate_lib = args.candidate_lib.resolve(strict=True)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="recovar-coarse-cuda-ab-") as temp_dir:
        temp = Path(temp_dir)
        baseline_dump = temp / "baseline.npz"
        candidate_dump = temp / "candidate.npz"
        _run_dump(script, baseline_lib, baseline_dump)
        _run_dump(script, candidate_lib, candidate_dump)
        passed, arrays = _compare(baseline_dump, candidate_dump)
    report = {
        "schema": "recovar.relion_coarse_cuda_binary_ab.v1",
        "passed": passed,
        "cases": [list(case) for case in _CASES],
        "baseline_lib": str(baseline_lib),
        "baseline_sha256": _sha256(baseline_lib),
        "candidate_lib": str(candidate_lib),
        "candidate_sha256": _sha256(candidate_lib),
        "arrays": arrays,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
