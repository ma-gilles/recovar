"""Strict loading and deterministic row ordering for BPref contribution captures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


MAGIC = "RECOVAR_BPREF_CONTRIBUTION_ROWS"
SCHEMA = "recovar-bpref-contribution-rows-v3"


def _scalar(values: dict[str, np.ndarray], key: str):
    if key not in values:
        raise ValueError(f"missing required BPref capture field {key!r}")
    value = np.asarray(values[key])
    if value.shape != ():
        raise ValueError(f"BPref capture field {key!r} must be scalar")
    return value.item()


def _array(values, key, *, ndim=None, dtype=None):
    if key not in values:
        raise ValueError(f"missing required BPref capture field {key!r}")
    value = np.asarray(values[key])
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"BPref capture field {key!r} must have ndim={ndim}")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ValueError(
            f"BPref capture field {key!r} must have dtype {np.dtype(dtype)}, got {value.dtype}"
        )
    return value


@dataclass(frozen=True)
class BPrefContributionShard:
    path: Path
    values: dict[str, np.ndarray]

    @property
    def row_count(self) -> int:
        return int(self.values["active_original_indices"].size)

    @property
    def row_identity(self) -> np.ndarray:
        return np.stack(
            (
                self.values["active_original_indices"],
                self.values["active_global_rotation_indices"],
                self.values["active_rotation_rows"],
            ),
            axis=1,
        )

    @property
    def execution_key(self) -> tuple[int, int, str]:
        return (
            int(self.values["call_index"]),
            int(self.values["dump_index"]),
            str(self.path),
        )


@dataclass(frozen=True)
class BPrefContributionBundle:
    shards: tuple[BPrefContributionShard, ...]

    @property
    def row_count(self) -> int:
        return sum(shard.row_count for shard in self.shards)

    def concatenate(self, order: str = "execution") -> dict[str, np.ndarray]:
        fields = (
            "active_original_indices",
            "active_global_rotation_indices",
            "active_rotation_rows",
            "active_summed",
            "active_ctf_probs",
            "active_rotations",
        )
        combined = {
            key: np.concatenate([shard.values[key] for shard in self.shards], axis=0)
            for key in fields
        }
        combined["source_shard"] = np.concatenate(
            [np.full(shard.row_count, index, dtype=np.int32) for index, shard in enumerate(self.shards)]
        )
        combined["source_row"] = np.concatenate(
            [np.arange(shard.row_count, dtype=np.int32) for shard in self.shards]
        )
        if order == "execution":
            indices = np.arange(self.row_count, dtype=np.int64)
        elif order == "canonical":
            indices = np.lexsort(
                (
                    combined["active_rotation_rows"],
                    combined["active_global_rotation_indices"],
                    combined["active_original_indices"],
                )
            )
        else:
            raise ValueError(f"unknown BPref row order {order!r}")
        return {key: value[indices] for key, value in combined.items()}


def load_bpref_contribution_shard(path: str | Path) -> BPrefContributionShard:
    """Load one v3 row capture and fail closed on incomplete identity/geometry."""

    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    if _scalar(values, "magic") != MAGIC or _scalar(values, "schema") != SCHEMA:
        raise ValueError(f"unsupported BPref contribution capture: {path}")
    if int(_scalar(values, "schema_version")) != 3 or int(_scalar(values, "pass_index")) != 2:
        raise ValueError("BPref contribution replay requires schema v3 pass_index=2")
    for key in ("iteration", "half", "current_size", "reconstruction_padding_factor"):
        if int(_scalar(values, key)) <= 0:
            raise ValueError(f"BPref capture field {key!r} must be positive")
    if int(_scalar(values, "half")) not in {1, 2}:
        raise ValueError("BPref capture half must be 1 or 2")
    checksum = str(_scalar(values, "source_stack_sha256"))
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum.lower()):
        raise ValueError("source_stack_sha256 must be a hexadecimal SHA256")

    image_shape = _array(values, "image_shape", ndim=1, dtype=np.int32)
    volume_shape = _array(values, "volume_shape", ndim=1, dtype=np.int32)
    if image_shape.shape != (2,) or volume_shape.shape != (3,):
        raise ValueError("BPref capture image/volume shapes are malformed")
    window = _array(values, "window_indices", ndim=1, dtype=np.int32)
    if window.size == 0 or np.unique(window).size != window.size:
        raise ValueError("BPref capture window_indices must be nonempty and unique")

    actual_counts = _array(values, "actual_counts", ndim=1)
    active_particle = _array(values, "active_particle_rows", ndim=1, dtype=np.int32)
    active_row = _array(values, "active_rotation_rows", ndim=1, dtype=np.int32)
    original = _array(values, "original_indices", ndim=1, dtype=np.int64)
    active_original = _array(values, "active_original_indices", ndim=1, dtype=np.int64)
    global_rotations = _array(values, "oversampled_rotation_indices", ndim=2)
    active_global = _array(values, "active_global_rotation_indices", ndim=1)
    q = active_particle.size
    if any(array.size != q for array in (active_row, active_original, active_global)):
        raise ValueError("BPref active row identity arrays have different lengths")
    if q == 0:
        raise ValueError("BPref contribution shard contains no active rows")
    if np.any((active_particle < 0) | (active_particle >= original.size)):
        raise ValueError("active_particle_rows lies outside the particle batch")
    if actual_counts.shape != (original.size,):
        raise ValueError("actual_counts does not match particle count")
    if np.any((active_row < 0) | (active_row >= actual_counts[active_particle])):
        raise ValueError("active_rotation_rows lies outside actual_counts")
    if global_rotations.shape[0] != original.size:
        raise ValueError("oversampled_rotation_indices does not match particle count")
    if not np.array_equal(active_original, original[active_particle]):
        raise ValueError("active original identities do not close against particle rows")
    if not np.array_equal(active_global, global_rotations[active_particle, active_row]):
        raise ValueError("active rotation identities do not close against candidate rows")

    summed = _array(values, "active_summed", ndim=2)
    weights = _array(values, "active_ctf_probs", ndim=2)
    rotations = _array(values, "active_rotations", ndim=3)
    if not np.issubdtype(summed.dtype, np.complexfloating):
        raise ValueError("BPref active_summed must have a complex dtype")
    if not np.issubdtype(weights.dtype, np.floating):
        raise ValueError("BPref active_ctf_probs must have a real floating dtype")
    if summed.shape != (q, window.size) or weights.shape != summed.shape:
        raise ValueError("BPref active operand shapes do not match rows/window")
    if rotations.shape != (q, 3, 3) or not np.issubdtype(rotations.dtype, np.floating):
        raise ValueError("BPref active_rotations must have shape (rows,3,3) and real dtype")
    if not all(np.all(np.isfinite(array)) for array in (summed, weights, rotations)):
        raise ValueError("BPref active operands contain nonfinite values")
    if np.any(weights < 0):
        raise ValueError("BPref active weights contain negative values")
    identity = np.stack((active_original, active_global, active_row), axis=1)
    if np.unique(identity, axis=0).shape[0] != q:
        raise ValueError("BPref row semantic identity is not unique within the shard")
    return BPrefContributionShard(path=path, values=values)


def load_bpref_contribution_bundle(paths) -> BPrefContributionBundle:
    """Load a complete boundary, preserving execution order and canonical identity."""

    shards = tuple(sorted((load_bpref_contribution_shard(path) for path in paths), key=lambda x: x.execution_key))
    if not shards:
        raise ValueError("at least one BPref contribution shard is required")
    reference = shards[0].values
    scalar_boundary = (
        "iteration",
        "half",
        "rank",
        "pass_index",
        "class_index",
        "current_size",
        "source_stack_sha256",
        "disc_type",
        "reconstruction_padding_factor",
    )
    array_boundary = ("image_shape", "volume_shape", "window_indices")
    for shard in shards[1:]:
        for key in scalar_boundary:
            if _scalar(shard.values, key) != _scalar(reference, key):
                raise ValueError(f"BPref shard boundary mismatch for {key}")
        for key in array_boundary:
            if not np.array_equal(shard.values[key], reference[key]):
                raise ValueError(f"BPref shard boundary mismatch for {key}")
    execution_keys = [(key[0], key[1]) for key in (shard.execution_key for shard in shards)]
    if len(set(execution_keys)) != len(execution_keys):
        raise ValueError("BPref shard call/dump execution identity is duplicated")
    identities = np.concatenate([shard.row_identity for shard in shards], axis=0)
    if np.unique(identities, axis=0).shape[0] != identities.shape[0]:
        raise ValueError("BPref semantic row identities overlap across shards")
    return BPrefContributionBundle(shards=shards)


def exact_array_metrics(left, right) -> dict[str, object]:
    """Exact and normed array differences; deliberately no correlation metric."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = left.astype(np.result_type(left.dtype, right.dtype), copy=False) - right
    absolute = np.abs(delta).astype(np.float64)
    denominator = max(float(np.linalg.norm(right.ravel())), np.finfo(np.float64).tiny)
    return {
        "shape_equal": True,
        "array_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "max_abs": float(np.max(absolute, initial=0.0)),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / denominator),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_bpref_contribution_bundle(bundle: BPrefContributionBundle) -> dict[str, object]:
    """Return a hash-bound, exact-metric inventory for one captured boundary."""

    reference = bundle.shards[0].values
    execution = bundle.concatenate("execution")
    canonical = bundle.concatenate("canonical")
    identity = np.stack(
        (
            canonical["active_original_indices"],
            canonical["active_global_rotation_indices"],
            canonical["active_rotation_rows"],
        ),
        axis=1,
    ).astype(np.int64, copy=False)
    identity_digest = hashlib.sha256(identity.tobytes(order="C")).hexdigest()
    return {
        "schema": "recovar-bpref-contribution-bundle-summary-v1",
        "status": "PASS",
        "boundary": {
            key: _scalar(reference, key)
            for key in (
                "run_id",
                "iteration",
                "half",
                "rank",
                "pass_index",
                "class_index",
                "current_size",
                "source_stack_sha256",
                "disc_type",
                "reconstruction_padding_factor",
            )
        },
        "image_shape": np.asarray(reference["image_shape"]).tolist(),
        "volume_shape": np.asarray(reference["volume_shape"]).tolist(),
        "window_pixel_count": int(np.asarray(reference["window_indices"]).size),
        "shard_count": len(bundle.shards),
        "row_count": bundle.row_count,
        "unique_particle_count": int(np.unique(identity[:, 0]).size),
        "canonical_row_identity": (
            "original_index_zero_based,global_rotation_index,particle_candidate_row"
        ),
        "canonical_row_identity_sha256": identity_digest,
        "captured_operand_dtypes": {
            "active_summed": sorted(
                {str(shard.values["active_summed"].dtype) for shard in bundle.shards}
            ),
            "active_ctf_probs": sorted(
                {str(shard.values["active_ctf_probs"].dtype) for shard in bundle.shards}
            ),
            "active_rotations": sorted(
                {str(shard.values["active_rotations"].dtype) for shard in bundle.shards}
            ),
        },
        "execution_vs_canonical_order": {
            "same_row_sequence": bool(
                np.array_equal(
                    np.stack(
                        (
                            execution["active_original_indices"],
                            execution["active_global_rotation_indices"],
                            execution["active_rotation_rows"],
                        ),
                        axis=1,
                    ),
                    identity,
                )
            ),
            "execution_to_canonical_moved_rows": int(
                np.count_nonzero(
                    (execution["source_shard"] != canonical["source_shard"])
                    | (execution["source_row"] != canonical["source_row"])
                )
            ),
        },
        "shards": [
            {
                "path": str(shard.path),
                "sha256": sha256_file(shard.path),
                "call_index": int(shard.values["call_index"]),
                "dump_index": int(shard.values["dump_index"]),
                "row_count": shard.row_count,
                "particle_count": int(np.asarray(shard.values["original_indices"]).size),
                "active_summed_dtype": str(shard.values["active_summed"].dtype),
                "active_ctf_probs_dtype": str(shard.values["active_ctf_probs"].dtype),
            }
            for shard in bundle.shards
        ],
        "quality_gate": "exact identity/schema/array validation; no correlation metric",
    }
