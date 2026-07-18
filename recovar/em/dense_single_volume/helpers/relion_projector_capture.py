"""Load schema-v2 RELION diagnostic projector captures for exact EM replay.

The native files come from the opt-in live-boundary hook on RELION d476e6f.
Every consumed byte is bound to its capture manifest before replay.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path

import numpy as np

STATE_RE = re.compile(
    r"state_iter(?P<iteration>\d+)_rank(?P<rank>\d+)_device(?P<device>\d+)_"
    r"class(?P<class_id>\d+)_state_schema_version\.bin$"
)
STATE_TOPOLOGY_RE = re.compile(
    r"state_iter(?P<iteration>\d+)_rank(?P<rank>\d+)_device(?P<device>\d+)_"
    r"class(?P<class_id>\d+)_(?:iteration|projector_(?:real|imag))\.bin$"
)
SUPPORTED_STATE_SCHEMA_VERSION = 2
NATIVE_LONG_DTYPE_V2 = np.dtype("<i8")


class ProjectorLoadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_scalar(path: Path, *, integral: bool = False) -> float | int:
    raw = Path(path).read_bytes()
    if len(raw) != 8:
        raise ProjectorLoadError(f"scalar must contain exactly eight bytes: {path}")
    value = struct.unpack("<d", raw)[0]
    if not math.isfinite(value):
        raise ProjectorLoadError(f"non-finite scalar: {path}")
    if integral:
        integer = int(value)
        if float(integer) != value:
            raise ProjectorLoadError(f"scalar must be exactly integral: {path}: {value}")
        return integer
    return value


def _read_vector(path: Path, dtype) -> np.ndarray:
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ProjectorLoadError(f"truncated vector header: {path}")
    count = struct.unpack("<q", raw[:8])[0]
    resolved = np.dtype(dtype)
    expected_bytes = 8 + count * resolved.itemsize
    if count < 0 or len(raw) != expected_bytes:
        raise ProjectorLoadError(
            f"vector payload length mismatch: {path}: count={count}, "
            f"bytes={len(raw)}, expected={expected_bytes}"
        )
    return np.frombuffer(raw, dtype=resolved, offset=8, count=count).copy()


def _parse_and_verify_manifest(path: Path) -> tuple[str, set[Path]]:
    path = Path(path).resolve()
    listed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, raw_name = line.split("  ", 1)
        except ValueError as exc:
            raise ProjectorLoadError(f"invalid manifest line {line_number}: {line!r}") from exc
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ProjectorLoadError(f"invalid manifest digest on line {line_number}")
        candidate = Path(raw_name)
        candidate = candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve()
        if candidate in listed:
            raise ProjectorLoadError(f"duplicate manifest path: {candidate}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ProjectorLoadError(f"manifest verification failed: {candidate}")
        listed.add(candidate)
    if not listed:
        raise ProjectorLoadError("source manifest is empty")
    return sha256_file(path), listed


def _prefix_file(prefix: Path, field: str) -> Path:
    return Path(str(prefix) + field + ".bin").resolve()


def _require_consumed(path: Path, listed: set[Path]) -> Path:
    path = Path(path).resolve()
    if path not in listed:
        raise ProjectorLoadError(f"consumed projector field is absent from manifest: {path}")
    return path


def _scalar(prefix: Path, field: str, listed: set[Path], *, integral: bool = True):
    return _read_scalar(_require_consumed(_prefix_file(prefix, field), listed), integral=integral)


def _vector(prefix: Path, field: str, dtype, listed: set[Path]) -> np.ndarray:
    return _read_vector(_require_consumed(_prefix_file(prefix, field), listed), dtype)


def build_relion_projector_replay_state(
    dump_dir: Path,
    *,
    manifest_path: Path,
    iteration: int,
    current_size: int,
    volume_shape,
    n_classes: int,
) -> dict[str, object]:
    """Return exactly the mapping accepted by ``relion_projector_state``."""

    dump_dir = Path(dump_dir).resolve()
    manifest_sha256, listed = _parse_and_verify_manifest(manifest_path)
    if list(dump_dir.rglob("*.partial")):
        raise ProjectorLoadError("live capture contains partial files")
    volume_shape = tuple(int(value) for value in volume_shape)
    if len(volume_shape) != 3 or any(value <= 0 for value in volume_shape):
        raise ProjectorLoadError("volume_shape must contain three positive values")
    if int(iteration) <= 0 or int(current_size) <= 0 or int(n_classes) <= 0:
        raise ProjectorLoadError("iteration/current_size/n_classes must be positive")

    rank_prefixes = {}
    for schema_path in dump_dir.glob(
        f"state_iter{int(iteration)}_rank*_device*_class0_state_schema_version.bin"
    ):
        match = STATE_RE.match(schema_path.name)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in rank_prefixes:
            raise ProjectorLoadError(f"multiple captured devices for MPI rank {rank}")
        rank_prefixes[rank] = Path(str(schema_path).removesuffix("state_schema_version.bin"))
    if not rank_prefixes:
        raise ProjectorLoadError("no captured rank-local projector state found")

    observed_topology = set()
    for path in dump_dir.glob(f"state_iter{int(iteration)}_rank*_device*_class*_*.bin"):
        match = STATE_TOPOLOGY_RE.match(path.name)
        if match is not None:
            observed_topology.add(
                (int(match.group("rank")), int(match.group("device")), int(match.group("class_id")))
            )
    expected_topology = set()
    for rank, prefix in rank_prefixes.items():
        match = STATE_RE.match(_prefix_file(prefix, "state_schema_version").name)
        if match is None:
            raise ProjectorLoadError(f"cannot recover device identity for rank {rank}")
        device = int(match.group("device"))
        expected_topology.update((rank, device, class_id) for class_id in range(int(n_classes)))
    if observed_topology != expected_topology:
        missing = sorted(expected_topology - observed_topology)
        extra = sorted(observed_topology - expected_topology)
        raise ProjectorLoadError(
            f"unexpected rank/device/class projector topology: missing={missing}, extra={extra}"
        )

    half_to_rank = {}
    for rank, prefix in rank_prefixes.items():
        schema_version = _scalar(prefix, "state_schema_version", listed)
        if schema_version != SUPPORTED_STATE_SCHEMA_VERSION:
            raise ProjectorLoadError(
                f"unsupported live-state schema for rank {rank}: {schema_version}; "
                f"expected {SUPPORTED_STATE_SCHEMA_VERSION}"
            )
        prefix_match = STATE_RE.match(_prefix_file(prefix, "state_schema_version").name)
        if prefix_match is None:
            raise ProjectorLoadError(f"cannot recover captured device for rank {rank}")
        filename_device = int(prefix_match.group("device"))
        scalar_device = _scalar(prefix, "device_id", listed)
        if scalar_device != filename_device:
            raise ProjectorLoadError(
                f"device identity mismatch for rank {rank}: filename={filename_device}, "
                f"scalar={scalar_device}"
            )
        if _scalar(prefix, "iteration", listed) != int(iteration):
            raise ProjectorLoadError(f"iteration mismatch for rank {rank}")
        if _scalar(prefix, "mpi_rank", listed) != rank:
            raise ProjectorLoadError(f"MPI rank metadata mismatch for rank {rank}")
        half = _scalar(prefix, "control_my_halfset", listed)
        if half not in (1, 2) or half in half_to_rank:
            raise ProjectorLoadError(f"invalid or duplicate half assignment for rank {rank}: {half}")
        # Schema v2 is emitted by the Della/Linux RELION build where native
        # ``std::vector<long>`` is exactly signed little-endian int64. Do not
        # guess this dtype from payload length or silently accept another ABI.
        current_sizes = _vector(prefix, "control_image_current_size", NATIVE_LONG_DTYPE_V2, listed)
        if current_sizes.size == 0 or np.any(current_sizes != int(current_size)):
            raise ProjectorLoadError(f"captured current-size schedule mismatch for rank {rank}")
        half_to_rank[half] = rank
    if set(half_to_rank) != {1, 2}:
        raise ProjectorLoadError(f"capture must contain exactly halves 1 and 2, got {sorted(half_to_rank)}")

    projectors_by_half = []
    r_max_by_half = []
    common_padding = None
    common_slab_shape = None
    for half in (1, 2):
        class_arrays = []
        class_r_max = []
        rank = half_to_rank[half]
        class0_prefix = rank_prefixes[rank]
        device_match = STATE_RE.match(_prefix_file(class0_prefix, "state_schema_version").name)
        if device_match is None:
            raise ProjectorLoadError(f"cannot recover device identity for rank {rank}")
        device = int(device_match.group("device"))
        for class_id in range(int(n_classes)):
            prefix = dump_dir / (
                f"state_iter{int(iteration)}_rank{rank}_device{device}_class{class_id}_"
            )
            if _scalar(prefix, "iteration", listed) != int(iteration):
                raise ProjectorLoadError(f"class {class_id} iteration mismatch for half {half}")
            if _scalar(prefix, "mpi_rank", listed) != rank:
                raise ProjectorLoadError(f"class {class_id} rank mismatch for half {half}")
            if _scalar(prefix, "device_id", listed) != device:
                raise ProjectorLoadError(f"class {class_id} device mismatch for half {half}")
            if _scalar(prefix, "class_id", listed) != class_id:
                raise ProjectorLoadError(f"class identity mismatch for half {half}, class {class_id}")
            zdim = _scalar(prefix, "projector_zdim", listed)
            ydim = _scalar(prefix, "projector_ydim", listed)
            xdim = _scalar(prefix, "projector_xdim", listed)
            if zdim != ydim or xdim != ydim // 2 + 1:
                raise ProjectorLoadError(f"nonstandard RELION half-projector shape {(zdim, ydim, xdim)}")
            starts = tuple(
                _scalar(prefix, field, listed)
                for field in ("projector_zinit", "projector_yinit", "projector_xinit")
            )
            if starts != (-(zdim // 2), -(ydim // 2), 0):
                raise ProjectorLoadError(f"nonstandard RELION projector starts {starts}")
            if _scalar(prefix, "projector_element_bytes", listed) != 4:
                raise ProjectorLoadError("captured accelerator projector is not XFLOAT float32")
            padding = _scalar(prefix, "projector_padding_factor", listed)
            if padding <= 0 or (common_padding is not None and padding != common_padding):
                raise ProjectorLoadError("projector padding factor differs across ranks/classes")
            common_padding = padding
            slab_shape = (zdim, ydim, xdim)
            if common_slab_shape is not None and slab_shape != common_slab_shape:
                raise ProjectorLoadError("projector slab shape differs across ranks/classes")
            common_slab_shape = slab_shape
            real = _vector(prefix, "projector_real", "<f4", listed)
            imag = _vector(prefix, "projector_imag", "<f4", listed)
            count = int(np.prod(slab_shape, dtype=np.int64))
            if real.size != count or imag.size != count:
                raise ProjectorLoadError("projector payload does not close over captured dimensions")
            projector = (real + np.complex64(1j) * imag).astype(np.complex64).reshape(slab_shape)
            if not np.isfinite(projector.real).all() or not np.isfinite(projector.imag).all():
                raise ProjectorLoadError("projector payload contains non-finite values")
            r_max = _scalar(prefix, "projector_r_max", listed)
            if r_max < 0 or r_max >= min(slab_shape):
                raise ProjectorLoadError(
                    f"projector r_max={r_max} is outside the captured slab {slab_shape}"
                )
            class_arrays.append(projector)
            class_r_max.append(r_max)
        if len(set(class_r_max)) != 1:
            raise ProjectorLoadError(f"class projector r_max differs within half {half}")
        projectors_by_half.append(np.ascontiguousarray(np.stack(class_arrays, axis=0)))
        r_max_by_half.append(class_r_max[0])

    return {
        "projector_half_by_half": projectors_by_half,
        "projector_r_max_by_half": r_max_by_half,
        "current_size": int(current_size),
        "padding_factor": int(common_padding),
        "volume_shape": list(volume_shape),
        "n_classes": int(n_classes),
        "source_manifest_sha256": manifest_sha256,
    }
