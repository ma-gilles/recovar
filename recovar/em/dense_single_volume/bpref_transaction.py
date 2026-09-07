"""Bound consecutive compatible BPref calls without changing particle order."""

import jax
import jax.numpy as jnp
import numpy as np


_POSITIONAL = (
    "data_volume",
    "weight_volume",
    "images",
    "ctf",
    "minvsigma2",
    "posterior_over_weight_norm",
    "translation_angles",
    "pixel_indices",
    "projector_full",
    "rotation_matrices",
    "image_shape",
    "volume_shape",
    "max_r",
    "projector_max_r",
    "projection_padding_factor",
)
_CARRY = ("data_volume", "weight_volume")
_PARTICLE = (
    "images",
    "ctf",
    "minvsigma2",
    "posterior_over_weight_norm",
    "rotation_matrices",
    "reconstruction_group_ids",
    "worker_lane_ids",
    "particle_trace_ids",
)
_NO_REPLAY = (
    "worker_lane_ids",
    "particle_trace_ids",
    "rotation_replay_order",
    "rotation_replay_counts",
    "particle_start_offsets_ns",
    "runtime_projector_radius",
)
_FALSE_REPLAY = (
    "serial_rotation_replay",
    "persistent_serial_rotation_replay",
    "float64_accumulator_replay",
    "reverse_rotation_replay",
    "rotation_replay_stride",
    "native_trace_shape_replay",
    "candidate_trace_active",
    "parallel_worker_replay",
)


@jax.jit
def _concatenate_fields(columns):
    return tuple(jnp.concatenate(column, axis=0) for column in columns)


class BprefTransactionQueue:
    """Combine only consecutive calls with identical shared operand objects.

    Bounds cover the number of images and compact particle-input bytes queued
    for concatenation, not total HBM. A single oversized or diagnostic call
    flushes previous work and executes unchanged. The caller must pass each
    returned accumulator to the next call and flush before using final state.
    """

    def __init__(self, *, max_images=256, max_input_bytes=128 * 1024**2):
        if type(max_images) is not int or type(max_input_bytes) is not int or min(max_images, max_input_bytes) <= 0:
            raise ValueError("BPref queue bounds must be positive integers")
        self.max_images = max_images
        self.max_input_bytes = max_input_bytes
        self._pending = []
        self._anchor = None
        self._callback = None
        self._key = None
        self._images = 0
        self._bytes = 0

    @staticmethod
    def _compatible_key(values):
        if any(values.get(name) is not None for name in _NO_REPLAY):
            return None
        for name in _FALSE_REPLAY:
            value = values.get(name)
            if value is not None and (type(value) not in (bool, int) or value != 0):
                return None
        key = []
        n = values["images"].shape[0]
        for name in sorted(values):
            value = values[name]
            if name in _CARRY or name == "return_denominator":
                continue
            if name in _PARTICLE:
                if value is None:
                    key.append((name, None))
                else:
                    if value.shape[0] != n:
                        raise ValueError("BPref particle axes differ")
                    key.append((name, tuple(value.shape[1:]), str(value.dtype)))
            elif hasattr(value, "shape") and hasattr(value, "dtype"):
                # No host reads or content-dependent equality on device arrays.
                key.append((name, id(value)))
            else:
                key.append((name, type(value), value))
        return tuple(key)

    def _check_carry(self, data, weight):
        if self._anchor is not None and (data is not self._anchor[0] or weight is not self._anchor[1]):
            raise RuntimeError("BPref carry changed while queued work was pending")

    def accumulate(self, callback, *args, **kwargs):
        if len(args) != len(_POSITIONAL) or set(kwargs).intersection(_POSITIONAL):
            raise ValueError("BPref queue requires the original fifteen positional operands")
        values = dict(zip(_POSITIONAL, args, strict=True))
        values.update(kwargs)
        data, weight = (values[name] for name in _CARRY)
        self._check_carry(data, weight)
        key = self._compatible_key(values)
        n = values["images"].shape[0]
        if n <= 0:
            raise ValueError("BPref batches must contain at least one image")
        byte_count = sum(
            values[name].size * values[name].dtype.itemsize for name in _PARTICLE if values.get(name) is not None
        )
        oversized = n > self.max_images or byte_count > self.max_input_bytes
        if self._pending and (
            key is None
            or key != self._key
            or callback is not self._callback
            or self._images + n > self.max_images
            or self._bytes + byte_count > self.max_input_bytes
        ):
            data, weight, _ = self.flush(data, weight)
            values.update(data_volume=data, weight_volume=weight)
        if key is None or oversized:
            return callback(**values)
        if not self._pending:
            self._anchor = (data, weight)
            self._callback = callback
            self._key = key
        self._pending.append(values)
        self._images += n
        self._bytes += byte_count
        return data, weight, None

    def flush(self, data, weight):
        self._check_carry(data, weight)
        if not self._pending:
            return data, weight, None
        merged = dict(self._pending[0])
        if len(self._pending) > 1:
            names = [name for name in _PARTICLE if merged.get(name) is not None]
            columns = [tuple(call[name] for call in self._pending) for name in names]
            # The old CUDA wrapper restarts arange(B) for each bucket. Explicit
            # IDs must keep that assignment without enabling parallel replay.
            counts = [call["images"].shape[0] for call in self._pending]
            names.extend(("worker_lane_ids", "particle_trace_ids"))
            columns.extend(
                (
                    tuple(np.arange(n, dtype=np.int32) % 8 for n in counts),
                    tuple(np.arange(n, dtype=np.int32) for n in counts),
                )
            )
            merged.update(zip(names, _concatenate_fields(tuple(columns)), strict=True))
            merged["parallel_worker_replay"] = False
        merged.update(data_volume=data, weight_volume=weight, return_denominator=False)
        result = self._callback(**merged)
        if len(result) != 3 or result[2] is not None:
            raise RuntimeError("Queued BPref callback did not omit its denominator")
        self._pending.clear()
        self._anchor = self._callback = self._key = None
        self._images = self._bytes = 0
        return result
