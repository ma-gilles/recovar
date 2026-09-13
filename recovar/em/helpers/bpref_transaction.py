"""Bound consecutive compatible BPref calls without changing particle order."""

from functools import partial

import jax
import jax.numpy as jnp

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
    "particle_tail_mask",
)


@jax.jit
def _concatenate_fields(columns):
    counts = tuple(batch.shape[0] for batch in columns[0])
    local_ids = tuple(jnp.arange(n, dtype=jnp.int32) for n in counts)
    return (
        *tuple(jnp.concatenate(column, axis=0) for column in columns),
        jnp.concatenate(tuple(ids % 8 for ids in local_ids)),
        jnp.concatenate(local_ids),
    )


@partial(jax.jit, static_argnums=(1, 2))
def _pad_particle_fields(columns, capacity, group_column):
    fields = _concatenate_fields(columns)
    padding = capacity - fields[0].shape[0]
    if padding < 0:
        raise ValueError("BPref physical capacity cannot truncate active particles")
    return tuple(
        jnp.pad(value, ((0, padding),) + ((0, 0),) * (value.ndim - 1),
                constant_values=-1 if i == group_column else 0)
        for i, value in enumerate(fields)
    )


class BprefTransactionQueue:
    """Combine only consecutive calls with identical shared operand objects.

    Bounds cover the number of images and compact particle-input bytes queued
    for concatenation, not total HBM. A single oversized or diagnostic call
    flushes previous work and executes unchanged. The caller must pass each
    returned accumulator to the next call and flush before using final state.

    ``stable_particle_capacity`` pads compatible grouped calls to a capacity
    determined only by non-particle shapes and these bounds. Generated worker
    and trace IDs count toward the physical byte limit. CUDA masks the suffix.
    """

    def __init__(self, *, max_images=256, max_input_bytes=128 * 1024**2, stable_particle_capacity=False, cuda_packing=False):
        if type(max_images) is not int or type(max_input_bytes) is not int or min(max_images, max_input_bytes) <= 0:
            raise ValueError("BPref queue bounds must be positive integers")
        if type(stable_particle_capacity) is not bool:
            raise TypeError("stable_particle_capacity must be a Python bool")
        if type(cuda_packing) is not bool:
            raise TypeError("cuda_packing must be a Python bool")
        if cuda_packing and (not stable_particle_capacity or max_images > 256):
            raise ValueError("CUDA BPref packing requires stable capacity of at most 256 images")
        self.max_images = max_images
        self.max_input_bytes = max_input_bytes
        self.stable_particle_capacity = stable_particle_capacity
        self.cuda_packing = cuda_packing
        self._capacity = 0
        self._pending = []
        self._anchor = None
        self._callback = None
        self._key = None
        self._images = 0
        self._bytes = 0
        self._scorer_carry = None

    def run_deferred_scorer(self, callback, arguments, options):
        """Keep real accumulators out of the scorer's donated pass-through slots."""
        if not all(
            options.get(name) is True
            for name in ("return_deferred_mstep_inputs", "disable_adjoint_y", "disable_adjoint_ctf")
        ):
            raise ValueError("Queued BPref requires a deferred scorer with both adjoints disabled")
        carry = arguments[7]
        self._check_carry(*carry)
        if self._scorer_carry is None:
            self._scorer_carry = tuple(jnp.zeros((0,), dtype=value.dtype) for value in carry)
        if any(dummy.dtype != value.dtype for dummy, value in zip(self._scorer_carry, carry, strict=True)):
            raise ValueError("Queued BPref scorer carry dtype changed")
        result = callback(*arguments[:7], carry._make(self._scorer_carry), *arguments[8:], **options)
        self._scorer_carry = (result.core.Ft_y, result.core.Ft_ctf)
        return result._replace(core=result.core._replace(Ft_y=carry[0], Ft_ctf=carry[1]))

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
        capacity = self.max_images
        if self.stable_particle_capacity and key is not None:
            if values.get("reconstruction_group_ids") is None or data.ndim != 2 or data.shape[0] <= 1:
                raise ValueError("BPref particle capacity requires explicit multiple accumulator groups")
            # Compatible inputs have no explicit worker/trace IDs. Include the
            # two generated S32 fields in the physical compact-input bound.
            row_bytes = byte_count // n + 2 * 4
            capacity = min(self.max_images, self.max_input_bytes // row_bytes)
        oversized = n > capacity or byte_count > self.max_input_bytes
        if self._pending and (
            key is None
            or key != self._key
            or callback is not self._callback
            or self._images + n > capacity
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
            self._capacity = capacity
        self._pending.append(values)
        self._images += n
        self._bytes += byte_count
        return data, weight, None

    def flush(self, data, weight):
        self._check_carry(data, weight)
        if not self._pending:
            return data, weight, None
        merged = dict(self._pending[0])
        if len(self._pending) > 1 or self.stable_particle_capacity:
            names = [name for name in _PARTICLE if merged.get(name) is not None]
            columns = [tuple(call[name] for call in self._pending) for name in names]
            # The old CUDA wrapper restarts arange(B) for each bucket. Explicit
            # IDs must keep that assignment without enabling parallel replay.
            names.extend(("worker_lane_ids", "particle_trace_ids"))
            if self.stable_particle_capacity:
                if self.cuda_packing:
                    from recovar.cuda_backproject import pack_bpref_particle_fields

                    if tuple(names) != _PARTICLE:
                        raise ValueError("CUDA BPref packing requires all six particle fields")
                    packed = pack_bpref_particle_fields(tuple(columns), self._capacity)
                else:
                    packed = _pad_particle_fields(tuple(columns), self._capacity, names.index("reconstruction_group_ids"))
                merged["particle_tail_mask"] = True
            else:
                packed = _concatenate_fields(tuple(columns))
            merged.update(zip(names, packed, strict=True))
            merged["parallel_worker_replay"] = False
        merged.update(data_volume=data, weight_volume=weight, return_denominator=False)
        result = self._callback(**merged)
        if len(result) != 3 or result[2] is not None:
            raise RuntimeError("Queued BPref callback did not omit its denominator")
        self._pending.clear()
        self._anchor = self._callback = self._key = None
        self._images = self._bytes = 0
        self._capacity = 0
        return result
