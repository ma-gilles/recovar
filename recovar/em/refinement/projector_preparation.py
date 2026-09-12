"""Prepare RELION projector slabs for scoring and validate captured geometry.

Native preparation converts references, reuses the existing disk cache and
writes optional projector dumps. Captured preparation checks the replay state
against the live scoring geometry. Neither path owns iteration scheduling.
The captured-state type and its serialized identity remain in relion_replay.
"""

from __future__ import annotations

import hashlib
import logging
import os

import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics.relion_replay import RelionProjectorReplayState

logger = logging.getLogger(__name__)


def prepare_initial_real_references(init_reference_real, *, volume_shape, n_classes, log):
    """Normalize direct real references to half/class axes without Fourier conversion.

    Preserve float64 source values, shared-half identity and per-half views.
    A missing handoff stays [None, None] for the existing Fourier fallback.
    """
    initial_real_references_by_half = [None, None]
    if init_reference_real is not None:
        expected_volume_shape = tuple(int(value) for value in volume_shape)

        def _as_class_real_references(value):
            array = np.asarray(value, dtype=np.float64)
            if n_classes == 1 and array.shape == expected_volume_shape:
                return array[None, ...]
            expected_class_shape = (n_classes,) + expected_volume_shape
            if array.shape == expected_class_shape:
                return array
            raise ValueError(
                "init_reference_real must be a shared real volume, a per-class "
                f"array, or a two-half collection; got {array.shape}, expected "
                f"{expected_volume_shape} or {expected_class_shape}",
            )

        if isinstance(init_reference_real, (list, tuple)) and len(init_reference_real) == 2:
            initial_real_references_by_half = [
                _as_class_real_references(init_reference_real[0]),
                _as_class_real_references(init_reference_real[1]),
            ]
        else:
            real_array = np.asarray(init_reference_real)
            per_half_shape = (2, n_classes) + expected_volume_shape
            if n_classes == 1 and real_array.shape == (2,) + expected_volume_shape:
                initial_real_references_by_half = [
                    _as_class_real_references(real_array[0]),
                    _as_class_real_references(real_array[1]),
                ]
            elif real_array.shape == per_half_shape:
                initial_real_references_by_half = [
                    _as_class_real_references(real_array[0]),
                    _as_class_real_references(real_array[1]),
                ]
            else:
                shared_real = _as_class_real_references(real_array)
                initial_real_references_by_half = [shared_real, shared_real]
        log.info(
            "RELION initial projector: preserving direct float64 real-reference handoff"
        )
    return initial_real_references_by_half


def prepare_local_projector_slab(projector_half, *, path_label="local RELION projector path"):
    """Return one (z, y, x_half) slab, preserving the input's JAX dtype.

    Local scoring accepts a slab or a singleton class axis. This is shape
    normalization only: no Fourier conversion, interpolation or precision policy.
    """
    slab = jnp.asarray(projector_half)
    if slab.ndim == 4:
        if int(slab.shape[0]) != 1:
            raise ValueError(
                f"{path_label} expected a single-class projector slab, got {slab.shape}",
            )
        slab = slab[0]
    if slab.ndim != 3:
        raise ValueError(
            f"{path_label} expected Projector::data shape (z, y, x_half), got {slab.shape}",
        )
    return slab


def _relion_projector_half_maps_for_scoring(
    means_k,
    *,
    volume_shape,
    current_size: int | None,
    padding_factor: int,
    n_classes: int,
    real_references=None,
    dump_label: str | None = None,
) -> tuple[np.ndarray, int]:
    """Build RELION ``Projector::data`` slabs from current Fourier references."""

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.vdam.dense_adapter import reference_to_relion_projector_half_maps

    refs_ft = np.asarray(means_k)
    if int(n_classes) == 1 and refs_ft.ndim == 1:
        refs_ft = refs_ft[None, :]
    if refs_ft.ndim != 2 or int(refs_ft.shape[0]) != int(n_classes):
        raise ValueError(
            "means_k must be a flat reference or a per-class reference array; "
            f"got shape {refs_ft.shape} for n_classes={n_classes}",
        )
    refs_real_override = None
    if real_references is not None:
        refs_real_override = np.asarray(real_references, dtype=np.float64)
        expected_shape = (int(n_classes),) + tuple(int(value) for value in volume_shape)
        if refs_real_override.shape != expected_shape:
            raise ValueError(
                "real_references must have one real-space volume per class; "
                f"got {refs_real_override.shape}, expected {expected_shape}",
            )
    resolved_current_size = int(current_size) if current_size is not None else int(volume_shape[0])
    cache_dir = os.environ.get("RECOVAR_RELION_PROJECTOR_CACHE_DIR", "").strip()
    cache_path = None
    if cache_dir:
        refs_for_hash = np.ascontiguousarray(
            refs_ft if refs_real_override is None else refs_real_override
        )
        hasher = hashlib.sha256()
        hasher.update(b"recovar-relion-projector-cache-v1")
        hasher.update(b"fourier-reference" if refs_real_override is None else b"real-reference")
        hasher.update(str(refs_for_hash.dtype).encode("utf-8"))
        hasher.update(np.asarray(refs_for_hash.shape, dtype=np.int64).tobytes())
        hasher.update(np.asarray(volume_shape, dtype=np.int64).tobytes())
        cache_params = np.asarray(
            [resolved_current_size, int(padding_factor), int(n_classes)],
            dtype=np.int64,
        )
        hasher.update(cache_params.tobytes())
        hasher.update(refs_for_hash.view(np.uint8))
        cache_path = os.path.join(cache_dir, f"projector_{hasher.hexdigest()[:24]}.npz")
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    projector_half = np.asarray(cached["projector_half"])
                    projector_r_max = int(np.asarray(cached["projector_r_max"]))
                    if (
                        int(np.asarray(cached["current_size"])) != resolved_current_size
                        or int(np.asarray(cached["padding_factor"])) != int(padding_factor)
                        or int(np.asarray(cached["n_classes"])) != int(n_classes)
                        or tuple(np.asarray(cached["volume_shape"], dtype=np.int64).tolist()) != tuple(volume_shape)
                    ):
                        raise ValueError("metadata mismatch")
                logger.info("RELION mode: loaded cached Projector::data from %s", cache_path)
                return projector_half, projector_r_max
            except Exception as exc:
                logger.warning("Ignoring unreadable RELION projector cache %s: %s", cache_path, exc)
    if refs_real_override is None:
        refs_real = []
        for class_index in range(int(n_classes)):
            ref_ft = jnp.asarray(refs_ft[class_index]).reshape(volume_shape)
            refs_real.append(np.asarray(ftu.get_idft3(ref_ft)).real)
        refs_real = np.asarray(refs_real, dtype=np.float64)
    else:
        refs_real = refs_real_override
    projector_half, projector_r_max = reference_to_relion_projector_half_maps(
        refs_real,
        current_size=resolved_current_size,
        padding_factor=int(padding_factor),
    )
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(os.path.join(cache_dir, "SAFE_TO_DELETE"), "a", encoding="utf-8"):
                pass
            tmp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
            np.savez(
                tmp_path,
                projector_half=np.asarray(projector_half),
                projector_r_max=np.int64(projector_r_max),
                current_size=np.int64(resolved_current_size),
                padding_factor=np.int64(padding_factor),
                volume_shape=np.asarray(volume_shape, dtype=np.int64),
                n_classes=np.int64(n_classes),
            )
            os.replace(tmp_path, cache_path)
            logger.info("RELION mode: saved Projector::data cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Could not write RELION projector cache %s: %s", cache_path, exc)
    dump_dir = os.environ.get("RECOVAR_RELION_PROJECTOR_DUMP_DIR")
    if dump_dir:
        label = dump_label or "projector"
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_dir, f"{safe_label}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half),
            reference_real=np.asarray(refs_real),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(resolved_current_size),
            padding_factor=np.int64(padding_factor),
            volume_shape=np.asarray(volume_shape, dtype=np.int64),
            n_classes=np.int64(n_classes),
        )
    return projector_half, projector_r_max


def _validate_captured_relion_projector_for_iteration(
    replay_state: RelionProjectorReplayState,
    *,
    current_size: int | None,
    volume_shape,
    padding_factor: int,
    n_classes: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Bind a captured projector state to one exact live replay geometry."""

    resolved_current_size = int(current_size) if current_size is not None else int(volume_shape[0])
    expected_volume_shape = tuple(int(value) for value in volume_shape)
    mismatches = []
    if replay_state.current_size != resolved_current_size:
        mismatches.append(
            f"current_size captured={replay_state.current_size} replay={resolved_current_size}"
        )
    if replay_state.padding_factor != int(padding_factor):
        mismatches.append(
            f"padding_factor captured={replay_state.padding_factor} replay={int(padding_factor)}"
        )
    if replay_state.volume_shape != expected_volume_shape:
        mismatches.append(
            f"volume_shape captured={replay_state.volume_shape} replay={expected_volume_shape}"
        )
    if replay_state.n_classes != int(n_classes):
        mismatches.append(
            f"n_classes captured={replay_state.n_classes} replay={int(n_classes)}"
        )
    if mismatches:
        raise ValueError(
            "captured RELION Projector::data does not match the live replay boundary: "
            + "; ".join(mismatches)
        )
    return (
        list(replay_state.projector_half_by_half),
        [int(value) for value in replay_state.projector_r_max_by_half],
    )
