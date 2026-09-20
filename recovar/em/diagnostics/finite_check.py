"""Opt-in finite checks for the pass-2 M-step operands and accumulators (P4-D).

Job 14178118's second compact control crashed at RELION iteration 15 with a
non-finite ``wsum_norm_correction``. Its saved accumulators localise the damage
but not its source: ``Ft_ctf`` for half 2 holds 2.35 M ``+inf`` entries and
``Ft_y`` follows to NaN, while half 1 is clean apart from the radius <= 18 ball
that ``joinTwoHalvesAtLowResolution`` copies across from half 2. The sums cannot
say which row produced the first ``inf``, because a sum hides its terms.

This module adds that missing step. It is off unless
``RECOVAR_EM_FINITE_CHECK=1``: the clean path reduces on the device and pulls
back one boolean per field, so it costs a reduction and a synchronisation
rather than a transfer, but it still serialises the pass it watches. An arm
with it set is a diagnostic arm and never a timing arm.

On the first non-finite value it reports the half, the bucket, the field, how
many entries are bad, the first bad flat index, and the operands at that index,
then raises unless ``RECOVAR_EM_FINITE_CHECK_WARN=1`` asks it to keep going and
collect every occurrence.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

FINITE_CHECK_ENV = "RECOVAR_EM_FINITE_CHECK"
FINITE_CHECK_WARN_ENV = "RECOVAR_EM_FINITE_CHECK_WARN"

__all__ = [
    "FINITE_CHECK_ENV",
    "FINITE_CHECK_WARN_ENV",
    "FiniteCheckError",
    "check_arrays",
    "check_per_image",
    "describe_context",
    "finite_check_enabled",
    "finite_check_warn_only",
]


class FiniteCheckError(AssertionError):
    """A pass-2 operand or accumulator held a non-finite value."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def finite_check_enabled() -> bool:
    """Whether the opt-in finite checks run at all."""

    return _flag(FINITE_CHECK_ENV)


def finite_check_warn_only() -> bool:
    """Whether a non-finite value is logged and survived rather than raised."""

    return _flag(FINITE_CHECK_WARN_ENV)


def _all_finite(value) -> bool:
    """Whether every entry is finite, reduced where the array already lives.

    Only the resulting scalar crosses the device boundary, so the check costs a
    reduction and one synchronisation rather than a full transfer.
    """

    try:
        import jax.numpy as jnp

        array = jnp.asarray(value)
        if array.dtype.kind not in "fgc":
            return True
        if array.dtype.kind == "c":
            ok = jnp.isfinite(array.real).all() & jnp.isfinite(array.imag).all()
        else:
            ok = jnp.isfinite(array).all()
        return bool(ok)
    except Exception:  # pragma: no cover - host arrays and exotic dtypes
        array = np.asarray(value)
        if array.dtype.kind not in "fgc":
            return True
        return bool(np.isfinite(array).all())


def describe_context(**fields) -> str:
    """Render the loop coordinates a report needs, skipping absent ones."""

    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


def _summarise(name, value, bad_index):
    """One field's shape, dtype, extremes and the value at ``bad_index``."""

    array = np.asarray(value)
    flat = array.reshape(-1)
    finite = np.isfinite(flat) if flat.dtype.kind != "c" else (
        np.isfinite(flat.real) & np.isfinite(flat.imag)
    )
    n_bad = int(flat.size - finite.sum())
    finite_values = flat[finite]
    extremes = ""
    if finite_values.size:
        magnitude = np.abs(finite_values)
        extremes = f" finite|max|={magnitude.max():.6g} finite|min|={magnitude.min():.6g}"
    at = ""
    if bad_index is not None and bad_index < flat.size:
        at = f" value_at_first_bad={flat[bad_index]!r}"
    return (
        f"{name}: shape={array.shape} dtype={array.dtype} nonfinite={n_bad}/{flat.size}"
        f"{extremes}{at}"
    )


def check_arrays(stage: str, arrays: dict, *, context: str = "", operands: dict | None = None):
    """Check ``arrays`` for non-finite entries; report the first offender.

    ``arrays`` are the values under test; ``operands`` are additional arrays to
    describe at the offending index, so the report says not only which sum went
    non-finite but what it was built from. Only a field that fails the device
    reduction is brought to the host, and only to build the report.
    """

    if not finite_check_enabled():
        return None

    # Fast path: reduce on whatever device holds the array and pull back one
    # boolean per field. The M-step sums are [images, rotations, pixels]; moving
    # them to the host every bucket would cost more than the pass being watched.
    suspect = [name for name, value in arrays.items() if value is not None and not _all_finite(value)]
    if not suspect:
        return None

    first_bad = None
    offender = None
    for name in suspect:
        flat = np.asarray(arrays[name]).reshape(-1)
        finite = (
            np.isfinite(flat.real) & np.isfinite(flat.imag)
            if flat.dtype.kind == "c"
            else np.isfinite(flat)
        )
        bad = int(np.flatnonzero(~finite)[0])
        if first_bad is None or bad < first_bad:
            first_bad, offender = bad, name

    if offender is None:
        return None

    lines = [f"non-finite value in pass-2 stage {stage!r}: {context}"]
    lines.append(f"  first offender: {offender} at flat index {first_bad}")
    for name, value in arrays.items():
        if value is not None:
            lines.append("  " + _summarise(name, value, first_bad))
    for name, value in (operands or {}).items():
        if value is not None:
            lines.append("  operand " + _summarise(name, value, None))
    message = "\n".join(lines)
    if finite_check_warn_only():
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)


def check_per_image(stage: str, arrays: dict, *, image_ids=None, context: str = ""):
    """Check per-image statistics and name the offending image ids.

    ``wsum_norm_correction`` is the statistic that failed in job 14178118, and
    it is per image, so the report is only useful if it says which particles.
    """

    if not finite_check_enabled():
        return None

    ids = None if image_ids is None else np.asarray(image_ids).reshape(-1)
    reports = []
    for name, value in arrays.items():
        if value is None or _all_finite(value):
            continue
        flat = np.asarray(value).reshape(-1)
        if flat.dtype.kind not in "fgc":
            continue
        finite = (
            np.isfinite(flat.real) & np.isfinite(flat.imag)
            if flat.dtype.kind == "c"
            else np.isfinite(flat)
        )
        if bool(finite.all()):
            continue
        bad = np.flatnonzero(~finite)
        named = bad if ids is None or ids.size != flat.size else ids[bad]
        reports.append(
            f"  {name}: {bad.size}/{flat.size} non-finite, "
            f"first rows {bad[:8].tolist()}, image ids {np.asarray(named[:8]).tolist()}"
        )
    if not reports:
        return None
    message = "\n".join([f"non-finite per-image statistic in stage {stage!r}: {context}", *reports])
    if finite_check_warn_only():
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)
