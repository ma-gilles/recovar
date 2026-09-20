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
    "check_bundle",
    "check_per_image",
    "check_posterior_bounds",
    "describe_context",
    "finite_check_enabled",
    "finite_check_warn_only",
    "report_tracked",
    "track_max",
    "track_max_host",
]

# ``reconstruction_probs`` are ``raw_weight / sum_weight`` where the numerator
# is one of the terms of the denominator, so every entry is at most one and a
# whole image's entries sum to at most one.  A value above that bound cannot be
# a rounding artefact; it means the denominator did not come from the weights
# it normalises.  One part in a thousand is far outside float32 rounding for a
# sum of ~600 positive terms and far inside the failure this hunts.
POSTERIOR_UPPER_BOUND = 1.0 + 1e-3


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


_TRACKED: dict[str, float] = {}


def track_max(name: str, value) -> float | None:
    """Record the largest magnitude seen for ``name`` across the run.

    The maximum is reduced on the device, so one scalar crosses the boundary.
    A run that never overflows still reports how close it came, which is the
    only evidence a non-reproducing repeat can give.
    """

    if not finite_check_enabled() or value is None:
        return None
    try:
        import jax.numpy as jnp

        array = jnp.asarray(value)
        if array.dtype.kind not in "fgc":
            return None
        magnitude = jnp.abs(array)
        finite = jnp.isfinite(magnitude)
        current = float(jnp.max(jnp.where(finite, magnitude, jnp.asarray(0, magnitude.dtype))))
        if not bool(jnp.all(finite)):
            current = float("inf")
    except Exception:  # pragma: no cover - host arrays
        array = np.asarray(value)
        if array.dtype.kind not in "fgc":
            return None
        magnitude = np.abs(array)
        finite = np.isfinite(magnitude)
        current = float(magnitude[finite].max()) if finite.any() else 0.0
        if not finite.all():
            current = float("inf")
    previous = _TRACKED.get(name)
    if previous is None or current > previous:
        _TRACKED[name] = current
    return current


def report_tracked(context: str = "") -> str | None:
    """Log every tracked maximum and clear them for the next pass."""

    if not finite_check_enabled() or not _TRACKED:
        return None
    parts = " ".join(f"{k}={v:.6g}" for k, v in sorted(_TRACKED.items()))
    message = f"finite-check maxima {context}: {parts}"
    logger.warning("%s", message)
    _TRACKED.clear()
    return message


def check_posterior_bounds(stage: str, probs, *, context: str = "", image_ids=None):
    """Assert the posterior bound that the M-step operands inherit.

    ``probs`` is ``[images, ...]``.  Both the largest entry and each image's
    total must stay at or below one.  This fires on a denominator that is only
    mildly wrong, long before the product overflows float32, so a repeat that
    never reaches ``inf`` still shows whether the bound is being broken.
    """

    if not finite_check_enabled() or probs is None:
        return None

    import jax.numpy as jnp

    array = jnp.asarray(probs)
    if array.dtype.kind not in "fg":
        return None
    rows = array.reshape(array.shape[0], -1)
    finite_rows = jnp.where(jnp.isfinite(rows), rows, jnp.float32(0.0))
    row_sums = jnp.sum(finite_rows, axis=1, dtype=jnp.float64)
    entry_max = float(jnp.max(finite_rows)) if rows.size else 0.0
    row_sum_max = float(jnp.max(row_sums)) if rows.size else 0.0
    track_max(f"{stage}.entry_max", jnp.asarray(entry_max))
    track_max(f"{stage}.image_sum_max", jnp.asarray(row_sum_max))

    if entry_max <= POSTERIOR_UPPER_BOUND and row_sum_max <= POSTERIOR_UPPER_BOUND:
        return None

    sums = np.asarray(row_sums)
    offenders = np.flatnonzero(sums > POSTERIOR_UPPER_BOUND)
    if offenders.size == 0:
        offenders = np.flatnonzero(np.asarray(jnp.max(finite_rows, axis=1)) > POSTERIOR_UPPER_BOUND)
    ids = None if image_ids is None else np.asarray(image_ids).reshape(-1)
    named = offenders if ids is None or ids.size != sums.size else ids[offenders]
    message = (
        f"posterior bound broken in stage {stage!r}: {context}\n"
        f"  entry_max={entry_max:.6g} image_sum_max={row_sum_max:.6g} "
        f"bound={POSTERIOR_UPPER_BOUND}\n"
        f"  offending rows {offenders[:8].tolist()} image ids "
        f"{np.asarray(named[:8]).tolist()} sums "
        f"{[float(x) for x in sums[offenders[:8]]]}"
    )
    if finite_check_warn_only():
        logger.error("%s", message)
        return message
    raise FiniteCheckError(message)


def check_bundle(
    stage: str,
    arrays: dict,
    *,
    posterior=None,
    posterior_name: str = "posterior",
    context: str = "",
    image_ids=None,
    operands: dict | None = None,
):
    """Check several arrays and the posterior bound with one synchronisation.

    ``check_arrays`` plus ``check_posterior_bounds`` plus ``track_max`` cost one
    device synchronisation each, and a bucket has half a dozen of them; at 200
    to 350 buckets a half that serialises the pass being watched and costs
    about four times the uninstrumented wall, which buys four times less
    exposure per GPU-hour on a defect measured at 1 in 671 half-iterations.

    Here every reduction is issued asynchronously and only the stacked scalars
    cross the boundary, so a clean bucket pays one synchronisation for the
    whole set. The expensive per-array host analysis runs only when a scalar
    says something is wrong, and then it is free to synchronise again.
    """

    if not finite_check_enabled():
        return None

    import jax.numpy as jnp

    names = [name for name, value in arrays.items() if value is not None]
    scalars = []
    for name in names:
        array = jnp.asarray(arrays[name])
        if array.dtype.kind not in "fgc":
            scalars.extend([jnp.float64(0.0), jnp.float64(1.0)])
            continue
        magnitude = jnp.abs(array)
        finite = jnp.isfinite(magnitude)
        scalars.append(jnp.max(jnp.where(finite, magnitude, jnp.zeros((), magnitude.dtype))).astype(jnp.float64))
        scalars.append(jnp.all(finite).astype(jnp.float64))

    has_posterior = posterior is not None
    if has_posterior:
        rows = jnp.asarray(posterior)
        rows = rows.reshape(rows.shape[0], -1)
        finite_rows = jnp.where(jnp.isfinite(rows), rows, jnp.zeros((), rows.dtype))
        scalars.append(jnp.max(finite_rows).astype(jnp.float64))
        scalars.append(jnp.max(jnp.sum(finite_rows, axis=1, dtype=jnp.float64)))

    if not scalars:
        return None
    host = np.asarray(jnp.stack(scalars))  # the single synchronisation

    offenders = []
    for index, name in enumerate(names):
        maximum = float(host[2 * index])
        all_finite = bool(host[2 * index + 1] > 0.5)
        track_max_host(name, float("inf") if not all_finite else maximum)
        if not all_finite:
            offenders.append(name)

    entry_max = row_sum_max = None
    if has_posterior:
        entry_max = float(host[-2])
        row_sum_max = float(host[-1])
        track_max_host(f"{posterior_name}.entry_max", entry_max)
        track_max_host(f"{posterior_name}.image_sum_max", row_sum_max)

    messages = []
    if offenders:
        messages.append(
            check_arrays(
                stage,
                {name: arrays[name] for name in names},
                context=context,
                operands=operands,
            )
        )
    if has_posterior and (
        entry_max > POSTERIOR_UPPER_BOUND or row_sum_max > POSTERIOR_UPPER_BOUND
    ):
        messages.append(
            check_posterior_bounds(
                posterior_name,
                posterior,
                context=context,
                image_ids=image_ids,
            )
        )
    return [m for m in messages if m] or None


def track_max_host(name: str, value: float) -> None:
    """Record a maximum already pulled to the host by a bundled check."""

    if not finite_check_enabled():
        return
    previous = _TRACKED.get(name)
    if previous is None or value > previous:
        _TRACKED[name] = value
