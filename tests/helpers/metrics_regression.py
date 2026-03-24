import math


LOWER_IS_BETTER_TOKENS = (
    "error",
    "locres",
    "angle",
    "loss",
    "rmse",
    "mse",
    "bias",
    "constrast",
    "contrast",
)

HIGHER_IS_BETTER_TOKENS = (
    "fsc",
    "correlation",
    "variance_explained",
    "relative_variance",
)


def metric_direction(metric_name):
    name = metric_name.lower()
    if any(tok in name for tok in LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(tok in name for tok in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return "ignore"


# Metrics that are inherently noisy across runs (depend on local resolution
# estimation which varies with dataset regeneration and GPU non-determinism).
# These get a looser tolerance than the default.
HIGH_VARIANCE_TOKENS = ("locres", "auc", "noise_max")
_HIGH_VARIANCE_MIN_TOL = 0.10
ABSOLUTE_TOLERANCE_FLOORS = (
    ("moving_piece_error_median", 2e-3),
    ("moving_piece_error_90pct", 3.5e-3),
)


def metric_tolerance(metric_name, default_tol_frac):
    """Return per-metric tolerance (high-variance metrics get a wider band)."""
    name = metric_name.lower()
    if any(tok in name for tok in HIGH_VARIANCE_TOKENS):
        return max(default_tol_frac, _HIGH_VARIANCE_MIN_TOL)
    return default_tol_frac


def metric_absolute_tolerance(metric_name):
    """Return an absolute tolerance floor for brittle near-zero metrics."""
    if metric_name is None:
        return 0.0
    name = metric_name.lower()
    for token, tol in ABSOLUTE_TOLERANCE_FLOORS:
        if token in name:
            return tol
    return 0.0


def compare_metric(current, baseline, direction, tol_frac, metric_name=None):
    """Compare current vs baseline with tolerance.

    If *metric_name* is provided and is a high-variance metric, the tolerance
    is widened automatically via metric_tolerance().
    """
    if not (math.isfinite(current) and math.isfinite(baseline)):
        return False, f"non-finite values current={current} baseline={baseline}"
    effective_tol = tol_frac
    absolute_tol = 0.0
    if metric_name is not None:
        effective_tol = metric_tolerance(metric_name, tol_frac)
        absolute_tol = metric_absolute_tolerance(metric_name)
    scale = max(abs(baseline), 1e-12)
    delta_raw = current - baseline
    delta = delta_raw / scale
    allowed = max(effective_tol * scale, absolute_tol)
    if direction == "lower":
        ok = delta_raw <= allowed
        msg = f"increase={delta:.4f} allowed={effective_tol:.4f}"
        if absolute_tol > 0:
            msg += f", abs_allowed={absolute_tol:.4g}"
        return ok, msg
    if direction == "higher":
        ok = delta_raw >= -allowed
        msg = f"drop={-delta:.4f} allowed={effective_tol:.4f}"
        if absolute_tol > 0:
            msg += f", abs_allowed={absolute_tol:.4g}"
        return ok, msg
    return True, "ignored"


def log_comparison_table(current, baseline, tol_frac, title="Metric Comparison",
                         direction_fn=None, skip_unknown=True):
    """Print a comparison table for all shared numeric metrics.

    Always prints every metric — not just failures. This makes it possible
    to review metric drift without re-running.

    Parameters
    ----------
    direction_fn : callable, optional
        Custom function ``(key) -> direction`` where direction is
        "higher", "lower", or "ignore".  Defaults to ``metric_direction``
        with outlier-style overrides (precision/recall/f1 → higher).
    skip_unknown : bool
        If True (default), skip metrics where direction is "ignore".
        If False, treat unknown metrics as "higher" (i.e. regression = drop).

    Returns (checked, failures) where failures is a list of error strings.
    """
    if direction_fn is None:
        direction_fn = metric_direction

    failures = []
    checked = 0
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  {title}")
    lines.append(f"{'=' * 80}")
    lines.append(f"  {'Metric':<45s} {'Current':>10s}  {'Baseline':>10s}  {'Delta':>8s}  {'Status'}")
    lines.append(f"  {'-' * 75}")

    for key in sorted(set(current) & set(baseline)):
        cur = current[key]
        base = baseline[key]
        if not (isinstance(cur, (int, float)) and isinstance(base, (int, float))):
            continue

        direction = direction_fn(key)
        if direction == "ignore":
            # Treat outlier_recall / outlier_precision / outlier_f1 as higher-is-better
            if any(tok in key for tok in ("recall", "precision", "f1")):
                direction = "higher"
            elif not skip_unknown:
                direction = "higher"

        scale = max(abs(base), 1e-12)
        delta_pct = (cur - base) / scale * 100

        if direction == "ignore":
            # Print for visibility but don't check
            lines.append(f"  {key:<45s} {cur:10.4f}  {base:10.4f}  {delta_pct:+7.1f}%  (info)")
            continue

        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=tol_frac, metric_name=key)
        checked += 1
        status = "OK" if ok else "FAIL"
        if not ok:
            failures.append(f"{key}: current={cur:.4f} baseline={base:.4f} ({msg})")

        lines.append(f"  {key:<45s} {cur:10.4f}  {base:10.4f}  {delta_pct:+7.1f}%  {status}")

    lines.append(f"  {'-' * 75}")
    lines.append(f"  Checked: {checked}, Failures: {len(failures)}")
    lines.append(f"{'=' * 80}")
    print("\n".join(lines))
    return checked, failures
