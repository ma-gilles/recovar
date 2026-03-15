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


def metric_tolerance(metric_name, default_tol_frac):
    """Return per-metric tolerance (high-variance metrics get a wider band)."""
    name = metric_name.lower()
    if any(tok in name for tok in HIGH_VARIANCE_TOKENS):
        return max(default_tol_frac, _HIGH_VARIANCE_MIN_TOL)
    return default_tol_frac


def compare_metric(current, baseline, direction, tol_frac, metric_name=None):
    """Compare current vs baseline with tolerance.

    If *metric_name* is provided and is a high-variance metric, the tolerance
    is widened automatically via metric_tolerance().
    """
    if not (math.isfinite(current) and math.isfinite(baseline)):
        return False, f"non-finite values current={current} baseline={baseline}"
    effective_tol = tol_frac
    if metric_name is not None:
        effective_tol = metric_tolerance(metric_name, tol_frac)
    scale = max(abs(baseline), 1e-12)
    delta = (current - baseline) / scale
    if direction == "lower":
        ok = delta <= effective_tol
        msg = f"increase={delta:.4f} allowed={effective_tol:.4f}"
        return ok, msg
    if direction == "higher":
        ok = delta >= -effective_tol
        msg = f"drop={-delta:.4f} allowed={effective_tol:.4f}"
        return ok, msg
    return True, "ignored"
