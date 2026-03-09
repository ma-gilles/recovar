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


def compare_metric(current, baseline, direction, tol_frac):
    if not (math.isfinite(current) and math.isfinite(baseline)):
        return False, f"non-finite values current={current} baseline={baseline}"
    scale = max(abs(baseline), 1e-12)
    delta = (current - baseline) / scale
    if direction == "lower":
        ok = delta <= tol_frac
        msg = f"increase={delta:.4f} allowed={tol_frac:.4f}"
        return ok, msg
    if direction == "higher":
        ok = delta >= -tol_frac
        msg = f"drop={-delta:.4f} allowed={tol_frac:.4f}"
        return ok, msg
    return True, "ignored"
