"""Memory model for the v0 ab-initio PPCA E-step + M-step.

Companion to ``docs/math/ppca_abinitio_memory_model.md``. The analytic
formulas there are reproduced here as Python; the doc is the source
of truth for the derivation.

API:
    estimate_peak_memory_bytes(...) -> dict[component_name, bytes]
    recommended_image_batch_size(..., budget_gb=60.0) -> int
"""

from __future__ import annotations


def _half_image_size(image_shape):
    N0, N1 = image_shape
    return N0 * (N1 // 2 + 1)


def _half_volume_size(volume_shape):
    N0, N1, N2 = volume_shape
    return N0 * N1 * (N2 // 2 + 1)


def estimate_peak_memory_bytes(
    n_img: int,
    volume_shape,
    image_shape,
    n_rot: int,
    n_trans: int,
    q: int,
) -> dict:
    """Per-component peak memory estimate in bytes.

    Components match the table in
    ``docs/math/ppca_abinitio_memory_model.md``. Sums to the
    expected steady-state peak; multiply by ~1.5 for the JIT-compile
    transient peak.
    """
    img_half = _half_image_size(image_shape)
    img_full = image_shape[0] * image_shape[1]
    V_half = _half_volume_size(volume_shape)

    # All tensors in float64 / complex128. complex128 = 16 bytes,
    # float64 = 8 bytes.
    components = {
        "u_proj_half": n_rot * q * img_half * 16,
        "post_mean": n_img * n_rot * n_trans * q * 8,
        "post_Hinv": n_img * n_rot * q * q * 8,
        "M_voxel": q * q * V_half * 16,
        "B_voxel": q * V_half * 16,
        "batch_full": n_img * img_full * 16,
    }
    components["total"] = sum(v for k, v in components.items() if k != "total")
    return components


def recommended_image_batch_size(
    n_img: int,
    volume_shape,
    image_shape,
    n_rot: int,
    n_trans: int,
    q: int,
    *,
    budget_gb: float = 60.0,
    runtime_overhead_factor: float = 1.5,
) -> int:
    """Largest power-of-two batch size whose predicted peak memory
    fits within ``budget_gb`` after a runtime-overhead multiplier.

    Returns ``n_img`` if the full dataset already fits.
    """
    if n_img <= 0:
        raise ValueError(f"n_img must be positive, got {n_img}")
    budget_bytes = budget_gb * (1024**3) / runtime_overhead_factor

    # Try descending powers of two until one fits.
    candidate = 1
    while candidate * 2 <= n_img:
        candidate *= 2
    while candidate > 1:
        cost = estimate_peak_memory_bytes(candidate, volume_shape, image_shape, n_rot, n_trans, q)
        if cost["total"] <= budget_bytes:
            return candidate
        candidate //= 2
    return 1


def format_memory_report(
    n_img: int,
    volume_shape,
    image_shape,
    n_rot: int,
    n_trans: int,
    q: int,
) -> str:
    """Pretty-print the per-component memory breakdown."""
    cost = estimate_peak_memory_bytes(n_img, volume_shape, image_shape, n_rot, n_trans, q)

    def _h(b):
        if b >= 1024**3:
            return f"{b / 1024**3:.2f} GB"
        if b >= 1024**2:
            return f"{b / 1024**2:.0f} MB"
        return f"{b / 1024:.0f} KB"

    lines = [
        f"memory model: n_img={n_img}, vol={tuple(volume_shape)}, image={tuple(image_shape)}, "
        f"n_rot={n_rot}, n_trans={n_trans}, q={q}",
    ]
    for k in ("u_proj_half", "post_mean", "post_Hinv", "M_voxel", "B_voxel", "batch_full"):
        lines.append(f"  {k:14s} {_h(cost[k])}")
    lines.append(f"  {'total':14s} {_h(cost['total'])}")
    return "\n".join(lines)
