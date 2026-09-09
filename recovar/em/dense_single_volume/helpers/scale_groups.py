"""Host validation and full group-axis sizing for EM scale statistics."""

from __future__ import annotations

import numpy as np


def prepare_scale_correction_groups(
    group_ids, scale_correction_group_count=None, *, n_images: int | None = None,
) -> tuple[np.ndarray | None, int]:
    """Normalize group IDs and retain the full scale-statistics axis.

    The explicit count is a lower bound, so subsets retain absent groups.
    Empty IDs still describe one group. Missing IDs retain the explicit count
    for routing; engines allocate scale statistics only when IDs are present.
    Supply ``n_images`` at engine boundaries to check the flattened image axis.
    Existing int64 conversion semantics, including integer-valued counts, remain.
    """
    group_ids_np = None
    explicit_scale_group_count = 0
    if scale_correction_group_count is not None:
        explicit_scale_group_count = int(scale_correction_group_count)
        if (
            explicit_scale_group_count < 0
            or not np.isfinite(float(scale_correction_group_count))
            or float(scale_correction_group_count) != float(explicit_scale_group_count)
        ):
            raise ValueError(
                "scale_correction_group_count must be a non-negative integer, "
                f"got {scale_correction_group_count!r}"
            )
    if group_ids is not None:
        group_ids_np = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if n_images is not None and group_ids_np.shape != (n_images,):
            raise ValueError(f"group_ids must have shape ({n_images},), got {group_ids_np.shape}")
        if group_ids_np.size and int(np.min(group_ids_np)) < 0:
            raise ValueError("group_ids must be non-negative")
        inferred_scale_group_count = int(np.max(group_ids_np)) + 1 if group_ids_np.size else 1
        n_scale_groups = max(explicit_scale_group_count, inferred_scale_group_count)
    else:
        n_scale_groups = explicit_scale_group_count
    return group_ids_np, n_scale_groups
