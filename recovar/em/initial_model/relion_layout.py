"""InitialModel adapters for shared RELION x-half public outputs."""

from __future__ import annotations

import numpy as np

from recovar.em.initial_model.layout import _as_centered_bpref_source, _bp_slab


def relion_x_public_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the shared RELION-x-half public-layout conversion.

    The shared M-step expands native RELION ``(z, y, xhalf)`` storage to a
    full cube and transposes it to RECOVAR's public ``(x, y, z)`` order.
    InitialModel consumes a native BPref again, so undo that transpose before
    selecting the positive-x slab.  The generic dense converter must remain
    unchanged because its input is already a centered RECOVAR Fourier cube.
    """

    if padding_factor not in (1, 2):
        raise NotImplementedError(f"padding_factor must be 1 or 2, got {padding_factor}")
    if r_max < 0:
        raise ValueError(f"r_max must be non-negative, got {r_max}")

    data_cube, data_center, data_radius = _as_centered_bpref_source(
        Ft_y,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    weight_cube, weight_center, weight_radius = _as_centered_bpref_source(
        Ft_ctf,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    if (data_center, data_radius) != (weight_center, weight_radius):
        raise ValueError("data and weight accumulators use different centered layouts")

    bp_data = _bp_slab(data_cube.transpose(2, 1, 0), data_radius, data_center)
    bp_weight = _bp_slab(weight_cube.transpose(2, 1, 0), weight_radius, weight_center)
    bp_weight_f64 = np.asarray(bp_weight.real, dtype=np.float64).copy()
    bp_weight_f64[np.abs(bp_weight_f64) < 1e-15] = 0.0
    return np.asarray(bp_data, dtype=np.complex128).copy(), bp_weight_f64
