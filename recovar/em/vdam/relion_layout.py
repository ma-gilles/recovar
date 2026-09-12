"""InitialModel adapters for shared RELION x-half public outputs."""

from __future__ import annotations

import numpy as np

from recovar.em.vdam.layout import _bp_slab, _bpref_slab_outputs, _centered_bpref_sources


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

    data_cube, weight_cube, center, radius = _centered_bpref_sources(
        Ft_y,
        Ft_ctf,
        ori_size=ori_size,
        r_max=r_max,
        padding_factor=padding_factor,
    )
    return _bpref_slab_outputs(
        _bp_slab(data_cube.transpose(2, 1, 0), radius, center),
        _bp_slab(weight_cube.transpose(2, 1, 0), radius, center),
    )
