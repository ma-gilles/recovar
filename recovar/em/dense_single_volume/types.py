"""Typed containers for the dense single-volume EM path."""

from typing import NamedTuple

import jax


class MeanStats(NamedTuple):
    """Accumulated M-step sufficient statistics.

    Both fields are additive over image batches and across devices,
    making this the natural unit for distributed all-reduce.

    Attributes:
        Ft_y: (volume_size,) complex -- weighted backprojected images.
        Ft_ctf: (volume_size,) real/complex -- weighted CTF^2 backprojection.
    """

    Ft_y: jax.Array
    Ft_ctf: jax.Array
