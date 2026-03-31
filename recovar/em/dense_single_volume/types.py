"""Typed containers for the dense single-volume EM path."""

from typing import NamedTuple

import jax
import numpy as np


class DensePoseGrid(NamedTuple):
    """Rotation + translation grids for dense E-step search.

    Attributes:
        rotations: (n_rot, 3, 3) float32 rotation matrices.
        translations: (n_trans, 2) float32 in-plane translations (pixels).
    """

    rotations: np.ndarray
    translations: np.ndarray

    @property
    def n_rotations(self) -> int:
        return self.rotations.shape[0]

    @property
    def n_translations(self) -> int:
        return self.translations.shape[0]

    @property
    def total_hidden(self) -> int:
        return self.n_rotations * self.n_translations


class DenseEMPlan(NamedTuple):
    """Memory plan for one dense EM iteration.

    All fields are integer batch sizes computed by plan.plan_em_iteration.

    Attributes:
        projection_batch: rotation batch size for slice_volume precompute.
        dot_product_batch: image batch size for cross-term GEMM.
        norm_batch: image batch size for CTF norm GEMM.
        prob_batch: image batch size for softmax normalization.
        image_batch: outer image batch (E_M_batches_2 level).
        mstep_image_batch: image batch size for M-step backprojection.
        mstep_rotation_batch: rotation block size for M-step accumulation.
    """

    projection_batch: int
    dot_product_batch: int
    norm_batch: int
    prob_batch: int
    image_batch: int
    mstep_image_batch: int
    mstep_rotation_batch: int


class MeanStats(NamedTuple):
    """Accumulated M-step sufficient statistics.

    Both fields are additive over image batches and across devices,
    making this the natural unit for distributed all-reduce.

    Attributes:
        Ft_y: (volume_size,) complex — weighted backprojected images.
        Ft_ctf: (volume_size,) real/complex — weighted CTF^2 backprojection.
    """

    Ft_y: jax.Array
    Ft_ctf: jax.Array
