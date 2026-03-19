"""Per-image metadata storage for poses and CTF parameters."""

from __future__ import annotations

import numpy as np


class MetadataStore:
    """Per-image metadata store.

    This class owns only metadata arrays. It has no loading, iteration,
    subset-view, or halfset logic.
    """

    __slots__ = (
        "_rotation_matrices",
        "_translations",
        "_ctf_params",
        "rotation_dtype",
        "ctf_dtype",
        "real_dtype",
    )

    def __init__(
        self,
        rotation_matrices,
        translations,
        ctf_params,
        *,
        rotation_dtype=np.float32,
        ctf_dtype=np.float32,
        real_dtype=np.float32,
    ):
        rotation_matrices = np.asarray(rotation_matrices, dtype=rotation_dtype)
        translations = np.asarray(translations, dtype=real_dtype)
        ctf_params = np.asarray(ctf_params, dtype=ctf_dtype)

        if rotation_matrices.ndim != 3 or rotation_matrices.shape[1:] != (3, 3):
            raise ValueError(
                "rotation_matrices must have shape (N, 3, 3), "
                f"got {rotation_matrices.shape}"
            )
        if translations.ndim != 2 or translations.shape[1] != 2:
            raise ValueError(
                "translations must have shape (N, 2), "
                f"got {translations.shape}"
            )
        if ctf_params.ndim != 2:
            raise ValueError(f"ctf_params must have shape (N, C), got {ctf_params.shape}")

        n_images = rotation_matrices.shape[0]
        if translations.shape[0] != n_images or ctf_params.shape[0] != n_images:
            raise ValueError(
                "metadata arrays must have matching leading dimension: "
                f"rots={rotation_matrices.shape[0]}, "
                f"trans={translations.shape[0]}, ctf={ctf_params.shape[0]}"
            )

        self._rotation_matrices = rotation_matrices
        self._translations = translations
        self._ctf_params = ctf_params
        self.rotation_dtype = rotation_dtype
        self.ctf_dtype = ctf_dtype
        self.real_dtype = real_dtype

    @property
    def n_images(self):
        return self._rotation_matrices.shape[0]

    def get_batch(self, indices):
        return (
            self._rotation_matrices[indices],
            self._translations[indices],
            self._ctf_params[indices],
        )

    def get_ctf_column(self, col):
        return self._ctf_params[:, col]

    def get_ctf_params_copy(self):
        return self._ctf_params.copy()

    def get_rotations_copy(self):
        return self._rotation_matrices.copy()

    def set_poses(self, rotation_matrices, translations):
        rotation_matrices = np.asarray(rotation_matrices, dtype=self.rotation_dtype)
        translations = np.asarray(translations, dtype=self.real_dtype)
        if rotation_matrices.shape != self._rotation_matrices.shape:
            raise ValueError(
                "rotation_matrices shape must stay fixed at "
                f"{self._rotation_matrices.shape}, got {rotation_matrices.shape}"
            )
        if translations.shape != self._translations.shape:
            raise ValueError(
                "translations shape must stay fixed at "
                f"{self._translations.shape}, got {translations.shape}"
            )
        self._rotation_matrices = rotation_matrices
        self._translations = translations

    def set_ctf(self, ctf_params):
        ctf_params = np.asarray(ctf_params, dtype=self.ctf_dtype)
        if ctf_params.shape != self._ctf_params.shape:
            raise ValueError(
                "ctf_params shape must stay fixed at "
                f"{self._ctf_params.shape}, got {ctf_params.shape}"
            )
        self._ctf_params = ctf_params

    def set_ctf_column(self, col, values):
        self._ctf_params[:, col] = values

    def scale_ctf_column(self, col, multipliers):
        self._ctf_params[:, col] *= multipliers

    def scale_ctf_element(self, row_indices, col, multiplier):
        self._ctf_params[row_indices, col] *= multiplier

    def subset(self, indices):
        return MetadataStore(
            self._rotation_matrices[indices],
            self._translations[indices],
            self._ctf_params[indices],
            rotation_dtype=self.rotation_dtype,
            ctf_dtype=self.ctf_dtype,
            real_dtype=self.real_dtype,
        )


Metadata = MetadataStore
