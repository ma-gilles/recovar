"""SOLVAR low-rank variability optimization."""

from .solvar import (
    SolvarFitResult,
    fit,
    make_loading_from_basis,
    make_random_loading,
    project_loading_to_mask,
    solvar_image_losses,
)

__all__ = [
    "SolvarFitResult",
    "fit",
    "make_loading_from_basis",
    "make_random_loading",
    "project_loading_to_mask",
    "solvar_image_losses",
]
