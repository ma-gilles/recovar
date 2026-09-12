"""Host input contracts for externally normalized EM scoring."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def optional_normalization_vector(value, *, name: str, n_images: int) -> np.ndarray | None:
    """Convert a present input to F64 without flattening its image axis.

    Domain checks belong to the caller: log inputs, probabilities and support
    thresholds have different valid values and mutual-exclusion rules.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (n_images,):
        raise ValueError(f"{name} must have shape ({n_images},), got {array.shape}")
    return array


class LocalNormalizationInputs(NamedTuple):
    """Validated external normalization and reconstruction support inputs."""

    log_z: np.ndarray | None
    log_evidence: np.ndarray | None
    max_posterior: np.ndarray | None
    reconstruction_threshold: np.ndarray | None


def prepare_local_normalization_inputs(
    *,
    n_images: int,
    normalization_log_z=None,
    normalization_log_evidence=None,
    normalization_max_posterior=None,
    reconstruction_probability_threshold=None,
) -> LocalNormalizationInputs:
    """Validate local-scoring inputs in their original exception order.

    Log normalizers are shape-checked only. Pmax must be finite in (0,1] and
    exclusive with either log normalizer. Reconstruction thresholds are finite
    and nonnegative, independent of the selected normalization mode.
    """
    log_z = optional_normalization_vector(normalization_log_z, name="normalization_log_z", n_images=n_images)
    log_evidence = optional_normalization_vector(
        normalization_log_evidence, name="normalization_log_evidence", n_images=n_images
    )
    if log_z is not None and log_evidence is not None:
        raise ValueError("Provide only one of normalization_log_z or normalization_log_evidence")
    max_posterior = optional_normalization_vector(
        normalization_max_posterior, name="normalization_max_posterior", n_images=n_images
    )
    if max_posterior is not None:
        if not np.all(np.isfinite(max_posterior)) or np.any(max_posterior <= 0.0) or np.any(max_posterior > 1.0):
            raise ValueError("normalization_max_posterior must contain finite probabilities in (0, 1]")
        if log_z is not None or log_evidence is not None:
            raise ValueError(
                "normalization_max_posterior is mutually exclusive with external log normalization",
            )
    threshold = optional_normalization_vector(
        reconstruction_probability_threshold, name="reconstruction_probability_threshold", n_images=n_images
    )
    if threshold is not None:
        if not np.all(np.isfinite(threshold)):
            raise ValueError("reconstruction_probability_threshold must be finite")
        if np.any(threshold < 0.0):
            raise ValueError("reconstruction_probability_threshold must be non-negative")
    return LocalNormalizationInputs(
        log_z,
        log_evidence,
        max_posterior,
        threshold,
    )
