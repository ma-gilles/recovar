"""Compatibility exports for the old InitialModel GPU module name.

The previous implementation mixed a single-volume ``run_em`` E-step,
hard-coded K=1 VDAM logic, and layout conversion in one file. InitialModel
now routes through ``dense_adapter`` so K=1 and K>1 share the same joint
class x pose E-step. New code should import from ``dense_adapter`` and
``layout`` directly.
"""

from __future__ import annotations

from .dense_adapter import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    dense_initial_model_expectation_step,
    run_dense_initial_model_estep,
    split_pseudo_halfset_particle_ids,
)
from .layout import bpref_to_run_em_output, relion_bpref_frame_scales, run_em_output_to_bpref

_split_halfset_particle_ids = split_pseudo_halfset_particle_ids


def run_iter_gpu_vdam(*args, **kwargs):
    """Deprecated diagnostic path retained for import compatibility."""

    raise NotImplementedError(
        "recovar.em.initial_model.gpu_pipeline.run_iter_gpu_vdam is stale on "
        "the native InitialModel branch; use scripts/run_ab_initio.py or "
        "recovar.em.initial_model.driver.run_native_initial_model instead"
    )


__all__ = [
    "DenseInitialModelEstepConfig",
    "DenseInitialModelEstepResult",
    "dense_initial_model_expectation_step",
    "run_dense_initial_model_estep",
    "run_iter_gpu_vdam",
    "split_pseudo_halfset_particle_ids",
    "_split_halfset_particle_ids",
    "run_em_output_to_bpref",
    "bpref_to_run_em_output",
    "relion_bpref_frame_scales",
]
