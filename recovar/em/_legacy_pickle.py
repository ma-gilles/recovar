"""Read saved Python objects from the retired refinement package namespace.

The implementation lives under recovar.em. These lazy module records preserve
old pickle GLOBAL lookups without importing engines or keeping a second source
tree. Only former class-owner modules are registered. New code imports current
owners directly; the three explicitly pinned result/support globals keep their
historical serialized names.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_CLASS_MODULES = {
    "recovar.em.dense_single_volume.batch_planning": "recovar.em.helpers.batch_planning",
    "recovar.em.dense_single_volume.bpref_transaction": "recovar.em.helpers.bpref_transaction",
    "recovar.em.dense_single_volume.dense_big_jit": "recovar.em.dense.dense_big_jit",
    "recovar.em.dense_single_volume.em_engine": "recovar.em.dense.em_engine",
    "recovar.em.dense_single_volume.fixed_capacity_local": "recovar.em.local.fixed_capacity_local",
    "recovar.em.dense_single_volume.frozen_boundary": "recovar.em.diagnostics.frozen_boundary",
    "recovar.em.dense_single_volume.half_scoring": "recovar.em.dense.half_scoring",
    "recovar.em.dense_single_volume.helpers.bpref_diagnostics": "recovar.em.diagnostics.bpref_diagnostics",
    "recovar.em.dense_single_volume.helpers.coarse_device_rescore": "recovar.em.scoring.coarse_device_rescore",
    "recovar.em.dense_single_volume.helpers.coarse_device_selection": "recovar.em.scoring.coarse_device_selection",
    "recovar.em.dense_single_volume.helpers.coarse_gaussian_diagnostics": "recovar.em.diagnostics.coarse_gaussian_diagnostics",
    "recovar.em.dense_single_volume.helpers.coarse_gaussian_gemm": "recovar.em.scoring.coarse_gaussian_gemm",
    "recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid": "recovar.em.scoring.coarse_gemm_hybrid",
    "recovar.em.dense_single_volume.helpers.coarse_gemm_streaming": "recovar.em.scoring.coarse_gemm_streaming",
    "recovar.em.dense_single_volume.helpers.coarse_partition": "recovar.em.scoring.coarse_partition",
    "recovar.em.dense_single_volume.helpers.compact_candidate_capture": "recovar.em.diagnostics.compact_candidate_capture",
    "recovar.em.dense_single_volume.helpers.compact_candidates": "recovar.em.scoring.compact_candidates",
    "recovar.em.dense_single_volume.helpers.convergence": "recovar.em.helpers.convergence",
    "recovar.em.dense_single_volume.helpers.deferred_vdam_host_pack": "recovar.em.helpers.deferred_vdam_host_pack",
    "recovar.em.dense_single_volume.helpers.dtype_policy": "recovar.em.helpers.dtype_policy",
    "recovar.em.dense_single_volume.helpers.expected_accuracy": "recovar.em.helpers.expected_accuracy",
    "recovar.em.dense_single_volume.helpers.flat_local_rows": "recovar.em.local.flat_local_rows",
    "recovar.em.dense_single_volume.helpers.fourier_window": "recovar.em.helpers.fourier_window",
    "recovar.em.dense_single_volume.helpers.half_spectrum": "recovar.em.helpers.half_spectrum",
    "recovar.em.dense_single_volume.helpers.iteration_history": "recovar.em.helpers.iteration_history",
    "recovar.em.dense_single_volume.helpers.normalization_inputs": "recovar.em.helpers.normalization_inputs",
    "recovar.em.dense_single_volume.helpers.orientation_priors": "recovar.em.helpers.orientation_priors",
    "recovar.em.dense_single_volume.helpers.projection_cache": "recovar.em.helpers.projection_cache",
    "recovar.em.dense_single_volume.helpers.relion_coarse_operands": "recovar.em.relion.relion_coarse_operands",
    "recovar.em.dense_single_volume.helpers.relion_projector_capture": "recovar.em.diagnostics.relion_projector_capture",
    "recovar.em.dense_single_volume.helpers.score_constraints": "recovar.em.scoring.score_constraints",
    "recovar.em.dense_single_volume.helpers.scoring": "recovar.em.scoring.scoring",
    "recovar.em.dense_single_volume.helpers.significance": "recovar.em.scoring.significance",
    "recovar.em.dense_single_volume.helpers.significant_samples": "recovar.em.scoring.significant_samples",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_bucket_plan": "recovar.em.sparse_pass2.sparse_pass2_bucket_plan",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed": "recovar.em.sparse_pass2.sparse_pass2_bucketed",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_dump": "recovar.em.diagnostics.sparse_pass2_dump",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_scoring": "recovar.em.sparse_pass2.sparse_pass2_scoring",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_wavg": "recovar.em.sparse_pass2.sparse_pass2_wavg",
    "recovar.em.dense_single_volume.helpers.sparse_pass2_window": "recovar.em.sparse_pass2.sparse_pass2_window",
    "recovar.em.dense_single_volume.helpers.state_swap_runtime": "recovar.em.diagnostics.state_swap_runtime",
    "recovar.em.dense_single_volume.helpers.timing": "recovar.em.helpers.timing",
    "recovar.em.dense_single_volume.helpers.types": "recovar.em.helpers.types",
    "recovar.em.dense_single_volume.iteration_loop": "recovar.em.refinement.iteration_loop",
    "recovar.em.dense_single_volume.k_class": "recovar.em.classification.k_class",
    "recovar.em.dense_single_volume.k_class_results": "recovar.em.classification.k_class_results",
    "recovar.em.dense_single_volume.local_big_jit": "recovar.em.local.local_big_jit",
    "recovar.em.dense_single_volume.local_bpref_capture": "recovar.em.diagnostics.local_bpref_capture",
    "recovar.em.dense_single_volume.local_bucket_stages": "recovar.em.local.local_bucket_stages",
    "recovar.em.dense_single_volume.local_caches": "recovar.em.local.local_caches",
    "recovar.em.dense_single_volume.local_debug": "recovar.em.diagnostics.local_debug",
    "recovar.em.dense_single_volume.local_layout": "recovar.em.local.local_layout",
    "recovar.em.dense_single_volume.local_projection_cache": "recovar.em.local.local_projection_cache",
    "recovar.em.dense_single_volume.local_search_iteration": "recovar.em.local.local_search_iteration",
    "recovar.em.dense_single_volume.local_timing": "recovar.em.local.local_timing",
    "recovar.em.dense_single_volume.mean_helpers": "recovar.em.refinement.mean_helpers",
    "recovar.em.dense_single_volume.ppca_bridge": "recovar.em.ppca_refinement.ppca_bridge",
    "recovar.em.dense_single_volume.refinement_options": "recovar.em.refinement.refinement_options",
    "recovar.em.dense_single_volume.relion_normalization": "recovar.em.relion.relion_normalization",
    "recovar.em.dense_single_volume.relion_replay": "recovar.em.diagnostics.relion_replay",
    "recovar.em.dense_single_volume.relion_worker_scale": "recovar.em.relion.relion_worker_scale",
    "recovar.em.dense_single_volume.score_outputs": "recovar.em.dense.score_outputs",
    "recovar.em.initial_model.native_options": "recovar.em.vdam.native_options",
    "recovar.em.initial_model.gt_registration": "recovar.em.vdam.gt_registration",
    "recovar.em.initial_model.schedules": "recovar.em.vdam.schedules",
    "recovar.em.initial_model.estep_common": "recovar.em.vdam.estep_common",
    "recovar.em.initial_model.gt_metrics": "recovar.em.vdam.gt_metrics",
    "recovar.em.initial_model.bootstrap_iref": "recovar.em.vdam.bootstrap_iref",
    "recovar.em.initial_model.mstep_accumulator": "recovar.em.vdam.mstep_accumulator",
    "recovar.em.initial_model.native_sampling": "recovar.em.vdam.native_sampling",
    "recovar.em.initial_model.subset": "recovar.em.vdam.subset",
    "recovar.em.initial_model.e_step": "recovar.em.vdam.e_step",
    "recovar.em.initial_model.star_io": "recovar.em.vdam.star_io",
    "recovar.em.initial_model.align_symmetry": "recovar.em.vdam.align_symmetry",
    "recovar.em.initial_model.state": "recovar.em.vdam.state",
    "recovar.em.initial_model.driver": "recovar.em.vdam.driver",
}


def _saved_module(old_name: str, owner: str) -> ModuleType:
    module = ModuleType(old_name, "Compatibility namespace for saved Python objects.")
    module.__package__ = old_name.rpartition(".")[0]

    def resolve(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(importlib.import_module(owner), name)

    module.__getattr__ = resolve
    return module


def install(parent: ModuleType) -> None:
    """Register lazy historical class owners; load no refinement implementation."""
    for suffix in ("dense_single_volume", "initial_model"):
        prefix = parent.__name__ + "." + suffix
        package = ModuleType(prefix, "Retired namespace for saved-object compatibility.")
        package.__path__ = []
        package.__package__ = prefix
        sys.modules.setdefault(prefix, package)
        setattr(parent, suffix, sys.modules[prefix])
        if suffix == "dense_single_volume":
            helpers = ModuleType(prefix + ".helpers")
            helpers.__path__ = []
            helpers.__package__ = helpers.__name__
            sys.modules.setdefault(helpers.__name__, helpers)
            package.helpers = sys.modules[helpers.__name__]
    for old_name, owner in _CLASS_MODULES.items():
        module = sys.modules.setdefault(old_name, _saved_module(old_name, owner))
        parent_name, _, leaf = old_name.rpartition(".")
        setattr(sys.modules[parent_name], leaf, module)
