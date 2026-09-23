"""FFI target coverage of the EM CUDA library (split from test_cuda_lib_symbol_check.py, relax split P2)."""

import pytest

from recovar import cuda_backproject as cb
from recovar.em.cuda import kernels as em_cuda_kernels

pytestmark = pytest.mark.unit


def test_em_ffi_registrations_cover_all_em_target_constants():
    """Every EM target is eager or explicitly optional in the EM library; optional ABIs stay lazy."""
    recovar_targets = {v for k, v in vars(cb).items() if k.startswith("_TARGET_") and isinstance(v, str)}
    # _TARGET_PROJECT_INDEXED is recovar's (imported by the EM wrappers)
    em_target_constants = {
        v
        for k, v in vars(em_cuda_kernels).items()
        if k.startswith("_TARGET_") and isinstance(v, str) and v not in recovar_targets
    }
    em_targets_in_table = {target for target, _symbol in em_cuda_kernels._FFI_REGISTRATIONS}
    em_optional_targets = {
        *em_cuda_kernels._OPTIONAL_FFI_REGISTRATIONS,
        em_cuda_kernels._TARGET_RELION_WAVG_NATIVE_PREFIX_F32,
        em_cuda_kernels._TARGET_RELION_WAVG_NATIVE_PREFIX_DEBUG_F32,
        em_cuda_kernels._TARGET_RELION_COARSE_POSTERIOR_TRANSACTION_F32,
        em_cuda_kernels._TARGET_RELION_COARSE_SHARED_PRETRANSLATED_RUNTIME_F32,
    }
    assert em_targets_in_table.isdisjoint(em_optional_targets)
    assert em_target_constants == em_targets_in_table | em_optional_targets


def test_em_ffi_registrations_have_unique_targets_and_symbols():
    targets = [t for t, _s in em_cuda_kernels._FFI_REGISTRATIONS]
    symbols = [s for _t, s in em_cuda_kernels._FFI_REGISTRATIONS]
    assert len(set(targets)) == len(targets)
    assert len(set(symbols)) == len(symbols)
