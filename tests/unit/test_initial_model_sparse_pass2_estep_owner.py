"""The InitialModel sparse pass-2 E-step and the shared E-step records have their own owners."""

import inspect

from recovar.em.initial_model import dense_adapter, estep_common, sparse_pass2_estep


def test_owners_hold_the_definitions_and_the_adapter_routes_to_them():
    adapter_src = inspect.getsource(dense_adapter)
    for name in ("_run_sparse_pass2_initial_model_estep", "_sparse_pass2_estep_meta", "_initial_model_pass2_layout", "_pop_sparse_pass2_options"):
        assert inspect.getmodule(getattr(sparse_pass2_estep, name)) is sparse_pass2_estep and f"\ndef {name}(" not in adapter_src
    for name in ("DenseInitialModelEstepConfig", "DenseInitialModelEstepResult", "_estep_meta", "_select_image_rows"):
        assert inspect.getmodule(getattr(estep_common, name)) is estep_common
    assert dense_adapter._run_sparse_pass2_initial_model_estep is sparse_pass2_estep._run_sparse_pass2_initial_model_estep
    assert dense_adapter.DenseInitialModelEstepConfig is estep_common.DenseInitialModelEstepConfig
    for mod in (sparse_pass2_estep, estep_common):
        assert "initial_model.dense_adapter import" not in inspect.getsource(mod)
