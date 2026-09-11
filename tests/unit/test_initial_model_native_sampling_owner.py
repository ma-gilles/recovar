"""The native InitialModel driver's sampling state/plan/accuracy logic lives in ``native_sampling``; its options in ``native_options``."""

import inspect

from recovar.em.initial_model import driver, native_options, native_sampling

SAMPLING = ("NativeOpticsState", "NativeSamplingPlan", "NativeSamplingState", "_build_sampling_plan", "_initial_sampling_state", "_estimate_native_sampling_accuracy", "_relion_update_native_sampling_state", "_prepare_native_sampling_for_iteration", "_random_perturbation_for_iteration")


def test_owners_hold_the_definitions_and_driver_only_imports_them():
    driver_src = inspect.getsource(driver)
    for name in SAMPLING:
        assert inspect.getmodule(getattr(native_sampling, name)) is native_sampling
        assert f"\ndef {name}(" not in driver_src and f"\nclass {name}(" not in driver_src
    assert inspect.getmodule(native_options.NativeInitialModelOptions) is native_options
    assert "\nclass NativeInitialModelOptions" not in driver_src
    assert driver.NativeInitialModelOptions is native_options.NativeInitialModelOptions
    assert "import recovar.em.initial_model.driver" not in inspect.getsource(native_sampling)
