"""The native InitialModel driver's STAR / artifact I/O lives in ``star_io``."""

import inspect

from recovar.em.vdam import driver, star_io

MOVED = ("NativeParticleState", "_particle_state_from_star", "_write_model_star", "_write_data_star", "_write_iteration_artifacts", "_write_final_outputs", "_star_column", "_stack_star_pair", "_experiment_read_order", "_micrograph_sort_order")


def test_star_io_owns_the_cluster_and_driver_only_imports_it():
    driver_src = inspect.getsource(driver)
    for name in MOVED:
        assert hasattr(star_io, name) and inspect.getmodule(getattr(star_io, name)) is star_io
        assert f"\ndef {name}(" not in driver_src and f"\nclass {name}(" not in driver_src
    assert "from recovar.em.vdam.star_io import (" in driver_src
    assert driver._write_iteration_artifacts is star_io._write_iteration_artifacts
