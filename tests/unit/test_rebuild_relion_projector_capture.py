import numpy as np
import pytest

from scripts.parity.rebuild_relion_projector_capture import _array_metrics, _validate_bind_path

pytestmark = pytest.mark.unit


def test_projector_rebuild_metrics_classify_exact_complex64():
    captured = np.asarray([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
    metrics = _array_metrics(captured.astype(np.complex128), captured)

    assert metrics["exact_after_complex64"]
    assert metrics["within_one_ulp_after_complex64"]
    assert metrics["max_real_ulp_after_complex64"] == 0
    assert metrics["max_imag_ulp_after_complex64"] == 0


def test_projector_rebuild_metrics_classify_one_ulp_as_numerical():
    captured = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    rebuilt = np.asarray(
        [np.nextafter(captured.real, np.float32(np.inf))[0] + 2.0j],
        dtype=np.complex128,
    )
    metrics = _array_metrics(rebuilt, captured)

    assert not metrics["exact_after_complex64"]
    assert metrics["within_one_ulp_after_complex64"]
    assert metrics["max_real_ulp_after_complex64"] == 1
    assert metrics["n_real_components_over_one_ulp"] == 0


def test_projector_rebuild_metrics_leave_two_ulp_unresolved():
    captured = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    one_ulp = np.nextafter(captured.real, np.float32(np.inf))
    two_ulp = np.nextafter(one_ulp, np.float32(np.inf))
    rebuilt = np.asarray([two_ulp[0] + 2.0j], dtype=np.complex128)
    metrics = _array_metrics(rebuilt, captured)

    assert not metrics["within_one_ulp_after_complex64"]
    assert metrics["max_real_ulp_after_complex64"] == 2
    assert metrics["n_real_components_over_one_ulp"] == 1


def test_projector_rebuild_binding_must_resolve_inside_requested_build(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    module = build / "_relion_bind_core.so"
    module.touch()

    assert _validate_bind_path(module, str(build)) == module.resolve()
    with pytest.raises(RuntimeError, match="outside RECOVAR_RELION_BIND_BUILD_DIR"):
        _validate_bind_path(module, str(tmp_path / "other_build"))
