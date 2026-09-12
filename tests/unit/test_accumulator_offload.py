"""Buffer ownership and ordering for the controller's explicit offload boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense import score_outputs

pytestmark = pytest.mark.unit


class DeviceArray:
    def __init__(self, name, host, events, delete_error=None):
        self.name = name
        self.host = host
        self.dtype = host.dtype
        self.size = host.size
        self.events = events
        self.delete_error = delete_error

    def delete(self):
        self.events.append(("delete", self.name))
        if self.delete_error is not None:
            raise self.delete_error


def test_offload_preserves_transfer_release_and_collection_order(monkeypatch):
    events = []
    y = np.array([1 + 2j], dtype=np.complex64)
    ctf = np.array([3], dtype=np.float32)
    result = SimpleNamespace(
        Ft_y=DeviceArray("y", y, events),
        Ft_ctf=DeviceArray("ctf", ctf, events),
        mstep_full_half_axis=0,
    )

    def device_get(value):
        events.append(("get", value.name))
        return value.host

    def log_info(message, *args):
        assert result.Ft_y is y and result.Ft_ctf is ctf
        events.append(("log", args))

    monkeypatch.setattr(score_outputs.jax, "device_get", device_get)
    monkeypatch.setattr(score_outputs.gc, "collect", lambda: events.append(("gc",)))
    actual = score_outputs._maybe_host_offload_half0_local_accumulators(
        half_index=0, use_local=True, k_class_enabled=False, score_result=result,
        log=SimpleNamespace(info=log_info),
    )
    assert actual is result
    assert events == [
        ("get", "y"), ("delete", "y"), ("get", "ctf"), ("delete", "ctf"),
        ("gc",), ("log", (y.nbytes / 1e9, ctf.nbytes / 1e9)),
    ]


@pytest.mark.parametrize(
    "half_index,use_local,k_class_enabled,axis",
    [(1, True, False, 0), (0, False, False, 0), (0, True, True, 0), (0, True, False, None)],
)
def test_ineligible_results_are_untouched(monkeypatch, half_index, use_local, k_class_enabled, axis):
    def unexpected(*args, **kwargs):
        pytest.fail("Ineligible result must not transfer, collect or log")

    monkeypatch.setattr(score_outputs.jax, "device_get", unexpected)
    monkeypatch.setattr(score_outputs.gc, "collect", unexpected)
    y, ctf = object(), object()
    result = SimpleNamespace(Ft_y=y, Ft_ctf=ctf, mstep_full_half_axis=axis)
    actual = score_outputs._maybe_host_offload_half0_local_accumulators(
        half_index=half_index, use_local=use_local, k_class_enabled=k_class_enabled,
        score_result=result, log=SimpleNamespace(info=unexpected),
    )
    assert actual is result and result.Ft_y is y and result.Ft_ctf is ctf


def test_host_array_is_returned_without_copy_or_device_access(monkeypatch):
    def unexpected(*args):
        pytest.fail("Host arrays must not pass through device_get")

    monkeypatch.setattr(score_outputs.jax, "device_get", unexpected)
    host = np.arange(4, dtype=np.float32)[::2]
    assert score_outputs._host_offload_array(host) is host


@pytest.mark.parametrize("delete_error", [None, RuntimeError("already released"), ValueError("unexpected error")])
def test_only_runtime_delete_errors_are_suppressed(monkeypatch, delete_error):
    events = []
    host = np.array([2.0], dtype=np.float32)
    value = DeviceArray("y", host, events, delete_error)
    monkeypatch.setattr(score_outputs.jax, "device_get", lambda value: value.host)
    if isinstance(delete_error, ValueError):
        with pytest.raises(ValueError, match="unexpected error"):
            score_outputs._host_offload_array(value)
    else:
        assert score_outputs._host_offload_array(value) is host
    assert events == [("delete", "y")]


def test_transfer_failure_does_not_delete_buffer(monkeypatch):
    events = []
    value = DeviceArray("y", np.ones(1, dtype=np.float32), events)

    def failed_get(value):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(score_outputs.jax, "device_get", failed_get)
    with pytest.raises(RuntimeError, match="transfer failed"):
        score_outputs._host_offload_array(value)
    assert events == []
