from __future__ import annotations

import inspect

import numpy as np

from scripts import replay_final_bpref_dump


def test_replay_final_bpref_dump_defaults_old_dumps_to_native_half_axis():
    assert replay_final_bpref_dump.resolve_tau2_full_half_axis({}) == -1


def test_replay_final_bpref_dump_reads_recorded_relion_x_half_axis():
    dump = {"mstep_full_half_axis": np.asarray(0, dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_tau2_full_half_axis(dump) == 0


def test_replay_final_bpref_dump_axis_override_wins():
    dump = {"mstep_full_half_axis": np.asarray(0, dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_tau2_full_half_axis(dump, override=-1) == -1


def test_replay_final_bpref_dump_resolves_recorded_accumulator_shape():
    dump = {"mstep_accumulator_shape": np.asarray([259, 259, 259], dtype=np.int32)}

    assert replay_final_bpref_dump.resolve_mstep_accumulator_shape(dump, (128, 128, 128), 2) == (
        259,
        259,
        259,
    )


def test_replay_final_bpref_dump_old_shape_defaults_to_even_padding():
    assert replay_final_bpref_dump.resolve_mstep_accumulator_shape({}, (128, 128, 128), 2) == (
        256,
        256,
        256,
    )


def test_replay_final_bpref_dump_uses_joined_half_weight_sum():
    source = inspect.getsource(replay_final_bpref_dump.main)

    assert 'weight_combination="sum"' in source
    assert '"tau2_weight_combination": "sum"' in source


def test_replay_fsc_uses_canonical_non_nyquist_shell_range():
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((16, 16, 16))

    fsc = replay_final_bpref_dump.shell_fsc(volume, volume)

    assert fsc.shape == (16 // 2 - 1,)
    np.testing.assert_allclose(fsc, 1.0, atol=1e-12)
