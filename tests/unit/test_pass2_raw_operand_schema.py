"""Round-trip the effective K1 raw operands through both diagnostic schemas."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import bpref_diagnostics
from recovar.em.diagnostics import pass2 as pass2_diagnostics

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raw_operand_capture_preserves_rows_dtypes_and_padding(
    tmp_path, monkeypatch, selected, dtype,
):
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_RAW_OPERANDS", "1")
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_CURRENT_SIZE", raising=False)
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_ITERATION", raising=False)
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_ROTATION_ROWS", raising=False)
    if selected:
        monkeypatch.setenv("RECOVAR_PASS2_DUMP_ROTATION_ROWS", "2,0")
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 3)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 2)

    # Image 42 is the second batch row, with only three of four padded rotations.
    rows = [0, 2] if selected else [0, 1, 2]
    raw = (np.arange(16).reshape(2, 4, 2) + 0.1).astype(dtype)
    shifted = (np.arange(12).reshape(2, 2, 3) + 0.2).astype(dtype) * (1 + 2j)
    ctf = (np.arange(6).reshape(2, 3) + 0.3).astype(dtype)
    projections = (np.arange(24).reshape(2, 4, 3) + 0.4).astype(dtype) * (2 - 1j)
    weights = np.asarray([0.5, 1.0, 0.5], dtype=dtype)
    highres = np.asarray([0.0, 1.1], dtype=dtype)
    lookup = np.asarray([-1, 0, 1, 2], dtype=np.int32)
    rotations = np.broadcast_to(np.eye(3), (3, 3, 3))
    count = pass2_diagnostics._maybe_dump_pass2_bucket(
        experiment_dataset=SimpleNamespace(dataset_indices=np.array([9, 42])),
        image_indices=np.array([0, 1]),
        per_image_inputs={
            "oversampled_rots": [rotations, rotations],
            "oversampled_rot_indices": [np.arange(3), np.arange(3)],
            "parent_map": [np.arange(3), np.arange(3)],
        },
        current_size=4,
        n_fine_trans=2,
        fine_translations=np.zeros((2, 2), dtype=dtype),
        scores=-raw,
        probs=np.full(raw.shape, 0.125, dtype=dtype),
        rotation_log_prior=np.zeros((2, 4), dtype=dtype),
        translation_log_prior=np.zeros((2, 2), dtype=dtype),
        candidate_mask=np.ones(raw.shape, dtype=bool),
        ctf2_over_nv_score=ctf,
        proj_half=projections,
        half_weights_used=weights,
        window_indices=np.arange(3),
        shifted_corrected_score_split=shifted,
        relion_raw_diff2=raw,
        relion_full_to_compact=lookup,
        relion_highres_xi2_half=highres,
    )
    assert count == 1
    assert [p.name for p in tmp_path.iterdir()] == ["pass2_orig000042_cs004.npz"]
    schema = "selected" if selected else "effective"
    expected = {
        "raw_operand_schema": np.asarray(f"recovar-k1-pass2-{schema}-raw-operands-v1"),
        "raw_operand_actual_rotation_count": np.int64(len(rows)),
        "raw_operand_raw_diff2": raw[1, rows].astype(np.float32),
        "raw_operand_shifted_corrected": shifted[1].astype(np.complex64),
        "raw_operand_corr_img_score": ctf[1].astype(np.float32),
        "raw_operand_proj_half": projections[1, rows].astype(np.complex64),
        "raw_operand_half_weights": weights.astype(np.float32),
        "raw_operand_relion_full_to_compact": lookup,
        "raw_operand_highres_xi2_half": np.float32(highres[1]),
        "relion_raw_diff2": raw[1, rows].astype(np.float32),
    }
    with np.load(tmp_path / "pass2_orig000042_cs004.npz", allow_pickle=False) as payload:
        assert {k for k in payload.files if k.startswith("raw_operand_")} == {
            k for k in expected if k.startswith("raw_operand_")
        }
        for key, value in expected.items():
            array = np.asarray(value)
            assert payload[key].dtype == array.dtype, key
            assert payload[key].shape == array.shape, key
            assert payload[key].tobytes() == array.tobytes(), key
        assert int(payload["original_index"]) == 42
        assert int(payload["local_index"]) == 1
        assert int(payload["iteration"]) == 3
        assert int(payload["half"]) == 2


@pytest.mark.parametrize("writer", [
    pass2_diagnostics._maybe_dump_pass2_bucket,
    pass2_diagnostics._maybe_dump_k_class_pass2_bucket,
])
@pytest.mark.parametrize("overrides,current_size", [
    ({"DIR": None, "ORIGINAL_INDICES": "invalid"}, 4),
    ({"ORIGINAL_INDICES": None, "CURRENT_SIZE": "invalid"}, 4),
    ({"CURRENT_SIZE": "5", "ITERATION": "invalid"}, 4),
    ({"CURRENT_SIZE": "invalid"}, None),
    ({"ITERATION": "4"}, 4),
])
def test_inactive_pass2_writer_preserves_validation_short_circuit(
    monkeypatch, tmp_path, writer, overrides, current_size,
):
    # Every required data operand is unusable: an inactive writer must return
    # before touching arrays, creating files or parsing later invalid controls.
    for name in ("DIR", "ORIGINAL_INDICES", "CURRENT_SIZE", "ITERATION", "CLASS"):
        monkeypatch.delenv("RECOVAR_PASS2_DUMP_" + name, raising=False)
    monkeypatch.delenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", raising=False)
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_DIR", str(tmp_path / "uncreated"))
    monkeypatch.setenv("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES", "42")
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 3)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 2)
    for name, value in overrides.items():
        if value is None:
            monkeypatch.delenv("RECOVAR_PASS2_DUMP_" + name, raising=False)
        else:
            monkeypatch.setenv("RECOVAR_PASS2_DUMP_" + name, value)
    kwargs = {
        name: None for name, parameter in inspect.signature(writer).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs["current_size"] = current_size
    assert writer(**kwargs) == 0
    assert not (tmp_path / "uncreated").exists()
