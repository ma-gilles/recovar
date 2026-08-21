from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import validate_relion_coarse_lane_capture as validator


def _headers() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    lane = [0] * 32
    lane[5:20] = [2, 17, 31, 1, 0, 1, 0, 64, 7, 3, 2, 5, 10, 2, 1]
    lane[20] = int(np.asarray([0.25], dtype=np.float32).view(np.uint32)[0])
    lane[21:31] = [1, 1, 1, 10_000, 1_000, 123, 456, 1, 1, 1]
    operand = [0] * 40
    operand[5] = 2
    operand[8] = 1
    operand[12:17] = [64, 7, 3, 2, 5]
    operand[37:40] = [10, 2, 1]
    component = [0] * 40
    component[5] = 2
    component[8] = 1
    component[12] = 3
    component[27] = 64
    return tuple(lane), tuple(operand), tuple(component)


def _fixture(monkeypatch, *, perturb_target: bool = False, perturb_model: bool = False):
    lane_header, operand_header, component_header = _headers()
    rotation_keys = np.asarray([1, 3], dtype=np.uint64)
    local_indices = np.asarray([0, 4], dtype=np.uint64)
    lanes = np.asarray(
        [
            [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0],
            [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.zeros((5, 3), dtype=np.float32)
    for rotation, key in enumerate(rotation_keys):
        for translation in range(3):
            thread_ids = translation + np.arange(3) * 3
            target[key, translation] = validator.possible_atomic_sums(
                lanes[rotation, thread_ids],
                initial=np.float32(0.25),
            )[0]
    if perturb_target:
        target[rotation_keys[0], 0] = np.nextafter(target[rotation_keys[0], 0], np.float32(np.inf))
        while np.any(
            validator._float32_bits(
                validator.possible_atomic_sums(
                    lanes[0, np.asarray([0, 3, 6])],
                    initial=np.float32(0.25),
                )
            )
            == validator._float32_bits(target[rotation_keys[0], 0:1])[0]
        ):
            target[rotation_keys[0], 0] = np.nextafter(target[rotation_keys[0], 0], np.float32(np.inf))
    modeled = lanes.copy()
    if perturb_model:
        modeled[0, 0] = np.nextafter(modeled[0, 0], np.float32(np.inf))
    monkeypatch.setattr(validator.operand_validator, "replay_production_lanes", lambda _: modeled)
    lane = validator.CoarseLaneCapture(
        path=Path("part17_stack31.p1-lane-v1.bin"),
        sha256="0" * 64,
        header=lane_header,
        rotation_keys=rotation_keys,
        local_rotation_indices=local_indices,
        lane_partials=lanes,
    )
    operand = SimpleNamespace(
        part_id=17,
        stack_index=31,
        mpi_rank=1,
        header=operand_header,
        rotation_keys=rotation_keys,
        local_rotation_indices=local_indices,
    )
    component = SimpleNamespace(
        part_id=17,
        stack_index=31,
        mpi_rank=1,
        header=component_header,
        raw_diff2=target,
    )
    return lane, operand, component


def test_possible_atomic_sums_preserve_distinct_float32_orders() -> None:
    outcomes = validator.possible_atomic_sums(np.asarray([16_777_216.0, -16_777_216.0, 1.0], dtype=np.float32))
    assert set(outcomes.tolist()) == {0.0, 1.0}


def test_possible_atomic_sums_starts_from_recorded_initial_term() -> None:
    outcomes = validator.possible_atomic_sums(
        np.asarray([0.5, 0.25], dtype=np.float32),
        initial=np.float32(2.0),
    )
    assert outcomes.tolist() == [2.75]


def test_loads_sealed_lane_artifact(tmp_path: Path) -> None:
    lane_header, _, _ = _headers()
    header = list(lane_header)
    header[:5] = [
        1,
        validator.HEADER_STRUCT.size,
        validator.FLOAT_DTYPE.itemsize,
        validator.UINT64_DTYPE.itemsize,
        validator.FOOTER_STRUCT.size,
    ]
    rotation_keys = np.asarray([1, 3], dtype="<u8")
    local_indices = np.asarray([0, 4], dtype="<u8")
    lanes = np.arange(20, dtype="<f4").reshape(2, 10)
    path = tmp_path / "part17_stack31.p1-lane-v1.bin"
    path.write_bytes(
        validator.HEADER_STRUCT.pack(validator.HEADER_MAGIC, *header)
        + rotation_keys.tobytes()
        + local_indices.tobytes()
        + lanes.tobytes()
        + validator.FOOTER_STRUCT.pack(validator.FOOTER_MAGIC, 2, 10)
    )
    loaded = validator.load_artifact(path)
    assert np.array_equal(loaded.rotation_keys, rotation_keys)
    assert np.array_equal(loaded.local_rotation_indices, local_indices)
    assert np.array_equal(loaded.lane_partials, lanes)


def test_lane_capture_accepts_exact_operands_and_legal_atomic_targets(monkeypatch) -> None:
    report = validator.validate_capture(*_fixture(monkeypatch))
    assert report["status"] == "pass"
    assert report["classification_ready"] is True
    assert report["fixed_metric"]["atomic_target_exactly_reachable_fraction"] == 1.0
    assert report["fixed_metric"]["operand_lane_values_bitwise_equal_fraction"] == 1.0


def test_lane_capture_rejects_target_outside_atomic_outcomes(monkeypatch) -> None:
    report = validator.validate_capture(*_fixture(monkeypatch, perturb_target=True))
    assert report["status"] == "rejected"
    assert report["fixed_metric"]["atomic_target_exactly_reachable"] < 6


def test_lane_capture_classifies_operand_lane_mismatch(monkeypatch) -> None:
    report = validator.validate_capture(*_fixture(monkeypatch, perturb_model=True))
    assert report["status"] == "pass"
    assert report["classification_ready"] is True
    assert report["operand_replay_qualified"] is False
    assert report["classification"] == "native_atomic_reduction_exact_but_passive_operand_replay_differs"
    assert report["fixed_metric"]["atomic_target_exactly_reachable"] == 6
    assert report["fixed_metric"]["operand_lane_values_bitwise_equal"] == 19
    assert report["operand_lane_first_mismatch"] == [0, 0]
