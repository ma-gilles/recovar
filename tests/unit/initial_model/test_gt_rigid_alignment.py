"""Rigid reporting geometry controls, separate from scientific EM execution."""

import dataclasses
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recovar.em.diagnostics import gt_registration as rigid

pytestmark = pytest.mark.unit


@pytest.fixture
def save(tmp_path):
    def record(name, values):
        (tmp_path / (name + ".json")).write_text(json.dumps(values, indent=2) + "\n")
        print(name + ": " + json.dumps(values), file=sys.stderr)

    return record


def analytic_volume(n=49, transform=None, translation=None):
    coords = np.indices((n, n, n), dtype=np.float64).reshape(3, -1) - (n - 1) / 2
    if transform is not None:
        # Independent analytic field sampled under a forward rigid transform.
        coords = np.linalg.solve(transform, coords - np.asarray(translation)[:, None])
    out = np.zeros(coords.shape[1])
    for amplitude, center, widths in [
        (1.0, [-8, -4, 3], [2.3, 3.2, 2.7]),
        (0.8, [4, 5, -6], [3.1, 2.1, 2.6]),
        (0.6, [9, -6, -3], [2.2, 2.9, 1.9]),
        (0.4, [-3, 9, 8], [2.4, 2.1, 3.3]),
    ]:
        delta = (coords - np.asarray(center)[:, None]) / np.asarray(widths)[:, None]
        out += amplitude * np.exp(-0.5 * np.sum(delta**2, axis=0))
    return out.reshape((n, n, n))


@pytest.mark.parametrize("hand", [False, True])
def test_forward_inverse_integer_transform_is_exact(hand):
    n = 13
    source = np.zeros((n, n, n))
    source[3, 4, 7] = 2
    source[7, 8, 5] = 3
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mirror = np.diag([-1 if hand else 1, 1, 1])
    translation = np.array([1, -2, 1])
    expected = np.zeros_like(source)
    for index in np.argwhere(source):
        destination = (6 + rotation @ mirror @ (index - 6) + translation).astype(int)
        expected[tuple(destination)] = source[tuple(index)]
    actual = rigid.apply_rigid_volume_transform(source, rotation, translation, mirror_x=hand, order=0)
    np.testing.assert_array_equal(actual, expected)
    inverse = (rotation @ mirror).T
    restored = rigid.apply_rigid_volume_transform(
        actual, inverse @ mirror, -inverse @ translation, mirror_x=hand, order=0
    )
    np.testing.assert_array_equal(restored, source)


def test_continuous_true_shift_beats_interpolation_knot(save):
    reference = analytic_volume()
    shift = np.array([2.25, -3.5, 1.125])
    moving = analytic_volume(transform=np.eye(3), translation=shift)
    objective = rigid._continuous_score(moving, reference, 8, 25)
    expected = -shift
    knot = np.round(expected / 2) * 2  # full-grid size49: (49-1)/(25-1)=2
    truth, wrong = objective(np.eye(3), expected), objective(np.eye(3), knot)
    save(
        "continuous_objective",
        {
            "true_shift_score": truth,
            "wrong_knot_score": wrong,
            "true_shift": expected.tolist(),
            "wrong_knot": knot.tolist(),
        },
    )
    assert truth > wrong


@pytest.mark.parametrize(
    "case,hand", [("translation", False), ("translation", True), ("arbitrary", False), ("arbitrary", True)]
)
def test_known_rigid_fit_and_no_mutation(case, hand, save):
    reference = analytic_volume()
    rotation = np.eye(3) if case == "translation" else Rotation.from_rotvec([0.31, -0.42, 0.27]).as_matrix()
    mirror = np.diag([-1.0 if hand else 1.0, 1.0, 1.0])
    transform = rotation @ mirror
    translation = np.array([2.0, -3.0, 1.0]) if case == "translation" else np.array([2.25, -3.5, 1.125])
    moving = analytic_volume(transform=transform, translation=translation)
    # Arbitrary rotations use the validated production order2/4608 grid.
    # A 24-rotation cube grid previously failed these arbitrary-rotation controls.
    from recovar.em.diagnostics.gt_metrics import relion_alignment_rotations

    grid = Rotation.create_group("O").as_matrix() if case == "translation" else relion_alignment_rotations(2)
    original = [v.copy() for v in (moving, reference, grid)]
    result = rigid.align_volume_rigid_to_reference(moving, reference, grid)
    for current, before in zip((moving, reference, grid), original):
        np.testing.assert_array_equal(current, before)
    expected_a, expected_t = transform.T, -transform.T @ translation
    actual_a = result.rotation_matrix @ np.diag([-1 if result.mirror_x else 1, 1, 1])
    hand_correct = result.mirror_x == hand
    angle = float(np.degrees(Rotation.from_matrix(actual_a @ expected_a.T).magnitude())) if hand_correct else None
    displacement = float(np.linalg.norm(result.translation_voxels - expected_t))
    save(
        case + "_hand" + str(int(hand)),
        {
            "hand_correct": hand_correct,
            "rotation_error_deg": angle,
            "translation_l2_error_voxels": displacement,
            "score": result.score,
            "full_corr": result.corr,
            "receipt": dataclasses.asdict(result.receipt),
            "expected_translation_voxels": expected_t.tolist(),
            "actual_translation_voxels": result.translation_voxels.tolist(),
            "control_limits": {"rotation_deg": 2, "translation_voxels": 0.5},
        },
    )
    # Existing v2 geometric control limits, not roundoff tolerances. All fitting
    # is already f64; no claim that discretization/fit errors are float32 noise.
    assert hand_correct and angle < 2 and displacement < 0.5
    assert result.receipt.optimizer_success
    assert result.sign == 1
    reapplied = rigid.apply_rigid_volume_transform(
        moving,
        result.rotation_matrix,
        result.translation_voxels,
        mirror_x=result.mirror_x,
        order=result.receipt.controls.final_interpolation_order,
    )
    np.testing.assert_array_equal(reapplied, result.aligned_volume)
    transport = rigid.RigidVolumeTransform.from_alignment(
        result,
        volume_shape=reference.shape,
        voxel_size=2.5,
        gt_sha256="a" * 64,
    )
    np.testing.assert_array_equal(
        transport.apply(moving, voxel_size=2.5, gt_sha256="a" * 64),
        result.aligned_volume,
    )
    with pytest.raises(ValueError, match="Declared grid"):
        rigid.RigidVolumeTransform.from_alignment(
            result,
            volume_shape=(17, 17, 17),
            voxel_size=2.5,
            gt_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((9, 9, 9)),
        np.full((9, 9, 9), np.nan),
        np.full((9, 9, 9), np.inf),
        np.zeros((9, 9, 8)),
        np.ones((9, 9, 9), dtype=np.complex128),
    ],
)
def test_reject_invalid_volume(bad):
    with pytest.raises(ValueError):
        rigid.align_volume_rigid_to_reference(bad, analytic_volume(9), np.eye(3)[None])


@pytest.mark.parametrize(
    "grid", [np.empty((0, 3, 3)), np.zeros((1, 3, 3)), np.diag([-1, 1, 1])[None], np.full((1, 3, 3), np.nan)]
)
def test_reject_invalid_rotation(grid):
    with pytest.raises(ValueError):
        rigid.align_volume_rigid_to_reference(analytic_volume(17), analytic_volume(17), grid)


@pytest.mark.parametrize(
    "controls",
    [
        rigid.RigidFitControls(coarse_sample_size=1),
        rigid.RigidFitControls(maxiter=0),
        rigid.RigidFitControls(xtol=np.nan),
        rigid.RigidFitControls(final_interpolation_order=6),
    ],
)
def test_reject_invalid_controls(controls):
    with pytest.raises(ValueError):
        rigid.align_volume_rigid_to_reference(
            analytic_volume(17), analytic_volume(17), np.eye(3)[None], controls=controls
        )


def test_optimizer_limit_is_reported_not_hidden():
    volume = analytic_volume(33)
    result = rigid.align_volume_rigid_to_reference(
        volume, volume, np.eye(3)[None], controls=rigid.RigidFitControls(maxfev=1, allow_mirror=False)
    )
    assert not result.receipt.optimizer_success
    assert result.receipt.optimizer_evaluations == 1
    assert result.receipt.optimizer_message
    assert result.sign == 1


@pytest.fixture
def transform():
    return rigid.RigidVolumeTransform(
        rotation_matrix=np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        translation_voxels=np.array([1.0, -2, 1]),
        mirror_x=True,
        volume_shape=(13, 13, 13),
        voxel_size=2.5,
        gt_sha256="a" * 64,
        interpolation_order=0,
    )


def test_transform_roundtrip_identity_and_apply_are_exact(transform, monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Transport must never refit")

    monkeypatch.setattr(rigid, "align_volume_rigid_to_reference", no_fit)
    payload = json.loads(json.dumps(transform.to_dict()))
    restored = rigid.RigidVolumeTransform.from_dict(dict(reversed(list(payload.items()))))
    assert restored == transform
    assert restored.identity_sha256 == transform.identity_sha256
    volume = np.zeros((13, 13, 13))
    volume[3, 4, 7] = 2
    expected = np.zeros_like(volume)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 6 + rotation @ np.diag([-1, 1, 1]) @ (np.array([3, 4, 7]) - 6) + np.array([1, -2, 1])
    expected[tuple(target)] = 2
    np.testing.assert_array_equal(restored.apply(volume, voxel_size=2.5, gt_sha256="a" * 64), expected)
    np.testing.assert_array_equal(volume[3, 4, 7], 2)
    changed = dataclasses.replace(transform, translation_voxels=(1.0, -1.0, 1.0))
    assert changed.identity_sha256 != transform.identity_sha256


def test_transform_owns_immutable_geometry(transform):
    rotation = np.asarray(transform.rotation_matrix).copy()
    translation = np.asarray(transform.translation_voxels).copy()
    copied = dataclasses.replace(transform, rotation_matrix=rotation, translation_voxels=translation)
    identity = copied.identity_sha256
    rotation[0, 0] = 9
    translation[0] = 9
    payload = copied.to_dict()
    payload["translation_voxels"][0] = 9
    assert copied.identity_sha256 == identity == transform.identity_sha256
    assert isinstance(copied.rotation_matrix, tuple) and isinstance(copied.rotation_matrix[0], tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        copied.sign = -1


@pytest.mark.parametrize(
    "key,value",
    [
        ("rotation_matrix", np.eye(3, dtype=np.complex128)),
        ("rotation_matrix", np.diag([-1, 1, 1])),
        ("translation_voxels", [1, 2, complex(3, 1)]),
        ("translation_voxels", [1, 2, np.nan]),
        ("translation_voxels", ["1", "2", "3"]),
        ("volume_shape", [13, 12, 13]),
        ("volume_shape", [13.0, 13.0, 13.0]),
        ("volume_shape", [2, 2, 2]),
        ("volume_shape", None),
        ("voxel_size", 0),
        ("voxel_size", np.inf),
        ("voxel_size", "2.5"),
        ("gt_sha256", "short"),
        ("gt_sha256", "A" * 64),
        ("mirror_x", 1),
        ("sign", -1),
        ("sign", True),
        ("interpolation_order", 1.5),
        ("interpolation_order", 6),
    ],
)
def test_transform_rejects_invalid_fields(transform, key, value):
    with pytest.raises(ValueError):
        dataclasses.replace(transform, **{key: value})


@pytest.mark.parametrize("mutation", ["unknown", "missing", "frame", "geometry", "units", "schema"])
def test_transform_json_schema_is_closed(transform, mutation):
    payload = transform.to_dict()
    if mutation == "unknown":
        payload["future_field"] = 1
    elif mutation == "missing":
        del payload["voxel_size"]
    else:
        key = {"frame": "coordinate_frame", "geometry": "geometry", "units": "translation_units", "schema": "schema"}[
            mutation
        ]
        payload[key] = "different"
    with pytest.raises(ValueError):
        rigid.RigidVolumeTransform.from_dict(payload)


@pytest.mark.parametrize(
    "shape,voxel,gt",
    [
        ((15, 15, 15), 2.5, "a" * 64),
        ((13, 13, 13), 2.6, "a" * 64),
        ((13, 13, 13), 2.5, "b" * 64),
        ((13, 13, 13), "2.5", "a" * 64),
    ],
)
def test_transform_apply_rejects_incompatible_grid_or_gt(transform, shape, voxel, gt):
    with pytest.raises(ValueError):
        transform.apply(np.zeros(shape), voxel_size=voxel, gt_sha256=gt)
