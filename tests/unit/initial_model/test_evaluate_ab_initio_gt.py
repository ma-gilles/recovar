from __future__ import annotations

import json

import numpy as np
import pytest

from recovar.em.diagnostics import gt_registration
from recovar.utils import helpers
from scripts import evaluate_ab_initio_gt as evaluator
from scripts.evaluate_ab_initio_gt import main


def _asymmetric_test_volume(n: int = 17) -> np.ndarray:
    grid = np.indices((n, n, n), dtype=np.float64)
    z, y, x = [(axis - (n - 1) / 2.0) / n for axis in grid]
    first = np.exp(-90.0 * ((x - 0.12) ** 2 + (y + 0.07) ** 2 + (z - 0.03) ** 2))
    second = 0.35 * np.exp(-130.0 * ((x + 0.18) ** 2 + (y - 0.11) ** 2 + (z + 0.09) ** 2))
    ridge = 0.08 * x + 0.03 * y - 0.05 * z
    return first + second + ridge


def test_evaluate_ab_initio_gt_cli_handles_relion_frame_outputs(tmp_path):
    gt = _asymmetric_test_volume()
    gt_path = tmp_path / "reference_gt.mrc"
    native_path = tmp_path / "run_it003_class001.mrc"
    out_npz = tmp_path / "metrics.npz"
    out_json = tmp_path / "metrics.json"

    helpers.write_mrc(str(gt_path), gt, voxel_size=2.5)
    helpers.write_relion_mrc(str(native_path), gt, voxel_size=2.5)

    rc = main(
        [
            "--volume",
            str(native_path),
            "--label",
            "native_it003",
            "--gt_volume",
            str(gt_path),
            "--volume_frame",
            "relion",
            "--gt_frame",
            "recovar",
            "--gt_align",
            "--gt_align_healpix_order",
            "0",
            "--gt_align_max_shell",
            "4",
            "--output_npz",
            str(out_npz),
            "--output_json",
            str(out_json),
        ]
    )

    assert rc == 0
    metrics = np.load(out_npz)
    summary = json.loads(out_json.read_text())

    assert summary["volume_frame"] == "relion"
    assert summary["gt_frame"] == "recovar"
    assert summary["gt_align_enabled"] is True
    assert summary["volumes"][0]["label"] == "native_it003"
    assert summary["volumes"][0]["corr_vs_gt"] > 0.999
    assert "aligned" in summary["volumes"][0]
    assert np.isfinite(summary["volumes"][0]["aligned"]["corr_vs_gt"])
    assert summary["volumes"][0]["aligned"]["sign"] == 1
    assert float(metrics["native_it003_corr_vs_gt"]) > 0.999
    assert np.isfinite(float(metrics["native_it003_aligned_corr_vs_gt"]))
    assert np.all(np.asarray(metrics["native_it003_fsc_vs_gt"])[1:5] > 0.999)


@pytest.fixture
def rigid_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "relion_alignment_rotations", lambda _: np.eye(3)[None])
    gt = _asymmetric_test_volume()
    paths = [tmp_path / name for name in ("gt.mrc", "candidate.mrc", "native.mrc")]
    helpers.write_mrc(str(paths[0]), gt, voxel_size=2.5)
    for path in paths[1:]:
        helpers.write_relion_mrc(str(path), np.roll(gt, 1, axis=0), voxel_size=2.5)
    return dict(
        volume_paths=[str(p) for p in paths[1:]],
        labels=["candidate", "native"],
        gt_volume_path=str(paths[0]),
        volume_frame="relion",
        gt_frame="recovar",
        voxel_size_override=None,
        gt_align=True,
        gt_align_healpix_order=0,
        gt_align_max_shell=4,
        gt_align_allow_mirror=False,
        gt_align_allow_sign=False,
        gt_align_rigid=True,
    )


@pytest.mark.unit
def test_rigid_shared_transform_fits_once_and_roundtrips_without_refit(rigid_inputs, monkeypatch, tmp_path):
    original = gt_registration.align_volume_rigid_to_reference
    calls = []

    def fit(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gt_registration, "align_volume_rigid_to_reference", fit)
    first_npz, first = evaluator.evaluate(**rigid_inputs, gt_align_fit_label="native")
    assert len(calls) == 1
    document = first["rigid_transform"]
    assert document["fit_reference"]["label"] == "native"

    def forbidden(*args, **kwargs):
        raise AssertionError("Applying a transform must not fit or build a rotation grid")

    monkeypatch.setattr(gt_registration, "align_volume_rigid_to_reference", forbidden)
    monkeypatch.setattr(evaluator, "relion_alignment_rotations", forbidden)
    second_npz, second = evaluator.evaluate(**rigid_inputs, gt_align_transform=json.loads(json.dumps(document)))
    assert second["rigid_transform"] == document
    assert second["gt_align_options_applied"] is False
    assert first["gt_align_options_applied"] is True
    for label, item in zip(rigid_inputs["labels"], second["volumes"]):
        np.testing.assert_array_equal(first_npz[f"{label}_aligned_fsc_vs_gt"], second_npz[f"{label}_aligned_fsc_vs_gt"])
        receipt = item["aligned"]["rigid_registration"]
        assert receipt["transform_identity_sha256"] == document["identity_sha256"]
        assert receipt["independently_fitted_per_volume"] is False
        assert item["aligned"]["score_vs_gt"] is None
    saved = tmp_path / "transform.json"
    saved.write_text(json.dumps(document))
    copied = tmp_path / "copy.json"
    assert (
        main(
            [
                *[arg for path in rigid_inputs["volume_paths"] for arg in ("--volume", path)],
                *[arg for label in rigid_inputs["labels"] for arg in ("--label", label)],
                "--gt_volume",
                rigid_inputs["gt_volume_path"],
                "--gt_frame",
                "recovar",
                "--gt_align",
                "--gt_align_rigid",
                "--gt_align_transform_json",
                str(saved),
                "--gt_align_transform_output",
                str(copied),
            ]
        )
        == 0
    )
    assert json.loads(copied.read_text()) == document


@pytest.mark.unit
def test_rigid_independent_receipt(rigid_inputs):
    _, summary = evaluator.evaluate(**rigid_inputs)
    assert "rigid_transform" not in summary
    for item in summary["volumes"]:
        receipt = item["aligned"]["rigid_registration"]
        assert receipt["independently_fitted_per_volume"] is True
        assert receipt["sign"] == 1
        assert receipt["quality_accepted"] is False
        assert len(receipt["translation_voxels"]) == 3


@pytest.mark.unit
@pytest.mark.parametrize("change", ["shape", "voxel", "anisotropic"])
def test_rigid_rejects_mismatched_grid(rigid_inputs, change):
    import mrcfile

    path = rigid_inputs["volume_paths"][0]
    if change == "shape":
        helpers.write_relion_mrc(path, np.zeros((15, 15, 15)), voxel_size=2.5)
    else:
        with mrcfile.open(path, mode="r+") as mrc:
            mrc.voxel_size = 3.0 if change == "voxel" else (2.5, 3.0, 2.5)
    with pytest.raises(ValueError, match="shape|voxel|grid"):
        evaluator.evaluate(**rigid_inputs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra",
    [
        dict(gt_align=False),
        dict(gt_align_allow_sign=True),
        dict(gt_align_fit_label="missing"),
        dict(voxel_size_override=float("nan")),
    ],
)
def test_rigid_rejects_incompatible_options(rigid_inputs, extra):
    with pytest.raises(ValueError):
        evaluator.evaluate(**(rigid_inputs | extra))


@pytest.mark.unit
def test_rigid_transport_rejects_changed_gt_and_identity(rigid_inputs):
    _, summary = evaluator.evaluate(**rigid_inputs, gt_align_fit_label="native")
    document = summary["rigid_transform"]
    bad = json.loads(json.dumps(document))
    bad["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        evaluator.evaluate(**rigid_inputs, gt_align_transform=bad)
    helpers.write_mrc(rigid_inputs["gt_volume_path"], _asymmetric_test_volume() * 2, voxel_size=2.5)
    with pytest.raises(ValueError, match="GT"):
        evaluator.evaluate(**rigid_inputs, gt_align_transform=document)


@pytest.mark.unit
@pytest.mark.parametrize(
    "flags",
    [
        ["--gt_align_rigid"],
        ["--gt_align_fit_label", "native"],
        ["--gt_align", "--gt_align_rigid", "--gt_align_allow_sign"],
        ["--gt_align", "--gt_align_rigid", "--gt_align_transform_output", "x"],
    ],
)
def test_rigid_cli_rejects_incompatible_flags(flags):
    with pytest.raises(SystemExit) as exc:
        evaluator._parse_args(["--volume", "unused", "--gt_volume", "unused", *flags])
    assert exc.value.code == 2
