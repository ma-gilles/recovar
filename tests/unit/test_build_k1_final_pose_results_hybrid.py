from pathlib import Path

import numpy as np
import starfile

from scripts.build_k1_final_pose_results_hybrid import build_pose_results_hybrid


def _write_star(path: Path, names: list[str], eulers=None, origins=None) -> None:
    import pandas as pd

    particles = pd.DataFrame({"rlnImageName": names})
    if eulers is not None:
        for column, values in zip(
            ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"), np.asarray(eulers).T, strict=True
        ):
            particles[column] = values
    if origins is not None:
        for column, values in zip(
            ("rlnOriginXAngst", "rlnOriginYAngst"), np.asarray(origins).T, strict=True
        ):
            particles[column] = values
    optics = pd.DataFrame({"rlnImagePixelSize": [2.0]})
    starfile.write({"optics": optics, "particles": particles}, path)


def _write_base_results(path: Path) -> None:
    zeros_euler = np.zeros((2, 3), dtype=np.float32)
    zeros_translation = np.zeros((2, 2), dtype=np.float32)
    np.savez(
        path,
        current_sizes=np.asarray([64]),
        half1_indices=np.asarray([2, 0]),
        half2_indices=np.asarray([3, 1]),
        best_rotation_eulers_iter_000_half0=zeros_euler,
        best_rotation_eulers_iter_000_half1=zeros_euler,
        best_rotation_eulers_iter_000=np.zeros((4, 3), dtype=np.float32),
        best_rotation_eulers_by_image_iter_000=np.zeros((4, 3), dtype=np.float32),
        best_rotation_eulers_final_by_image=np.zeros((4, 3), dtype=np.float32),
        best_translations_iter_000_half0=zeros_translation,
        best_translations_iter_000_half1=zeros_translation,
        best_translations_iter_000=np.zeros((4, 2), dtype=np.float32),
        best_translations_by_image_iter_000=np.zeros((4, 2), dtype=np.float32),
        best_translations_final_by_image=np.zeros((4, 2), dtype=np.float32),
        untouched=np.asarray([17]),
    )


def test_build_pose_results_hybrid_joins_relion_by_identity(tmp_path: Path) -> None:
    names = [f"{index}@stack.mrcs" for index in range(1, 5)]
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    _write_star(input_star, names)
    relion_order = [names[3], names[1], names[0], names[2]]
    eulers = np.asarray([[40, 41, 42], [20, 21, 22], [10, 11, 12], [30, 31, 32]])
    origins = np.asarray([[8, 10], [4, 6], [2, 4], [6, 8]], dtype=np.float64)
    _write_star(relion_star, relion_order, eulers=eulers, origins=origins)
    base_results = tmp_path / "base.npz"
    output_results = tmp_path / "hybrid.npz"
    _write_base_results(base_results)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    expected_translations = (
        np.asarray([[3, 4], [1, 2]], dtype=np.float32),
        np.asarray([[4, 5], [2, 3]], dtype=np.float32),
    )
    for half, values in enumerate(expected_translations):
        np.savez(
            manifest_dir / f"manifest_final_half{half}.npz",
            absolute_previous_translations=values,
        )

    report = build_pose_results_hybrid(
        base_results=base_results,
        input_particle_star=input_star,
        relion_data_star=relion_star,
        aligned_manifest_dir=manifest_dir,
        output_results=output_results,
    )

    with np.load(output_results) as result:
        np.testing.assert_array_equal(
            result["best_rotation_eulers_iter_000_half0"], [[30, 31, 32], [10, 11, 12]]
        )
        np.testing.assert_array_equal(
            result["best_translations_iter_000_half1"], [[4, 5], [2, 3]]
        )
        np.testing.assert_array_equal(
            result["best_translations_by_image_iter_000"],
            [[1, 2], [2, 3], [3, 4], [4, 5]],
        )
        np.testing.assert_array_equal(result["untouched"], [17])
    assert report["status"] == "complete"
    assert report["pose_iteration_label"] == "000"


def test_build_pose_results_hybrid_rejects_manifest_translation_mismatch(tmp_path: Path) -> None:
    names = [f"{index}@stack.mrcs" for index in range(1, 5)]
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    _write_star(input_star, names)
    _write_star(
        relion_star,
        names,
        eulers=np.zeros((4, 3)),
        origins=np.zeros((4, 2)),
    )
    base_results = tmp_path / "base.npz"
    _write_base_results(base_results)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for half in range(2):
        np.savez(
            manifest_dir / f"manifest_final_half{half}.npz",
            absolute_previous_translations=np.ones((2, 2), dtype=np.float32),
        )

    with np.testing.assert_raises_regex(ValueError, "do not match the aligned manifest"):
        build_pose_results_hybrid(
            base_results=base_results,
            input_particle_star=input_star,
            relion_data_star=relion_star,
            aligned_manifest_dir=manifest_dir,
            output_results=tmp_path / "hybrid.npz",
        )

    assert not (tmp_path / "hybrid.npz").exists()


def test_build_pose_results_hybrid_can_replace_only_eulers(tmp_path: Path) -> None:
    names = [f"{index}@stack.mrcs" for index in range(1, 5)]
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    _write_star(input_star, names)
    _write_star(
        relion_star,
        names,
        eulers=np.arange(12, dtype=np.float64).reshape(4, 3),
        origins=np.arange(8, dtype=np.float64).reshape(4, 2) * 2.0,
    )
    base_results = tmp_path / "base.npz"
    output_results = tmp_path / "eulers_only.npz"
    _write_base_results(base_results)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    translations = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.savez(
        manifest_dir / "manifest_final_half0.npz",
        absolute_previous_translations=translations[[2, 0]],
    )
    np.savez(
        manifest_dir / "manifest_final_half1.npz",
        absolute_previous_translations=translations[[3, 1]],
    )

    report = build_pose_results_hybrid(
        base_results=base_results,
        input_particle_star=input_star,
        relion_data_star=relion_star,
        aligned_manifest_dir=manifest_dir,
        output_results=output_results,
        replace_translations=False,
    )

    with np.load(output_results) as result:
        np.testing.assert_array_equal(
            result["best_rotation_eulers_by_image_iter_000"],
            np.arange(12, dtype=np.float32).reshape(4, 3),
        )
        np.testing.assert_array_equal(
            result["best_translations_by_image_iter_000"], np.zeros((4, 2), dtype=np.float32)
        )
    assert report["replaced_components"] == {"eulers": True, "translations": False}


def test_build_pose_results_hybrid_can_replace_only_translations(tmp_path: Path) -> None:
    names = [f"{index}@stack.mrcs" for index in range(1, 5)]
    input_star = tmp_path / "input.star"
    relion_star = tmp_path / "relion.star"
    _write_star(input_star, names)
    eulers = np.arange(12, dtype=np.float64).reshape(4, 3)
    translations = np.arange(8, dtype=np.float32).reshape(4, 2)
    _write_star(relion_star, names, eulers=eulers, origins=translations * 2.0)
    base_results = tmp_path / "base.npz"
    output_results = tmp_path / "translations_only.npz"
    _write_base_results(base_results)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    np.savez(
        manifest_dir / "manifest_final_half0.npz",
        absolute_previous_translations=translations[[2, 0]],
    )
    np.savez(
        manifest_dir / "manifest_final_half1.npz",
        absolute_previous_translations=translations[[3, 1]],
    )

    report = build_pose_results_hybrid(
        base_results=base_results,
        input_particle_star=input_star,
        relion_data_star=relion_star,
        aligned_manifest_dir=manifest_dir,
        output_results=output_results,
        replace_eulers=False,
    )

    with np.load(output_results) as result:
        np.testing.assert_array_equal(
            result["best_rotation_eulers_by_image_iter_000"], np.zeros((4, 3), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result["best_translations_by_image_iter_000"], translations
        )
    assert report["replaced_components"] == {"eulers": False, "translations": True}


def test_build_pose_results_hybrid_requires_a_component(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "at least one pose component"):
        build_pose_results_hybrid(
            base_results=tmp_path / "unused.npz",
            input_particle_star=tmp_path / "unused.star",
            relion_data_star=tmp_path / "unused_relion.star",
            aligned_manifest_dir=tmp_path,
            output_results=tmp_path / "unused_output.npz",
            replace_eulers=False,
            replace_translations=False,
        )
