"""Source geometry survives loading independently of computational precision."""
import pickle

import mrcfile
import numpy as np
import pandas as pd
import pytest

from recovar.data_io import cryoem_dataset, load_utils, metadata_readers
from recovar.data_io.starfile import StarFile, write_star

pytestmark = pytest.mark.unit
PIXEL = 1.6375


def _star(tmp_path, *, old=False, size_field=True, pixels=(PIXEL,), origins="angstrom"):
    stack = tmp_path / "particles.mrcs"
    with mrcfile.new(stack, overwrite=True) as mrc:
        mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
    groups = ["7", "3", "7", "3"] if len(pixels) == 2 else ["7"] * 4
    data = pd.DataFrame({
        "_rlnImageName": [f"{i + 1}@{stack}" for i in range(4)],
        "_rlnOpticsGroup": groups,
        "_rlnMicrographName": ["mic"] * 4,
        "_rlnAngleRot": [0.0] * 4, "_rlnAngleTilt": [0.0] * 4,
        "_rlnAnglePsi": [0.0] * 4,
        "_rlnDefocusU": [10000.0] * 4, "_rlnDefocusV": [11000.0] * 4,
        "_rlnDefocusAngle": [0.0] * 4,
    })
    if origins == "angstrom":
        data["_rlnOriginXAngst"] = 2 * PIXEL
        data["_rlnOriginYAngst"] = -PIXEL
    else:
        data["_rlnOriginX"] = 2.0
        data["_rlnOriginY"] = -1.0
    optics = pd.DataFrame({
        "_rlnOpticsGroup": ["7", "3"][:len(pixels)],
        "_rlnImagePixelSize": pixels,
        "_rlnVoltage": [300.0] * len(pixels),
        "_rlnSphericalAberration": [2.7] * len(pixels),
        "_rlnAmplitudeContrast": [0.1] * len(pixels),
    })
    if size_field:
        optics["_rlnImageSize"] = 8
    if old:
        for field in ("_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast"):
            data[field] = optics[field].iloc[0]
        data["_rlnDetectorPixelSize"] = 6.55
        data["_rlnMagnification"] = 40000.0
        if size_field:
            data["_rlnImageSize"] = 8
        optics = None
    path = tmp_path / "particles.star"
    write_star(str(path), data, optics)
    return str(path)


@pytest.mark.parametrize("old", [False, True])
def test_canonical_star_pixels_preserve_legacy_apix(tmp_path, old):
    sf = StarFile.load(_star(tmp_path, old=old))
    np.testing.assert_array_equal(sf.source_pixel_sizes_angstrom, np.full(4, PIXEL))
    assert sf.source_pixel_sizes_angstrom.dtype == np.float64
    np.testing.assert_array_equal(sf.apix, np.full(4, PIXEL, dtype=np.float32))
    assert sf.apix.dtype == np.float32


@pytest.mark.parametrize("old,size_field", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("target", [8, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_loaded_host_pixel_and_computation_dtypes(tmp_path, old, size_field, target, dtype):
    path = _star(tmp_path, old=old, size_field=size_field)
    ds = cryoem_dataset.load_dataset(path, downsample_D=target, dtype=dtype)
    expected = PIXEL * 8 / target
    assert type(ds.voxel_size) is float
    assert ds.voxel_size == expected
    assert ds.subset([3, 1]).voxel_size == expected
    real_dtype = np.empty((), dtype=dtype).real.dtype
    assert ds.CTF_params.dtype == real_dtype
    assert ds.rotation_matrices.dtype == real_dtype
    assert ds.translations.dtype == real_dtype
    assert ds.dtype == dtype
    np.testing.assert_array_equal(ds.translations, np.tile([2 * target / 8, -target / 8], (4, 1)))
    # Shared SPA CTF evaluation still takes its computation dtype from arrays.
    ctf = ds.compute_CTF(ds.CTF_params[:1])
    assert ctf.dtype == real_dtype
    if dtype == np.complex64:
        expected_ctf = ds.ctf_evaluator(ds.CTF_params[:1], ds.image_shape, np.float32(expected))
        np.testing.assert_array_equal(ctf, expected_ctf)
    images = ds.image_source.process_images(np.zeros((1, target, target), dtype=real_dtype))
    assert images.dtype == dtype


@pytest.mark.parametrize("origins", ["angstrom", "pixel"])
@pytest.mark.parametrize("target", [8, 4])
def test_origin_conversion_uses_source_geometry(tmp_path, origins, target):
    path = _star(tmp_path, origins=origins)
    _, fractions = metadata_readers.parse_poses_from_star(path, target)
    np.testing.assert_array_equal(fractions, np.tile([2 / 8, -1 / 8], (4, 1)))


def test_optics_mapping_selection_and_uniformity(tmp_path):
    path = _star(tmp_path, pixels=(PIXEL, 2.1))
    sf = StarFile.load(path)
    np.testing.assert_array_equal(sf.source_pixel_sizes_angstrom, [PIXEL, 2.1, PIXEL, 2.1])
    ds = cryoem_dataset.load_dataset(path, ind=np.array([2, 0]))
    assert ds.voxel_size == PIXEL
    with pytest.raises(ValueError, match="All voxel sizes"):
        cryoem_dataset.load_dataset(path)


def test_multiple_optics_with_common_pixel_size(tmp_path):
    ds = cryoem_dataset.load_dataset(_star(tmp_path, pixels=(PIXEL, PIXEL)))
    assert ds.voxel_size == PIXEL


def test_missing_source_pixel_is_explicit():
    sf = StarFile(data=pd.DataFrame({"_rlnImageName": ["1@unavailable.mrcs"]}))
    assert sf.source_pixel_sizes_angstrom is None
    assert sf.apix is None


def test_manually_supplied_voxel_size_is_not_reconstructed(tmp_path):
    ds = cryoem_dataset.load_dataset(_star(tmp_path))
    ds.voxel_size = np.float32(PIXEL)
    assert ds.subset([0]).voxel_size is ds.voxel_size


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_source_pixel_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="[Pp]ixel"):
        cryoem_dataset.load_dataset(_star(tmp_path, pixels=(bad,)))


@pytest.mark.parametrize("source_dtype", [np.float32, np.float64])
def test_explicit_pickle_geometry_is_authoritative_and_not_mutated(tmp_path, source_dtype, monkeypatch):
    path = _star(tmp_path, pixels=(2.1,))
    rows = np.tile([8, PIXEL, 10000, 11000, 0, 300, 2.7, 0.1, 0], (4, 1)).astype(source_dtype)
    original = rows.copy()
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: rows)
    ds = cryoem_dataset.load_dataset(path, ctf_file="ctf.pkl", ind=np.array([3, 1]), downsample_D=4)
    assert ds.voxel_size == float(source_dtype(PIXEL)) * 2
    assert ds.CTF_params.dtype == np.float32
    np.testing.assert_array_equal(rows, original)


def test_pickle_selection_precedes_common_geometry_check(tmp_path):
    path = _star(tmp_path)
    rows = np.tile([8, PIXEL, 10000, 11000, 0, 300, 2.7, 0.1, 0], (4, 1)).astype(np.float64)
    rows[1, 1] = 2.0
    pkl = tmp_path / "ctf.pkl"
    pkl.write_bytes(pickle.dumps(rows))
    ds = cryoem_dataset.load_dataset(path, ctf_file=str(pkl), ind=np.array([3, 2]))
    assert ds.voxel_size == PIXEL
    with pytest.raises(ValueError, match="All voxel sizes"):
        cryoem_dataset.load_dataset(path, ctf_file=str(pkl))


@pytest.mark.parametrize("size", [0, -8, 7.5, np.nan, np.inf])
def test_invalid_source_dimensions_are_rejected(size):
    with pytest.raises(ValueError, match="dimension"):
        metadata_readers.pixel_sizes_at_grid([PIXEL], [size], 8)


def test_cs_scaling_preserves_serialized_pixel_value(tmp_path):
    dtype = [("blob/psize_A", "f4"), ("blob/shape", "i4", (2,))]
    dtype += [(name, "f4") for name in ("ctf/df1_A", "ctf/df2_A", "ctf/df_angle_rad", "ctf/accel_kv", "ctf/cs_mm", "ctf/amp_contrast")]
    data = np.ones(3, dtype=dtype)
    data["blob/psize_A"] = PIXEL
    data["blob/shape"] = [8, 8]
    path = tmp_path / "particles.cs"
    with path.open("wb") as stream:
        np.save(stream, data)
    ctf = metadata_readers.parse_ctf_from_cs(str(path), 4)
    np.testing.assert_array_equal(ctf[:, 0], np.full(3, float(np.float32(PIXEL)) * 2))


def test_initial_model_consumers_receive_one_loaded_scalar(tmp_path, monkeypatch):
    from recovar.em.dense_single_volume.helpers.resolution import shell_index_to_resolution_angstrom
    from recovar.em.initial_model import driver, native_options, native_sampling

    path = _star(tmp_path)
    sf = StarFile.load(path)
    ds = cryoem_dataset.load_dataset(path)
    opts = native_options.NativeInitialModelOptions(fn_img=path, particle_diameter=10, offset_range_px=6, offset_step_px=2)
    seen = {}
    backend = ds.image_source.backend
    monkeypatch.setattr(backend, "set_relion_image_mask", lambda **kwargs: seen.update(mask=kwargs))
    driver._configure_relion_image_mask(ds, opts)
    assert seen["mask"]["pixel_size"] == PIXEL
    assert driver._single_optics_scalars(sf.df, sf.data_optics, ds)[-1] == PIXEL
    sampling = native_sampling._initial_sampling_state(opts, pixel_size=ds.voxel_size)
    assert sampling.offset_range_angstrom == 6 * PIXEL
    assert sampling.offset_step_angstrom == 2 * PIXEL
    assert shell_index_to_resolution_angstrom(2, 8, ds.voxel_size) == 4 * PIXEL

    class CapturedBootstrap(Exception):
        pass

    def average(_images, **kwargs):
        seen["average"] = kwargs
        return np.zeros((8, 8)), np.zeros((1, 5))

    def bootstrap(**kwargs):
        seen["bootstrap"] = kwargs
        raise CapturedBootstrap

    monkeypatch.setattr(driver, "compute_avg_unaligned_and_sigma2", average)
    monkeypatch.setattr(driver, "compute_bootstrap_iref_via_cpp", bootstrap)
    with pytest.raises(CapturedBootstrap):
        driver._initial_state_from_particles(ds, sf.df, sf.data_optics, opts, ds.rotation_matrices)
    assert seen["average"]["pixel_size"] == PIXEL
    assert seen["bootstrap"]["pixel_size"] == PIXEL
