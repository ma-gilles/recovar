import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

pytest.importorskip("jax")

from helpers import tiny_synthetic
from recovar import core, utils
from recovar.data_io import dataset, starfile, cryo_dataset

pytestmark = pytest.mark.unit

tilt_dataset = cryo_dataset


@pytest.fixture(scope="module")
def sim_tiny_tilt_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_tiny_tilt")
    return tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        out,
        grid_size=8,
        n_images=24,
        n_tilts=3,
        n_volumes=4,
    )


def _dataset_image_batches(cryo, *, batch_size, subset_indices=None):
    return list(
        cryo.iter_batches(
            batch_size=batch_size,
            indices=subset_indices,
            by_image=True,
            prefetch=False,
        )
    )


def _dataset_group_batches(cryo, *, batch_size, subset_indices=None):
    return list(
        cryo.iter_batches(
            batch_size=batch_size,
            indices=subset_indices,
            by_image=False,
            prefetch=False,
        )
    )


def _batch_images(batch):
    return np.asarray(batch[0])


def _batch_particle_indices(batch):
    return np.asarray(batch[5]).reshape(-1)


def _batch_image_indices(batch):
    return np.asarray(batch[6]).reshape(-1)


def test_load_dataset_tiny_spa_files(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ind = np.array([1, 4], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=ind,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )

    assert cryo.tilt_series_flag is False
    assert cryo.n_images == ind.size
    assert cryo.n_units == ind.size
    np.testing.assert_array_equal(cryo.dataset_indices, ind)
    assert cryo.CTF_params.shape[0] == ind.size
    assert cryo.ctf_evaluator.mode == core.CTFMode.SPA


def test_load_dataset_tiny_spa_boolean_ind_preserves_dataset_indices_and_image_identity(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    mask = np.array([False, True, True, False, True, False], dtype=bool)
    selected = np.flatnonzero(mask).astype(np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=mask,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices), selected)
    assert cryo.n_images == selected.size
    assert cryo.CTF_params.shape[0] == selected.size

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.arange(selected.size, dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[selected], atol=1e-6)


def test_load_dataset_tiny_spa_duplicate_ind_preserves_order_duplicates_and_ctf_alignment(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    requested = np.array([4, 1, 4, 0], dtype=np.int32)
    ctf = np.asarray(utils.pickle_load(files["ctf_pkl"]))

    cryo = dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    assert cryo.n_images == requested.size

    # CTF rows should align exactly with requested (including duplicates).
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFU],
        ctf[requested, 2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFV],
        ctf[requested, 3],
        atol=1e-7,
    )

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.arange(requested.size, dtype=np.int32))

    source_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, source_images[requested], atol=1e-6)


def test_load_dataset_tiny_spa_full_length_duplicate_ind_keeps_pose_alignment(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    requested = np.array([5, 0, 5, 1, 2, 3], dtype=np.int32)  # len(requested) == total n_images

    # Overwrite poses with deterministic, row-distinct values to verify alignment.
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6, axis=0)
    rots[:, 0, 0] = np.arange(1, 7, dtype=np.float32)
    trans_frac = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    utils.pickle_dump((rots, trans_frac), files["poses_pkl"])

    cryo = dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    np.testing.assert_allclose(np.asarray(cryo.rotation_matrices)[:, 0, 0], rots[requested, 0, 0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(cryo.translations), trans_frac[requested] * 8.0, atol=1e-7)


def test_load_dataset_tiny_spa_rejects_nonfinite_ctf_params(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ctf = np.asarray(utils.pickle_load(files["ctf_pkl"]))
    ctf[0, 2] = np.nan
    bad_ctf_pkl = Path(tmp_path) / "ctf_nonfinite.pkl"
    utils.pickle_dump(ctf, str(bad_ctf_pkl))

    with pytest.raises(ValueError, match="non-finite"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=str(bad_ctf_pkl),
            datadir=str(tmp_path),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_load_dataset_tiny_tilt_series_files_and_subset_generators(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    assert cryo.tilt_series_flag is True
    assert cryo.n_images == files["n_images"]
    assert cryo.n_units == 3  # number of particle groups in fixture
    assert cryo.ctf_evaluator.mode == core.CTFMode.CRYO_ET

    # Particle-subset generator should preserve requested particle order.
    subset_particles = np.array([2, 0], dtype=np.int32)
    batches = _dataset_group_batches(cryo, batch_size=8, subset_indices=subset_particles)
    got_particles = [int(_batch_particle_indices(batch)[0]) for batch in batches]
    assert got_particles == subset_particles.tolist()

    # Image-subset generator should only emit requested image indices, in order.
    subset_images = np.array([5, 1, 4], dtype=np.int32)
    image_batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=subset_images)
    got_images = []
    for batch in image_batches:
        got_images.extend(_batch_image_indices(batch).tolist())
    assert got_images == subset_images.tolist()


def test_load_dataset_tiny_tilt_series_image_subset_preserves_duplicate_order_and_identity(
    sim_tiny_tilt_files,
):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    subset_images = np.array([7, 1, 7, 4], dtype=np.int32)
    image_batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=subset_images)

    got_indices = np.concatenate([_batch_image_indices(batch) for batch in image_batches], axis=0)
    got_images = np.concatenate([_batch_images(batch) for batch in image_batches], axis=0)
    source_images = utils.load_mrc(files["particles_mrcs"])

    np.testing.assert_array_equal(got_indices, subset_images)
    np.testing.assert_allclose(got_images, source_images[subset_images], atol=1e-6)


def test_load_dataset_tiny_tilt_series_boolean_ind_preserves_dataset_indices(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    mask = np.zeros(files["n_images"], dtype=bool)
    mask[[1, 4, 7, 10]] = True
    selected = np.flatnonzero(mask).astype(np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=mask,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    assert cryo.tilt_series_flag is True
    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices), selected)
    assert cryo.n_images == selected.size
    assert cryo.CTF_params.shape[0] == selected.size

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.arange(selected.size, dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[selected], atol=1e-6)


def test_load_dataset_tiny_tilt_series_rejects_out_of_range_star_image_indices(sim_tiny_tilt_files, tmp_path):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)

    sf = starfile.StarFile.load(files["particles_star"])
    image_name = str(sf.df["_rlnImageName"].iloc[0])
    _, rel_path = image_name.split("@", 1)
    sf.df.loc[sf.df.index[0], "_rlnImageName"] = f"999@{rel_path}"

    bad_star = tmp_path / "bad_out_of_range_particles.star"
    starfile.write_star(str(bad_star), data=sf.df)

    with pytest.raises(ValueError, match="out of range"):
        dataset.load_dataset(
            particles_file=str(bad_star),
            poses_file=files["poses_pkl"],
            ctf_file=files["ctf_pkl"],
            datadir=datadir,
            lazy=True,
            tilt_series=True,
            tilt_series_ctf="relion5",
        )


def test_load_dataset_tiny_tilt_series_full_length_duplicate_ind_keeps_pose_alignment(
    sim_tiny_tilt_files,
    tmp_path,
):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    n = int(files["n_images"])
    requested = np.arange(n, dtype=np.int32)
    requested = np.roll(requested, 3)
    requested[0] = requested[1]  # keep len == n but include duplicate/reordered indices

    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n, axis=0)
    rots[:, 0, 0] = np.arange(1, n + 1, dtype=np.float32)
    trans_frac = np.zeros((n, 2), dtype=np.float32)
    trans_frac[:, 0] = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    trans_frac[:, 1] = np.linspace(0.5, -0.5, n, dtype=np.float32)
    poses_pkl = Path(tmp_path) / "poses_override.pkl"
    utils.pickle_dump((rots, trans_frac), str(poses_pkl))

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=str(poses_pkl),
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=requested,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    np.testing.assert_allclose(
        np.asarray(cryo.rotation_matrices)[:, 0, 0],
        rots[requested, 0, 0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.translations),
        trans_frac[requested] * 8.0,
        atol=1e-7,
    )


def test_load_dataset_tiny_tilt_series_duplicate_ind_preserves_order_duplicates_and_image_identity(
    sim_tiny_tilt_files,
):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    requested = np.array([9, 2, 9, 0], dtype=np.int32)
    ctf = np.asarray(utils.pickle_load(files["ctf_pkl"]))

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=requested,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    assert cryo.n_images == requested.size
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFU],
        ctf[requested, 2],
        atol=1e-7,
    )

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.arange(requested.size, dtype=np.int32))

    source_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, source_images[requested], atol=1e-6)


def test_load_dataset_rejects_invalid_ind_values(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)

    with pytest.raises(IndexError, match="out of range|negative"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=files["ctf_pkl"],
            datadir=str(tmp_path),
            ind=np.array([0, -1], dtype=np.int32),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )

    with pytest.raises(IndexError, match="out.of.range"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=files["ctf_pkl"],
            datadir=str(tmp_path),
            ind=np.array([0, 9], dtype=np.int32),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )

    with pytest.raises(ValueError, match="boolean mask length"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=files["ctf_pkl"],
            datadir=str(tmp_path),
            ind=np.array([True, False], dtype=bool),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_tiny_spa_loading_rejects_ctf_count_mismatch_without_subset(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ctf = utils.pickle_load(files["ctf_pkl"])
    utils.pickle_dump(ctf[:-1], files["ctf_pkl"])

    with pytest.raises(ValueError, match="CTF parameter count"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=files["ctf_pkl"],
            datadir=str(tmp_path),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_tiny_spa_loading_allows_short_ctf_when_subset_fits(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ctf = utils.pickle_load(files["ctf_pkl"])
    short_ctf_pkl = Path(tmp_path) / "ctf_short.pkl"
    utils.pickle_dump(ctf[:4], str(short_ctf_pkl))
    requested = np.array([0, 2, 3], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_mrcs"],
        poses_file=files["poses_pkl"],
        ctf_file=str(short_ctf_pkl),
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )
    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    assert cryo.n_images == requested.size


def test_tiny_spa_loading_rejects_short_ctf_when_subset_exceeds_bounds(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ctf = utils.pickle_load(files["ctf_pkl"])
    short_ctf_pkl = Path(tmp_path) / "ctf_short.pkl"
    utils.pickle_dump(ctf[:4], str(short_ctf_pkl))

    with pytest.raises(IndexError, match="out-of-range"):
        dataset.load_dataset(
            particles_file=files["particles_mrcs"],
            poses_file=files["poses_pkl"],
            ctf_file=str(short_ctf_pkl),
            datadir=str(tmp_path),
            ind=np.array([0, 4], dtype=np.int32),
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_simulator_tiny_tilt_loading_rejects_ctf_count_mismatch_without_subset(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ctf = utils.pickle_load(files["ctf_pkl"])
    bad_ctf_pkl = Path(datadir) / "ctf_bad_short.pkl"
    utils.pickle_dump(ctf[:-2], str(bad_ctf_pkl))

    with pytest.raises(ValueError, match="CTF parameter count"):
        dataset.load_dataset(
            particles_file=files["particles_star"],
            poses_file=files["poses_pkl"],
            ctf_file=str(bad_ctf_pkl),
            datadir=datadir,
            lazy=True,
            tilt_series=True,
            tilt_series_ctf="relion5",
        )


def test_simulator_tiny_tilt_loading_allows_short_ctf_when_subset_fits(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ctf = utils.pickle_load(files["ctf_pkl"])
    short_ctf_pkl = Path(datadir) / "ctf_short_subset_ok.pkl"
    utils.pickle_dump(ctf[:10], str(short_ctf_pkl))
    requested = np.array([0, 7, 9], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=str(short_ctf_pkl),
        datadir=datadir,
        ind=requested,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )
    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    assert cryo.n_images == requested.size


def test_simulator_tiny_tilt_loading_rejects_short_ctf_when_subset_exceeds_bounds(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ctf = utils.pickle_load(files["ctf_pkl"])
    short_ctf_pkl = Path(datadir) / "ctf_short_subset_bad.pkl"
    utils.pickle_dump(ctf[:10], str(short_ctf_pkl))

    with pytest.raises(IndexError, match="out-of-range"):
        dataset.load_dataset(
            particles_file=files["particles_star"],
            poses_file=files["poses_pkl"],
            ctf_file=str(short_ctf_pkl),
            datadir=datadir,
            ind=np.array([0, 11], dtype=np.int32),
            lazy=True,
            tilt_series=True,
            tilt_series_ctf="relion5",
        )


def test_load_dataset_tiny_tilt_series_warp_path(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="warp",
        angle_per_tilt=3.0,
    )

    assert cryo.tilt_series_flag is True
    assert cryo.n_images == files["n_images"]
    assert cryo.ctf_evaluator.mode == core.CTFMode.CRYO_ET
    # v2 branch appends dose and angle channels.
    assert cryo.CTF_params.shape[1] >= 11

    subset_images = np.array([5, 1, 4], dtype=np.int32)
    image_batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=subset_images)
    got_images = []
    for batch in image_batches:
        got_images.extend(_batch_image_indices(batch).tolist())
    assert got_images == subset_images.tolist()


def test_load_dataset_tiny_tilt_series_from_simulator_files(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(Path(files["particles_star"]).parent),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    assert cryo.tilt_series_flag is True
    assert cryo.n_images == files["n_images"]
    assert cryo.ctf_evaluator.mode == core.CTFMode.CRYO_ET

    subset_images = np.array([7, 2, 7, 11], dtype=np.int32)
    image_batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=subset_images)
    got_images = []
    for batch in image_batches:
        got_images.extend(_batch_image_indices(batch).tolist())
    assert got_images == subset_images.tolist()


def test_simulator_tilt_dataset_subset_generator_images_mode_preserves_particle_order_and_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=1,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    subset_particles = np.array([2, 0, 2], dtype=np.int32)
    batches = list(ds.get_dataset_subset_generator(batch_size=1, subset_indices=subset_particles, mode="images"))
    got_particles = []
    got_images = []
    got_tilts = []
    for imgs, pidx, tidx in batches:
        got_particles.extend(np.asarray(pidx).reshape(-1).tolist())
        got_tilts.extend(np.asarray(tidx).reshape(-1).tolist())
        got_images.append(np.asarray(imgs))

    # One selected tilt per particle (num_tilts=1), preserving subset order including duplicates.
    assert got_particles == [2, 0, 2]
    assert len(got_tilts) == 3

    # Verify emitted image payload matches source MRCS at emitted tilt indices.
    source_images = utils.load_mrc(files["particles_mrcs"])
    got_images_arr = np.concatenate(got_images, axis=0)
    np.testing.assert_allclose(got_images_arr, source_images[np.asarray(got_tilts, dtype=np.int32)], atol=1e-5)


def test_get_split_tilt_indices_simulator_generated_with_ntilts_cap(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)
    half0 = np.arange(0, n_particles, 2, dtype=np.int32)
    half1 = np.arange(1, n_particles, 2, dtype=np.int32)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=str(Path(files["particles_star"]).parent),
        ntilts=1,
        particle_halfset_indices_file=[half0, half1],
    )
    assert len(split) == 2

    # Build reverse map and ensure each particle contributes at most one tilt in each half.
    tilt_to_particle = {}
    for p_idx, tilt_inds in enumerate(particles_to_tilts):
        for t in tilt_inds:
            tilt_to_particle[int(t)] = p_idx

    for half_imgs in split:
        counts = {}
        for t in np.asarray(half_imgs).astype(int):
            p = tilt_to_particle[t]
            counts[p] = counts.get(p, 0) + 1
        assert all(v <= 1 for v in counts.values())


def test_get_split_tilt_indices_simulator_with_zero_tilts_returns_empty_halves(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=str(Path(files["particles_star"]).parent),
        ntilts=0,
    )
    assert len(split) == 2
    np.testing.assert_array_equal(np.asarray(split[0]), np.array([], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(split[1]), np.array([], dtype=np.int32))


def test_get_split_tilt_indices_simulator_with_negative_tilts_returns_empty_halves(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=str(Path(files["particles_star"]).parent),
        ntilts=-1,
    )
    assert len(split) == 2
    np.testing.assert_array_equal(np.asarray(split[0]), np.array([], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(split[1]), np.array([], dtype=np.int32))


def test_simulator_tiny_tilt_split_indices_sanitizes_tilt_ind_file_values(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)
    keep_a = int(n_particles - 1)
    keep_b = 0

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=datadir,
        # duplicate + invalid ids
        tilt_ind_file=np.array([keep_a, keep_a, -5, 999, keep_b], dtype=np.int32),
        particle_halfset_indices_file=[
            np.array([keep_a, keep_b], dtype=np.int32),
            np.array([], dtype=np.int32),
        ],
    )
    expected_half0 = np.concatenate(
        [
            np.asarray(particles_to_tilts[keep_a], dtype=np.int32),
            np.asarray(particles_to_tilts[keep_b], dtype=np.int32),
        ]
    )
    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), expected_half0)
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), np.array([], dtype=np.int32))


def test_simulator_tiny_tilt_split_indices_with_only_invalid_tilt_ids_returns_empty(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=datadir,
        tilt_ind_file=np.array([-9, 10_000], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), np.array([], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), np.array([], dtype=np.int32))


def test_simulator_tilt_dataset_random_tilts_clamps_when_requested_exceeds_available(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=7,          # intentionally larger than available per particle in this tiny set
        random_tilts=True,
        tilt_file_option="relion5",
    )
    # Should not raise; should clamp to available tilts.
    imgs, pidx, selected = ds[0]
    assert pidx == 0
    assert imgs.shape[0] == selected.shape[0]
    assert selected.shape[0] <= 7
    assert selected.shape[0] > 0
    assert len(np.unique(selected)) == selected.shape[0]


def test_simulator_tilt_dataset_images_mode_rejects_too_small_batch(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=3,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    with pytest.raises(ValueError, match="Batch size"):
        list(ds.get_dataset_generator(batch_size=2, mode="images"))


def test_simulator_tilt_image_subset_none_matches_full_generator(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    full = list(ds.get_image_generator(batch_size=5))
    subset_none = list(ds.get_image_subset_generator(batch_size=5, subset_indices=None))

    full_imgs = np.concatenate([np.asarray(b[0]) for b in full], axis=0)
    full_tilt_idx = np.concatenate([np.asarray(b[2]).reshape(-1) for b in full], axis=0)
    sub_imgs = np.concatenate([np.asarray(b[0]) for b in subset_none], axis=0)
    sub_tilt_idx = np.concatenate([np.asarray(b[2]).reshape(-1) for b in subset_none], axis=0)

    np.testing.assert_array_equal(sub_tilt_idx, full_tilt_idx)
    np.testing.assert_allclose(sub_imgs, full_imgs, atol=1e-6)


def test_simulator_tiny_tilt_loading_with_duplicate_ind_preserves_image_identity(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    duplicated_ind = np.array([7, 2, 7, 11], dtype=np.int32)
    datadir = str(Path(files["particles_star"]).parent)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=duplicated_ind,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2, 3], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[duplicated_ind], atol=1e-5)


def test_simulator_tiny_tilt_figure_out_halfsets_applies_n_images_cap(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)

    class _Args:
        pass

    args = _Args()
    args.halfsets = None
    args.tilt_series = True
    args.tilt_series_ctf = "relion5"
    args.particles = files["particles_star"]
    args.ind = None
    args.tilt_ind = None
    args.ntilts = 1
    args.datadir = datadir
    args.strip_prefix = None
    args.n_images = 6

    full = dataset.get_split_tilt_indices(
        args.particles,
        ind_file=args.ind,
        tilt_ind_file=args.tilt_ind,
        ntilts=args.ntilts,
        datadir=args.datadir,
    )
    expected = [h[: args.n_images // 2] for h in full]
    got = dataset.figure_out_halfsets(args)

    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def test_simulator_tiny_tilt_split_indices_with_halfset_file_and_filters_preserves_particle_order(sim_tiny_tilt_files, tmp_path):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    tilt_ds = tilt_dataset.TiltSeriesDataset(files["particles_star"], datadir=datadir)
    tilt_numbers = np.asarray(tilt_ds.tilt_numbers)

    # Restrict to a subset of particles and provide an explicit halfset order.
    selected_particles = np.array([0, 2, 4], dtype=np.int32)
    halfsets_path = tmp_path / "particle_halfsets.pkl"
    tilt_ind_path = tmp_path / "tilt_ind.pkl"
    import pickle
    with open(halfsets_path, "wb") as f:
        # The first halfset is intentionally out-of-order; this order should be preserved.
        pickle.dump([np.array([4, 0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)], f)
    with open(tilt_ind_path, "wb") as f:
        pickle.dump(selected_particles, f)

    # Keep at most two tilts per particle and allow only this shuffled image subset.
    allowed_images = np.concatenate([particles_to_tilts[2], particles_to_tilts[4], particles_to_tilts[0]])[[0, 1, 2, 3, 4, 5]]
    ind_path = tmp_path / "allowed_images.pkl"
    with open(ind_path, "wb") as f:
        pickle.dump(np.asarray(allowed_images, dtype=np.int32), f)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        ind_file=str(ind_path),
        tilt_ind_file=str(tilt_ind_path),
        ntilts=2,
        datadir=datadir,
        particle_halfset_indices_file=str(halfsets_path),
    )
    assert len(split) == 2

    # Build expected first half from explicit particle order [4, 0, 2], applying ntilts then allowed-image filter.
    expected_half0 = []
    for p in [4, 0, 2]:
        imgs = np.asarray(particles_to_tilts[p], dtype=np.int32)
        imgs = imgs[tilt_numbers[imgs] < 2]
        imgs = imgs[np.isin(imgs, allowed_images)]
        expected_half0.extend(imgs.tolist())

    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), np.asarray(expected_half0, dtype=np.int32))
    # Second halfset particles [1,3] are removed by tilt_ind intersection.
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), np.array([], dtype=np.int32))


def test_simulator_tiny_tilt_split_indices_ignores_out_of_range_particle_ids_in_halfset_file(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=datadir,
        particle_halfset_indices_file=[
            np.array([n_particles - 1, -3, 0, 999], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
    )
    assert len(split) == 2

    expected_half0 = np.concatenate(
        [
            np.asarray(particles_to_tilts[n_particles - 1], dtype=np.int32),
            np.asarray(particles_to_tilts[0], dtype=np.int32),
        ]
    )
    expected_half1 = np.asarray(particles_to_tilts[1], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), expected_half0)
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), expected_half1)


def test_simulator_tiny_tilt_split_indices_deduplicates_halfset_particle_ids(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)
    keep_a = int(n_particles - 1)
    keep_b = 0

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=datadir,
        # keep_a appears twice and should be deduplicated.
        particle_halfset_indices_file=[
            np.array([keep_a, keep_b, keep_a], dtype=np.int32),
            np.array([], dtype=np.int32),
        ],
    )
    assert len(split) == 2

    expected_half0 = np.concatenate(
        [
            np.asarray(particles_to_tilts[keep_a], dtype=np.int32),
            np.asarray(particles_to_tilts[keep_b], dtype=np.int32),
        ]
    )
    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), expected_half0)
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), np.array([], dtype=np.int32))


def test_load_dataset_simulator_tilt_ctf_in_spa_mode_preserves_reordered_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    reordered_ind = np.array([8, 1, 8, 5], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=reordered_ind,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="relion5",
    )
    assert cryo.tilt_series_flag is False
    assert cryo.ctf_evaluator.mode == core.CTFMode.CRYO_ET

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2, 3], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[reordered_ind], atol=1e-5)


def test_tiny_tilt_get_split_tilt_indices_with_particle_subset_file(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    # Keep only particle ids 0 and 2.
    tilt_ind_file = tmp_path / "tilt_ind.pkl"
    import pickle
    with open(tilt_ind_file, "wb") as f:
        pickle.dump(np.array([0, 2], dtype=np.int32), f)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        tilt_ind_file=str(tilt_ind_file),
        datadir=str(tmp_path),
    )
    assert len(split) == 2
    merged = np.sort(np.concatenate(split))
    # With 3 particles and cyclic assignment over 6 images:
    # g1 -> [0,3], g2 -> [1,4], g3 -> [2,5]. Keeping g1 and g3 gives [0,2,3,5].
    np.testing.assert_array_equal(merged, np.array([0, 2, 3, 5], dtype=np.int32))
    assert np.intersect1d(split[0], split[1]).size == 0


def test_tiny_tilt_figure_out_halfsets_respects_ind_intersection(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    import pickle
    halfsets_path = tmp_path / "halfsets.pkl"
    ind_path = tmp_path / "ind.pkl"
    with open(halfsets_path, "wb") as f:
        pickle.dump([np.array([0, 1, 2]), np.array([3, 4, 5])], f)
    with open(ind_path, "wb") as f:
        pickle.dump(np.array([1, 4, 5], dtype=np.int32), f)

    class _Args:
        pass

    args = _Args()
    args.halfsets = str(halfsets_path)
    args.tilt_series = False
    args.tilt_series_ctf = "cryoem"
    args.particles = files["particles_star"]
    args.ind = str(ind_path)
    args.tilt_ind = None
    args.ntilts = None
    args.datadir = str(tmp_path)
    args.strip_prefix = None
    args.n_images = -1

    hs = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(hs[0], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(hs[1], np.array([4, 5], dtype=np.int32))


def test_tiny_tilt_figure_out_halfsets_with_particle_halfsets_and_image_filter(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    import pickle

    halfsets_path = tmp_path / "particle_halfsets.pkl"
    ind_path = tmp_path / "ind.pkl"
    with open(halfsets_path, "wb") as f:
        # Particle-space halfsets for tilt-series mode.
        pickle.dump([np.array([0, 2], dtype=np.int32), np.array([1], dtype=np.int32)], f)
    with open(ind_path, "wb") as f:
        # Keep only images from particles g1/g3.
        pickle.dump(np.array([0, 2, 3, 5], dtype=np.int32), f)

    class _Args:
        pass

    args = _Args()
    args.halfsets = str(halfsets_path)
    args.tilt_series = True
    args.tilt_series_ctf = "relion5"
    args.particles = files["particles_star"]
    args.ind = str(ind_path)
    args.tilt_ind = None
    args.ntilts = None
    args.datadir = str(tmp_path)
    args.strip_prefix = None
    args.n_images = -1

    hs = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(hs[0], np.array([0, 3, 2, 5], dtype=np.int32))
    np.testing.assert_array_equal(hs[1], np.array([], dtype=np.int32))


def test_tiny_tilt_loading_with_reordered_ind_preserves_image_identity(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    reordered_ind = np.array([5, 1, 4], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=reordered_ind,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    # Image loader returns subset-local indices [0..n-1], but image payloads must
    # correspond exactly to the requested original rows in `reordered_ind`.
    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[reordered_ind], atol=1e-6)


def test_tiny_tilt_loading_with_duplicate_ind_preserves_duplicates_and_order(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    duplicated_ind = np.array([5, 1, 5, 4], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=duplicated_ind,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    batches = _dataset_image_batches(cryo, batch_size=2)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    got_local_idx = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2, 3], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[duplicated_ind], atol=1e-6)


def test_tiny_tilt_particle_subset_generator_preserves_duplicates(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    subset_particles = np.array([2, 0, 2], dtype=np.int32)
    batches = _dataset_group_batches(cryo, batch_size=8, subset_indices=subset_particles)
    got_particles = [int(_batch_particle_indices(batch)[0]) for batch in batches]
    assert got_particles == [2, 0, 2]

def test_tiny_tilt_split_indices_accepts_in_memory_halfsets_and_arrays(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        # Keep particle groups g1 and g3 only.
        tilt_ind_file=np.array([0, 2], dtype=np.int32),
        # Restrict to these allowed image indices.
        ind_file=np.array([0, 2, 3, 5], dtype=np.int32),
        # Deterministic particle halfsets in-memory, no pickle file.
        particle_halfset_indices_file=[np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)],
        datadir=str(tmp_path),
        ntilts=1,
    )
    # ntilts=1 keeps one image per particle according to deterministic tilt ordering.
    np.testing.assert_array_equal(split[0], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([2], dtype=np.int32))


def test_simulator_tiny_tilt_split_indices_accepts_boolean_masks(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesDataset.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)

    keep_particles = np.array([0, n_particles - 1], dtype=np.int32)
    particle_mask = np.zeros(n_particles, dtype=bool)
    particle_mask[keep_particles] = True

    # Build expected one-tilt-per-particle set under deterministic ordering.
    tilt_ds = tilt_dataset.TiltSeriesDataset(files["particles_star"], datadir=datadir)
    tilt_numbers = np.asarray(tilt_ds.tilt_numbers)
    half0_expected = np.asarray(particles_to_tilts[keep_particles[0]], dtype=np.int32)
    half0_expected = half0_expected[tilt_numbers[half0_expected] < 1]
    half1_expected = np.asarray(particles_to_tilts[keep_particles[1]], dtype=np.int32)
    half1_expected = half1_expected[tilt_numbers[half1_expected] < 1]

    image_mask = np.zeros(files["n_images"], dtype=bool)
    image_mask[half0_expected] = True
    image_mask[half1_expected] = True

    split = dataset.get_split_tilt_indices(
        particles_file=files["particles_star"],
        datadir=datadir,
        tilt_ind_file=particle_mask,
        ind_file=image_mask,
        ntilts=1,
        particle_halfset_indices_file=[
            np.array([keep_particles[0]], dtype=np.int32),
            np.array([keep_particles[1]], dtype=np.int32),
        ],
    )
    np.testing.assert_array_equal(np.asarray(split[0], dtype=np.int32), np.asarray(half0_expected, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(split[1], dtype=np.int32), np.asarray(half1_expected, dtype=np.int32))


def test_simulator_tiny_tilt_split_indices_rejects_non_1d_masks(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        dataset.get_split_tilt_indices(
            particles_file=files["particles_star"],
            datadir=datadir,
            tilt_ind_file=np.array([[True, False, True]], dtype=bool),
        )

    with pytest.raises(ValueError, match="ind_file boolean mask must be 1D"):
        dataset.get_split_tilt_indices(
            particles_file=files["particles_star"],
            datadir=datadir,
            tilt_ind_file=np.array([0, 1], dtype=np.int32),
            ind_file=np.array([[True, False], [False, True]], dtype=bool),
        )


def test_tiny_tilt_loading_with_strip_prefix_resolves_prefixed_star_paths(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path / "tiny_data", grid_size=8, n_images=6, n_particles=3)
    bad_prefix = "/definitely/not/a/real/path"
    mrcs_name = Path(files["particles_mrcs"]).name

    star = starfile.StarFile.load(files["particles_star"])
    star.df["_rlnImageName"] = [f"{i+1}@{bad_prefix}/{mrcs_name}" for i in range(files["n_images"])]
    prefixed_star = tmp_path / "particles_prefixed.star"
    starfile.write_star(str(prefixed_star), data=star.df)

    base_kwargs = dict(
        particles_file=str(prefixed_star),
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(Path(files["particles_mrcs"]).parent),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    # Without strip_prefix this should fail because STAR points to nonexistent prefixed path.
    with pytest.raises(Exception):
        dataset.load_dataset(**base_kwargs)

    # With strip_prefix, loader should recover and resolve to datadir/particles.mrcs.
    cryo = dataset.load_dataset(strip_prefix=bad_prefix + "/", **base_kwargs)
    assert cryo.tilt_series_flag is True
    assert cryo.n_images == files["n_images"]

    batches = _dataset_image_batches(
        cryo,
        batch_size=2,
        subset_indices=np.array([5, 1, 4], dtype=np.int32),
    )
    got = []
    for batch in batches:
        got.extend(_batch_image_indices(batch).tolist())
    assert got == [5, 1, 4]


def test_simulator_tiny_tilt_subsample_cryoem_dataset_preserves_local_order_and_image_identity(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    # Use unique original indices so the remap table is bijective.
    requested = np.array([7, 2, 5, 1], dtype=np.int32)
    sub = dataset.subsample_cryoem_dataset(cryo, requested)
    np.testing.assert_array_equal(np.asarray(sub.dataset_indices), requested)
    assert sub.n_images == requested.size
    assert sub.tilt_series_flag is True

    original_images = utils.load_mrc(files["particles_mrcs"])

    full_local = np.arange(sub.n_images, dtype=np.int32)
    full_batches = _dataset_image_batches(sub, batch_size=2, subset_indices=full_local)
    got_full_images = np.concatenate([_batch_images(batch) for batch in full_batches], axis=0)
    got_full_local = np.concatenate([_batch_image_indices(batch) for batch in full_batches], axis=0)
    np.testing.assert_array_equal(got_full_local, full_local)
    np.testing.assert_allclose(got_full_images, original_images[requested], atol=1e-6)

    local_subset = np.array([3, 0, 2, 1], dtype=np.int32)
    subset_batches = _dataset_image_batches(sub, batch_size=2, subset_indices=local_subset)
    got_subset_images = np.concatenate([_batch_images(batch) for batch in subset_batches], axis=0)
    got_subset_local = np.concatenate([_batch_image_indices(batch) for batch in subset_batches], axis=0)
    np.testing.assert_array_equal(got_subset_local, local_subset)
    np.testing.assert_allclose(got_subset_images, original_images[requested[local_subset]], atol=1e-6)


def test_simulator_tiny_tilt_load_dataset_from_args_preserves_halfset_image_identity(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    args = SimpleNamespace(
        particles=files["particles_star"],
        poses=files["poses_pkl"],
        ctf=files["ctf_pkl"],
        datadir=datadir,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="false",
        tilt_series=True,
        tilt_series_ctf="relion5",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
        strip_prefix=None,
        halfsets=None,
        tilt_ind=None,
        ntilts=None,
    )

    cryos = dataset.load_dataset_from_args(args, lazy=True)
    assert len(cryos.halfset_indices) == 2

    half0 = np.asarray(cryos.halfset_indices[0], dtype=np.int32)
    half1 = np.asarray(cryos.halfset_indices[1], dtype=np.int32)
    assert np.intersect1d(half0, half1).size == 0
    all_selected = np.sort(np.concatenate([half0, half1]))
    np.testing.assert_array_equal(all_selected, np.arange(cryos.n_images, dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    dataset_indices = np.asarray(cryos.dataset_indices, dtype=np.int32)
    for half_idx in cryos.halfset_indices:
        half_dataset_indices = dataset_indices[half_idx]
        batches = _dataset_image_batches(cryos, batch_size=3, subset_indices=half_idx)
        got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
        got_local = np.concatenate([_batch_image_indices(batch) for batch in batches], axis=0)
        np.testing.assert_array_equal(got_local, half_idx)
        np.testing.assert_allclose(
            got_images,
            original_images[half_dataset_indices],
            atol=1e-6,
        )


def test_simulator_tiny_tilt_get_split_datasets_preserves_half_order_and_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ind_split = [
        np.array([7, 2, 7, 1], dtype=np.int32),
        np.array([0, 5, 0], dtype=np.int32),
    ]
    cryos = dataset.get_split_datasets(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind_split=ind_split,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    assert len(cryos.halfset_indices) == 2
    original_images = utils.load_mrc(files["particles_mrcs"])
    dataset_indices = np.asarray(cryos.dataset_indices, dtype=np.int32)
    for half, half_idx in zip(ind_split, cryos.halfset_indices):
        half_dataset_indices = dataset_indices[half_idx]
        np.testing.assert_array_equal(half_dataset_indices, half)
        batches = _dataset_image_batches(cryos, batch_size=2, subset_indices=half_idx)
        got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
        np.testing.assert_allclose(got_images, original_images[half], atol=1e-6)


def test_simulator_tiny_tilt_load_dataset_preserves_ind_order_and_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    requested = np.array([7, 2, 7, 1], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=requested,
        tilt_series=True,
        tilt_series_ctf="relion5",
        lazy=True,
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices), requested)
    assert cryo.tilt_series_flag is True
    assert cryo.n_images == requested.size

    original_images = utils.load_mrc(files["particles_mrcs"])
    local = np.arange(requested.size, dtype=np.int32)
    batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=local)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    np.testing.assert_allclose(got_images, original_images[requested], atol=1e-6)


def test_simulator_tiny_tilt_get_split_datasets_matches_direct_api(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ind_split = [
        np.array([7, 2, 7, 1], dtype=np.int32),
        np.array([0, 5, 0], dtype=np.int32),
    ]
    loader_dict = {
        "particles_file": files["particles_star"],
        "poses_file": files["poses_pkl"],
        "ctf_file": files["ctf_pkl"],
        "datadir": datadir,
        "tilt_series": True,
        "tilt_series_ctf": "relion5",
    }

    by_wrapper = dataset.get_split_datasets(**loader_dict, ind_split=ind_split, lazy=True)
    direct = dataset.get_split_datasets(ind_split=ind_split, lazy=True, **loader_dict)

    assert len(by_wrapper.halfset_indices) == len(direct.halfset_indices) == 2
    np.testing.assert_array_equal(np.asarray(by_wrapper.dataset_indices), np.asarray(direct.dataset_indices))
    assert by_wrapper.n_images == direct.n_images
    for h0, h1 in zip(by_wrapper.halfset_indices, direct.halfset_indices):
        np.testing.assert_array_equal(h0, h1)


def test_simulator_tiny_tilt_load_dataset_from_args_with_explicit_split_skips_halfset_builder(
    sim_tiny_tilt_files, monkeypatch
):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ind_split = [
        np.array([7, 2, 7, 1], dtype=np.int32),
        np.array([0, 5, 0], dtype=np.int32),
    ]

    def _should_not_be_called(_args):
        raise AssertionError("figure_out_halfsets should not be called when ind_split is explicitly provided")

    monkeypatch.setattr(dataset, "figure_out_halfsets", _should_not_be_called)

    args = SimpleNamespace(
        particles=files["particles_star"],
        poses=files["poses_pkl"],
        ctf=files["ctf_pkl"],
        datadir=datadir,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="false",
        tilt_series=True,
        tilt_series_ctf="relion5",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
        strip_prefix=None,
        halfsets=None,
        tilt_ind=None,
        ntilts=None,
    )

    cryos = dataset.load_dataset_from_args(args, lazy=True, ind_split=ind_split)
    assert len(cryos.halfset_indices) == 2
    original_images = utils.load_mrc(files["particles_mrcs"])
    dataset_indices = np.asarray(cryos.dataset_indices, dtype=np.int32)
    for half, half_idx in zip(ind_split, cryos.halfset_indices):
        half_dataset_indices = dataset_indices[half_idx]
        np.testing.assert_array_equal(half_dataset_indices, half)
        batches = _dataset_image_batches(cryos, batch_size=2, subset_indices=half_idx)
        got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
        np.testing.assert_allclose(got_images, original_images[half], atol=1e-6)


def test_simulator_tiny_tilt_load_dataset_eager_mode_preserves_identity(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    requested = np.array([7, 2, 7, 1], dtype=np.int32)

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
        ind=requested,
        tilt_series=True,
        tilt_series_ctf="relion5",
        lazy=False,
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices), requested)
    assert cryo.n_images == requested.size
    original_images = utils.load_mrc(files["particles_mrcs"])

    local = np.arange(requested.size, dtype=np.int32)
    batches = _dataset_image_batches(cryo, batch_size=2, subset_indices=local)
    got_images = np.concatenate([_batch_images(batch) for batch in batches], axis=0)
    np.testing.assert_allclose(got_images, original_images[requested], atol=1e-6)


def test_tiny_tilt_loading_preserves_pose_and_ctf_row_alignment_with_reordered_duplicates(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=8, n_particles=4)
    n = files["n_images"]

    # Build unique per-row CTF/pose values so index shuffles are observable.
    ctf = np.zeros((n, 9), dtype=np.float32)
    ctf[:, 0] = float(files["grid_size"])  # D
    ctf[:, 1] = 1.5  # Apix
    ctf[:, 2] = 10_000.0 + np.arange(n, dtype=np.float32) * 101.0  # DFU
    ctf[:, 3] = 11_000.0 + np.arange(n, dtype=np.float32) * 103.0  # DFV
    ctf[:, 4] = np.arange(n, dtype=np.float32) * 2.0               # DFANG
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1
    ctf[:, 8] = np.arange(n, dtype=np.float32) * 0.25              # phase shift
    utils.pickle_dump(ctf, files["ctf_pkl"])

    rots = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
    rots[:, 0, 0] = np.arange(1, n + 1, dtype=np.float32)
    trans_frac = np.stack(
        [
            np.linspace(0.0, 0.7, n, dtype=np.float32),
            np.linspace(0.1, 0.8, n, dtype=np.float32),
        ],
        axis=1,
    )
    utils.pickle_dump((rots, trans_frac), files["poses_pkl"])

    requested = np.array([6, 2, 6, 1], dtype=np.int32)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    np.testing.assert_allclose(np.asarray(cryo.rotation_matrices), rots[requested], atol=1e-7)
    np.testing.assert_allclose(np.asarray(cryo.translations), trans_frac[requested] * files["grid_size"], atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFU],
        ctf[requested, 2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFV],
        ctf[requested, 3],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.PHASE_SHIFT],
        ctf[requested, 8],
        atol=1e-7,
    )


def test_tiny_spa_with_tilt_ctf_preserves_pose_and_ctf_row_alignment(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=8, n_particles=4)
    n = files["n_images"]

    ctf = np.zeros((n, 9), dtype=np.float32)
    ctf[:, 0] = float(files["grid_size"])  # D
    ctf[:, 1] = 1.5
    ctf[:, 2] = 9_000.0 + np.arange(n, dtype=np.float32) * 89.0
    ctf[:, 3] = 9_500.0 + np.arange(n, dtype=np.float32) * 97.0
    ctf[:, 4] = np.arange(n, dtype=np.float32) * 1.5
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1
    ctf[:, 8] = np.arange(n, dtype=np.float32) * 0.3
    utils.pickle_dump(ctf, files["ctf_pkl"])

    rots = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
    rots[:, 1, 1] = np.arange(1, n + 1, dtype=np.float32)
    trans_frac = np.stack(
        [
            np.linspace(0.0, 0.6, n, dtype=np.float32),
            np.linspace(0.2, 0.8, n, dtype=np.float32),
        ],
        axis=1,
    )
    utils.pickle_dump((rots, trans_frac), files["poses_pkl"])

    requested = np.array([7, 1, 7, 3], dtype=np.int32)
    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="relion5",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    np.testing.assert_allclose(np.asarray(cryo.rotation_matrices), rots[requested], atol=1e-7)
    np.testing.assert_allclose(np.asarray(cryo.translations), trans_frac[requested] * files["grid_size"], atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFU],
        ctf[requested, 2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFV],
        ctf[requested, 3],
        atol=1e-7,
    )


def test_tiny_tilt_get_split_datasets_preserves_pose_and_ctf_alignment(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=8, n_particles=4)
    n = files["n_images"]

    ctf = np.zeros((n, 9), dtype=np.float32)
    ctf[:, 0] = float(files["grid_size"])  # D
    ctf[:, 1] = 1.5
    ctf[:, 2] = 12_000.0 + np.arange(n, dtype=np.float32) * 41.0
    ctf[:, 3] = 13_000.0 + np.arange(n, dtype=np.float32) * 43.0
    ctf[:, 4] = np.arange(n, dtype=np.float32) * 1.25
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1
    ctf[:, 8] = np.arange(n, dtype=np.float32) * 0.2
    utils.pickle_dump(ctf, files["ctf_pkl"])

    rots = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
    rots[:, 2, 2] = np.arange(1, n + 1, dtype=np.float32)
    trans_frac = np.stack(
        [
            np.linspace(0.0, 0.6, n, dtype=np.float32),
            np.linspace(0.2, 0.8, n, dtype=np.float32),
        ],
        axis=1,
    )
    utils.pickle_dump((rots, trans_frac), files["poses_pkl"])

    ind_split = [
        np.array([7, 1, 7, 3], dtype=np.int32),
        np.array([0, 5, 0], dtype=np.int32),
    ]
    cryos = dataset.get_split_datasets(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind_split=ind_split,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    assert len(cryos.halfset_indices) == 2
    dataset_indices = np.asarray(cryos.dataset_indices, dtype=np.int32)
    for half, half_idx in zip(ind_split, cryos.halfset_indices):
        half_dataset_indices = dataset_indices[half_idx]
        np.testing.assert_array_equal(half_dataset_indices, half)
        np.testing.assert_allclose(np.asarray(cryos.rotation_matrices)[half_idx], rots[half], atol=1e-7)
        np.testing.assert_allclose(np.asarray(cryos.translations)[half_idx], trans_frac[half] * files["grid_size"], atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(cryos.CTF_params)[half_idx, core.CTFParamIndex.DFU],
            ctf[half, 2],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(cryos.CTF_params)[half_idx, core.CTFParamIndex.DFV],
            ctf[half, 3],
            atol=1e-7,
        )


@pytest.mark.parametrize("tilt_series", [True, False])
def test_tiny_relion5_ctf_channels_preserve_requested_order_with_duplicates(tmp_path, tilt_series):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=8, n_particles=4)
    n = files["n_images"]
    requested = np.array([7, 1, 7, 3], dtype=np.int32)

    star = starfile.StarFile.load(files["particles_star"])
    exposure = np.array([9.0, 1.0, 7.0, 3.0, 8.0, 2.0, 6.0, 4.0], dtype=np.float32)
    scale = np.array([0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    star.df["_rlnMicrographPreExposure"] = exposure
    star.df["_rlnCtfScalefactor"] = scale
    starfile.write_star(files["particles_star"], data=star.df)

    # Keep CTF pickle valid but distinctive in DFU/DFV as an extra sanity check.
    ctf = np.zeros((n, 9), dtype=np.float32)
    ctf[:, 0] = float(files["grid_size"])
    ctf[:, 1] = 1.5
    ctf[:, 2] = 14_000.0 + np.arange(n, dtype=np.float32) * 31.0
    ctf[:, 3] = 15_000.0 + np.arange(n, dtype=np.float32) * 37.0
    ctf[:, 4] = 0.0
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1
    ctf[:, 8] = 0.0
    utils.pickle_dump(ctf, files["ctf_pkl"])

    cryo = dataset.load_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=requested,
        lazy=True,
        tilt_series=tilt_series,
        tilt_series_ctf="relion5",
    )

    np.testing.assert_array_equal(np.asarray(cryo.dataset_indices, dtype=np.int32), requested)
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFU],
        ctf[requested, 2],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DFV],
        ctf[requested, 3],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.CONTRAST],
        scale[requested],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.DOSE],
        exposure[requested],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cryo.CTF_params)[:, core.CTFParamIndex.TILT_ANGLE],
        np.zeros(requested.size, dtype=np.float32),
        atol=1e-7,
    )


def test_tiny_tilt_image_count_batch_loader_subset_len_matches_emitted_images(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=7, n_particles=3)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=str(tmp_path),
        lazy=True,
        num_tilts=None,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    # Duplicate/reordered subset over particles with uneven tilt counts.
    subset = cryo_dataset.ParticleSubset(ds, np.array([2, 2], dtype=np.int32))
    loader = cryo_dataset.ImageCountBatchLoader(subset, batch_size=4, pad_to_batch=False)

    # Particle 2 has 2 tilts in this tiny STAR, duplicated twice => 4 total images.
    assert loader.total_images == 4
    assert len(loader) == 1

    batches = list(loader)
    assert len(batches) == len(loader)
    total_emitted = int(sum(np.asarray(b[0]).shape[0] for b in batches))
    assert total_emitted == loader.total_images

    pidx = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    tidx = np.concatenate([np.asarray(b[2]).reshape(-1) for b in batches], axis=0)
    assert np.all(pidx == 2)
    np.testing.assert_array_equal(np.sort(np.unique(tidx)), np.sort(np.asarray(ds._particle_tilts[2], dtype=np.int32)))


def test_simulator_tiny_tilt_image_count_batch_loader_reports_consistent_lengths(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=2,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    subset = cryo_dataset.ParticleSubset(ds, np.array([2, 0, 2], dtype=np.int32))
    loader = cryo_dataset.ImageCountBatchLoader(subset, batch_size=3, pad_to_batch=False)

    batches = list(loader)
    assert len(loader) == len(batches)
    total_emitted = int(sum(np.asarray(b[0]).shape[0] for b in batches))
    assert total_emitted == loader.total_images


def test_simulator_tiny_tilt_image_count_batch_loader_nested_torch_subsets(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=2,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    subset_lvl1 = cryo_dataset._SimpleSubset(ds, [3, 1, 0])
    subset_lvl2 = cryo_dataset._SimpleSubset(subset_lvl1, [2, 0, 2])  # maps to base particle ids [0, 3, 0]
    loader = cryo_dataset.ImageCountBatchLoader(subset_lvl2, batch_size=3, pad_to_batch=False)

    batches = list(loader)
    assert len(loader) == len(batches)
    assert loader.total_images == 6

    emitted_particles = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    emitted_tilts = np.concatenate([np.asarray(b[2]).reshape(-1) for b in batches], axis=0)

    np.testing.assert_array_equal(emitted_particles, np.array([0, 0, 3, 3, 0, 0], dtype=np.int32))

    expected_tilts = np.concatenate(
        [
            np.asarray(ds[0][2]).reshape(-1),
            np.asarray(ds[3][2]).reshape(-1),
            np.asarray(ds[0][2]).reshape(-1),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(emitted_tilts, expected_tilts)


def test_simulator_tiny_tilt_image_count_batch_loader_num_tilts_zero_emits_no_batches(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=0,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    assert ds._max_tilts_per_particle() == 0
    first_imgs, _first_particle, first_tilts = ds[0]
    assert np.asarray(first_imgs).shape[0] == 0
    assert np.asarray(first_tilts).size == 0

    loader = cryo_dataset.ImageCountBatchLoader(ds, batch_size=4, pad_to_batch=False)
    assert loader.total_images == 0
    assert len(loader) == 0
    assert list(loader) == []

    # Same invariant through the dataset generator API in image-batching mode.
    gen = ds.get_dataset_generator(batch_size=4, mode="images")
    assert list(gen) == []


def test_simulator_tiny_tilt_negative_num_tilts_matches_all_tilts(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds_all = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=None,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    ds_neg = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=-1,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    assert ds_neg.num_tilts is None
    assert ds_neg._max_tilts_per_particle() == ds_all._max_tilts_per_particle()

    for pidx in range(len(ds_all)):
        imgs_all, _p_all, tilts_all = ds_all[pidx]
        imgs_neg, _p_neg, tilts_neg = ds_neg[pidx]
        np.testing.assert_array_equal(np.asarray(tilts_neg), np.asarray(tilts_all))
        assert np.asarray(imgs_neg).shape == np.asarray(imgs_all).shape


def test_simulator_tiny_tilt_series_generator_forces_batch_size_one_and_particle_order(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesDataset(
        files["particles_star"],
        datadir=datadir,
        lazy=True,
        num_tilts=1,
        random_tilts=False,
        tilt_file_option="relion5",
    )

    loader = ds.get_dataset_generator(batch_size=128, mode="tilt_series")
    assert loader.batch_size == 1

    got_particles = []
    for idx, batch in enumerate(loader):
        pidx = np.asarray(batch[1]).reshape(-1)
        got_particles.append(int(pidx[0]))
        if idx >= 4:
            break

    np.testing.assert_array_equal(np.asarray(got_particles, dtype=np.int32), np.arange(len(got_particles), dtype=np.int32))
