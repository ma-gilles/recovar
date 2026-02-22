import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("jax")
pytest.importorskip("torch")

from helpers import tiny_synthetic
from recovar import dataset, core, utils, starfile

pytestmark = pytest.mark.unit


def test_load_cryodrgn_dataset_tiny_spa_files(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    ind = np.array([1, 4], dtype=np.int32)

    cryo = dataset.load_cryodrgn_dataset(
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
    assert cryo.CTF_fun_inp is core.evaluate_ctf_wrapper


def test_load_cryodrgn_dataset_tiny_tilt_series_files_and_subset_generators(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_cryodrgn_dataset(
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
    assert cryo.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2

    # Particle-subset generator should preserve requested particle order.
    subset_particles = np.array([2, 0], dtype=np.int32)
    batches = list(cryo.get_dataset_subset_generator(batch_size=8, subset_indices=subset_particles, mode="tilt_series"))
    got_particles = [int(np.array(b[1]).reshape(-1)[0]) for b in batches]
    assert got_particles == subset_particles.tolist()

    # Image-subset generator should only emit requested image indices, in order.
    subset_images = np.array([5, 1, 4], dtype=np.int32)
    image_batches = list(cryo.get_image_subset_generator(batch_size=2, subset_indices=subset_images))
    got_images = []
    for b in image_batches:
        got_images.extend(np.array(b[2]).reshape(-1).tolist())
    assert got_images == subset_images.tolist()


def test_load_cryodrgn_dataset_tiny_tilt_series_warp_path(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_cryodrgn_dataset(
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
    assert cryo.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    # v2 branch appends dose and angle channels.
    assert cryo.CTF_params.shape[1] >= 11

    subset_images = np.array([5, 1, 4], dtype=np.int32)
    image_batches = list(cryo.get_image_subset_generator(batch_size=2, subset_indices=subset_images))
    got_images = []
    for b in image_batches:
        got_images.extend(np.array(b[2]).reshape(-1).tolist())
    assert got_images == subset_images.tolist()


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

    cryo = dataset.load_cryodrgn_dataset(
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
    batches = list(cryo.get_image_generator(batch_size=2))
    got_images = np.concatenate([np.array(b[0]) for b in batches], axis=0)
    got_local_idx = np.concatenate([np.array(b[2]).reshape(-1) for b in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[reordered_ind], atol=1e-6)


def test_tiny_tilt_loading_with_duplicate_ind_preserves_duplicates_and_order(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    duplicated_ind = np.array([5, 1, 5, 4], dtype=np.int32)

    cryo = dataset.load_cryodrgn_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        ind=duplicated_ind,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    batches = list(cryo.get_image_generator(batch_size=2))
    got_images = np.concatenate([np.array(b[0]) for b in batches], axis=0)
    got_local_idx = np.concatenate([np.array(b[2]).reshape(-1) for b in batches], axis=0)
    np.testing.assert_array_equal(got_local_idx, np.array([0, 1, 2, 3], dtype=np.int32))

    original_images = utils.load_mrc(files["particles_mrcs"])
    np.testing.assert_allclose(got_images, original_images[duplicated_ind], atol=1e-6)


def test_tiny_tilt_particle_subset_generator_preserves_duplicates(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    cryo = dataset.load_cryodrgn_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=str(tmp_path),
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )

    subset_particles = np.array([2, 0, 2], dtype=np.int32)
    batches = list(cryo.get_dataset_subset_generator(batch_size=8, subset_indices=subset_particles, mode="tilt_series"))
    got_particles = [int(np.array(b[1]).reshape(-1)[0]) for b in batches]
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


def test_tiny_tilt_loading_with_strip_prefix_resolves_prefixed_star_paths(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path / "tiny_data", grid_size=8, n_images=6, n_particles=3)
    bad_prefix = "/definitely/not/a/real/path"
    mrcs_name = Path(files["particles_mrcs"]).name

    star = starfile.Starfile.load(files["particles_star"])
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
        dataset.load_cryodrgn_dataset(**base_kwargs)

    # With strip_prefix, loader should recover and resolve to datadir/particles.mrcs.
    cryo = dataset.load_cryodrgn_dataset(strip_prefix=bad_prefix + "/", **base_kwargs)
    assert cryo.tilt_series_flag is True
    assert cryo.n_images == files["n_images"]

    batches = list(cryo.get_image_subset_generator(batch_size=2, subset_indices=np.array([5, 1, 4], dtype=np.int32)))
    got = []
    for batch in batches:
        got.extend(np.array(batch[2]).reshape(-1).tolist())
    assert got == [5, 1, 4]
