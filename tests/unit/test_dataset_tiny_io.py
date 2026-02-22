import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("jax")
pytest.importorskip("torch")

from helpers import tiny_synthetic
from recovar import dataset, core, utils, starfile
from recovar import tilt_dataset

pytestmark = pytest.mark.unit


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


def test_load_cryodrgn_dataset_tiny_tilt_series_from_simulator_files(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    cryo = dataset.load_cryodrgn_dataset(
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
    assert cryo.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2

    subset_images = np.array([7, 2, 7, 11], dtype=np.int32)
    image_batches = list(cryo.get_image_subset_generator(batch_size=2, subset_indices=subset_images))
    got_images = []
    for b in image_batches:
        got_images.extend(np.array(b[2]).reshape(-1).tolist())
    assert got_images == subset_images.tolist()


def test_simulator_tiny_tilt_series_to_images_matches_particle_mapping(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
    subset_particles = np.array([2, 0, 2], dtype=np.int32)

    mapped = tilt_dataset.tilt_series_indices_to_image_indices(subset_particles, files["particles_star"])
    expected = np.concatenate(
        [
            np.asarray(particles_to_tilts[2], dtype=np.int32),
            np.asarray(particles_to_tilts[0], dtype=np.int32),
            np.asarray(particles_to_tilts[2], dtype=np.int32),
        ]
    )
    np.testing.assert_array_equal(np.asarray(mapped, dtype=np.int32), expected)


def test_simulator_tiny_tilt_series_to_images_with_subset_preserves_order_and_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
    subset_particles = np.array([2, 0, 2], dtype=np.int32)
    full = np.concatenate(
        [
            np.asarray(particles_to_tilts[2], dtype=np.int32),
            np.asarray(particles_to_tilts[0], dtype=np.int32),
            np.asarray(particles_to_tilts[2], dtype=np.int32),
        ]
    )

    image_subset = np.array([int(full[1]), int(full[-1]), int(full[1])], dtype=np.int32)
    mapped = tilt_dataset.tilt_series_indices_to_image_indices(
        subset_particles,
        files["particles_star"],
        image_subset=image_subset,
    )
    expected = full[np.isin(full, image_subset)]
    np.testing.assert_array_equal(np.asarray(mapped, dtype=np.int32), expected)


def test_simulator_tiny_tilt_series_to_images_accepts_boolean_image_subset_mask(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
    subset_particles = np.array([1, 0], dtype=np.int32)
    full = np.concatenate(
        [
            np.asarray(particles_to_tilts[1], dtype=np.int32),
            np.asarray(particles_to_tilts[0], dtype=np.int32),
        ]
    )

    mask = np.zeros(files["n_images"], dtype=bool)
    mask[int(full[0])] = True
    mask[int(full[-1])] = True
    mapped = tilt_dataset.tilt_series_indices_to_image_indices(
        subset_particles,
        files["particles_star"],
        image_subset=mask,
    )
    expected = full[np.isin(full, np.flatnonzero(mask))]
    np.testing.assert_array_equal(np.asarray(mapped, dtype=np.int32), expected)


def test_simulator_tiny_tilt_series_to_images_rejects_wrong_length_boolean_mask(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    with pytest.raises(ValueError, match="must match number of images"):
        tilt_dataset.tilt_series_indices_to_image_indices(
            np.array([0, 1], dtype=np.int32),
            files["particles_star"],
            image_subset=np.array([True, False, True], dtype=bool),
        )


def test_simulator_tilt_dataset_subset_generator_images_mode_preserves_particle_order_and_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    ds = tilt_dataset.TiltSeriesData(
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
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
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
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
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
    ds = tilt_dataset.TiltSeriesData(
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
    ds = tilt_dataset.TiltSeriesData(
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
    ds = tilt_dataset.TiltSeriesData(
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

    cryo = dataset.load_cryodrgn_dataset(
        particles_file=files["particles_star"],
        poses_file=files["poses_pkl"],
        ctf_file=files["ctf_pkl"],
        datadir=datadir,
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
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
    tilt_ds = tilt_dataset.TiltSeriesData(files["particles_star"], datadir=datadir)
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
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
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
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
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


def test_load_cryodrgn_dataset_simulator_tilt_ctf_in_spa_mode_preserves_reordered_duplicates(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    reordered_ind = np.array([8, 1, 8, 5], dtype=np.int32)

    cryo = dataset.load_cryodrgn_dataset(
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
    assert cryo.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2

    batches = list(cryo.get_image_generator(batch_size=2))
    got_images = np.concatenate([np.array(b[0]) for b in batches], axis=0)
    got_local_idx = np.concatenate([np.array(b[2]).reshape(-1) for b in batches], axis=0)
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


def test_simulator_tiny_tilt_split_indices_accepts_boolean_masks(sim_tiny_tilt_files):
    files = sim_tiny_tilt_files
    datadir = str(Path(files["particles_star"]).parent)
    particles_to_tilts, _ = tilt_dataset.TiltSeriesData.parse_particle_tilt(files["particles_star"])
    n_particles = len(particles_to_tilts)

    keep_particles = np.array([0, n_particles - 1], dtype=np.int32)
    particle_mask = np.zeros(n_particles, dtype=bool)
    particle_mask[keep_particles] = True

    # Build expected one-tilt-per-particle set under deterministic ordering.
    tilt_ds = tilt_dataset.TiltSeriesData(files["particles_star"], datadir=datadir)
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
