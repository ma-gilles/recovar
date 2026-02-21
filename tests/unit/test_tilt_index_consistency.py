import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from recovar import cryo_dataset, dataset, starfile

pytestmark = pytest.mark.unit


def _write_mrcs(path: Path, data: np.ndarray) -> None:
    import mrcfile

    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))


def _make_tilt_fixture(tmp_path: Path):
    # 8 images, 3 particles (groups), deliberately interleaved in STAR row order.
    rows = [
        ("gB", 10.0, -2.0),
        ("gA", 1.0, -1.0),
        ("gC", 7.0, -3.0),
        ("gB", 30.0, -6.0),
        ("gA", 20.0, -4.0),
        ("gC", 5.0, -5.0),
        ("gB", 20.0, -7.0),
        ("gC", 40.0, -8.0),
    ]
    n = len(rows)
    D = 8
    stack = np.arange(n * D * D, dtype=np.float32).reshape(n, D, D)
    mrcs_path = tmp_path / "tilt_stack.mrcs"
    _write_mrcs(mrcs_path, stack)

    df = pd.DataFrame(
        {
            "_rlnImageName": [f"{i + 1}@{mrcs_path.name}" for i in range(n)],
            "_rlnGroupName": [r[0] for r in rows],
            "_rlnMicrographPreExposure": [r[1] for r in rows],
            "_rlnCtfScalefactor": np.ones(n, dtype=np.float32),
            "_rlnCtfBfactor": np.array([r[2] for r in rows], dtype=np.float32),
        }
    )
    star_path = tmp_path / "tilt_particles.star"
    starfile.write_star(str(star_path), data=df)
    return star_path


def _load_golden():
    golden_path = Path(__file__).resolve().parents[1] / "baselines" / "tilt_index_consistency" / "v1.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_particle_tilt_mapping_matches_golden_reference(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    golden = _load_golden()

    particles_to_tilts, tilts_to_particles = cryo_dataset.TiltSeriesDataset.parse_particle_tilt(str(star_path))
    particles_to_tilts = [arr.tolist() for arr in particles_to_tilts]
    tilts_to_particles = {str(k): int(v) for k, v in sorted(tilts_to_particles.items())}

    assert cryo_dataset.get_canonical_group_names(starfile.Starfile.load(str(star_path)).df) == golden["canonical_groups"]
    assert particles_to_tilts == golden["particles_to_tilts"]
    assert tilts_to_particles == golden["tilts_to_particles"]

    subset_particles = np.array(golden["subset_particles"], dtype=np.int32)
    subset_images = np.array([1, 2, 4, 7], dtype=np.int32)
    mapped = cryo_dataset.tilt_series_to_images(subset_particles, str(star_path), image_subset=subset_images)
    np.testing.assert_array_equal(mapped, np.array(golden["subset_image_intersection"], dtype=np.int32))


def test_tilt_series_subset_generator_preserves_subset_order_and_indices(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        lazy=True,
        random_tilts=False,
        num_tilts=2,
        tilt_file_option="relion5",
    )
    subset = np.array([2, 0], dtype=np.int32)
    gen = ds.get_dataset_subset_generator(
        batch_size=8,
        subset_indices=subset,
        mode="tilt_series",
    )
    batches = list(gen)
    assert len(batches) == len(subset)

    returned_particles = []
    returned_tilts = []
    for batch in batches:
        # simple_dataloader enforces batch_size=1 for tilt-series mode.
        returned_particles.append(int(np.array(batch[1]).reshape(-1)[0]))
        returned_tilts.append(np.array(batch[2]).reshape(-1))

    assert returned_particles == subset.tolist()
    for i, pidx in enumerate(subset):
        expected_tilts = np.array(ds[pidx][2]).reshape(-1)
        np.testing.assert_array_equal(returned_tilts[i], expected_tilts)


def test_image_subset_generator_returns_only_requested_tilt_indices(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        lazy=True,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    subset_images = np.array([7, 1, 4], dtype=np.int32)
    gen = ds.get_image_subset_generator(batch_size=2, subset_indices=subset_images)

    got = []
    for batch in gen:
        got.extend(np.array(batch[2]).reshape(-1).tolist())
    assert got == subset_images.tolist()


def test_get_split_tilt_indices_with_precomputed_particle_halfsets_is_stable(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)

    ind_file = tmp_path / "ind.pkl"
    tilt_ind_file = tmp_path / "tilt_ind.pkl"
    halfsets_file = tmp_path / "halfsets.pkl"

    with open(ind_file, "wb") as f:
        # Keep only a strict subset of image indices.
        pickle.dump(np.array([1, 2, 7], dtype=np.int32), f)
    with open(tilt_ind_file, "wb") as f:
        # Keep only particles 0 and 2 in canonical order.
        pickle.dump(np.array([0, 2], dtype=np.int32), f)
    with open(halfsets_file, "wb") as f:
        # One half contains particle 0, other half contains particle 2.
        pickle.dump([np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)], f)

    split = dataset.get_split_tilt_indices(
        particles_file=str(star_path),
        ind_file=str(ind_file),
        tilt_ind_file=str(tilt_ind_file),
        datadir=str(tmp_path),
        particle_halfset_indices_file=str(halfsets_file),
    )

    # particle 0 -> tilts [1, 4] intersect [1,2,7] => [1]
    # particle 2 -> tilts [2, 5, 7] intersect [1,2,7] => [2, 7]
    np.testing.assert_array_equal(split[0], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([2, 7], dtype=np.int32))

    # No overlap, complete coverage of allowed image subset.
    assert np.intersect1d(split[0], split[1]).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.array([1, 2, 7], dtype=np.int32))


def test_get_split_tilt_indices_precomputed_halfsets_respected_with_image_filter(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)

    ind_file = tmp_path / "ind_subset.pkl"
    halfsets_file = tmp_path / "halfsets.pkl"

    with open(ind_file, "wb") as f:
        # Keep only images from canonical particles 0 and 2.
        pickle.dump(np.array([1, 2, 4, 7], dtype=np.int32), f)
    with open(halfsets_file, "wb") as f:
        # Explicit split: half0 has particles 0 and 2, half1 has particle 1.
        pickle.dump([np.array([0, 2], dtype=np.int32), np.array([1], dtype=np.int32)], f)

    split = dataset.get_split_tilt_indices(
        particles_file=str(star_path),
        ind_file=str(ind_file),
        datadir=str(tmp_path),
        particle_halfset_indices_file=str(halfsets_file),
    )

    np.testing.assert_array_equal(split[0], np.array([1, 2, 4, 7], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([], dtype=np.int32))
    assert np.intersect1d(split[0], split[1]).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.array([1, 2, 4, 7], dtype=np.int32))


@pytest.mark.parametrize(
    "tilt_file_option",
    ["relion5", "warp"],
)
def test_tilt_dataset_subset_images_mode_only_emits_subset_particles(tmp_path, tilt_file_option):
    star_path = _make_tilt_fixture(tmp_path)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        lazy=True,
        random_tilts=False,
        num_tilts=2,
        tilt_file_option=tilt_file_option,
    )
    subset_particles = np.array([2, 0], dtype=np.int32)
    gen = ds.get_dataset_subset_generator(
        batch_size=4,
        subset_indices=subset_particles,
        mode="images",
    )
    seen_particles = []
    seen_tilts = []
    for batch in gen:
        seen_particles.extend(np.array(batch[1]).reshape(-1).tolist())
        seen_tilts.extend(np.array(batch[2]).reshape(-1).tolist())

    assert set(seen_particles).issubset(set(subset_particles.tolist()))
    expected_tilts = np.concatenate([ds[pidx][2] for pidx in subset_particles]).astype(int)
    np.testing.assert_array_equal(np.sort(np.asarray(seen_tilts, dtype=int)), np.sort(expected_tilts))


def test_tilt_dataset_subset_none_matches_full_sequence(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        lazy=True,
        random_tilts=False,
        num_tilts=2,
        tilt_file_option="relion5",
    )

    full_batches = list(ds.get_dataset_generator(batch_size=16, mode="tilt_series"))
    subset_none_batches = list(ds.get_dataset_subset_generator(batch_size=16, subset_indices=None, mode="tilt_series"))

    assert len(full_batches) == len(subset_none_batches) == len(ds)
    for b_full, b_subset in zip(full_batches, subset_none_batches):
        np.testing.assert_array_equal(np.array(b_full[1]), np.array(b_subset[1]))
        np.testing.assert_array_equal(np.array(b_full[2]), np.array(b_subset[2]))


def test_tilt_series_ind_filtering_preserves_original_indices(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    # Keep rows from all groups but sparse and unordered.
    ind = np.array([7, 1, 3, 4], dtype=np.int32)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        ind=ind,
        lazy=True,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    # Remaining groups should be gA:[1,4], gB:[3], gC:[7]
    p2t, t2p = cryo_dataset.TiltSeriesDataset.parse_particle_tilt(str(star_path), indices=ind)
    expected = [np.array([1, 4], dtype=int), np.array([3], dtype=int), np.array([7], dtype=int)]
    for got, exp in zip(p2t, expected):
        np.testing.assert_array_equal(got, exp)
    assert t2p[1] == 0 and t2p[4] == 0 and t2p[3] == 1 and t2p[7] == 2
    assert ds.num_particles == 3


def test_random_tilt_selection_respects_num_tilts_and_membership(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    np.random.seed(0)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        lazy=True,
        random_tilts=True,
        num_tilts=1,
        tilt_file_option="relion5",
    )
    for pidx in range(len(ds)):
        _, _, selected = ds[pidx]
        assert selected.size == 1
        all_group_tilts = list(ds.particle_groups.values())[pidx]
        assert int(selected[0]) in set(all_group_tilts.tolist())


def test_get_split_tilt_indices_random_split_has_disjoint_complete_coverage(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    ind_file = tmp_path / "ind_all.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.arange(8, dtype=np.int32), f)

    split = dataset.get_split_tilt_indices(
        particles_file=str(star_path),
        ind_file=str(ind_file),
        datadir=str(tmp_path),
    )
    assert len(split) == 2
    assert np.intersect1d(split[0], split[1]).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.arange(8, dtype=np.int32))


def test_get_split_tilt_indices_with_ntilts_filters_per_particle(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    # Canonical particles:
    # gA -> [1,4], gB -> [0,3,6], gC -> [2,5,7]
    # RELION5 ordering in fixture (higher dose first):
    # gA keeps tilt-order ranks [4,1] => with ntilts=1 keep one per particle in get_split_tilt_indices by tilt_numbers < 1.
    split = dataset.get_split_tilt_indices(
        particles_file=str(star_path),
        ntilts=1,
        datadir=str(tmp_path),
    )
    # exactly one tilt per particle in whichever halfset each particle is assigned.
    kept = np.sort(np.concatenate(split))
    assert kept.size == 3
    p2t, _ = cryo_dataset.TiltSeriesDataset.parse_particle_tilt(str(star_path))
    for particle_tilts in p2t:
        assert np.intersect1d(kept, particle_tilts).size == 1


def test_particle_subset_max_tilts_matches_parent_counts(tmp_path):
    star_path = _make_tilt_fixture(tmp_path)
    ds = cryo_dataset.TiltSeriesDataset(
        str(star_path),
        datadir=str(tmp_path),
        num_tilts=2,
        random_tilts=False,
        tilt_file_option="relion5",
    )
    subset = cryo_dataset.ParticleSubset(ds, np.array([1, 2], dtype=np.int32))
    # both chosen particles have >=2 tilts; max should be 2 due to num_tilts cap.
    assert subset._max_tilts_per_particle() == 2
