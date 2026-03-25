"""
Unit tests for cryo-ET I/O and particle/image subset handling.

Covers the brittle --ind / --particle-ind / --image-ind indexing paths that
underpin cryo-ET data loading.  All tests use tiny on-disk data created by
the simulator so that exact byte-level STAR/MRC compatibility is exercised.

Tested paths
------------
* load_dataset(tilt_series=True, ind=...)
  - image-level subset that leaves some particles with fewer tilts
  - boolean mask subset (all-True identity, all-False empty)
  - duplicate image indices propagate through CTF / poses
* get_split_tilt_indices
  - particle-level filter only (tilt_ind_file)
  - image-level filter only (ind_file)
  - combined ind_file + tilt_ind_file (intersection logic)
  - ntilts per-particle capping
  - precomputed particle halfsets with out-of-range / duplicate ids
* reorder_to_original_indexing_from_halfsets
  - round-trips through two non-overlapping halfsets
  - NaN fill for images absent from both halfsets
  - raises on duplicate dataset indices
* CTF / pose shape alignment after every kind of subset

NOTE on parse_particle_tilt return type
----------------------------------------
TiltSeriesDataset.parse_particle_tilt(starfile) returns:
  particles_to_tilts: List[np.ndarray]  – one array of image indices per particle
  tilts_to_particles: Dict[int, int]    – image_idx -> particle_idx
Particle i has images particles_to_tilts[i].
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from helpers import tiny_synthetic
from recovar.data_io import cryoem_dataset as recovar_dataset, halfsets, image_backends as cryo_dataset

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Module-level fixture – create files once, share across all tests in file
# ---------------------------------------------------------------------------

# 3 tilts × 8 particles = 24 images; 4 distinct volumes
_GRID_SIZE = 8
_N_TILTS = 3
_N_VOLUMES = 4
_N_IMAGES = 24  # n_tilts × n_particles
_N_PARTICLES = _N_IMAGES // _N_TILTS  # 8


@pytest.fixture(scope="module")
def tilt_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("cryo_et_io_tiny")
    return tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        out,
        grid_size=_GRID_SIZE,
        n_images=_N_IMAGES,
        n_tilts=_N_TILTS,
        n_volumes=_N_VOLUMES,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tilt_cryo(tilt_files, ind=None):
    """Load tiny cryo-ET dataset with optional image-level ind."""
    return recovar_dataset.load_dataset(
        particles_file=tilt_files["particles_star"],
        poses_file=tilt_files["poses_pkl"],
        ctf_file=tilt_files["ctf_pkl"],
        ind=ind,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )


def _parse_p2t(tilt_files):
    """Return (particles_to_tilts: List[ndarray], tilts_to_particles: dict)."""
    return cryo_dataset.TiltSeriesDataset.parse_particle_tilt(tilt_files["particles_star"])


def _all_image_indices(p2t):
    """Concatenate all image indices from a particles_to_tilts list."""
    return np.concatenate(p2t).astype(np.int32)


# ---------------------------------------------------------------------------
# 1. Basic cryo-ET loading (no subset)
# ---------------------------------------------------------------------------


def test_load_tilt_no_ind_n_images_and_particles(tilt_files):
    cryo = _load_tilt_cryo(tilt_files, ind=None)
    assert cryo.tilt_series_flag is True
    assert cryo.n_images == _N_IMAGES
    assert cryo.n_units == _N_PARTICLES


def test_load_tilt_no_ind_ctf_pose_shapes(tilt_files):
    cryo = _load_tilt_cryo(tilt_files, ind=None)
    assert cryo.CTF_params.shape[0] == _N_IMAGES
    assert cryo.rotation_matrices.shape == (_N_IMAGES, 3, 3)
    assert cryo.translations.shape == (_N_IMAGES, 2)


# ---------------------------------------------------------------------------
# 2. Image-level subset (--ind): sparse particles
# ---------------------------------------------------------------------------


def test_load_tilt_with_image_ind_keeps_correct_image_count(tilt_files):
    """Selecting a proper subset of images gives a smaller dataset."""
    p2t, _ = _parse_p2t(tilt_files)
    # Keep only images from the first two particles
    selected_images = np.concatenate([p2t[0], p2t[1]]).astype(np.int32)

    cryo = _load_tilt_cryo(tilt_files, ind=selected_images)
    assert cryo.n_images == selected_images.size


def test_load_tilt_with_image_ind_ctf_pose_align(tilt_files):
    """CTF and pose arrays have exactly n_images rows after image-level subset."""
    p2t, _ = _parse_p2t(tilt_files)
    selected = np.concatenate([p2t[0], p2t[1]]).astype(np.int32)

    cryo = _load_tilt_cryo(tilt_files, ind=selected)
    n = cryo.n_images
    assert cryo.CTF_params.shape[0] == n
    assert cryo.rotation_matrices.shape[0] == n
    assert cryo.translations.shape[0] == n


def test_load_tilt_with_image_ind_removes_some_tilts_from_a_particle(tilt_files):
    """
    Selecting only some tilts of a particle yields a particle with fewer tilts.
    The particle must still appear in the dataset (not completely dropped).
    """
    p2t, _ = _parse_p2t(tilt_files)
    # All tilts of first particle + only first tilt of second particle
    first_p_tilts = np.asarray(p2t[0], dtype=np.int32)
    second_p_first_tilt = np.asarray(p2t[1], dtype=np.int32)[:1]
    selected = np.concatenate([first_p_tilts, second_p_first_tilt])

    cryo = _load_tilt_cryo(tilt_files, ind=selected)
    assert cryo.n_images == selected.size
    # Both particles should still be present
    assert cryo.n_units == 2


def test_load_tilt_image_ind_all_finite_ctf(tilt_files):
    """All CTF params are finite after image-level subset."""
    p2t, _ = _parse_p2t(tilt_files)
    selected = _all_image_indices(p2t)[:6]
    cryo = _load_tilt_cryo(tilt_files, ind=selected)
    assert np.all(np.isfinite(cryo.CTF_params))


# ---------------------------------------------------------------------------
# 3. Boolean mask variants
# ---------------------------------------------------------------------------


def test_load_tilt_boolean_mask_all_true_is_identity(tilt_files):
    """Boolean all-True mask is equivalent to loading with no ind."""
    mask = np.ones(_N_IMAGES, dtype=bool)
    cryo_mask = _load_tilt_cryo(tilt_files, ind=mask)
    cryo_none = _load_tilt_cryo(tilt_files, ind=None)
    assert cryo_mask.n_images == cryo_none.n_images
    assert cryo_mask.n_units == cryo_none.n_units


def test_load_tilt_boolean_mask_subset_aligns_ctf(tilt_files):
    """Boolean mask with some True values: CTF shape matches selected count."""
    mask = np.zeros(_N_IMAGES, dtype=bool)
    # Select the first tilt of every particle (index 0, 3, 6, …)
    mask[::_N_TILTS] = True
    selected_count = int(mask.sum())

    cryo = _load_tilt_cryo(tilt_files, ind=mask)
    assert cryo.n_images == selected_count
    assert cryo.CTF_params.shape[0] == selected_count
    assert cryo.rotation_matrices.shape[0] == selected_count


# ---------------------------------------------------------------------------
# 4. Duplicate image indices propagate through CTF / poses
# ---------------------------------------------------------------------------


def test_load_tilt_duplicate_ind_creates_repeated_rows(tilt_files):
    """
    Providing duplicate indices (e.g., [a, a, b]) should result in a dataset
    where CTF / pose arrays have 3 rows (with row a repeated).
    """
    p2t, _ = _parse_p2t(tilt_files)
    first_tilt = int(p2t[0][0])
    second_tilt = int(p2t[0][1] if len(p2t[0]) > 1 else p2t[1][0])
    ind_with_dup = np.array([first_tilt, first_tilt, second_tilt], dtype=np.int32)

    cryo = _load_tilt_cryo(tilt_files, ind=ind_with_dup)
    assert cryo.n_images == 3
    assert cryo.CTF_params.shape[0] == 3
    assert cryo.rotation_matrices.shape[0] == 3
    # Rows 0 and 1 come from the same original image → CTF must be identical
    np.testing.assert_array_equal(cryo.CTF_params[0], cryo.CTF_params[1])


# ---------------------------------------------------------------------------
# 5. get_split_tilt_indices – particle-level filter
# ---------------------------------------------------------------------------


def test_get_split_tilt_indices_particle_ind_only_no_overlap(tilt_files):
    """
    When using only tilt_ind_file (particle filter), returned halfsets must
    be non-overlapping and cover exactly the selected particles' images.
    """
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    n_particles = len(p2t)
    # Choose the first half of particles for the filter
    n_keep = max(n_particles // 2, 1)
    particle_ind = np.arange(n_keep, dtype=np.int32)

    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        tilt_ind_file=particle_ind,
    )
    half0 = np.asarray(half0, dtype=np.int32)
    half1 = np.asarray(half1, dtype=np.int32)

    # No overlap between halfsets
    assert np.intersect1d(half0, half1).size == 0
    # Union is a subset of the selected particles' images
    allowed = np.concatenate([p2t[i] for i in range(n_keep)]).astype(np.int32)
    combined = np.concatenate([half0, half1])
    assert set(combined.tolist()).issubset(set(allowed.tolist()))


def test_get_split_tilt_indices_particle_ind_all_images_accounted(tilt_files):
    """Every particle image must appear in exactly one halfset."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    all_particle_ids = np.arange(len(p2t), dtype=np.int32)

    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        tilt_ind_file=all_particle_ids,
    )
    half0 = np.asarray(half0, dtype=np.int32)
    half1 = np.asarray(half1, dtype=np.int32)
    combined = np.sort(np.concatenate([half0, half1]))
    all_images = np.sort(_all_image_indices(p2t))
    np.testing.assert_array_equal(combined, all_images)


# ---------------------------------------------------------------------------
# 6. get_split_tilt_indices – image-level filter (ind_file)
# ---------------------------------------------------------------------------


def test_get_split_tilt_indices_image_ind_only_no_overlap(tilt_files):
    """Image-level filter: halfsets are non-overlapping and within allowed set."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    # Keep only even-numbered elements from sorted image index list
    all_idx = np.sort(_all_image_indices(p2t))
    ind_file = all_idx[::2]

    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        ind_file=ind_file,
    )
    half0 = np.asarray(half0, dtype=np.int32)
    half1 = np.asarray(half1, dtype=np.int32)

    assert np.intersect1d(half0, half1).size == 0
    allowed = set(ind_file.tolist())
    assert set(half0.tolist()).issubset(allowed)
    assert set(half1.tolist()).issubset(allowed)


# ---------------------------------------------------------------------------
# 7. get_split_tilt_indices – combined ind_file + tilt_ind_file
# ---------------------------------------------------------------------------


def test_get_split_tilt_indices_combined_ind_is_intersection(tilt_files):
    """
    When both ind_file and tilt_ind_file are provided, output must be within
    the intersection: particle's images AND allowed images.
    """
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)

    # Particle filter: keep only first particle (index 0)
    particle_ind = np.array([0], dtype=np.int32)
    first_p_images = np.asarray(p2t[0], dtype=np.int32)

    # Image filter: keep only the very first tilt of the first particle
    ind_file = first_p_images[:1]

    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        ind_file=ind_file,
        tilt_ind_file=particle_ind,
    )
    combined = np.concatenate([np.asarray(half0), np.asarray(half1)]).astype(np.int32)
    # Must be within the image filter
    assert set(combined.tolist()).issubset(set(ind_file.tolist()))
    # Must be within the particle's images
    assert set(combined.tolist()).issubset(set(first_p_images.tolist()))


def test_get_split_tilt_indices_combined_out_of_range_particle_ids_ignored(tilt_files):
    """Out-of-range particle ids in tilt_ind_file are silently dropped."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    n_particles = len(p2t)

    # Mix valid and out-of-range particle ids
    particle_ind = np.array([0, n_particles + 999], dtype=np.int32)

    # Should not raise
    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        tilt_ind_file=particle_ind,
    )
    combined = np.concatenate([np.asarray(half0), np.asarray(half1)]).astype(np.int32)
    # Out-of-range particles contribute no images; only particle 0's images present
    valid_images = set(np.asarray(p2t[0], dtype=np.int32).tolist())
    assert set(combined.tolist()).issubset(valid_images)


# ---------------------------------------------------------------------------
# 8. get_split_tilt_indices – ntilts per-particle cap
# ---------------------------------------------------------------------------


def test_get_split_tilt_indices_ntilts_caps_per_particle(tilt_files):
    """ntilts=1 keeps at most one tilt per particle in each halfset."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)

    half0, half1 = halfsets.get_split_tilt_indices(
        particles_file=star,
        ntilts=1,
    )
    half0 = np.asarray(half0, dtype=np.int32)
    half1 = np.asarray(half1, dtype=np.int32)

    for pidx, tilts in enumerate(p2t):
        particle_tilts = set(np.asarray(tilts, dtype=np.int32).tolist())
        in_half0 = particle_tilts & set(half0.tolist())
        in_half1 = particle_tilts & set(half1.tolist())
        assert len(in_half0) <= 1, f"particle {pidx} has {len(in_half0)} tilts in half0"
        assert len(in_half1) <= 1, f"particle {pidx} has {len(in_half1)} tilts in half1"


# ---------------------------------------------------------------------------
# 9. get_split_tilt_indices – precomputed particle halfsets
# ---------------------------------------------------------------------------


def test_get_split_tilt_indices_precomputed_halfsets_are_respected(tilt_files):
    """Precomputed particle halfsets must assign particles to the correct side."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    n_particles = len(p2t)

    # Even-index particles → halfset 0; odd-index → halfset 1
    half0_particles = np.arange(0, n_particles, 2, dtype=np.int32)
    half1_particles = np.arange(1, n_particles, 2, dtype=np.int32)

    half0_images, half1_images = halfsets.get_split_tilt_indices(
        particles_file=star,
        particle_halfset_indices_file=[half0_particles, half1_particles],
    )
    half0_images = np.asarray(half0_images, dtype=np.int32)
    half1_images = np.asarray(half1_images, dtype=np.int32)

    # Verify no overlap
    assert np.intersect1d(half0_images, half1_images).size == 0

    # Images of even particles must only appear in half0
    for i, tilts in enumerate(p2t):
        particle_images = set(np.asarray(tilts, dtype=np.int32).tolist())
        if i % 2 == 0:
            assert particle_images.issubset(set(half0_images.tolist())), f"even particle {i} images leaked into half1"
        else:
            assert particle_images.issubset(set(half1_images.tolist())), f"odd particle {i} images leaked into half0"


def test_get_split_tilt_indices_precomputed_with_duplicates_deduped(tilt_files):
    """Duplicate particle ids in precomputed halfsets are deduplicated."""
    star = tilt_files["particles_star"]
    p2t, _ = _parse_p2t(tilt_files)
    n_particles = len(p2t)

    dup_half0 = np.array([0, 0, 1], dtype=np.int32)  # particle 0 listed twice
    half1_p = np.arange(2, n_particles, dtype=np.int32) if n_particles > 2 else np.array([], dtype=np.int32)

    # Should not raise; particle 0 images appear at most once in half0
    half0_images, _ = halfsets.get_split_tilt_indices(
        particles_file=star,
        particle_halfset_indices_file=[dup_half0, half1_p],
    )
    half0_images = np.asarray(half0_images, dtype=np.int32)

    # Particle 0 images must not be duplicated
    p0_images = np.asarray(p2t[0], dtype=np.int32)
    for img in p0_images:
        count = int(np.sum(half0_images == img))
        assert count <= 1, f"image {img} appears {count} times in half0"


# ---------------------------------------------------------------------------
# 10. reorder_to_original_indexing_from_halfsets
# ---------------------------------------------------------------------------


def test_reorder_roundtrip_two_halfsets(tilt_files):
    """
    Values written through two non-overlapping halfsets are placed at the
    correct original positions; gaps (absent images) are NaN.
    """
    p2t, _ = _parse_p2t(tilt_files)
    all_images = np.sort(_all_image_indices(p2t))
    n_total = int(all_images.max()) + 1

    half0_idx = all_images[::2]
    half1_idx = all_images[1::2]

    val0 = np.arange(half0_idx.size, dtype=np.float32)
    val1 = np.arange(half1_idx.size, dtype=np.float32) + 100.0
    arr = np.concatenate([val0, val1])
    halfsets = [half0_idx, half1_idx]

    out = recovar_dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images=n_total)
    assert out.shape == (n_total,)
    np.testing.assert_array_equal(out[half0_idx], val0)
    np.testing.assert_array_equal(out[half1_idx], val1)


def test_reorder_raises_on_duplicate_halfset_indices():
    """Duplicate dataset indices across halfsets must raise ValueError."""
    halfsets = [np.array([0, 1, 2], dtype=np.int32), np.array([2, 3, 4], dtype=np.int32)]  # index 2 in both
    arr = np.ones(6, dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate"):
        recovar_dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)


def test_reorder_absent_images_are_nan():
    """Images not in either halfset have NaN in the output."""
    halfsets = [
        np.array([0, 2], dtype=np.int32),  # skip index 1
        np.array([3], dtype=np.int32),
    ]
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out = recovar_dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images=4)
    assert out.shape == (4,)
    assert np.isnan(out[1])
    assert float(out[0]) == pytest.approx(10.0)
    assert float(out[2]) == pytest.approx(20.0)
    assert float(out[3]) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 11. Generator alignment after image-level subset
# ---------------------------------------------------------------------------


def test_tilt_generator_local_image_indices_in_range_after_image_ind(tilt_files):
    """
    After an image-level subset, the generator yields contiguous local image
    indices 0 … n_images-1 in b[2] (consistent with SPA generator contract).
    """
    p2t, _ = _parse_p2t(tilt_files)
    # Select all images of first two particles
    selected = np.concatenate([p2t[0], p2t[1]]).astype(np.int32)

    cryo = _load_tilt_cryo(tilt_files, ind=selected)
    batches = list(
        cryo.iter_batches(
            batch_size=cryo.n_images,
            by_image=True,
            prefetch=False,
        )
    )
    assert len(batches) >= 1
    # Local image indices (b[2]) must cover [0, n_images)
    all_local = np.concatenate([np.asarray(b[6]).ravel() for b in batches]).astype(np.int32)
    np.testing.assert_array_equal(np.sort(all_local), np.arange(cryo.n_images, dtype=np.int32))


# ---------------------------------------------------------------------------
# 12. Edge-case: ind selects only images from a single particle
# ---------------------------------------------------------------------------


def test_load_tilt_single_particle_subset(tilt_files):
    """Selecting all tilts of exactly one particle → n_units == 1."""
    p2t, _ = _parse_p2t(tilt_files)
    first_particle_images = np.asarray(p2t[0], dtype=np.int32)
    cryo = _load_tilt_cryo(tilt_files, ind=first_particle_images)
    assert cryo.n_units == 1
    assert cryo.n_images == first_particle_images.size


def test_load_tilt_single_tilt_per_particle_subset(tilt_files):
    """Selecting exactly one tilt per particle → n_units == n_particles."""
    p2t, _ = _parse_p2t(tilt_files)
    # First tilt (index 0 within each particle's tilt list) of each particle
    one_per_particle = np.array([int(tilts[0]) for tilts in p2t], dtype=np.int32)
    cryo = _load_tilt_cryo(tilt_files, ind=one_per_particle)
    assert cryo.n_units == _N_PARTICLES
    assert cryo.n_images == _N_PARTICLES  # one tilt each
