"""File-backed image backends used underneath :mod:`recovar.data_io.image_sources`.

This module owns the low-level Grain-backed loaders for:

- single-particle image stacks
- cryo-ET tilt-series grouped by particle

It does not own metadata, halfset policy, or the top-level dataset view.
Those live in ``image_metadata.py``, ``halfsets.py``, and
``cryoem_dataset.py`` respectively.
"""

import logging
import queue
import threading
import time
from collections import OrderedDict, Counter
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

import grain.python as grain

from recovar.data_io._index_utils import normalize_indices
from recovar.data_io.image_loader import ImageLoader
from recovar.data_io import starfile
from recovar.core import mask

from recovar.utils.nvtx_shim import nvtx

logger = logging.getLogger(__name__)

NVTX_DOMAIN_DATA_IO = "data_io"


class _SimpleSubset:
    """Lightweight subset wrapper (replaces torch.utils.data.Subset).

    Exposes the same ``dataset`` and ``indices`` attributes that
    ``_ImageCountBatchLoader`` walks when resolving particle tilts.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]

    def __len__(self):
        return len(self.indices)


class _ImageView:
    """Thin wrapper that exposes individual images from an ImageSource."""

    def __init__(self, source, num_images):
        self.source = source
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img = self.source.images(idx)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        return img, np.inf, idx


# ---------------------------------------------------------------------------
# ParticleImageDataset
# ---------------------------------------------------------------------------

class ParticleImageDataset:
    """Dataset for cryo-EM particle images.

    Implements ``__getitem__`` / ``__len__`` which is the protocol expected
    by both ``grain.RandomAccessDataSource`` and the downstream loaders.
    """

    def __init__(self, image_file: str, lazy: bool = True, ind: Optional[np.ndarray] = None,
                 invert_data: bool = False, datadir: str = "", padding: int = 0,
                 max_threads: int = 16, strip_prefix: Optional[str] = None,
                 downsample_D: Optional[int] = None, device=None, **kwargs):
        if padding != 0:
            raise NotImplementedError("Padding not yet supported")

        self.source = ImageLoader.from_file(
            image_file, lazy=lazy, datadir=datadir or "",
            indices=ind, max_threads=max_threads, strip_prefix=strip_prefix
        )

        if downsample_D is not None:
            from recovar.data_io.image_loader import DownsamplingImageLoader
            self.source = DownsamplingImageLoader(self.source, downsample_D)

        self.image_size = self.source.D
        self.num_images = self.source.n
        self.lazy = lazy
        self.invert_data = invert_data
        self.padding = padding
        self.device = device

        if self.image_size % 2 != 0:
            raise ValueError(f"Image size must be even, got {self.image_size}")

        self.dtype = np.complex64
        self.image_shape = (self.image_size, self.image_size)
        self.total_pixels = self.image_size * self.image_size
        self.image_mask = np.array(mask.window_mask(self.image_size, 0.85, 0.99))
        self.data_multiplier = -1 if invert_data else 1

        # Compatibility aliases
        self.N = self.num_images
        self.n_images = self.num_images
        self.D = self.image_size
        self.unpadded_D = self.image_size
        self.mask = self.image_mask
        self.mult = self.data_multiplier

    def __len__(self) -> int:
        return self.num_images

    def __repr__(self) -> str:
        return f"ParticleImageDataset(N={self.num_images}, D={self.image_size})"

    @nvtx.annotate("ParticleImageDataset.__getitem__", color="yellow",
                    domain=NVTX_DOMAIN_DATA_IO)
    def __getitem__(self, index):
        images = self.source.images(index)
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        return images, index, index

    def process_images(self, images: np.ndarray, apply_image_mask: bool = False) -> np.ndarray:
        if apply_image_mask:
            images = images * self.image_mask
        import recovar.core.padding as pad
        images = pad.padded_dft(images * self.data_multiplier, self.D, self.padding)
        return images.astype(self.dtype, copy=False)

    def process_images_half(self, images: np.ndarray, apply_image_mask: bool = False) -> np.ndarray:
        """Return half-spectrum images using the legacy full-FFT path.

        The old pipeline applied ``process_images`` first and then converted
        the full FFT layout to half-spectrum storage.  Direct ``rfft`` is
        mathematically close, but it is not numerically identical and that
        drift is enough to change downstream PCA / outlier regressions.
        """
        processed = self.process_images(images, apply_image_mask=apply_image_mask)
        import recovar.core.fourier_transform_utils as fourier_transform_utils
        half_images = fourier_transform_utils.full_image_to_half_image(
            processed,
            self.image_shape,
        )
        return half_images.astype(self.dtype, copy=False)

    def get_dataset_generator(self, batch_size: int, num_workers: int = 0,
                              pad_to_batch_size: bool = False, mode: str = 'tilt_series',
                              **kwargs):
        return _GrainBatchLoader(self, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    def get_dataset_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                     num_workers: int = 0, pad_to_batch_size: bool = False,
                                     mode: str = 'tilt_series', **kwargs):
        subset_indices = normalize_indices(
            subset_indices, n_total=int(self.num_images), name="subset_indices"
        )
        subset = _SimpleSubset(self, subset_indices)
        return _GrainBatchLoader(subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    def get_image_generator(self, batch_size: int, num_workers: int = 0):
        return self.get_dataset_generator(batch_size, num_workers)

    def get_image_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                   num_workers: int = 0):
        return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers)


# ---------------------------------------------------------------------------
# TiltSeriesDataset
# ---------------------------------------------------------------------------

class TiltSeriesDataset(ParticleImageDataset):
    """Dataset for tilt series with automatic particle grouping."""

    def __init__(self, starfile_path: str, lazy: bool = True,
                 num_tilts: Optional[int] = None, random_tilts: bool = False,
                 ind: Optional[np.ndarray] = None,
                 voltage: Optional[float] = None, dose_per_tilt: Optional[float] = None,
                 angle_per_tilt: Optional[float] = None, expected_res: Optional[float] = None,
                 tilt_file_option: str = 'relion5', **kwargs):
        start_time = time.time()

        if num_tilts is not None:
            if not isinstance(num_tilts, (int, np.integer)):
                raise TypeError("num_tilts must be an integer or None")
            num_tilts = int(num_tilts)
            if num_tilts < 0:
                logger.warning("num_tilts=%d < 0; using all available tilts per particle",
                               num_tilts)
                num_tilts = None

        super().__init__(starfile_path, lazy=lazy, ind=ind, **kwargs)
        logger.info("Base dataset loaded in %.2fs", time.time() - start_time)

        star = starfile.StarFile.load(starfile_path)
        logger.info("STAR file parsed in %.2fs", time.time() - start_time)

        canonical_groups = self._get_canonical_groups(star.df)

        if ind is not None:
            star.df = star.df.loc[ind]

        self.particle_groups = self._build_particle_groups(star.df, canonical_groups)
        self._particle_tilts = list(self.particle_groups.values())
        self.num_particles = len(self.particle_groups)
        self.dataset_tilt_indices = [
            canonical_groups.index(gn) for gn in self.particle_groups.keys()
        ]

        self.ctfscalefactor = np.asarray(star.df["_rlnCtfScalefactor"], dtype=np.float32)
        if '_rlnCtfBfactor' in star.df.columns:
            self.ctfBfactor = np.asarray(star.df["_rlnCtfBfactor"], dtype=np.float32)
        elif tilt_file_option == 'warp':
            raise ValueError(
                "Warp tilt ordering requires '_rlnCtfBfactor' column in the "
                "STAR file, but it was not found. Check that your STAR file "
                "was exported from Warp with B-factor information."
            )
        if tilt_file_option == 'relion5':
            self.dose = np.asarray(star.df["_rlnMicrographPreExposure"], dtype=np.float32)

        self._compute_tilt_ordering(tilt_file_option)

        self.num_tilts = num_tilts
        self.random_tilts = random_tilts
        self.voltage = voltage
        self.dose_per_tilt = dose_per_tilt
        self.tilt_angles = None

        group_counts = Counter(star.df["_rlnGroupName"])
        logger.info("Loaded %d tilts for %d particles", self.N, self.num_particles)
        logger.info("Tilts per particle: %s", set(group_counts.values()))
        logger.info("Dataset loaded in %.2fs", time.time() - start_time)

        # Compatibility aliases
        self.Np = self.num_particles
        self.particles = list(self.particle_groups.values())
        self.counts = group_counts
        self.tilt_numbers = self.tilt_order
        self.ntilts = num_tilts

    @staticmethod
    def _get_canonical_groups(df) -> List[str]:
        if '_rlnGroupName' not in df.columns:
            raise ValueError("STAR data is missing required column: _rlnGroupName")
        return sorted(df['_rlnGroupName'].unique())

    @staticmethod
    def _build_particle_groups(df, canonical_groups: List[str]) -> OrderedDict:
        groups: Dict[str, list] = {}
        for idx, group_name in enumerate(df["_rlnGroupName"]):
            groups.setdefault(group_name, []).append(idx)

        ordered = OrderedDict()
        for gn in canonical_groups:
            if gn in groups:
                ordered[gn] = np.array(groups[gn], dtype=int)
        return ordered

    def _compute_tilt_ordering(self, method: str):
        if method == 'relion5':
            logger.info("Ordering tilts by dose (_rlnMicrographPreExposure) - RELION 5")
        elif method == 'warp':
            logger.info("Ordering tilts by B-factor - Warp")
        else:
            raise ValueError(f"Invalid tilt ordering method: {method}")

        self.tilt_order = np.zeros(self.N, dtype=int)

        for particle_tilts in self.particle_groups.values():
            if method == 'relion5':
                sort_indices = np.argsort(-self.dose[particle_tilts])
            else:
                sort_indices = np.argsort(self.ctfBfactor[particle_tilts])

            ranks = np.empty_like(sort_indices)
            ranks[sort_indices[::-1]] = np.arange(len(particle_tilts))
            self.tilt_order[particle_tilts] = ranks

    def __len__(self) -> int:
        return self.num_particles

    def __getitem__(self, particle_index: int):
        particle_tilts = self._particle_tilts[particle_index]

        if self.random_tilts and self.num_tilts is not None:
            n_select = min(int(self.num_tilts), len(particle_tilts))
            if n_select <= 0:
                selected = particle_tilts[:0]
            else:
                selected = np.random.choice(particle_tilts, n_select, replace=False)
        else:
            tilt_orders = self.tilt_order[particle_tilts]
            sorted_idx = np.argsort(tilt_orders)
            n_select = self.num_tilts if self.num_tilts is not None else len(particle_tilts)
            selected = particle_tilts[sorted_idx[:n_select]]

        images = self.source.images(selected)
        return images, particle_index, selected

    @classmethod
    def parse_particle_tilt(cls, starfile_path: str,
                            indices: Optional[np.ndarray] = None):
        star = starfile.StarFile.load(starfile_path)
        df = star.df

        if '_rlnGroupName' not in df.columns:
            raise ValueError("STAR data is missing required column: _rlnGroupName")

        if indices is not None:
            norm_indices = normalize_indices(indices, n_total=len(df), name="indices")
            df = df.loc[norm_indices]

        canonical_groups = sorted(df['_rlnGroupName'].unique())
        grouped = df.groupby('_rlnGroupName').groups
        particles_to_tilts = [np.asarray(grouped[gn], dtype=int) for gn in canonical_groups]

        tilts_to_particles = {}
        for particle_idx, tilt_indices in enumerate(particles_to_tilts):
            for tilt_idx in tilt_indices:
                tilts_to_particles[int(tilt_idx)] = particle_idx

        return particles_to_tilts, tilts_to_particles

    @classmethod
    def parse_micrograph_tilt_mapping(cls, starfile_path: str):
        star = starfile.StarFile.load(starfile_path)
        df = star.df

        tilt_col = None
        for col in ['_rlnTiltName', 'rlnTiltName']:
            if col in df.columns:
                tilt_col = col
                break

        if tilt_col is None:
            raise ValueError(f"No tilt name column found. Available: {list(df.columns)}")

        grouped = df.groupby(tilt_col).groups
        tomogram_tilts = [np.asarray(indices, dtype=int) for indices in grouped.values()]

        reverse_map = {}
        for group_idx, tilt_indices in enumerate(tomogram_tilts):
            for tilt_idx in tilt_indices:
                reverse_map[int(tilt_idx)] = group_idx

        return tomogram_tilts, reverse_map

    def _max_tilts_per_particle(self) -> int:
        max_tilts = 0
        for tilts in self._particle_tilts:
            n_available = len(tilts)
            n_actual = (
                min(self.num_tilts, n_available) if self.num_tilts is not None
                else n_available
            )
            max_tilts = max(max_tilts, n_actual)
        return max_tilts

    def get_image_generator(self, batch_size: int, num_workers: int = 0):
        view = _ImageView(self.source, self.source.n)
        return _GrainBatchLoader(view, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    def get_image_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                   num_workers: int = 0):
        if subset_indices is None:
            return self.get_image_generator(batch_size, num_workers)
        subset_indices = normalize_indices(
            subset_indices, n_total=int(self.source.n), name="subset_indices"
        )
        view = _ImageView(self.source, self.source.n)
        subset = _SimpleSubset(view, subset_indices)
        return _GrainBatchLoader(subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    def get_dataset_generator(self, batch_size: int, num_workers: int = 0,
                              pad_to_batch_size: bool = False, mode: str = 'tilt_series',
                              **kwargs):
        if mode == 'images':
            max_tilts = _max_tilts_per_dataset_view(self)
            if batch_size < max_tilts:
                raise ValueError(
                    f"Batch size ({batch_size}) < max tilts per particle ({max_tilts}). "
                    f"Use larger batch_size or mode='tilt_series'"
                )
            return _ImageCountBatchLoader(self, batch_size, num_workers, pad_to_batch_size)
        elif mode == 'tilt_series':
            return _GrainBatchLoader(self, batch_size=1, shuffle=False, num_workers=num_workers)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_dataset_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                     num_workers: int = 0, pad_to_batch_size: bool = False,
                                     mode: str = 'tilt_series', **kwargs):
        if subset_indices is None:
            return self.get_dataset_generator(batch_size, num_workers, pad_to_batch_size, mode)
        subset_indices = normalize_indices(
            subset_indices, n_total=len(self), name="subset_indices"
        )

        if mode == 'images':
            subset = _SimpleSubset(self, subset_indices)
            max_tilts = _max_tilts_per_dataset_view(subset)
            if batch_size < max_tilts:
                raise ValueError(f"Batch size ({batch_size}) < max tilts ({max_tilts})")
            return _ImageCountBatchLoader(subset, batch_size, num_workers, pad_to_batch_size)
        else:
            subset = _SimpleSubset(self, subset_indices)
            return _GrainBatchLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

@nvtx.annotate("collate_to_jax", color="magenta", domain=NVTX_DOMAIN_DATA_IO)
def _collate_batch_to_jax(batch):
    """Convert a batch of dataset items to JAX arrays.

    Handles multiple input formats:
    - ``None`` → ``None``
    - A single numpy array → ``jnp.asarray``
    - A list of numpy arrays → concatenate along axis 0
    - A list of tuples/lists → transpose then collate each column
    """
    if batch is None:
        return None
    if isinstance(batch, np.ndarray):
        return jnp.asarray(batch)

    if isinstance(batch[0], np.ndarray):
        if len(batch) == 1:
            return jnp.asarray(batch[0])
        return jnp.asarray(np.concatenate(batch, axis=0))

    if isinstance(batch[0], (tuple, list)):
        return [_collate_batch_to_jax(list(samples)) for samples in zip(*batch)]

    return jnp.asarray(batch)


# ---------------------------------------------------------------------------
# PrefetchIterator
# ---------------------------------------------------------------------------

_SENTINEL = object()


class _PrefetchIterator:
    """Wraps any iterable with background-thread prefetching.

    While the consumer processes batch N, the background thread loads
    batch N+1 from disk, overlapping I/O with computation.
    """

    def __init__(self, iterable, buffer_size: int = 2):
        self._iterable = iterable
        self._buffer_size = max(buffer_size, 1)

    def __iter__(self):
        q = queue.Queue(maxsize=self._buffer_size)

        def _producer():
            try:
                for item in self._iterable:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Grain Batch Loader
# ---------------------------------------------------------------------------

class _GrainBatchLoader:
    """Iterable that batches a dataset and yields JAX arrays with prefetching.

    Uses Grain's ``MapDataset`` for efficient batched iteration with
    configurable background-thread prefetching.
    """

    # Default prefetch threads used when callers don't specify num_workers.
    # 4 threads lets Grain read ahead while the GPU computes, overlapping
    # disk I/O with computation for lazy-loaded datasets.
    DEFAULT_NUM_THREADS = 4

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False,
                 num_workers: int = 0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._num_threads = max(num_workers, 1) if num_workers > 0 else self.DEFAULT_NUM_THREADS
        self._prefetch_buffer = min(max(2, self._num_threads), 8)

    def _iter_batches(self):
        """Yield collated batches from Grain-backed iteration."""
        ds = grain.MapDataset.source(self.dataset)
        if self.shuffle:
            ds = ds.shuffle(seed=42)
        # Don't use Grain's .batch() — it uses np.stack which adds an extra
        # dimension.  Our __getitem__ returns images as (1, D, D) so stacking
        # would give (batch, 1, D, D) instead of (batch, D, D).
        # Instead, iterate over individual items with Grain's prefetching
        # and batch manually using _collate_batch_to_jax (which uses np.concatenate).
        it = ds.to_iter_dataset(
            grain.ReadOptions(
                num_threads=self._num_threads,
                prefetch_buffer_size=self._prefetch_buffer,
            )
        )
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield _collate_batch_to_jax(batch)
                batch = []
        if batch:
            yield _collate_batch_to_jax(batch)

    def __iter__(self):
        # Wrap with PrefetchIterator so the next collated batch is prepared
        # in a background thread while the GPU processes the current one.
        return iter(_PrefetchIterator(self._iter_batches(), buffer_size=2))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Image Count Batch Loader
# ---------------------------------------------------------------------------
def _resolve_particle_tilts(dataset, effective_num_tilts):
    """Walk dataset/subset chain to find ``_particle_tilts`` on the root."""
    cursor = dataset
    mapped_indices = np.arange(len(dataset), dtype=np.int32)

    while cursor is not None:
        if effective_num_tilts is None:
            effective_num_tilts = getattr(cursor, "num_tilts", None)

        if hasattr(cursor, "_particle_tilts"):
            return (
                [cursor._particle_tilts[i] for i in mapped_indices],
                effective_num_tilts,
            )

        # Walk up subset chain (works for both _SimpleSubset and torch Subset)
        subset_indices = getattr(cursor, "indices", None)
        parent = getattr(cursor, "dataset", None)
        if subset_indices is None or parent is None:
            break

        parent_idx = normalize_indices(
            np.asarray(subset_indices), n_total=len(parent), name="subset indices"
        )
        mapped_indices = parent_idx[mapped_indices]
        cursor = parent

    return None, effective_num_tilts

def _max_tilts_per_dataset_view(dataset) -> int:
    """Return the largest per-particle tilt count visible through a dataset/subset view."""
    effective_num_tilts = getattr(dataset, "num_tilts", None)
    particle_tilts_list, effective_num_tilts = _resolve_particle_tilts(
        dataset,
        effective_num_tilts,
    )

    if particle_tilts_list is None:
        particle_groups = getattr(dataset, "particle_groups", None)
        if particle_groups is None:
            raise AttributeError(
                "dataset must expose _particle_tilts, or provide particle_groups "
                "(or subset indices with parent _particle_tilts)"
            )
        particle_tilts_list = list(particle_groups.values())[:len(dataset)]

    max_tilts = 0
    for particle_tilts in particle_tilts_list:
        n_available = len(particle_tilts)
        n_actual = (
            min(effective_num_tilts, n_available)
            if effective_num_tilts is not None
            else n_available
        )
        max_tilts = max(max_tilts, n_actual)
    return max_tilts


class _ImageCountBatchLoader:
    """DataLoader that batches by total image count across tilt series.

    Wraps iteration with ``PrefetchIterator`` for background I/O.
    """

    def __init__(self, dataset, batch_size: int, num_workers: int = 0,
                 pad_to_batch: bool = False):
        if not isinstance(batch_size, (int, np.integer)):
            raise TypeError(
                f"batch_size must be an integer, got {type(batch_size).__name__}"
            )
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_to_batch = pad_to_batch

        effective_num_tilts = getattr(dataset, "num_tilts", None)
        particle_tilts_list, effective_num_tilts = _resolve_particle_tilts(
            dataset, effective_num_tilts
        )

        if particle_tilts_list is None:
            particle_groups = getattr(dataset, "particle_groups", None)
            if particle_groups is None:
                raise AttributeError(
                    "_ImageCountBatchLoader dataset must expose _particle_tilts, "
                    "or provide particle_groups (or subset indices with parent "
                    "_particle_tilts)."
                )
            particle_tilts_list = list(particle_groups.values())[:len(dataset)]

        self.tilts_counts = []
        for particle_tilts in particle_tilts_list:
            n_available = len(particle_tilts)
            n_actual = (
                min(effective_num_tilts, n_available)
                if effective_num_tilts is not None
                else n_available
            )
            self.tilts_counts.append(n_actual)

        self.tilts_counts = np.asarray(self.tilts_counts, dtype=np.int32)
        self.total_images = int(np.sum(self.tilts_counts, dtype=np.int64))
        self._n_batches = self._compute_n_batches(self.tilts_counts, self.batch_size)

    @staticmethod
    def _compute_n_batches(tilts_counts: np.ndarray, batch_size: int) -> int:
        n_batches = 0
        particle_idx = 0
        n_particles = len(tilts_counts)

        while particle_idx < n_particles:
            current_count = 0
            while particle_idx < n_particles and current_count < batch_size:
                n_images = int(tilts_counts[particle_idx])
                if current_count + n_images > batch_size and current_count > 0:
                    break
                current_count += n_images
                particle_idx += 1

            if current_count == 0 and particle_idx < n_particles:
                current_count = int(tilts_counts[particle_idx])
                particle_idx += 1

            if current_count > 0:
                n_batches += 1

        return n_batches

    def _generate_batches(self):
        """Core batch generation logic."""
        particle_idx = 0

        while particle_idx < len(self.dataset):
            batch_images = []
            batch_particle_ids = []
            batch_tilt_ids = []
            current_count = 0

            while particle_idx < len(self.dataset) and current_count < self.batch_size:
                images, p_idx, t_indices = self.dataset[particle_idx]
                n_images = images.shape[0]

                if n_images <= 0:
                    particle_idx += 1
                    continue

                if current_count + n_images > self.batch_size and current_count > 0:
                    break

                batch_images.append(images)
                batch_particle_ids.append(np.full(n_images, p_idx, dtype=np.int32))
                batch_tilt_ids.append(t_indices)
                current_count += n_images
                particle_idx += 1

            if len(batch_images) == 0 and particle_idx < len(self.dataset):
                images, p_idx, t_indices = self.dataset[particle_idx]
                if images.shape[0] > 0:
                    batch_images.append(images)
                    batch_particle_ids.append(
                        np.full(images.shape[0], p_idx, dtype=np.int32)
                    )
                    batch_tilt_ids.append(t_indices)
                    current_count = images.shape[0]
                particle_idx += 1

            if len(batch_images) > 0:
                batch_images = np.concatenate(batch_images, axis=0)
                batch_particle_ids = np.concatenate(batch_particle_ids, axis=0)
                batch_tilt_ids = np.concatenate(batch_tilt_ids, axis=0)

                if self.pad_to_batch and current_count < self.batch_size:
                    pad_size = self.batch_size - current_count
                    img_shape = batch_images.shape[1:]
                    batch_images = np.concatenate([
                        batch_images,
                        np.zeros((pad_size,) + img_shape, dtype=batch_images.dtype),
                    ])
                    batch_particle_ids = np.concatenate([
                        batch_particle_ids,
                        np.full(pad_size, -1, dtype=np.int32),
                    ])
                    batch_tilt_ids = np.concatenate([
                        batch_tilt_ids,
                        np.full(pad_size, -1, dtype=batch_tilt_ids.dtype),
                    ])

                yield batch_images, batch_particle_ids, batch_tilt_ids

    def __iter__(self):
        return iter(_PrefetchIterator(self._generate_batches(), buffer_size=2))

    def __len__(self) -> int:
        return self._n_batches
