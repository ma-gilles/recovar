"""
Utilities for loading cryo-EM particle images from various file formats.

Supported formats:
- MRC/MRCS: Single or multi-image MRC stacks
- STAR: RELION star files referencing MRC stacks
- CS: cryoSPARC particle files
- TXT: Text file listing MRC paths

All loaders share the ImageLoader base class which provides a uniform
interface for indexing, batching, and caching.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor
import logging

try:
    import nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    import functools as _functools

    class _NvtxStub:
        """No-op stub used when nvtx is not installed."""

        @staticmethod
        def annotate(msg="", color=None, domain=None):
            class _NoOp:
                def __call__(self, fn):
                    @_functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        return fn(*args, **kwargs)
                    return wrapper

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return _NoOp()

    nvtx = _NvtxStub()
    _NVTX_AVAILABLE = False

logger = logging.getLogger(__name__)

NVTX_DOMAIN_DATA_IO = "data_io"


def _swap_mrc_ext(filepath: str) -> Optional[str]:
    """Return filepath with .mrc/.mrcs swapped, or None if not applicable."""
    if filepath.endswith('.mrc'):
        return filepath + 's'
    elif filepath.endswith('.mrcs'):
        return filepath[:-1]
    return None


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _normalize_selection_indices(indices, n_total: int, name: str) -> np.ndarray:
    """Normalize optional subset indices used at loader construction time."""
    if indices is None:
        return np.arange(int(n_total), dtype=np.int32)
    arr = np.asarray(indices)
    if arr.dtype == bool:
        if arr.ndim != 1:
            raise ValueError(f"{name} boolean mask must be 1D")
        if arr.size != int(n_total):
            raise ValueError(
                f"{name} boolean mask length {arr.size} must match available length {int(n_total)}"
            )
        return np.flatnonzero(arr).astype(np.int32, copy=False)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} indices must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} indices must be integer or boolean mask")
    arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0) or np.any(arr >= int(n_total)):
        raise IndexError(f"{name} indices out of range [0, {int(n_total)})")
    return arr.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_images(filepath: str, indices: Optional[np.ndarray] = None,
                datadir: str = "", lazy: bool = True, max_threads: int = 1,
                strip_prefix: Optional[str] = None):
    """Load cryo-EM images from file.

    Args:
        filepath: Path to data file (.mrcs, .star, .txt, .cs)
        indices: Optional subset of image indices to load
        datadir: Base directory for resolving relative paths
        lazy: If True, defer loading until access
        max_threads: Number of threads for parallel I/O
        strip_prefix: Prefix to strip from paths in metadata

    Returns:
        ImageLoader instance for the specified file
    """
    ext = filepath.rsplit('.', 1)[-1].lower()

    loaders = {
        'mrcs': lambda: MRCLoader(filepath, indices, lazy),
        'mrc': lambda: MRCLoader(filepath, indices, lazy),
        'star': lambda: StarLoader(filepath, indices, datadir, lazy, max_threads, strip_prefix),
        'txt': lambda: MultiMRCLoader.from_txt(filepath, indices, lazy, max_threads),
        'cs': lambda: CryoSparcLoader(filepath, indices, datadir, lazy, max_threads, strip_prefix),
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: .{ext}")

    return loaders[ext]()


# ---------------------------------------------------------------------------
# Base loader
# ---------------------------------------------------------------------------

class ImageLoader:
    """Base class for loading particle images.

    Provides a uniform interface for indexing (int, slice, array, bool mask),
    lazy/eager loading, caching, and batched iteration.
    """

    @staticmethod
    def from_file(filepath: str, lazy: bool = True, indices: Optional[np.ndarray] = None,
                  datadir: str = "", max_threads: int = 1, strip_prefix: Optional[str] = None):
        """Compatibility alias for load_images()."""
        return load_images(filepath, indices=indices, datadir=datadir, lazy=lazy,
                           max_threads=max_threads, strip_prefix=strip_prefix)

    def __init__(self, num_images: int, image_size: int, dtype=np.float32):
        self._num_images = num_images
        self._image_size = image_size
        self._dtype = dtype
        self._cached = None

    # -- Properties ----------------------------------------------------------

    @property
    def num_images(self) -> int:
        return self._num_images

    @property
    def n(self) -> int:
        """Compatibility alias for num_images."""
        return self._num_images

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def D(self) -> int:
        """Compatibility alias for image_size."""
        return self._image_size

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._num_images, self._image_size, self._image_size)

    def __len__(self) -> int:
        return self._num_images

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self._num_images}, D={self._image_size})"

    # -- Access --------------------------------------------------------------

    def __getitem__(self, key) -> np.ndarray:
        """Get images using indexing syntax."""
        return self.get(key)

    def get(self, indices=None) -> np.ndarray:
        """Get images at specified indices.

        Args:
            indices: Indices to retrieve (int, slice, array, or None for all)

        Returns:
            Array of shape (N, image_size, image_size)
        """
        idx_array = self._parse_indices(indices)

        if self._cached is not None:
            result = self._cached[idx_array]
        else:
            result = self._load(idx_array)

        return result.astype(self._dtype, copy=False)

    def images(self, indices=None, require_contiguous: bool = False) -> np.ndarray:
        """Compatibility alias for get()."""
        return self.get(indices)

    # -- Index parsing -------------------------------------------------------

    def _parse_indices(self, indices) -> np.ndarray:
        """Convert various index formats to int32 array."""
        if indices is None:
            return np.arange(self._num_images, dtype=np.int32)

        if isinstance(indices, (int, np.integer)):
            idx = int(indices)
            if idx < 0 or idx >= self._num_images:
                raise IndexError(f"Index {idx} out of range [0, {self._num_images})")
            return np.array([idx], dtype=np.int32)

        if isinstance(indices, slice):
            return np.arange(*indices.indices(self._num_images), dtype=np.int32)

        if isinstance(indices, (list, tuple)):
            indices = np.array(indices)

        if isinstance(indices, np.ndarray):
            if indices.dtype == bool:
                if indices.ndim != 1:
                    raise ValueError("Boolean indices must be a 1D mask")
                if indices.size != self._num_images:
                    raise ValueError(
                        f"Boolean index mask length {indices.size} "
                        f"must match number of images {self._num_images}"
                    )
                return np.flatnonzero(indices).astype(np.int32, copy=False)
            if indices.ndim == 0:
                idx = int(indices)
                if idx < 0 or idx >= self._num_images:
                    raise IndexError(f"Index {idx} out of range [0, {self._num_images})")
                return np.array([idx], dtype=np.int32)
            if indices.ndim != 1:
                raise ValueError("Indices array must be 1D")
            if indices.dtype.kind not in ("i", "u"):
                raise TypeError(f"Indices array must be integer or bool dtype, got {indices.dtype}")
            if np.any(indices < 0) or np.any(indices >= self._num_images):
                raise IndexError("Index out of range")
            return indices.astype(np.int32, copy=False)

        raise TypeError(f"Cannot index with type {type(indices)}")

    # -- I/O -----------------------------------------------------------------

    def _load(self, indices: np.ndarray) -> np.ndarray:
        """Load images at indices. Must be implemented by subclasses."""
        raise NotImplementedError

    def iter_batches(self, batch_size: int = 1000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over images in batches.

        Yields:
            (indices, images) tuples
        """
        for start in range(0, self._num_images, batch_size):
            end = min(start + batch_size, self._num_images)
            idx = np.arange(start, end)
            yield idx, self.get(idx)

    def chunks(self, chunksize: int = 1000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Compatibility alias for iter_batches()."""
        return self.iter_batches(chunksize)

    def load_all(self):
        """Load and cache all images in memory."""
        if self._cached is None:
            self._cached = self._load(np.arange(self._num_images))


# ---------------------------------------------------------------------------
# MRC loader with memory-mapped sequential reads
# ---------------------------------------------------------------------------

class MRCLoader(ImageLoader):
    """Load images from a single MRC/MRCS file.

    Uses contiguous seek+fromfile for sequential reads and individual
    seek+fromfile for scattered random access.  A lazy ``np.memmap`` view
    is available via ``_get_memmap()`` for bulk access patterns.
    """

    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None, lazy: bool = True):
        import mrcfile

        self._filepath = filepath
        self._memmap = None  # lazy-created on first sequential read

        with mrcfile.open(filepath, mode='r', permissive=True) as mrc:
            nz, ny, nx = mrc.data.shape
            self._file_dtype = mrc.data.dtype
            extended_header_size = (
                mrc.extended_header.nbytes if mrc.extended_header is not None else 0
            )

        if ny != nx:
            raise ValueError(f"Non-square images not supported: {ny} x {nx}")

        self._total_file_images = nz
        self._data_start = 1024 + extended_header_size
        self._pixels_per_image = ny * nx
        self._bytes_per_image = self._file_dtype.itemsize * self._pixels_per_image

        self._file_indices = _normalize_selection_indices(indices, nz, "MRCLoader indices")

        super().__init__(len(self._file_indices), ny, self._file_dtype)

        if not lazy:
            self.load_all()

    def __repr__(self) -> str:
        return (
            f"MRCLoader(filepath={self._filepath!r}, "
            f"n={self._num_images}, D={self._image_size})"
        )

    # -- Memory-mapped access ------------------------------------------------

    def _get_memmap(self) -> np.memmap:
        """Lazily create a read-only memory-mapped view of the MRC data."""
        if self._memmap is None:
            self._memmap = np.memmap(
                self._filepath,
                dtype=self._file_dtype,
                mode='r',
                offset=self._data_start,
                shape=(self._total_file_images, self._image_size, self._image_size),
            )
        return self._memmap

    def close(self):
        """Release memory-mapped file resources."""
        if self._memmap is not None:
            del self._memmap
            self._memmap = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- Loading -------------------------------------------------------------

    @nvtx.annotate("MRCLoader._load", color="blue", domain=NVTX_DOMAIN_DATA_IO)
    def _load(self, indices: np.ndarray) -> np.ndarray:
        """Load images from MRC file."""
        file_idx = self._file_indices[indices]
        if len(file_idx) == 0:
            return np.empty((0, self._image_size, self._image_size), dtype=self._file_dtype)

        # De-duplicate to avoid redundant disk reads.
        unique_idx, inverse = np.unique(file_idx, return_inverse=True)
        has_duplicates = unique_idx.size != file_idx.size
        read_idx = unique_idx if has_duplicates else file_idx

        read_output = np.empty(
            (len(read_idx), self._image_size, self._image_size), dtype=self._file_dtype
        )

        with nvtx.annotate(f"disk_read_{len(file_idx)}_images", color="cyan",
                           domain=NVTX_DOMAIN_DATA_IO):
            sorted_order = np.argsort(read_idx)
            sorted_idx = read_idx[sorted_order]
            is_sequential = (
                np.all(np.diff(sorted_idx) == 1) if len(sorted_idx) > 1 else True
            )

            if is_sequential:
                with nvtx.annotate("sequential_read", color="green",
                                   domain=NVTX_DOMAIN_DATA_IO):
                    offset = (
                        self._data_start
                        + int(sorted_idx[0]) * self._bytes_per_image
                    )
                    with open(self._filepath, 'rb') as f:
                        f.seek(offset)
                        data = np.fromfile(
                            f, dtype=self._file_dtype,
                            count=self._pixels_per_image * len(sorted_idx),
                        )
                    data = data.reshape(
                        len(sorted_idx), self._image_size, self._image_size
                    )
                    read_output[sorted_order] = data
            else:
                with nvtx.annotate("random_access_read", color="red",
                                   domain=NVTX_DOMAIN_DATA_IO):
                    with open(self._filepath, 'rb') as f:
                        for i, idx in enumerate(read_idx):
                            offset = self._data_start + int(idx) * self._bytes_per_image
                            f.seek(offset)
                            data = np.fromfile(
                                f, dtype=self._file_dtype,
                                count=self._pixels_per_image,
                            )
                            read_output[i] = data.reshape(
                                self._image_size, self._image_size
                            )

        if has_duplicates:
            return read_output[inverse]
        return read_output


# ---------------------------------------------------------------------------
# Multi-file loader
# ---------------------------------------------------------------------------

class MultiMRCLoader(ImageLoader):
    """Load images distributed across multiple MRC files."""

    def __init__(self, file_map: pd.DataFrame, indices: Optional[np.ndarray] = None,
                 lazy: bool = True, max_threads: int = 1,
                 raw_paths: Optional[list] = None):
        self._file_map = file_map.copy()
        self._max_threads = max_threads
        self._raw_paths = raw_paths  # original paths from metadata (for error hints)

        if indices is not None:
            iloc_idx = _normalize_selection_indices(
                indices, len(self._file_map), "MultiMRCLoader indices"
            )
            self._file_map = self._file_map.iloc[iloc_idx].reset_index(drop=True)

        if len(self._file_map) == 0:
            raise ValueError("No images selected for MultiMRCLoader")

        mrc_index_values = np.asarray(self._file_map["mrc_index"])
        if mrc_index_values.dtype.kind not in ("i", "u"):
            raise ValueError("mrc_index values must be integers")
        self._file_map["mrc_index"] = mrc_index_values.astype(np.int64, copy=False)

        self._loaders: dict[str, MRCLoader] = {}
        missing_files = []
        ext_swaps: dict[str, str] = {}  # original -> swapped path
        for filepath in self._file_map['mrc_file'].unique():
            try:
                self._loaders[filepath] = MRCLoader(filepath, lazy=True)
            except FileNotFoundError:
                # Try .mrc <-> .mrcs swap (common in cryo-EM workflows)
                swapped = _swap_mrc_ext(filepath)
                if swapped and os.path.isfile(swapped):
                    self._loaders[filepath] = MRCLoader(swapped, lazy=True)
                    ext_swaps[filepath] = swapped
                    logger.info("File not found: %s, using %s instead", filepath, swapped)
                else:
                    missing_files.append(filepath)

        # Update file_map to use the swapped paths so _load() uses correct keys
        if ext_swaps:
            self._file_map['mrc_file'] = self._file_map['mrc_file'].replace(ext_swaps)
            # Re-key loaders under the new paths
            for old, new in ext_swaps.items():
                self._loaders[new] = self._loaders.pop(old)

        if missing_files:
            n_missing = len(missing_files)
            sample = missing_files[0]
            basename = os.path.basename(sample)
            hint = (
                f"\n\nTo fix broken paths, try:\n"
                f"  --datadir /path/to/directory/containing/{basename}\n"
            )
            # Use raw metadata path for strip-prefix hint (before datadir was joined)
            if self._raw_paths:
                raw_sample = self._raw_paths[0]
                if '/' in raw_sample:
                    raw_prefix = raw_sample.rsplit('/', 1)[0]
                    hint += f"  --strip-prefix {raw_prefix}\n"
            elif '/' in sample:
                prefix = sample.rsplit('/', 1)[0]
                hint += f"  --strip-prefix {prefix}\n"
            raise FileNotFoundError(
                f"Cannot find {n_missing} MRC file(s) referenced in the metadata.\n"
                f"  First missing: {sample}\n"
                f"  File basename: {basename}"
                f"{hint}"
            )

        # Get image size and dtype from the first successfully loaded file.
        first_loader = next(iter(self._loaders.values()))
        img_size = first_loader.image_size
        dtype = first_loader._file_dtype

        # Validate per-file indices eagerly.
        if (self._file_map["mrc_index"] < 0).any():
            raise ValueError("mrc_index values must be non-negative")
        for filepath, group in self._file_map.groupby("mrc_file"):
            max_valid = int(self._loaders[filepath].num_images) - 1
            requested = group["mrc_index"].to_numpy()
            bad = (requested < 0) | (requested > max_valid)
            if np.any(bad):
                bad_values = np.unique(requested[bad]).tolist()
                raise ValueError(
                    f"mrc_index values {bad_values} out of range for file {filepath}; "
                    f"valid range is [0, {max_valid}]"
                )

        super().__init__(len(self._file_map), img_size, dtype)

        if not lazy:
            self.load_all()

    def __repr__(self) -> str:
        n_files = len(self._loaders)
        return f"MultiMRCLoader(n={self._num_images}, D={self._image_size}, files={n_files})"

    def close(self):
        """Release resources for all sub-loaders."""
        for loader in self._loaders.values():
            loader.close()

    def _load(self, indices: np.ndarray) -> np.ndarray:
        n_out = int(len(indices))
        output = np.empty((n_out, self._image_size, self._image_size), dtype=self._dtype)
        if n_out == 0:
            return output

        subset = self._file_map.iloc[indices]
        out_pos = np.arange(n_out, dtype=np.int32)
        file_paths = subset["mrc_file"].to_numpy()
        mrc_indices = subset["mrc_index"].to_numpy(dtype=np.int64, copy=False)

        # Group by file using numpy (avoids DataFrame groupby overhead).
        unique_paths, path_group = np.unique(file_paths, return_inverse=True)
        group_order = np.argsort(path_group, kind="stable")
        split_points = np.flatnonzero(np.diff(path_group[group_order])) + 1
        grouped_positions = np.split(group_order, split_points)

        if self._max_threads > 1 and unique_paths.size > 1:
            with ThreadPoolExecutor(max_workers=self._max_threads) as executor:
                futures = {}
                for pos in grouped_positions:
                    group_id = int(path_group[int(pos[0])])
                    filepath = unique_paths[group_id]
                    future = executor.submit(
                        self._loaders[filepath]._load, mrc_indices[pos]
                    )
                    futures[future] = out_pos[pos]

                for future, group_out_pos in futures.items():
                    output[group_out_pos] = future.result()
        else:
            for pos in grouped_positions:
                group_id = int(path_group[int(pos[0])])
                filepath = unique_paths[group_id]
                images = self._loaders[filepath]._load(mrc_indices[pos])
                output[out_pos[pos]] = images

        return output

    @staticmethod
    def from_txt(filepath: str, indices: Optional[np.ndarray] = None,
                 lazy: bool = True, max_threads: int = 1) -> 'MultiMRCLoader':
        """Create loader from text file listing MRC paths."""
        base = os.path.dirname(filepath)

        mrc_files = []
        mrc_indices = []

        with open(filepath) as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                if not os.path.isabs(path):
                    path = os.path.join(base, path)
                loader = MRCLoader(path, lazy=True)
                n = loader.num_images
                mrc_files.extend([path] * n)
                mrc_indices.extend(range(n))

        df = pd.DataFrame({'mrc_file': mrc_files, 'mrc_index': mrc_indices})
        return MultiMRCLoader(df, indices, lazy, max_threads)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

class StarLoader(MultiMRCLoader):
    """Load images from RELION STAR file."""

    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None,
                 datadir: str = "", lazy: bool = True, max_threads: int = 1,
                 strip_prefix: Optional[str] = None):
        from recovar.data_io.starfile import StarFile

        star = StarFile.load(filepath)
        df = star.df.copy()

        if '_rlnImageName' not in df.columns:
            raise ValueError("STAR file is missing required column: _rlnImageName")

        image_names = df["_rlnImageName"].astype(str)
        parts = image_names.str.split('@', n=1, expand=True)
        if parts.shape[1] < 2 or parts[1].isna().any():
            raise ValueError(
                "Malformed _rlnImageName entries: expected '<index>@<path>' format"
            )
        try:
            df['mrc_index'] = parts[0].astype(int) - 1
        except Exception as exc:
            raise ValueError(
                "Malformed _rlnImageName entries: index part is not an integer"
            ) from exc
        if (df['mrc_index'] < 0).any():
            raise ValueError(
                "Malformed _rlnImageName entries: image indices must be >= 1"
            )
        df['mrc_file'] = parts[1]

        if strip_prefix:
            if not df['mrc_file'].str.startswith(strip_prefix).any():
                raise ValueError(f"No paths match strip_prefix: {strip_prefix}")
            df['mrc_file'] = df['mrc_file'].str.removeprefix(strip_prefix).str.lstrip('/')

        if not datadir:
            datadir = os.path.abspath(os.path.dirname(filepath))
        else:
            datadir = os.path.abspath(datadir)

        # Save raw paths (after strip but before datadir join) for error hints
        raw_paths = df['mrc_file'].unique().tolist()

        df['mrc_file'] = df['mrc_file'].apply(lambda p: os.path.join(datadir, p))

        super().__init__(df[['mrc_file', 'mrc_index']], indices, lazy, max_threads,
                         raw_paths=raw_paths)


class CryoSparcLoader(MultiMRCLoader):
    """Load images from cryoSPARC CS file."""

    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None,
                 datadir: str = "", lazy: bool = True, max_threads: int = 1,
                 strip_prefix: Optional[str] = None):
        cs_data = np.load(filepath)

        if 'blob/idx' not in cs_data.dtype.names or 'blob/path' not in cs_data.dtype.names:
            raise ValueError("CS file must contain fields: 'blob/idx' and 'blob/path'")

        blob_idx = cs_data['blob/idx']
        blob_paths = cs_data['blob/path']
        if np.any(blob_idx < 0):
            raise ValueError("CS file contains negative blob/idx values")

        clean_paths = []
        for p in blob_paths:
            if isinstance(p, (bytes, np.bytes_)):
                p = p.decode("utf-8", errors="replace")
            else:
                p = str(p)
            clean_paths.append(p.lstrip('>'))

        # Strip prefix if provided (same as StarLoader)
        if strip_prefix:
            stripped = []
            for p in clean_paths:
                if p.startswith(strip_prefix):
                    stripped.append(p[len(strip_prefix):].lstrip('/'))
                else:
                    stripped.append(p)
            clean_paths = stripped

        if not datadir:
            datadir = os.path.dirname(filepath)
        elif not os.path.isabs(datadir):
            datadir = os.path.join(os.path.dirname(filepath), datadir)

        # Save raw paths (after strip but before datadir join) for error hints
        raw_paths = list(dict.fromkeys(clean_paths))  # unique, order-preserving

        full_paths = [os.path.join(datadir, p) for p in clean_paths]

        df = pd.DataFrame({'mrc_file': full_paths, 'mrc_index': blob_idx})

        super().__init__(df, indices, lazy, max_threads, raw_paths=raw_paths)


# ---------------------------------------------------------------------------
# Downsampling wrapper
# ---------------------------------------------------------------------------

class DownsamplingImageLoader(ImageLoader):
    """Wrapper that Fourier-crops images on the fly during loading."""

    def __init__(self, base_loader: ImageLoader, target_D: int):
        if target_D > base_loader.D:
            raise ValueError(
                f"target_D ({target_D}) must be <= source image size ({base_loader.D})"
            )
        if target_D % 2 != 0:
            raise ValueError(f"target_D must be even, got {target_D}")

        self._base = base_loader
        self._target_D = target_D
        super().__init__(base_loader.num_images, target_D, base_loader._dtype)

    def _load(self, indices: np.ndarray) -> np.ndarray:
        images = self._base._load(indices)
        if self._target_D == self._base.D:
            return images
        from recovar.data_io.downsample import downsample_images
        return downsample_images(images, self._target_D)


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

ImageSource = ImageLoader
MRCFileSource = MRCLoader
StarfileSource = StarLoader
TxtFileSource = MultiMRCLoader
CsSource = CryoSparcLoader
