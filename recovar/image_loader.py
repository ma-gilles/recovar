"""
Utilities for loading cryo-EM particle images from various file formats.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor
import logging
import nvtx

logger = logging.getLogger(__name__)

# NVTX domain for data I/O profiling
NVTX_DOMAIN_DATA_IO = "data_io"


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
    ext = filepath.split('.')[-1].lower()
    
    loaders = {
        'mrcs': lambda: MRCLoader(filepath, indices, lazy),
        'mrc': lambda: MRCLoader(filepath, indices, lazy),
        'star': lambda: StarLoader(filepath, indices, datadir, lazy, max_threads, strip_prefix),
        'txt': lambda: MultiMRCLoader.from_txt(filepath, indices, lazy, max_threads),
        'cs': lambda: CryoSparcLoader(filepath, indices, datadir, lazy, max_threads),
    }
    
    if ext not in loaders:
        raise ValueError(f"Unsupported format: .{ext}")
    
    return loaders[ext]()


class ImageLoader:
    """Base class for loading particle images."""
    
    @staticmethod
    def from_file(filepath: str, lazy: bool = True, indices: Optional[np.ndarray] = None,
                  datadir: str = "", max_threads: int = 1, strip_prefix: Optional[str] = None):
        """Compatibility alias for load_images()."""
        return load_images(filepath, indices=indices, datadir=datadir, lazy=lazy,
                          max_threads=max_threads, strip_prefix=strip_prefix)
    
    def __init__(self, num_images: int, image_size: int, dtype=np.float32):
        """Initialize loader.
        
        Args:
            num_images: Total number of images
            image_size: Side length of square images
            dtype: Data type for loaded images
        """
        self._num_images = num_images
        self._image_size = image_size
        self._dtype = dtype
        self._cached = None
    
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
        
        return result.astype(self._dtype)
    
    def images(self, indices=None, require_contiguous: bool = False) -> np.ndarray:
        """Compatibility alias for get().
        
        Args:
            indices: Indices to retrieve
            require_contiguous: Ignored for compatibility
            
        Returns:
            Array of shape (N, image_size, image_size)
        """
        return self.get(indices)
    
    def _parse_indices(self, indices) -> np.ndarray:
        """Convert various index formats to array."""
        if indices is None:
            return np.arange(self._num_images)
        
        # Handle int and numpy integer types
        if isinstance(indices, (int, np.integer)):
            idx = int(indices)  # Convert numpy int to python int
            if idx < 0 or idx >= self._num_images:
                raise IndexError(f"Index {idx} out of range [0, {self._num_images})")
            return np.array([idx])
        
        if isinstance(indices, slice):
            return np.arange(*indices.indices(self._num_images))
        
        if isinstance(indices, (list, tuple)):
            indices = np.array(indices)
        
        if isinstance(indices, np.ndarray):
            if indices.dtype == bool:
                indices = np.where(indices)[0]
            # Handle 0-d arrays (scalars)
            if indices.ndim == 0:
                idx = int(indices)
                if idx < 0 or idx >= self._num_images:
                    raise IndexError(f"Index {idx} out of range [0, {self._num_images})")
                return np.array([idx])
            if np.any(indices < 0) or np.any(indices >= self._num_images):
                raise IndexError("Index out of range")
            return indices
        
        raise TypeError(f"Cannot index with type {type(indices)}")
    
    def _load(self, indices: np.ndarray) -> np.ndarray:
        """Load images at indices. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def iter_batches(self, batch_size: int = 1000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over images in batches.
        
        Args:
            batch_size: Number of images per batch
            
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


class MRCLoader(ImageLoader):
    """Load images from single MRC/MRCS file."""
    
    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None, lazy: bool = True):
        """Initialize MRC loader.
        
        Args:
            filepath: Path to .mrc/.mrcs file
            indices: Optional subset of indices
            lazy: If False, load all images immediately
        """
        from recovar.cryodrgn_mrcfile import MRCHeader
        
        self._filepath = filepath
        self._header = MRCHeader.parse(filepath)
        
        # Get dimensions
        nz = self._header.fields["nz"]
        ny = self._header.fields["ny"]
        nx = self._header.fields["nx"]
        
        if ny != nx:
            raise ValueError(f"Non-square images not supported: {ny} x {nx}")
        
        # Calculate file layout
        self._data_start = 1024 + self._header.fields["next"]
        self._pixels_per_image = ny * nx
        self._bytes_per_image = self._header.dtype().itemsize * self._pixels_per_image
        self._file_dtype = self._header.dtype
        
        # Set up index mapping
        self._file_indices = indices if indices is not None else np.arange(nz)
        
        super().__init__(len(self._file_indices), ny, self._file_dtype)
        
        if not lazy:
            self.load_all()
    
    @nvtx.annotate("MRCLoader._load", color="blue", domain=NVTX_DOMAIN_DATA_IO)
    def _load(self, indices: np.ndarray) -> np.ndarray:
        """Load images from MRC file."""
        # Map requested indices to file indices
        file_idx = self._file_indices[indices]
        
        output = np.empty((len(file_idx), self._image_size, self._image_size), 
                         dtype=self._file_dtype)
        
        with nvtx.annotate(f"disk_read_{len(file_idx)}_images", color="cyan", domain=NVTX_DOMAIN_DATA_IO):
            with open(self._filepath, 'rb') as f:
                # Check if we can do a single contiguous read
                if len(file_idx) > 0:
                    sorted_order = np.argsort(file_idx)
                    sorted_idx = file_idx[sorted_order]
                    
                    is_sequential = np.all(np.diff(sorted_idx) == 1) if len(sorted_idx) > 1 else True
                    
                    if is_sequential:
                        # Single read for contiguous block
                        with nvtx.annotate("sequential_read", color="green", domain=NVTX_DOMAIN_DATA_IO):
                            offset = self._data_start + sorted_idx[0] * self._bytes_per_image
                            f.seek(offset)
                            data = np.fromfile(f, dtype=self._file_dtype, 
                                              count=self._pixels_per_image * len(sorted_idx))
                            data = data.reshape(len(sorted_idx), self._image_size, self._image_size)
                            output[sorted_order] = data
                    else:
                        # Individual reads
                        with nvtx.annotate("random_access_read", color="red", domain=NVTX_DOMAIN_DATA_IO):
                            for i, idx in enumerate(file_idx):
                                offset = self._data_start + idx * self._bytes_per_image
                                f.seek(offset)
                                data = np.fromfile(f, dtype=self._file_dtype, 
                                                 count=self._pixels_per_image)
                                output[i] = data.reshape(self._image_size, self._image_size)
        
        return output


class MultiMRCLoader(ImageLoader):
    """Load images distributed across multiple MRC files."""
    
    def __init__(self, file_map: pd.DataFrame, indices: Optional[np.ndarray] = None,
                 lazy: bool = True, max_threads: int = 1):
        """Initialize multi-file loader.
        
        Args:
            file_map: DataFrame with 'mrc_file' and 'mrc_index' columns
            indices: Optional subset of images
            lazy: If False, load all immediately
            max_threads: Number of parallel I/O threads
        """
        self._file_map = file_map.copy()
        self._max_threads = max_threads
        
        # Get image size from first file
        first_file = str(file_map['mrc_file'].iloc[0])
        first_loader = MRCLoader(first_file, lazy=True)
        img_size = first_loader.image_size
        dtype = first_loader._file_dtype
        
        # Apply index filter if provided
        if indices is not None:
            self._file_map = self._file_map.iloc[indices].reset_index(drop=True)
        
        # Create loaders for unique files
        self._loaders = {}
        for filepath in file_map['mrc_file'].unique():
            try:
                self._loaders[filepath] = MRCLoader(filepath, lazy=True)
            except FileNotFoundError:
                logger.error(f"Cannot find MRC file: {filepath}")
                raise
        
        super().__init__(len(self._file_map), img_size, dtype)
        
        if not lazy:
            self.load_all()
    
    def _load(self, indices: np.ndarray) -> np.ndarray:
        """Load images from multiple files."""
        subset = self._file_map.iloc[indices]
        output = np.empty((len(indices), self._image_size, self._image_size), 
                         dtype=self._dtype)
        
        # Group by file for efficient loading
        groups = subset.groupby('mrc_file')
        
        if self._max_threads > 1 and len(groups) > 1:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self._max_threads) as executor:
                futures = {}
                for filepath, group in groups:
                    loader = self._loaders[filepath]
                    mrc_idx = group['mrc_index'].to_numpy()
                    future = executor.submit(loader._load, mrc_idx)
                    futures[future] = group.index.to_numpy()
                
                for future, output_idx in futures.items():
                    images = future.result()
                    for i, pos in enumerate(output_idx):
                        # Map to position in output array
                        output_pos = np.where(indices == pos)[0][0]
                        output[output_pos] = images[i]
        else:
            # Sequential loading
            for filepath, group in groups:
                loader = self._loaders[filepath]
                mrc_idx = group['mrc_index'].to_numpy()
                images = loader._load(mrc_idx)
                
                for i, original_idx in enumerate(group.index):
                    output_pos = np.where(indices == original_idx)[0][0]
                    output[output_pos] = images[i]
        
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
                
                # Resolve relative paths
                if not os.path.isabs(path):
                    path = os.path.join(base, path)
                
                # Count images in this file
                loader = MRCLoader(path, lazy=True)
                n = loader.num_images
                
                mrc_files.extend([path] * n)
                mrc_indices.extend(range(n))
        
        df = pd.DataFrame({'mrc_file': mrc_files, 'mrc_index': mrc_indices})
        return MultiMRCLoader(df, indices, lazy, max_threads)


class StarLoader(MultiMRCLoader):
    """Load images from RELION STAR file."""
    
    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None,
                 datadir: str = "", lazy: bool = True, max_threads: int = 1,
                 strip_prefix: Optional[str] = None):
        """Initialize STAR file loader.
        
        Args:
            filepath: Path to .star file
            indices: Optional image subset
            datadir: Base directory for MRC files
            lazy: If False, load all immediately
            max_threads: Threads for parallel I/O
            strip_prefix: Prefix to remove from paths
        """
        from recovar.starfile import Starfile
        
        # Parse STAR file
        star = Starfile.load(filepath)
        df = star.df.copy()
        
        # Parse image names (format: "index@filepath")
        parts = df['_rlnImageName'].str.split('@', n=1, expand=True)
        df['mrc_index'] = parts[0].astype(int) - 1  # Convert to 0-indexed
        df['mrc_file'] = parts[1]
        
        # Apply prefix stripping
        if strip_prefix:
            if not df['mrc_file'].str.startswith(strip_prefix).any():
                raise ValueError(f"No paths match strip_prefix: {strip_prefix}")
            df['mrc_file'] = df['mrc_file'].str.removeprefix(strip_prefix).str.lstrip('/')
        
        # Resolve paths with datadir
        if not datadir:
            datadir = os.path.abspath(os.path.dirname(filepath))
        else:
            datadir = os.path.abspath(datadir)
        
        df['mrc_file'] = df['mrc_file'].apply(lambda p: os.path.join(datadir, p))
        
        super().__init__(df[['mrc_file', 'mrc_index']], indices, lazy, max_threads)


class CryoSparcLoader(MultiMRCLoader):
    """Load images from cryoSPARC CS file."""
    
    def __init__(self, filepath: str, indices: Optional[np.ndarray] = None,
                 datadir: str = "", lazy: bool = True, max_threads: int = 1):
        """Initialize cryoSPARC loader.
        
        Args:
            filepath: Path to .cs file
            indices: Optional image subset
            datadir: Base directory for blob files
            lazy: If False, load all immediately
            max_threads: Threads for parallel I/O
        """
        # Load cryoSPARC metadata
        cs_data = np.load(filepath)
        
        blob_idx = cs_data['blob/idx']
        blob_paths = cs_data['blob/path']
        
        # Clean paths (remove leading '>')
        clean_paths = [str(p).lstrip('>') for p in blob_paths]
        
        # Resolve datadir
        if not datadir:
            datadir = os.path.dirname(filepath)
        elif not os.path.isabs(datadir):
            datadir = os.path.join(os.path.dirname(filepath), datadir)
        
        full_paths = [os.path.join(datadir, p) for p in clean_paths]
        
        df = pd.DataFrame({'mrc_file': full_paths, 'mrc_index': blob_idx})
        
        super().__init__(df, indices, lazy, max_threads)


# Compatibility aliases for existing code
ImageSource = ImageLoader
MRCFileSource = MRCLoader
StarfileSource = StarLoader
TxtFileSource = MultiMRCLoader
CsSource = CryoSparcLoader

