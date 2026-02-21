"""
PyTorch Dataset classes for cryo-EM single particle and tilt series data.

Core classes:
- ParticleImageDataset: Base dataset for single particle images
- TiltSeriesDataset: Dataset for tilt series with automatic grouping
- Batch utilities for flexible data loading
"""

import numpy as np
import logging
import time
from collections import OrderedDict, Counter
from typing import Optional, Tuple, List, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import jax.numpy as jnp

from recovar.image_loader import ImageSource
from recovar import starfile
from recovar import mask
import nvtx

logger = logging.getLogger(__name__)

# NVTX domain for data I/O profiling
NVTX_DOMAIN_DATA_IO = "data_io"


def create_window_mask(image_size: int, inner_radius: float = 0.85, 
                       outer_radius: float = 0.99, dtype=np.complex64) -> np.ndarray:
    """Generate a circular window mask for image processing.
    
    Args:
        image_size: Size of square image
        inner_radius: Inner radius as fraction of image size
        outer_radius: Outer radius asiss fraction of image size  
        dtype: Data type for mask (not used, for compatibility)
        
    Returns:
        Window mask array
    """
    return mask.window_mask(image_size, inner_radius, outer_radius)


class ParticleImageDataset(Dataset):
    """PyTorch Dataset for cryo-EM particle images.
    
    Loads images from MRC/MRCS files with optional preprocessing.
    """
    
    def __init__(self, image_file: str, lazy: bool = True, ind: Optional[np.ndarray] = None,
                 invert_data: bool = False, datadir: str = "", padding: int = 0,
                 max_threads: int = 16, strip_prefix: Optional[str] = None,
                 device: Union[str, torch.device] = "cpu", **kwargs):
        """Initialize cryo-EM dataset.
        
        Args:
            image_file: Path to MRC/MRCS file
            lazy: If True, defer loading images until accessed
            ind: Optional subset of image indices
            invert_data: If True, multiply images by -1
            datadir: Base directory for relative paths
            padding: Padding to add around images
            max_threads: Threads for parallel loading
            strip_prefix: Prefix to strip from paths
            device: PyTorch device (for compatibility, not used)
        """
        if padding != 0:
            raise NotImplementedError("Padding not yet supported")
        
        # Load image source
        self.source = ImageSource.from_file(
            image_file, lazy=lazy, datadir=datadir or "",
            indices=ind, max_threads=max_threads, strip_prefix=strip_prefix
        )
        
        # Store parameters
        self.image_size = self.source.D
        self.num_images = self.source.n
        self.lazy = lazy
        self.invert_data = invert_data
        self.padding = padding
        self.device = device
        
        # Validate image size
        if self.image_size % 2 != 0:
            raise ValueError(f"Image size must be even, got {self.image_size}")
        
        # Setup image properties
        self.dtype = np.complex64
        self.image_shape = (self.image_size, self.image_size)
        self.total_pixels = self.image_size * self.image_size
        
        # Create image mask
        self.image_mask = np.array(create_window_mask(self.image_size, dtype=self.dtype))
        self.data_multiplier = -1 if invert_data else 1
        
        # Aliases for compatibility
        self.N = self.num_images
        self.n_images = self.num_images
        self.D = self.image_size
        self.unpadded_D = self.image_size
        self.unpadded_image_shape = self.image_shape
        self.image_size_alias = self.total_pixels
        self.mask = self.image_mask
        self.mult = self.data_multiplier
        self.ind = ind
        self.src = self.source
    
    def __len__(self) -> int:
        """Number of images in dataset."""
        return self.num_images
    
    @nvtx.annotate("ParticleImageDataset.__getitem__", color="yellow", domain=NVTX_DOMAIN_DATA_IO)
    def __getitem__(self, index: Union[int, np.ndarray, slice]) -> Tuple[np.ndarray, Union[int, np.ndarray], Union[int, np.ndarray]]:
        """Get images at index.
        
        Args:
            index: Single index, array of indices, or slice
            
        Returns:
            Tuple of (images, particle_indices, tilt_indices)
        """
        with nvtx.annotate("load_from_source", color="orange", domain=NVTX_DOMAIN_DATA_IO):
            images = self.source.images(index)
        
        # Ensure 3D output (add batch dimension if needed)
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        
        # Log access
        if isinstance(index, (int, np.integer)):
            logger.debug(f"Loaded image at index {index}")
        else:
            idx_arr = index if isinstance(index, np.ndarray) else np.arange(len(images))
            logger.debug(f"Loaded {len(idx_arr)} images (indices {idx_arr[0]}..{idx_arr[-1]})")
        
        return images, index, index
    
    def get_slice(self, start: int, stop: int) -> Tuple[np.ndarray, None]:
        """Get contiguous slice of images.
        
        Args:
            start: Start index
            stop: Stop index (exclusive)
            
        Returns:
            Tuple of (images, None)
        """
        images = self.source.images(slice(start, stop), require_contiguous=True)
        return images, None
    
    def apply_preprocessing(self, images: np.ndarray, use_mask: bool = False) -> np.ndarray:
        """Apply preprocessing to images.
        
        Args:
            images: Input images
            use_mask: Whether to apply circular mask
            
        Returns:
            Processed images
        """
        if use_mask:
            images = images * self.image_mask
        
        # Apply padding and DFT if needed
        import recovar.padding as pad
        images = pad.padded_dft(images * self.data_multiplier, self.D, self.padding)
        
        return images.astype(self.dtype, copy=False)
    
    def process_images(self, images: np.ndarray, apply_image_mask: bool = False) -> np.ndarray:
        """Compatibility alias for apply_preprocessing."""
        return self.apply_preprocessing(images, use_mask=apply_image_mask)
    
    def create_dataloader(self, batch_size: int, num_workers: int = 0, 
                         **kwargs) -> DataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return JAXDataLoader(self, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def get_dataset_generator(self, batch_size: int, num_workers: int = 0, 
                             pad_to_batch_size: bool = False, mode: str = 'tilt_series', 
                             **kwargs):
        """Compatibility method for getting data generator."""
        return self.create_dataloader(batch_size, num_workers)
    
    def get_dataset_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                    num_workers: int = 0, pad_to_batch_size: bool = False,
                                    mode: str = 'tilt_series', **kwargs):
        """Create data generator for subset."""
        subset = Subset(self, subset_indices)
        return JAXDataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def get_image_generator(self, batch_size: int, num_workers: int = 0):
        """Get image generator (for SPA, wraps the dataset directly).
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            DataLoader instance
        """
        return self.get_dataset_generator(batch_size, num_workers)
    
    def get_image_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                   num_workers: int = 0):
        """Get image subset generator.
        
        Args:
            batch_size: Batch size
            subset_indices: Indices to include
            num_workers: Number of workers
            
        Returns:
            DataLoader for subset
        """
        return self.get_dataset_subset_generator(batch_size, subset_indices, num_workers)


class TiltSeriesDataset(ParticleImageDataset):
    """Dataset for tilt series with automatic particle grouping.
    
    Groups tilts by particle ID and handles tilt ordering.
    """
    
    def __init__(self, starfile_path: str, lazy: bool = True, num_tilts: Optional[int] = None,
                 random_tilts: bool = False, ind: Optional[np.ndarray] = None,
                 voltage: Optional[float] = None, dose_per_tilt: Optional[float] = None,
                 angle_per_tilt: Optional[float] = None, expected_res: Optional[float] = None,
                 tilt_file_option: str = 'relion5', **kwargs):
        """Initialize tilt series dataset.
        
        Args:
            starfile_path: Path to STAR file
            lazy: Lazy loading mode
            num_tilts: Number of tilts per particle (None = all)
            random_tilts: Randomly select tilts instead of by order
            ind: Optional tilt indices to include
            voltage: Microscope voltage
            dose_per_tilt: Electron dose per tilt
            angle_per_tilt: Tilt angle increment
            expected_res: Expected resolution
            tilt_file_option: 'relion5' or 'warp' for tilt ordering
        """
        start_time = time.time()
        
        # Initialize base dataset
        super().__init__(starfile_path, lazy=lazy, ind=ind, **kwargs)
        logger.info(f"Base dataset loaded in {time.time() - start_time:.2f}s")
        
        # Parse STAR file
        star = starfile.Starfile.load(starfile_path)
        logger.info(f"STAR file parsed in {time.time() - start_time:.2f}s")
        
        # Get canonical group ordering
        canonical_groups = self._get_canonical_groups(star.df)
        
        # Apply index filter if needed
        if ind is not None:
            star.df = star.df.loc[ind]
        
        # Group tilts by particle
        self.particle_groups = self._build_particle_groups(star.df, canonical_groups)
        self.num_particles = len(self.particle_groups)
        self.dataset_tilt_indices = [canonical_groups.index(gn) for gn in self.particle_groups.keys()]
        
        # Extract CTF parameters
        self.ctfscalefactor = np.asarray(star.df["_rlnCtfScalefactor"], dtype=np.float32)
        
        if '_rlnCtfBfactor' in star.df.columns:
            self.ctfBfactor = np.asarray(star.df["_rlnCtfBfactor"], dtype=np.float32)
        
        if tilt_file_option == 'relion5':
            self.dose = np.asarray(star.df["_rlnMicrographPreExposure"], dtype=np.float32)
        
        # Determine tilt ordering
        self._compute_tilt_ordering(tilt_file_option)
        
        # Store parameters
        self.num_tilts = num_tilts
        self.random_tilts = random_tilts
        self.voltage = voltage
        self.dose_per_tilt = dose_per_tilt
        self.tilt_angles = None  # Dose-symmetric scheme (Hagen et al. JSB 2017)
        
        # Log statistics
        group_counts = Counter(star.df["_rlnGroupName"])
        logger.info(f"Loaded {self.N} tilts for {self.num_particles} particles")
        logger.info(f"Tilts per particle: {set(group_counts.values())}")
        logger.info(f"Dataset loaded in {time.time() - start_time:.2f}s")
        
        # Compatibility attributes
        self.Np = self.num_particles
        self.particles = list(self.particle_groups.values())
        self.counts = group_counts
        self.tilt_numbers = self.tilt_order
        self.ntilts = num_tilts
    
    def _get_canonical_groups(self, df: 'pd.DataFrame') -> List[str]:
        """Get sorted list of unique group names for consistent ordering."""
        return sorted(df['_rlnGroupName'].unique())
    
    def _build_particle_groups(self, df: 'pd.DataFrame', canonical_groups: List[str]) -> OrderedDict:
        """Build ordered mapping of particles to tilt indices."""
        groups = OrderedDict()
        for idx, group_name in enumerate(df["_rlnGroupName"]):
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(idx)
        
        # Convert to arrays and reorder canonically
        ordered = OrderedDict()
        for gn in canonical_groups:
            if gn in groups:
                ordered[gn] = np.array(groups[gn], dtype=int)
        
        return ordered
    
    def _compute_tilt_ordering(self, method: str):
        """Compute tilt ordering based on dose or B-factor.
        
        Args:
            method: 'relion5' (use dose) or 'warp' (use B-factor)
        """
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
            else:  # warp
                sort_indices = np.argsort(self.ctfBfactor[particle_tilts])
            
            # Assign ranks
            ranks = np.empty_like(sort_indices)
            ranks[sort_indices[::-1]] = np.arange(len(particle_tilts))
            self.tilt_order[particle_tilts] = ranks
    
    def __len__(self) -> int:
        """Number of particles (not tilts)."""
        return self.num_particles
    
    def __getitem__(self, particle_index: int) -> Tuple[np.ndarray, int, np.ndarray]:
        """Get tilts for a particle.
        
        Args:
            particle_index: Index of particle
            
        Returns:
            Tuple of (images, particle_index, tilt_indices)
        """
        particle_tilts = list(self.particle_groups.values())[particle_index]
        
        if self.random_tilts and self.num_tilts is not None:
            # Random selection
            selected = np.random.choice(particle_tilts, self.num_tilts, replace=False)
        else:
            # Ordered selection
            tilt_orders = self.tilt_order[particle_tilts]
            sorted_idx = np.argsort(tilt_orders)
            n_select = self.num_tilts if self.num_tilts is not None else len(particle_tilts)
            selected = particle_tilts[sorted_idx[:n_select]]
        
        images = self.source.images(selected)
        return images, particle_index, selected
    
    def get_tilt(self, tilt_index: int) -> Tuple[np.ndarray, int, int]:
        """Get individual tilt image (not particle group).
        
        Args:
            tilt_index: Index of tilt image
            
        Returns:
            Tuple of (image, tilt_index, tilt_index)
        """
        return super().__getitem__(tilt_index)
    
    @classmethod
    def parse_particle_tilt(cls, starfile_path: str, 
                           indices: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], Dict[int, int]]:
        """Parse particle-to-tilt and tilt-to-particle mappings.
        
        Args:
            starfile_path: Path to STAR file
            indices: Optional subset of tilts
            
        Returns:
            Tuple of (particles_to_tilts, tilts_to_particles)
        """
        star = starfile.Starfile.load(starfile_path)
        df = star.df
        
        if indices is not None:
            df = df.loc[indices]
        
        # Get canonical ordering
        canonical_groups = sorted(df['_rlnGroupName'].unique())
        
        # Build forward mapping
        grouped = df.groupby('_rlnGroupName').groups
        particles_to_tilts = [np.asarray(grouped[gn], dtype=int) for gn in canonical_groups]
        
        # Build reverse mapping
        tilts_to_particles = {}
        for particle_idx, tilt_indices in enumerate(particles_to_tilts):
            for tilt_idx in tilt_indices:
                tilts_to_particles[int(tilt_idx)] = particle_idx
        
        return particles_to_tilts, tilts_to_particles
    
    @classmethod
    def parse_micrograph_tilt_mapping(cls, starfile_path: str) -> Tuple[List[np.ndarray], Dict[int, int]]:
        """Parse tomogram tilt mappings.
        
        Args:
            starfile_path: Path to STAR file
            
        Returns:
            Tuple of (tomogram_tilts_to_tilts, tilts_to_tomogram_tilts)
        """
        star = starfile.Starfile.load(starfile_path)
        df = star.df
        
        # Find tilt name column
        tilt_col = None
        for col in ['_rlnTiltName', 'rlnTiltName']:
            if col in df.columns:
                tilt_col = col
                break
        
        if tilt_col is None:
            raise ValueError(f"No tilt name column found. Available: {list(df.columns)}")
        
        # Group by tilt name
        grouped = df.groupby(tilt_col).groups
        tomogram_tilts = [np.asarray(indices, dtype=int) for indices in grouped.values()]
        
        # Reverse mapping
        reverse_map = {}
        for group_idx, tilt_indices in enumerate(tomogram_tilts):
            for tilt_idx in tilt_indices:
                reverse_map[int(tilt_idx)] = group_idx
        
        return tomogram_tilts, reverse_map
    
    @staticmethod
    def particles_to_tilts(particle_tilt_map: List[np.ndarray], 
                          particle_indices: np.ndarray) -> np.ndarray:
        """Convert particle indices to tilt indices.
        
        Args:
            particle_tilt_map: Mapping from particles to tilts
            particle_indices: Particle indices to convert
            
        Returns:
            Array of tilt indices
        """
        tilts = [particle_tilt_map[int(i)] for i in particle_indices]
        return np.concatenate(tilts)
    
    @staticmethod
    def tilts_to_particles(tilt_particle_map: Dict[int, int], 
                          tilt_indices: np.ndarray) -> np.ndarray:
        """Convert tilt indices to unique particle indices.
        
        Args:
            tilt_particle_map: Mapping from tilts to particles
            tilt_indices: Tilt indices to convert
            
        Returns:
            Array of unique particle indices (sorted)
        """
        particles = {tilt_particle_map[int(i)] for i in tilt_indices}
        return np.array(sorted(particles))
    
    def _max_tilts_per_particle(self) -> int:
        """Get maximum number of tilts in any particle."""
        max_tilts = 0
        for tilts in self.particle_groups.values():
            n_available = len(tilts)
            n_actual = min(self.num_tilts, n_available) if self.num_tilts else n_available
            max_tilts = max(max_tilts, n_actual)
        return max_tilts
    
    def create_image_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Create dataloader that iterates over individual images (not particles).
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            DataLoader for individual images
        """
        class ImageView(Dataset):
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
        
        view = ImageView(self.source, self.source.n)
        return JAXDataLoader(view, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def get_image_generator(self, batch_size: int, num_workers: int = 0):
        """Compatibility method for image-level iteration."""
        return self.create_image_dataloader(batch_size, num_workers)
    
    def get_image_subset_generator(self, batch_size: int, subset_indices: np.ndarray, 
                                   num_workers: int = 0):
        """Create image dataloader for subset of images."""
        if subset_indices is None:
            return self.create_image_dataloader(batch_size, num_workers)
        
        class ImageView(Dataset):
            def __init__(self, source, num_images):
                self.source = source
                self.num_images = num_images
            
            def __len__(self):
                return self.num_images
            
            def __getitem__(self, idx):
                if isinstance(idx, (list, tuple, np.ndarray)):
                    idx = idx[0]
                img = self.source.images(idx)
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                return img, np.inf, idx
        
        view = ImageView(self.source, self.source.n)
        subset = Subset(view, subset_indices)
        return JAXDataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def get_dataset_generator(self, batch_size: int, num_workers: int = 0,
                             pad_to_batch_size: bool = False, mode: str = 'tilt_series', 
                             **kwargs):
        """Create data generator with different batching modes.
        
        Args:
            batch_size: Target batch size
            num_workers: Number of workers
            pad_to_batch_size: Pad last batch to exact size
            mode: 'images' (batch by image count) or 'tilt_series' (batch by particles)
            
        Returns:
            DataLoader instance
        """
        if mode == 'images':
            max_tilts = self._max_tilts_per_particle()
            if batch_size < max_tilts:
                raise ValueError(
                    f"Batch size ({batch_size}) < max tilts per particle ({max_tilts}). "
                    f"Use larger batch_size or mode='tilt_series'"
                )
            return ImageCountBatchLoader(self, batch_size, num_workers, pad_to_batch_size)
        elif mode == 'tilt_series':
            return simple_dataloader(self, batch_size=batch_size, num_workers=num_workers)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_dataset_subset_generator(self, batch_size: int, subset_indices: np.ndarray,
                                    num_workers: int = 0, pad_to_batch_size: bool = False,
                                    mode: str = 'tilt_series', **kwargs):
        """Create data generator for particle subset."""
        if subset_indices is None:
            return self.get_dataset_generator(batch_size, num_workers, pad_to_batch_size, mode)
        
        if mode == 'images':
            subset = ParticleSubset(self, subset_indices)
            max_tilts = subset._max_tilts_per_particle()
            if batch_size < max_tilts:
                raise ValueError(f"Batch size ({batch_size}) < max tilts ({max_tilts})")
            return ImageCountBatchLoader(subset, batch_size, num_workers, pad_to_batch_size)
        else:
            subset = Subset(self, subset_indices)
            return simple_dataloader(subset, batch_size=batch_size, num_workers=num_workers)


def simple_dataloader(dataset, batch_size: int, num_workers: int = 0, 
                     shuffle: bool = False, buffer_size: int = 0):
    """Create simple dataloader for dataset.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size (forced to 1 for tilt series)
        num_workers: Number of workers
        shuffle: Whether to shuffle
        buffer_size: Buffer size for shuffling
        
    Returns:
        DataLoader instance
    """
    batch_size = 1  # Force batch size of 1 for tilt series
    return JAXDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@nvtx.annotate("collate_to_jax", color="magenta", domain=NVTX_DOMAIN_DATA_IO)
def collate_to_jax(batch):
    """Collate function that converts batches to JAX arrays.
    
    Args:
        batch: Batch from PyTorch DataLoader
        
    Returns:
        JAX arrays
    """
    with nvtx.annotate("numpy_to_jax_transfer", color="purple", domain=NVTX_DOMAIN_DATA_IO):
        if isinstance(batch[0], np.ndarray):
            return jnp.concatenate(batch, axis=0)
        elif isinstance(batch[0], (tuple, list)):
            return [collate_to_jax(samples) for samples in zip(*batch)]
        elif batch is None:
            return None
        else:
            return jnp.array(batch)


class JAXDataLoader(DataLoader):
    """DataLoader that returns JAX arrays instead of PyTorch tensors."""
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, 
                 num_workers: int = 0, **kwargs):
        """Initialize JAX DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
        """
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_to_jax,
            **kwargs
        )


class ImageCountBatchLoader:
    """DataLoader that batches by total image count across tilt series."""
    
    def __init__(self, dataset, batch_size: int, num_workers: int = 0, 
                 pad_to_batch: bool = False):
        """Initialize image count batch loader.
        
        Args:
            dataset: Tilt series dataset
            batch_size: Target number of images per batch
            num_workers: Number of workers (not used)
            pad_to_batch: Whether to pad last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_to_batch = pad_to_batch
        
        # Precompute tilts per particle
        self.tilts_counts = []
        for particle_idx in range(len(dataset)):
            particle_tilts = list(dataset.particle_groups.values())[particle_idx]
            n_available = len(particle_tilts)
            n_actual = min(dataset.num_tilts, n_available) if dataset.num_tilts else n_available
            self.tilts_counts.append(n_actual)
        
        self.tilts_counts = np.array(self.tilts_counts)
        self.total_images = np.sum(self.tilts_counts)
    
    def __iter__(self):
        """Iterate over batches of images."""
        particle_idx = 0
        
        while particle_idx < len(self.dataset):
            batch_images = []
            batch_particle_ids = []
            batch_tilt_ids = []
            current_count = 0
            
            while particle_idx < len(self.dataset) and current_count < self.batch_size:
                images, p_idx, t_indices = self.dataset[particle_idx]
                n_images = images.shape[0]
                
                # Check if adding would exceed batch size
                if current_count + n_images > self.batch_size and current_count > 0:
                    break
                
                batch_images.append(images)
                batch_particle_ids.append(np.full(n_images, p_idx, dtype=np.int32))
                batch_tilt_ids.append(t_indices)
                
                current_count += n_images
                particle_idx += 1
            
            # Handle empty batch or oversized single particle
            if len(batch_images) == 0 and particle_idx < len(self.dataset):
                images, p_idx, t_indices = self.dataset[particle_idx]
                batch_images.append(images)
                batch_particle_ids.append(np.full(images.shape[0], p_idx, dtype=np.int32))
                batch_tilt_ids.append(t_indices)
                current_count = images.shape[0]
                particle_idx += 1
            
            if len(batch_images) > 0:
                # Concatenate batch
                batch_images = np.concatenate(batch_images, axis=0)
                batch_particle_ids = np.concatenate(batch_particle_ids, axis=0)
                batch_tilt_ids = np.concatenate(batch_tilt_ids, axis=0)
                
                # Pad if requested
                if self.pad_to_batch and current_count < self.batch_size:
                    pad_size = self.batch_size - current_count
                    img_shape = batch_images.shape[1:]
                    
                    pad_images = np.zeros((pad_size,) + img_shape, dtype=batch_images.dtype)
                    pad_particles = np.full(pad_size, -1, dtype=np.int32)
                    pad_tilts = np.full(pad_size, -1, dtype=batch_tilt_ids.dtype)
                    
                    batch_images = np.concatenate([batch_images, pad_images])
                    batch_particle_ids = np.concatenate([batch_particle_ids, pad_particles])
                    batch_tilt_ids = np.concatenate([batch_tilt_ids, pad_tilts])
                
                yield batch_images, batch_particle_ids, batch_tilt_ids
    
    def __len__(self) -> int:
        """Estimate number of batches."""
        return int(np.ceil(self.total_images / self.batch_size))


class ParticleSubset:
    """Subset of particles from tilt series dataset."""
    
    def __init__(self, dataset, indices: np.ndarray):
        """Initialize subset.
        
        Args:
            dataset: Parent dataset
            indices: Particle indices to include
        """
        self.dataset = dataset
        self.indices = indices
        self.num_tilts = getattr(dataset, 'num_tilts', None)
        self.particle_groups = getattr(dataset, 'particle_groups', None)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    
    def _max_tilts_per_particle(self) -> int:
        """Get max tilts in subset."""
        max_tilts = 0
        for idx in self.indices:
            particle_tilts = list(self.dataset.particle_groups.values())[idx]
            n_available = len(particle_tilts)
            n_actual = min(self.dataset.num_tilts, n_available) if self.dataset.num_tilts else n_available
            max_tilts = max(max_tilts, n_actual)
        return max_tilts


def tilt_series_to_images(tilt_series_indices: np.ndarray, starfile_path: str,
                          image_subset: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert tilt series indices to image indices.
    
    Args:
        tilt_series_indices: Particle indices
        starfile_path: Path to STAR file
        image_subset: Optional subset of images to intersect with
        
    Returns:
        Array of image indices
    """
    particle_tilts, _ = TiltSeriesDataset.parse_particle_tilt(starfile_path)
    image_indices = np.concatenate([particle_tilts[i] for i in tilt_series_indices])
    
    if image_subset is not None:
        image_indices = np.intersect1d(image_indices, image_subset)
    
    return image_indices


def get_canonical_group_names(df: 'pd.DataFrame', group_column: str = '_rlnGroupName') -> List[str]:
    """Get sorted list of unique group names for consistent ordering.
    
    Args:
        df: DataFrame with group column
        group_column: Name of grouping column
        
    Returns:
        Sorted list of unique group names
    """
    return sorted(df[group_column].unique())


# Compatibility aliases
ImageDataset = ParticleImageDataset
TiltSeriesData = TiltSeriesDataset
NumpyLoader = JAXDataLoader
make_dataloader = simple_dataloader
numpy_collate = collate_to_jax
ImageBatchDataLoader = ImageCountBatchLoader
TiltSeriesSubset = ParticleSubset
tilt_series_indices_to_image_indices = tilt_series_to_images

def set_standard_mask(D, dtype):
    """Compatibility wrapper for create_window_mask (dtype ignored)."""
    return create_window_mask(D, inner_radius=0.85, outer_radius=0.99)
