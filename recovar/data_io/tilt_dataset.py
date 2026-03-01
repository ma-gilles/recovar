"""
Compatibility module - re-exports from cryo_dataset.

This module exists for backward compatibility. All functionality
has been moved to cryo_dataset.py with a clean implementation.
"""

# Re-export everything from cryo_dataset
from recovar.data_io.cryo_dataset import *

# Explicitly list what we're exporting for clarity
__all__ = [
    'ParticleImageDataset',
    'TiltSeriesDataset',
    'JAXDataLoader',
    'ImageCountBatchLoader',
    'ParticleSubset',
    'create_window_mask',
    'set_standard_mask',
    'collate_to_jax',
    'simple_dataloader',
    'tilt_series_to_images',
    # Aliases
    'ImageDataset',
    'TiltSeriesData',
    'NumpyLoader',
    'ImageBatchDataLoader',
    'TiltSeriesSubset',
    'tilt_series_indices_to_image_indices',
]
