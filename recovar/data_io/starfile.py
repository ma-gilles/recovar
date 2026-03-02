"""
Utilities for reading and writing RELION .star files.

Supports both RELION 3.0 (single data table) and RELION 3.1 (with optics table).

Equivalent to cryodrgn/starfile
"""

import functools
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Union, TextIO, List
from typing_extensions import Self

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _read_star_cached(filepath: str, mtime: float) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Cached implementation — filepath + mtime together are the cache key.

    Including mtime means any write to the file automatically invalidates the
    cached entry (new mtime → cache miss → re-parse).  The stat() call in
    read_star() is negligible compared to parsing.
    """
    return _parse_star_file(filepath)


def read_star(filepath: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Parse a RELION .star file into main data and optional optics tables.

    Results are cached by normalised absolute path + file mtime so that
    repeated calls for the same unchanged file (e.g. once for halfsets, once
    per halfset for CTF/poses/image loading) incur only one disk read, while
    any write to the file automatically triggers a re-parse.

    Args:
        filepath: Path to .star file

    Returns:
        Tuple of (main_data, optics_data) where optics_data is None for RELION 3.0
    """
    abs_path = os.path.realpath(os.path.abspath(filepath))
    mtime = os.path.getmtime(abs_path)
    main_df, optics_df = _read_star_cached(abs_path, mtime)
    # Return copies so callers can mutate without corrupting the shared cache entry.
    return main_df.copy(), (optics_df.copy() if optics_df is not None else None)


def _parse_star_file(filepath: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Internal: read and parse a .star file (no caching)."""
    if not filepath.endswith('.star'):
        raise ValueError(f"Expected .star file, got: {filepath}")

    data_blocks: dict = {}
    current_block_name: Optional[str] = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('data_'):
                current_block_name = line
                if current_block_name in data_blocks:
                    raise ValueError(f"Duplicate data block: {current_block_name}")
                data_blocks[current_block_name] = {'columns': [], 'rows': []}

            elif line.startswith('_'):
                if current_block_name is None:
                    continue
                data_blocks[current_block_name]['columns'].append(line.split()[0])

            elif line and current_block_name is not None:
                if not line.startswith('#') and line != 'loop_':
                    values = line.split()
                    if values:
                        data_blocks[current_block_name]['rows'].append(values)

    dfs: dict = {}
    for block_name, block_data in data_blocks.items():
        if not block_data['rows']:
            continue
        cols = block_data['columns']

        row_lengths = [len(row) for row in block_data['rows']]
        if len(set(row_lengths)) != 1:
            raise ValueError(f"Inconsistent row lengths in {block_name}")
        if row_lengths[0] != len(cols):
            raise ValueError(
                f"Column count mismatch in {block_name}: "
                f"{row_lengths[0]} values vs {len(cols)} headers"
            )

        rows_array = np.array(block_data['rows'])
        dfs[block_name] = pd.DataFrame(rows_array, columns=cols)

    # Extract optics table if present
    optics_df = dfs.pop('data_optics', None)

    # Find main data table (largest non-optics table)
    main_df = None
    for name, df in dfs.items():
        if name.startswith('data_') and name != 'data_optics':
            if main_df is None or len(df) > len(main_df):
                main_df = df

    if main_df is None:
        raise ValueError(f"No data table found in {filepath}")

    logger.debug("Parsed %s: %d particles, %d columns, optics=%s",
                 filepath, len(main_df), len(main_df.columns),
                 optics_df is not None)
    return main_df, optics_df


def write_star(filepath: str, data: pd.DataFrame, 
               data_optics: Optional[pd.DataFrame] = None) -> None:
    """Write data to a RELION .star file.
    
    Args:
        filepath: Output file path
        data: Main data table
        data_optics: Optional optics table (for RELION 3.1 format)
    """
    with open(filepath, 'w') as f:
        # Header comment
        f.write(f"# Created {datetime.now()}\n\n")
        
        # RELION 3.1 format (with optics)
        if data_optics is not None:
            _write_block(f, data_optics, 'data_optics')
            f.write("\n\n")
            _write_block(f, data, 'data_particles')
        
        # RELION 3.0 format (no optics)
        else:
            _write_block(f, data, 'data_')


def _write_block(f: TextIO, df: pd.DataFrame, block_name: str) -> None:
    """Write a single data block to file.
    
    Args:
        f: File handle
        df: DataFrame to write
        block_name: Name of the data block
    """
    f.write(f"{block_name}\n\n")
    f.write("loop_\n")
    
    # Write column headers
    for col in df.columns:
        f.write(f"{col}\n")
    
    # Write data rows
    for _, row in df.iterrows():
        f.write(" ".join(str(val) for val in row.values))
        f.write("\n")


class StarFile:
    """Container for RELION .star file data with convenient access methods.
    
    Attributes:
        df: Main data table
        data_optics: Optics table (None for RELION 3.0)
    """
    
    def __init__(self, starfile: Optional[str] = None, *,
                 data: Optional[pd.DataFrame] = None,
                 data_optics: Optional[pd.DataFrame] = None):
        """Initialize from file or data tables.
        
        Args:
            starfile: Path to .star file (mutually exclusive with data)
            data: Main data table (keyword only)
            data_optics: Optics table (keyword only)
        """
        # Validate arguments
        if (starfile is None) == (data is None):
            raise ValueError("Provide exactly one of: starfile path or data DataFrame")
        
        # Load from file if provided
        if starfile is not None:
            data, data_optics = read_star(starfile)

        self.df = data
        self.data_optics = data_optics
        self._star_path = starfile
        
        # Set up optics table index if present
        if self.has_optics:
            if '_rlnOpticsGroup' not in self.data_optics.columns:
                raise ValueError("Optics table missing _rlnOpticsGroup column")
            self.data_optics = self.data_optics.set_index('_rlnOpticsGroup', drop=False)
    
    @classmethod
    def load(cls, filepath: str) -> Self:
        """Load from .star file (convenience method)."""
        return cls(starfile=filepath)
    
    def save(self, filepath: str) -> None:
        """Save to .star file."""
        write_star(filepath, self.df, self.data_optics)
    
    def write(self, filepath: str) -> None:
        """Alias for save()."""
        self.save(filepath)
    
    @property
    def has_optics(self) -> bool:
        """Whether this is RELION 3.1 format with optics table."""
        return self.data_optics is not None
    
    @property
    def relion31(self) -> bool:
        """Alias for has_optics (compatibility)."""
        return self.has_optics
    
    def __len__(self) -> int:
        """Number of particles in main data table."""
        return len(self.df)
    
    def __eq__(self, other: Self) -> bool:
        """Check equality with another StarFile."""
        if not self.df.equals(other.df):
            return False
        if self.has_optics != other.has_optics:
            return False
        if self.has_optics and not self.data_optics.equals(other.data_optics):
            return False
        return True
    
    def get_optics_values(self, field: str, dtype: Optional[np.dtype] = None) -> Optional[np.ndarray]:
        """Get per-particle values for a field, consulting optics table if available.
        
        Args:
            field: Field name to retrieve
            dtype: Optional dtype to cast values to
            
        Returns:
            Array of values (one per particle) or None if field not found
        """
        values = None
        
        # Try optics table first (RELION 3.1)
        if self.has_optics and field in self.data_optics.columns:
            # Map optics group to each particle
            if '_rlnOpticsGroup' in self.df.columns:
                # Vectorised: pandas .map() is O(N) in C vs an O(N) Python loop
                optics_series = self.data_optics[field]
                values = self.df['_rlnOpticsGroup'].map(optics_series).to_numpy()
            else:
                # Single optics group for all particles
                single_value = self.data_optics[field].iloc[0]
                values = np.full(len(self.df), single_value)
        
        # Fall back to main data table
        elif field in self.df.columns:
            values = self.df[field].values
        
        # Cast to requested dtype
        if values is not None and dtype is not None:
            values = values.astype(dtype)
        
        return values
    
    def set_optics_values(self, field: str, values: Union[float, List, np.ndarray]) -> None:
        """Set per-particle values for a field in appropriate table.
        
        Args:
            field: Field name to set
            values: Single value or array of values
        """
        # Normalize to list
        if not hasattr(values, '__iter__'):
            values = [values]
        else:
            values = list(values)
        
        # Validate length
        valid_lengths = {1, len(self)}
        if self.has_optics:
            valid_lengths.add(len(self.data_optics))
        
        if len(values) not in valid_lengths:
            raise ValueError(
                f"Values length {len(values)} not in valid set {valid_lengths}"
            )
        
        # Update optics table if field exists there
        if self.has_optics and field in self.data_optics.columns:
            if '_rlnOpticsGroup' in self.df.columns:
                if len(values) in {1, len(self.data_optics)}:
                    self.data_optics.loc[:, field] = values
                else:
                    # Move field from optics to main table
                    self.df.loc[:, field] = values
                    self.data_optics.drop(field, axis=1, inplace=True)
            else:
                if len(values) != 1:
                    raise ValueError(
                        f"Without optics groups, can only set single value (got {len(values)})"
                    )
                self.data_optics.loc[:, field] = values[0]
        
        # Update main data table
        elif field in self.df.columns:
            # Map from optics groups if needed
            if self.has_optics and len(values) == len(self.data_optics):
                optics_groups = self.df['_rlnOpticsGroup'].values
                mapped_values = [
                    values[self.data_optics['_rlnOpticsGroup'].tolist().index(g)]
                    for g in optics_groups
                ]
                self.df.loc[:, field] = mapped_values
            else:
                self.df.loc[:, field] = values
        else:
            raise ValueError(f"Field {field} not found in .star file")
    
    @property
    def apix(self) -> Optional[np.ndarray]:
        """Pixel size (Angstroms/pixel) for each particle.

        Tries _rlnImagePixelSize first (RELION 3.1+), then falls back to
        _rlnDetectorPixelSize * 1e4 / _rlnMagnification (older RELION).
        """
        values = self.get_optics_values('_rlnImagePixelSize', dtype=np.float32)
        if values is not None:
            return values
        # Old-format STAR: pixel_size = detector_pixel_size (um) * 1e4 / magnification
        det = self.get_optics_values('_rlnDetectorPixelSize', dtype=np.float64)
        mag = self.get_optics_values('_rlnMagnification', dtype=np.float64)
        if det is not None and mag is not None:
            return (det * 1e4 / mag).astype(np.float32)
        return None

    @property
    def resolution(self) -> Optional[np.ndarray]:
        """Image size (pixels) for each particle.

        Tries ``_rlnImageSize`` first (RELION 3.1 optics table).
        Falls back to reading the MRC header of the first particle stack
        referenced in ``_rlnImageName`` (RELION 3.0 files).
        """
        vals = self.get_optics_values('_rlnImageSize', dtype=np.float32)
        if vals is not None:
            return vals.astype(np.int64)
        # RELION 3.0 fallback: read dimension from MRC header
        image_name = self.get_optics_values('_rlnImageName', dtype=str)
        if image_name is not None:
            try:
                # _rlnImageName format: "idx@path/to/stack.mrcs"
                first = str(image_name[0])
                mrcs_path = first.split('@')[-1] if '@' in first else first
                # Resolve relative to STAR file directory
                import os
                if not os.path.isabs(mrcs_path) and hasattr(self, '_star_path'):
                    mrcs_path = os.path.join(
                        os.path.dirname(self._star_path), mrcs_path
                    )
                if os.path.exists(mrcs_path):
                    import mrcfile
                    with mrcfile.open(mrcs_path, mode='r', header_only=True) as mrc:
                        D = int(mrc.header.ny)
                    return np.full(len(self), D, dtype=np.int64)
            except (ImportError, FileNotFoundError, OSError, KeyError) as e:
                logger.debug("Could not read MRC header for resolution: %s", e)
        return None
    
    def flatten_to_relion30(self) -> pd.DataFrame:
        """Convert to RELION 3.0 format by flattening optics into main table.
        
        Returns:
            DataFrame with all optics fields merged into main table
        """
        result = self.df.copy()
        
        if self.has_optics:
            # Add optics fields to main table
            for field in self.data_optics.columns:
                if field not in result.columns and 'OpticsGroup' not in field:
                    result[field] = self.get_optics_values(field)
        
        return result
    
    def to_relion30(self) -> pd.DataFrame:
        """Alias for flatten_to_relion30 (compatibility)."""
        return self.flatten_to_relion30()
