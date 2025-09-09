"""
Data copying utilities for recovar pipeline.

This module provides functions to copy data to temporary folders for faster processing,
with automatic cleanup of temporary files when processing is complete.

Behavior:
- Input data files are copied to a temporary folder for faster I/O
- Output results are written directly to the original --o directory (no copy-back needed)
- By default, temporary files and directories are automatically cleaned up after pipeline completion
- Use the --no-cleanup flag to preserve temporary files for chaining multiple commands
- Temporary files are created in /tmp with unique names to avoid conflicts
- File copying is cached to avoid redundant copies of the same files

Note: --no-cleanup is different from --delete-rounds (in pipeline_with_outliers):
- --no-cleanup: Controls cleanup of temporary copied data files
- --delete-rounds: Controls cleanup of intermediate round results in outlier detection

Usage:
    # Default behavior - temp files are cleaned up automatically
    recovar pipeline --copy-to-folder /tmp -o /path/to/results ...
    
    # Preserve temp files for chaining multiple calls efficiently
    recovar pipeline --copy-to-folder /tmp -o /path/to/results --no-cleanup ...
    
    # Chain multiple commands without re-copying data
    recovar pipeline --copy-to-folder /tmp -o /path/to/results1 --no-cleanup ...
    recovar analyze /path/to/results1 --outdir /path/to/analyze1 --copy-to-folder /tmp --no-cleanup ...
    recovar analyze /path/to/results1 --outdir /path/to/analyze2 --copy-to-folder /tmp ...  # Last call cleans up
"""

import os
import shutil
import logging
import time
import tempfile
import hashlib
import json

logger = logging.getLogger(__name__)

# Global cache to track copied files
_file_copy_cache = {}

def _get_file_hash(filepath):
    """Get a hash of file metadata (path, size, modification time) for caching."""
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    metadata = f"{filepath}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(metadata.encode()).hexdigest()

def _is_file_cached(filepath, temp_folder):
    """Check if a file is already cached in the temp folder."""
    if filepath not in _file_copy_cache:
        return False
    
    cache_entry = _file_copy_cache[filepath]
    if cache_entry['temp_folder'] != temp_folder:
        return False
    
    # Check if the file has changed since last copy
    current_hash = _get_file_hash(filepath)
    return current_hash == cache_entry['file_hash']

def _update_file_cache(filepath, temp_folder, temp_path):
    """Update the cache with file copy information."""
    file_hash = _get_file_hash(filepath)
    _file_copy_cache[filepath] = {
        'temp_folder': temp_folder,
        'temp_path': temp_path,
        'file_hash': file_hash,
        'copy_time': time.time()
    }

def copy_data_to_temp_folder(args):
    """
    Copy data files to a temporary folder if --copy_to_folder is specified.
    Uses smart caching to avoid redundant copies.
    
    Args:
        args: Parsed arguments containing copy_to_folder path
        
    Returns:
        dict: Dictionary with original paths and temporary paths, or None if no copying needed
    """
    if not hasattr(args, 'copy_to_folder') or args.copy_to_folder is None:
        return None

    if args.copy_to_folder == 'auto':
        # Use Python's tempfile module for robust, unique temp directory creation
        temp_folder = tempfile.mkdtemp(prefix="recovar_copy_")
    else:
        temp_folder = args.copy_to_folder
        os.makedirs(temp_folder, exist_ok=True)
    logger.info(f"Copying data to temporary folder: {temp_folder}")
    
    # Store original and temporary paths
    path_mapping = {
        'original_particles': args.particles,
        'original_poses': args.poses,
        'original_ctf': args.ctf,
        'original_datadir': args.datadir,
        'temp_folder': temp_folder
    }
    
    # Copy particles file with caching
    if args.particles and os.path.exists(args.particles):
        temp_particles = os.path.join(temp_folder, os.path.basename(args.particles))
        if _is_file_cached(args.particles, temp_folder):
            logger.info(f"Particles file already cached, skipping copy: {temp_particles}")
        else:
            shutil.copy2(args.particles, temp_particles)
            _update_file_cache(args.particles, temp_folder, temp_particles)
            logger.info(f"Copied particles file to: {temp_particles}")
        args.particles = temp_particles
        path_mapping['temp_particles'] = temp_particles

        # If particles is a .star file and --copy-to-folder is used, enforce --datadir and check all referenced files
        if temp_particles.endswith('.star'):
            if not args.datadir:
                raise RuntimeError("When using --copy-to-folder with a .star file as particles input, you must also provide --datadir pointing to the directory containing the referenced image files.")
            # Parse the .star file to find referenced image files
            referenced_files = []
            with open(temp_particles, 'r') as f:
                for line in f:
                    if '@' in line:
                        parts = line.strip().split('@')
                        if len(parts) == 2:
                            referenced_files.append(parts[1].split()[0])
            # Apply strip-prefix if present
            strip_prefix = getattr(args, 'strip_prefix', None)
            for ref_file in referenced_files:
                rel_path = ref_file
                if strip_prefix and rel_path.startswith(strip_prefix):
                    rel_path = rel_path[len(strip_prefix):].lstrip('/')
                abs_path = os.path.join(args.datadir, rel_path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"File referenced in star file not found: {abs_path}.\nThis usually means --datadir is incorrect or missing files. Make sure all files referenced in the star file exist in the directory specified by --datadir (after applying --strip-prefix if used).\nFailing file: {ref_file}")
                # Copy the referenced file into the temp folder
                temp_ref_path = os.path.join(temp_folder, os.path.basename(rel_path))
                if not os.path.exists(temp_ref_path):
                    shutil.copy2(abs_path, temp_ref_path)
    
    # Copy poses file with caching
    if args.poses and os.path.exists(args.poses):
        temp_poses = os.path.join(temp_folder, os.path.basename(args.poses))
        if _is_file_cached(args.poses, temp_folder):
            logger.info(f"Poses file already cached, skipping copy: {temp_poses}")
        else:
            shutil.copy2(args.poses, temp_poses)
            _update_file_cache(args.poses, temp_folder, temp_poses)
            logger.info(f"Copied poses file to: {temp_poses}")
        args.poses = temp_poses
        path_mapping['temp_poses'] = temp_poses
    
    # Copy CTF file with caching
    if args.ctf and os.path.exists(args.ctf):
        temp_ctf = os.path.join(temp_folder, os.path.basename(args.ctf))
        if _is_file_cached(args.ctf, temp_folder):
            logger.info(f"CTF file already cached, skipping copy: {temp_ctf}")
        else:
            shutil.copy2(args.ctf, temp_ctf)
            _update_file_cache(args.ctf, temp_folder, temp_ctf)
            logger.info(f"Copied CTF file to: {temp_ctf}")
        args.ctf = temp_ctf
        path_mapping['temp_ctf'] = temp_ctf
    
    # Copy datadir if it exists and is a directory
    if args.datadir and os.path.isdir(args.datadir):
        temp_datadir = os.path.join(temp_folder, 'datadir')
        shutil.copytree(args.datadir, temp_datadir, dirs_exist_ok=True)
        args.datadir = temp_datadir
        path_mapping['temp_datadir'] = temp_datadir
        logger.info(f"Copied datadir to: {temp_datadir}")
    
    # Copy mask file if it exists
    if hasattr(args, 'mask') and args.mask and os.path.exists(args.mask):
        original_mask = args.mask  # Store original path before changing it
        temp_mask = os.path.join(temp_folder, os.path.basename(args.mask))
        shutil.copy2(args.mask, temp_mask)
        args.mask = temp_mask
        path_mapping['original_mask'] = original_mask
        path_mapping['temp_mask'] = temp_mask
        logger.info(f"Copied mask file to: {temp_mask}")
    
    # Copy focus mask file if it exists
    if hasattr(args, 'focus_mask') and args.focus_mask and os.path.exists(args.focus_mask):
        original_focus_mask = args.focus_mask  # Store original path before changing it
        temp_focus_mask = os.path.join(temp_folder, os.path.basename(args.focus_mask))
        shutil.copy2(args.focus_mask, temp_focus_mask)
        args.focus_mask = temp_focus_mask
        path_mapping['original_focus_mask'] = original_focus_mask
        path_mapping['temp_focus_mask'] = temp_focus_mask
        logger.info(f"Copied focus mask file to: {temp_focus_mask}")
    
    # Copy index files if they exist
    if hasattr(args, 'ind') and args.ind and os.path.exists(args.ind):
        original_ind = args.ind  # Store original path before changing it
        temp_ind = os.path.join(temp_folder, os.path.basename(args.ind))
        shutil.copy2(args.ind, temp_ind)
        args.ind = temp_ind
        path_mapping['original_ind'] = original_ind
        path_mapping['temp_ind'] = temp_ind
        logger.info(f"Copied index file to: {temp_ind}")
    
    if hasattr(args, 'tilt_ind') and args.tilt_ind and os.path.exists(args.tilt_ind):
        original_tilt_ind = args.tilt_ind  # Store original path before changing it
        temp_particle_ind = os.path.join(temp_folder, os.path.basename(args.tilt_ind))
        shutil.copy2(args.tilt_ind, temp_particle_ind)
        args.tilt_ind = temp_particle_ind
        path_mapping['original_tilt_ind'] = original_tilt_ind
        path_mapping['temp_particle_ind'] = temp_particle_ind
        logger.info(f"Copied particle index file to: {temp_particle_ind}")
    
    if hasattr(args, 'halfsets') and args.halfsets and os.path.exists(args.halfsets):
        original_halfsets = args.halfsets  # Store original path before changing it
        temp_halfsets = os.path.join(temp_folder, os.path.basename(args.halfsets))
        shutil.copy2(args.halfsets, temp_halfsets)
        args.halfsets = temp_halfsets
        path_mapping['original_halfsets'] = original_halfsets
        path_mapping['temp_halfsets'] = temp_halfsets
        logger.info(f"Copied halfsets file to: {temp_halfsets}")
    
    logger.info("Data copying completed")
    return path_mapping


def copy_data_from_pipeline_output(pipeline_output, copy_to_folder_path):
    """
    Copy data from pipeline output to a new temporary folder.
    
    This function loads the input_args from a pipeline output and copies the data
    to a new temporary location specified by copy_to_folder_path.
    
    Args:
        pipeline_output: PipelineOutput object or path to pipeline output directory
        copy_to_folder_path: Path to the new temporary folder for copying data
        
    Returns:
        dict: Path mapping dictionary, or None if no copying needed
    """
    from recovar import output as o
    
    # Load pipeline output if path is provided
    if isinstance(pipeline_output, str):
        po = o.PipelineOutput(pipeline_output + '/')
    else:
        po = pipeline_output
    
    # Get input_args from pipeline output
    input_args = po.get('input_args')
    if input_args is None:
        logger.warning("No input_args found in pipeline output, skipping data copy")
        return None
    
    # Set the new copy_to_folder path and copy data
    input_args.copy_to_folder = copy_to_folder_path
    return copy_data_to_temp_folder(input_args)


def save_original_paths_info(path_mapping, output_dir):
    """
    Save original paths information to a text file.
    
    Args:
        path_mapping: Dictionary containing original and temporary paths
        output_dir: Output directory where to save the paths file
    """
    if path_mapping is None:
        return
    
    paths_file = os.path.join(output_dir, "original_paths.txt")
    with open(paths_file, "w") as text_file:
        text_file.write("Original data paths (before copying to temporary folder):\n")
        text_file.write(f"Original particles file: {path_mapping['original_particles']}\n")
        text_file.write(f"Original poses file: {path_mapping['original_poses']}\n")
        text_file.write(f"Original CTF file: {path_mapping['original_ctf']}\n")
        if path_mapping['original_datadir']:
            text_file.write(f"Original datadir: {path_mapping['original_datadir']}\n")
        text_file.write(f"Temporary folder: {path_mapping['temp_folder']}\n")
        text_file.write("\nTemporary paths used during processing:\n")
        for key, value in path_mapping.items():
            if key.startswith('temp_'):
                text_file.write(f"{key}: {value}\n")
    
    logger.info(f"Original paths saved to: {paths_file}")


def copy_specific_files_to_temp(file_paths, temp_folder, file_types=None):
    """
    Copy specific files to a temporary folder.
    
    Args:
        file_paths: Dictionary mapping file types to file paths
        temp_folder: Temporary folder to copy files to
        file_types: List of file types to copy (if None, copy all)
        
    Returns:
        dict: Dictionary with original and temporary paths
    """
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder, exist_ok=True)
    
    path_mapping = {'temp_folder': temp_folder}
    
    for file_type, file_path in file_paths.items():
        if file_types is not None and file_type not in file_types:
            continue
            
        if file_path and os.path.exists(file_path):
            if os.path.isdir(file_path):
                # Copy directory
                temp_path = os.path.join(temp_folder, os.path.basename(file_path))
                shutil.copytree(file_path, temp_path, dirs_exist_ok=True)
                path_mapping[f'original_{file_type}'] = file_path
                path_mapping[f'temp_{file_type}'] = temp_path
                logger.info(f"Copied directory {file_type} to: {temp_path}")
            else:
                # Copy file
                temp_path = os.path.join(temp_folder, os.path.basename(file_path))
                shutil.copy2(file_path, temp_path)
                path_mapping[f'original_{file_type}'] = file_path
                path_mapping[f'temp_{file_type}'] = temp_path
                logger.info(f"Copied file {file_type} to: {temp_path}")
    
    return path_mapping


def clear_file_copy_cache():
    """Clear the file copy cache."""
    global _file_copy_cache
    _file_copy_cache.clear()
    logger.info("File copy cache cleared")

def get_cache_stats():
    """Get statistics about the file copy cache."""
    return {
        'cached_files': len(_file_copy_cache),
        'cache_entries': list(_file_copy_cache.keys())
    }

def cleanup_temp_files(path_mapping):
    """
    Clean up temporary files and folders.
    
    Args:
        path_mapping: Dictionary containing temporary paths
    """
    if path_mapping is None:
        return
    
    temp_folder = path_mapping.get('temp_folder')
    if temp_folder and os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
            logger.info(f"Cleaned up temporary folder: {temp_folder}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary folder {temp_folder}: {e}")
            # Try to remove individual files if rmtree fails
            try:
                for root, dirs, files in os.walk(temp_folder, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except:
                            pass
                os.rmdir(temp_folder)
                logger.info(f"Cleaned up temporary folder using fallback method: {temp_folder}")
            except Exception as e2:
                logger.error(f"Failed to clean up temporary folder {temp_folder} even with fallback: {e2}")
    
    # Clear cache entries for this temp folder
    global _file_copy_cache
    files_to_remove = []
    for filepath, cache_entry in _file_copy_cache.items():
        if cache_entry['temp_folder'] == temp_folder:
            files_to_remove.append(filepath)
    
    for filepath in files_to_remove:
        del _file_copy_cache[filepath]
    
    if files_to_remove:
        logger.info(f"Cleared {len(files_to_remove)} cache entries for temp folder: {temp_folder}") 