"""
Junk Particle Detection Command

This module provides functionality to detect junk particles from latent space using clustering and FSC analysis.

Features:
- K-means clustering on latent embeddings
- FSC-based quality assessment for each cluster
- Multiple outlier detection methods (half-map FSC, vs-mean FSC)
- Comprehensive visualization and analysis plots
- Optional reconstruction saving (halfmaps and combined volumes as MRC files)

Usage:
    python -m recovar.commands.junk_particle_detection input_dir [options]

New in this version:
- Added --save-reconstructions flag to save cluster reconstructions as MRC files
- Reconstructions include halfmap1.mrc, halfmap2.mrc, combined.mrc, and particle_indices.npz
- Each cluster gets its own subdirectory under reconstructions/
- Reconstruction metadata is saved in reconstructions_info_{zdim_key}.pkl
"""

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans
from recovar import utils
from recovar.reconstruction import relion_functions
from recovar.output import output, plot_utils


def _safe_savefig(filepath, **kwargs):
    """Save the current figure, handling tight_layout overflow gracefully.

    When tight_layout fails to fit all decorations (common with degenerate data),
    bbox_inches='tight' can compute an absurdly large bounding box. This helper
    catches the resulting error and retries without bbox_inches='tight'.
    (ValueError for oversized pixel dimensions, TypeError for int overflow in RendererAgg.)
    """
    try:
        plt.savefig(filepath, **kwargs)
    except (ValueError, TypeError):
        kwargs.pop('bbox_inches', None)
        plt.savefig(filepath, **kwargs)
import recovar.core.fourier_transform_utils as fourier_transform_utils
import jax.numpy as jnp
import seaborn as sns
import mrcfile
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import umap
import argparse

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

logger = logging.getLogger(__name__)


def compute_fsc_auc(fsc_curve, grid_size, voxel_size, threshold=1/7):
    """
    Compute the Area Under Curve (AUC) of the FSC curve above threshold.
    Higher AUC indicates better quality.
    """
    freq = fourier_transform_utils.get_1d_frequency_grid(2*grid_size, voxel_size=0.5*voxel_size, scaled=True)
    freq = freq[freq >= 0]
    
    # Ensure both arrays have the same length
    min_length = min(len(freq), len(fsc_curve))
    freq = freq[:min_length]
    fsc_curve = fsc_curve[:min_length]
    
    # Skip the first element (DC component)
    if min_length > 1:
        freq = freq[1:]
        fsc_curve = fsc_curve[1:]
    else:
        return 0.0

    # Find where FSC drops below threshold
    above_threshold = fsc_curve >= threshold
    if np.all(above_threshold):
        return np.trapezoid(fsc_curve, freq)

    if np.all(~above_threshold):
        return 0.0

    # Find first index below threshold
    idx = np.argmin(above_threshold)
    if idx == 0:
        return 0.0

    # Compute AUC up to that point
    auc = np.trapezoid(fsc_curve[:idx], freq[:idx])
    return auc


def compute_cluster_fsc_scores(pipeline_output, cluster_centers, cluster_indices, 
                              zdim_key, batch_size=100, n_particles_per_cluster=10, 
                              save_reconstructions=False, output_folder=None, 
                              filter_resolution=None, filter_fourier_shells=10):
    """
    Compute FSC scores for each cluster by generating halfmaps and comparing them.
    
    Parameters:
    - pipeline_output: PipelineOutput object
    - cluster_centers: K-means cluster centers
    - cluster_indices: Cluster assignments for each particle
    - zdim_key: Dimension key for embeddings
    - batch_size: Batch size for reconstruction
    - n_particles_per_cluster: Number of particles to use per halfmap (so 2*n_particles_per_cluster total)
    - save_reconstructions: Whether to save reconstructions to file
    - output_folder: Output folder for saving reconstructions
    - filter_resolution: Resolution in Angstroms to filter combined reconstructions (if None, no filtering)
    - filter_fourier_shells: Number of Fourier shells to keep when filtering (default: 10)
    
    Returns:
    - fsc_scores: Dictionary with FSC scores for each cluster
    - fsc_auc_scores: Dictionary with FSC AUC scores for each cluster
    - particle_usage: Dictionary with particle indices used for each cluster
    - reconstructions: Dictionary with reconstructions (if save_reconstructions=True)
    
    Note:
    - Uses the n_particles_per_cluster closest particles in latent space to each cluster center
    - Particles can come from any cluster if they're closer in latent space
    - Creates two halfmaps with n_particles_per_cluster particles each
    - Uses relion-style reconstruction with existing dataset splits from pipeline output
    - Maps global particle indices to local indices for each halfset dataset
    - If filter_resolution is provided, combined reconstructions are low-pass filtered
    """
    cryos = pipeline_output.get('dataset')  # This returns [cryo1, cryo2] for the two halfsets
    zs = pipeline_output.get('zs')[zdim_key]
    volume_shape = pipeline_output.get('volume_shape')
    voxel_size = pipeline_output.get('voxel_size')
    
    # Get mean reconstruction for comparison
    mean_volume = pipeline_output.get('mean').reshape(volume_shape)
    mean_real = fourier_transform_utils.get_idft3(mean_volume)
    
    fsc_scores = {}
    fsc_auc_scores = {}
    particle_usage = {}
    reconstructions = {}
    
    # Create reconstructions directory if saving
    if save_reconstructions and output_folder is not None:
        reconstructions_dir = os.path.join(output_folder, 'reconstructions')
        os.makedirs(reconstructions_dir, exist_ok=True)
        logger.info(f"Saving reconstructions to {reconstructions_dir}")
        
        # Create single folder for all individual .mrc files
        individual_mrc_dir = os.path.join(output_folder, 'individual_mrc_files')
        os.makedirs(individual_mrc_dir, exist_ok=True)
        logger.info(f"Saving individual .mrc files to {individual_mrc_dir}")
        
        # Initialize list to collect all combined reconstructions for stacking
        all_combined_reconstructions = []
        
        # Track file paths for individual .mrc files
        individual_mrc_files = []
    
    for cluster_idx in range(len(cluster_centers)):
        logger.info(f"Processing cluster {cluster_idx + 1}/{len(cluster_centers)}")
        
        cluster_center = cluster_centers[cluster_idx]
        
        # Create temporary directory for this cluster
        temp_dir = f"/tmp/cluster_{cluster_idx}_reconstruction"
        os.makedirs(temp_dir, exist_ok=True)
        
        halfmaps = [None, None]
        zs_subsets = [ zs[:cryos[0].n_units], zs[cryos[0].n_units:] ]
        used_particles = [[], []]  # Track which particles are used for each halfmap
        
        for i, zs_subset in enumerate(zs_subsets):
            distances = np.linalg.norm(zs_subset - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:n_particles_per_cluster]
            used_particles[i] = closest_indices.copy()
            
            # Map to global indices
            if i == 0:
                global_indices = closest_indices
            else:
                global_indices = closest_indices + cryos[0].n_units
            used_particles[i] = global_indices
            
            logger.info(f"Cluster {cluster_idx}: Using {len(closest_indices)} closest particles for half-map {i} (min distance: {distances[closest_indices[0]]:.3f}, max distance: {distances[closest_indices[-1]]:.3f}). Average distance over all particles: {np.mean(distances):.3f}")
            
            if cryos[i].tilt_series_flag:
                # For tilt series, noise must be fetched per-batch because
                # closest_indices are particle indices but noise.get() needs
                # image indices. Pass None to let relion_style_triangular_kernel
                # fetch noise per-batch from the dataset's noise model.
                noise_variance = None
            else:
                noise_variance = cryos[i].noise.get(closest_indices)
                if noise_variance is None:
                    noise_variance = np.ones(len(closest_indices), dtype=np.float32)

            cryos[i].update_volume_upsampling_factor(2)
            Ft_ctf, F_ty = relion_functions.relion_style_triangular_kernel(
                cryos[i], noise_variance, batch_size=None,
                disc_type='linear_interp',
                data_generator=cryos[i].get_dataset_subset_generator(batch_size, closest_indices, mode = 'images')
            )
            halfmap = relion_functions.post_process_from_filter(
                cryos[i], Ft_ctf, F_ty, 
                disc_type='linear_interp', use_spherical_mask=True, 
                grid_correct=True, gridding_correct="square"
            )
            halfmaps[i] = halfmap

        if halfmaps[0] is None or halfmaps[1] is None:
            raise Exception("Half-map generation failed")

        # Convert to real space for FSC computation
        halfmap1_real = fourier_transform_utils.get_idft3(halfmaps[0].reshape(volume_shape))
        halfmap2_real = fourier_transform_utils.get_idft3(halfmaps[1].reshape(volume_shape))
        
        # Compute combined reconstruction
        combined_recon = (halfmaps[0] + halfmaps[1]) / 2
        combined_real = fourier_transform_utils.get_idft3(combined_recon.reshape(volume_shape))
        
        # Apply low-pass filtering if requested
        if filter_resolution is not None:
            from recovar.reconstruction import regularization
            
            # Convert resolution to frequency
            freq_threshold = 1.0 / filter_resolution  # 1/Angstrom
            
            # Get frequency grid
            freq_grid = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size=voxel_size, scaled=True)
            freq_magnitude = jnp.linalg.norm(freq_grid, axis=-1)
            
            # Create low-pass filter
            # Keep only the first filter_fourier_shells shells
            max_freq = freq_threshold * (filter_fourier_shells / 10.0)  # Scale based on requested shells
            low_pass_filter = (freq_magnitude <= max_freq).astype(np.float32)
            
            # Apply filter in Fourier space
            combined_recon_filtered = combined_recon * low_pass_filter.reshape(-1)
            combined_real_filtered = fourier_transform_utils.get_idft3(combined_recon_filtered.reshape(volume_shape))
            
            logger.info(f"Cluster {cluster_idx}: Applied low-pass filter at {filter_resolution:.1f} Angstroms ({filter_fourier_shells} Fourier shells)")
            
            # Use filtered version for saving
            combined_real = combined_real_filtered
            combined_recon = combined_recon_filtered
        
        # Save reconstructions if requested
        if save_reconstructions and output_folder is not None:
            cluster_reconstructions_dir = os.path.join(reconstructions_dir, f'cluster_{cluster_idx:03d}')
            os.makedirs(cluster_reconstructions_dir, exist_ok=True)
            
            # Save halfmaps
            halfmap1_path = os.path.join(cluster_reconstructions_dir, 'halfmap1.mrc')
            halfmap2_path = os.path.join(cluster_reconstructions_dir, 'halfmap2.mrc')
            
            # Save as MRC files
            with mrcfile.new(halfmap1_path, overwrite=True) as mrc:
                mrc.set_data(halfmap1_real.astype(np.float32))
                mrc.voxel_size = voxel_size
            with mrcfile.new(halfmap2_path, overwrite=True) as mrc:
                mrc.set_data(halfmap2_real.astype(np.float32))
                mrc.voxel_size = voxel_size
            
            # Save combined reconstruction
            combined_path = os.path.join(cluster_reconstructions_dir, 'combined.mrc')
            with mrcfile.new(combined_path, overwrite=True) as mrc:
                mrc.set_data(combined_real.astype(np.float32))
                mrc.voxel_size = voxel_size
            
            # Save particle indices used
            particle_indices_path = os.path.join(cluster_reconstructions_dir, 'particle_indices.npz')
            np.savez(particle_indices_path, 
                    halfmap1_particles=used_particles[0],
                    halfmap2_particles=used_particles[1],
                    all_particles=np.concatenate([used_particles[0], used_particles[1]]))
            
            # Save individual .mrc files in single folder with volXXXX.mrc naming
            vol_idx = cluster_idx * 1  # Each cluster gets 1 volume (combined only)
            
            # Save combined as volXXXX.mrc
            vol_combined_path = os.path.join(individual_mrc_dir, f'vol{vol_idx:04d}.mrc')
            with mrcfile.new(vol_combined_path, overwrite=True) as mrc:
                mrc.set_data(combined_real.astype(np.float32))
                mrc.voxel_size = voxel_size
            individual_mrc_files.append({
                'path': vol_combined_path,
                'type': 'combined',
                'cluster': cluster_idx,
                'vol_index': vol_idx
            })
            
            # Store reconstruction info
            reconstructions[cluster_idx] = {
                'halfmap1_path': halfmap1_path,
                'halfmap2_path': halfmap2_path,
                'combined_path': combined_path,
                'particle_indices_path': particle_indices_path,
                'halfmap1_real': halfmap1_real,
                'halfmap2_real': halfmap2_real,
                'combined_real': combined_real,
                'individual_mrc_files': {
                    'combined': vol_combined_path
                }
            }
            
            # Collect combined reconstruction for stacking
            all_combined_reconstructions.append(combined_real.astype(np.float32))
            
            logger.info(f"Saved reconstructions for cluster {cluster_idx} to {cluster_reconstructions_dir}")
            logger.info(f"Saved individual .mrc files: vol{vol_idx:04d}.mrc (combined)")
        
        # Compute FSC between halfmaps
        fsc_curve = plot_utils.FSC(halfmaps[0], halfmaps[1])
        fsc_score = plot_utils.fsc_score(fsc_curve, volume_shape[0], voxel_size, threshold=1/7)
        fsc_auc = compute_fsc_auc(fsc_curve, volume_shape[0], voxel_size, threshold=1/7)
        
        # Compute FSC between cluster reconstruction and mean
        fsc_vs_mean_curve = plot_utils.FSC(combined_recon, mean_volume)
        fsc_vs_mean_score = plot_utils.fsc_score(fsc_vs_mean_curve, volume_shape[0], voxel_size, threshold=1/7)
        fsc_vs_mean_auc = compute_fsc_auc(fsc_vs_mean_curve, volume_shape[0], voxel_size, threshold=1/7)
        
        fsc_scores[cluster_idx] = {
            'halfmap_fsc': fsc_score,
            'vs_mean_fsc': fsc_vs_mean_score,
            'halfmap_curve': fsc_curve,
            'vs_mean_curve': fsc_vs_mean_curve
        }
        
        fsc_auc_scores[cluster_idx] = {
            'halfmap_auc': fsc_auc,
            'vs_mean_auc': fsc_vs_mean_auc
        }
        
        particle_usage[cluster_idx] = {
            'halfmap1_particles': used_particles[0],
            'halfmap2_particles': used_particles[1],
            'all_particles': np.concatenate([used_particles[0], used_particles[1]])
        }
        
        logger.info(f"Cluster {cluster_idx}: FSC={fsc_score:.3f}, AUC={fsc_auc:.3f}, vs-mean_FSC={fsc_vs_mean_score:.3f}")
            
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Save all combined reconstructions in a single stacked file
    if save_reconstructions and output_folder is not None and len(all_combined_reconstructions) > 0:
        stacked_path = os.path.join(reconstructions_dir, 'all_combined_reconstructions.mrc')
        
        # Stack all reconstructions along a new axis
        stacked_reconstructions = np.stack(all_combined_reconstructions, axis=0)
        
        with mrcfile.new(stacked_path, overwrite=True) as mrc:
            mrc.set_data(stacked_reconstructions)
            mrc.voxel_size = voxel_size
        
        logger.info(f"Saved all {len(all_combined_reconstructions)} combined reconstructions to {stacked_path}")
        logger.info(f"Stacked volume shape: {stacked_reconstructions.shape}")
        
        # Add stacked file info to reconstructions
        reconstructions['stacked_file'] = {
            'path': stacked_path,
            'shape': stacked_reconstructions.shape,
            'n_clusters': len(all_combined_reconstructions)
        }
        
        # Add individual .mrc files info
        reconstructions['individual_mrc_files'] = {
            'directory': individual_mrc_dir,
            'files': individual_mrc_files,
            'total_files': len(individual_mrc_files),
            'n_clusters': len(cluster_centers),
            'files_per_cluster': 1  # combined only
        }
        
        logger.info(f"Saved {len(individual_mrc_files)} individual .mrc files to {individual_mrc_dir}")
        logger.info(f"File naming: vol0000.mrc (cluster 0 combined), vol0001.mrc (cluster 1 combined), etc.")
    
    if save_reconstructions:
        return fsc_scores, fsc_auc_scores, particle_usage, reconstructions
    else:
        return fsc_scores, fsc_auc_scores, particle_usage


def plot_junk_detection_results(zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, 
                               particle_usage, output_folder, zdim_key):
    """
    Create plots for junk particle detection results with improved styling and clarity.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up improved styling based on output.py patterns
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Use professional color scheme
    colors = {
        'background': '#E0E0E0',  # Light gray
        'good': '#2E8B57',        # Sea green
        'junk': '#DC143C',        # Crimson red
        'mean': '#4169E1',        # Royal blue
        'grid': '#F0F0F0',        # Very light gray
        'scatter': 'cornflowerblue'
    }
    
    # Extract FSC scores
    halfmap_fscs = [fsc_scores[i]['halfmap_fsc'] for i in range(len(cluster_centers))]
    vs_mean_fscs = [fsc_scores[i]['vs_mean_fsc'] for i in range(len(cluster_centers))]
    halfmap_aucs = [fsc_auc_scores[i]['halfmap_auc'] for i in range(len(cluster_centers))]
    vs_mean_aucs = [fsc_auc_scores[i]['vs_mean_auc'] for i in range(len(cluster_centers))]

    # --- Save all FSC curves ---
    all_fsc_curves = [fsc_scores[i]['halfmap_curve'] for i in range(len(cluster_centers))]
    all_vs_mean_curves = [fsc_scores[i]['vs_mean_curve'] for i in range(len(cluster_centers))]
    
    with open(os.path.join(output_folder, f'all_fsc_curves_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(all_fsc_curves, f)
    with open(os.path.join(output_folder, f'all_vs_mean_fsc_curves_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(all_vs_mean_curves, f)

    # --- Plot all half-map FSC curves with improved styling ---
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(all_fsc_curves[0]))
    
    # Create frequency axis (assuming Nyquist frequency)
    freq_axis = x / (2 * len(all_fsc_curves[0]))
    
    # Use hexbin density plot for background like in output.py
    # try:
    #     # Create 2D histogram of FSC curves vs frequency
    curve_data = np.array(all_fsc_curves)
    freq_mesh, curve_mesh = np.meshgrid(freq_axis, np.arange(len(curve_data)))
    hb = ax.hexbin(freq_mesh.flatten(), curve_data.flatten(), 
                  gridsize=30, cmap='Blues', alpha=0.3, mincnt=1)
    # except:
    #     pass
    
    # Color code curves by FSC score with better transparency
    colors_curves = cm.viridis(np.array(halfmap_fscs))
    for i, curve in enumerate(all_fsc_curves):
        ax.plot(freq_axis, curve, color=colors_curves[i], alpha=0.3, linewidth=0.5)
    
    # Mean and IQR with better styling
    all_fsc_array = np.array(all_fsc_curves)
    mean_curve = np.mean(all_fsc_array, axis=0)
    q25 = np.percentile(all_fsc_array, 25, axis=0)
    q75 = np.percentile(all_fsc_array, 75, axis=0)
    ax.plot(freq_axis, mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25, q75, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)', fontweight='bold')
    ax.set_ylabel('Fourier Shell Correlation', fontweight='bold')
    ax.set_title(f'Half-map FSC Curves for All Clusters (n={len(cluster_centers)})', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=mcolors.Normalize(vmin=min(halfmap_fscs), vmax=max(halfmap_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Half-map FSC Score', fontweight='bold')
    
    plt.tight_layout()
    _safe_savefig(os.path.join(output_folder, f'all_halfmap_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot all vs-mean FSC curves with improved styling ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use hexbin density plot for background
    # try:
    curve_data = np.array(all_vs_mean_curves)
    freq_mesh, curve_mesh = np.meshgrid(freq_axis, np.arange(len(curve_data)))
    hb = ax.hexbin(freq_mesh.flatten(), curve_data.flatten(), 
                  gridsize=30, cmap='Blues', alpha=0.3, mincnt=1)
    # except:
    #     pass
    
    # Color code curves by vs-mean FSC score
    colors_curves = cm.viridis(np.array(vs_mean_fscs))
    for i, curve in enumerate(all_vs_mean_curves):
        ax.plot(freq_axis, curve, color=colors_curves[i], alpha=0.3, linewidth=0.5)
    
    # Mean and IQR
    all_vs_mean_array = np.array(all_vs_mean_curves)
    mean_vs_mean_curve = np.mean(all_vs_mean_array, axis=0)
    q25_vs_mean = np.percentile(all_vs_mean_array, 25, axis=0)
    q75_vs_mean = np.percentile(all_vs_mean_array, 75, axis=0)
    ax.plot(freq_axis, mean_vs_mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25_vs_mean, q75_vs_mean, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)', fontweight='bold')
    ax.set_ylabel('Fourier Shell Correlation', fontweight='bold')
    ax.set_title(f'vs-Mean FSC Curves for All Clusters (n={len(cluster_centers)})', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(vs_mean_fscs), vmax=max(vs_mean_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('vs-Mean FSC Score', fontweight='bold')
    
    plt.tight_layout()
    _safe_savefig(os.path.join(output_folder, f'all_vs_mean_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Individual cluster FSC plots (top 10 and bottom 10) with better layout ---
    sorted_indices = np.argsort(halfmap_fscs)
    top_10 = sorted_indices[-10:]
    bottom_10 = sorted_indices[:10]
    
    # Plot top 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Top 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98, fontweight='bold')
    
    for i, cluster_idx in enumerate(top_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {len(sorted_indices)-i})', fontsize=10, fontweight='bold')
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
        ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    _safe_savefig(os.path.join(output_folder, f'top_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot bottom 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Bottom 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98, fontweight='bold')
    
    for i, cluster_idx in enumerate(bottom_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {i+1})', fontsize=10, fontweight='bold')
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
        ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    _safe_savefig(os.path.join(output_folder, f'bottom_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pad to at least 2 columns for 2D scatter/hexbin plots
    zs_2d = zs if zs.shape[1] >= 2 else np.column_stack([zs, np.zeros(len(zs))])
    centers_2d = (cluster_centers if cluster_centers.shape[1] >= 2
                  else np.column_stack([cluster_centers, np.zeros(len(cluster_centers))]))

    # --- Create main summary plot with improved hexbin visualizations ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Junk Particle Detection Summary (zdim={zdim_key}, n_clusters={len(cluster_centers)})',
                 fontsize=20, y=0.95, fontweight='bold')

    # 1. Latent space colored by cluster (hexbin density plot like output.py)
    ax = axes[0, 0]

    # Create hexbin density plot for background
    hb = ax.hexbin(zs_2d[:, 0], zs_2d[:, 1], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)

    # Main scatter plot with improved styling (like output.py)
    ax.scatter(zs_2d[:, 0], zs_2d[:, 1], s=1, alpha=0.6, c=colors['scatter'], edgecolors='none', rasterized=True)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=100,
               linewidth=2, label='Cluster Centers', zorder=10)
    ax.set_title('Latent Space by Cluster ID', fontweight='bold')
    ax.set_xlabel('z₁', fontweight='bold')
    ax.set_ylabel('z₂', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')

    # 2. Latent space colored by Half-map FSC AUC (hexbin)
    ax = axes[0, 1]
    particle_halfmap_aucs = np.array([halfmap_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs_2d[:, 0], zs_2d[:, 1], C=particle_halfmap_aucs, gridsize=30,
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean Half-map FSC AUC', fontweight='bold')
    ax.set_title('Latent Space by Half-map FSC AUC', fontweight='bold')
    ax.set_xlabel('z₁', fontweight='bold')
    ax.set_ylabel('z₂', fontweight='bold')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')

    # 3. Latent space colored by FSC vs Mean AUC (hexbin)
    ax = axes[1, 0]
    particle_vs_mean_aucs = np.array([vs_mean_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs_2d[:, 0], zs_2d[:, 1], C=particle_vs_mean_aucs, gridsize=30,
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean FSC vs Mean AUC', fontweight='bold')
    ax.set_title('Latent Space by FSC vs Mean AUC', fontweight='bold')
    ax.set_xlabel('z₁', fontweight='bold')
    ax.set_ylabel('z₂', fontweight='bold')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # 4. Histogram of particle counts per cluster with improved styling
    ax = axes[1, 1]
    cluster_counts = np.bincount(cluster_indices)
    
    # Use adaptive binning for better histogram readability
    n_clusters = len(cluster_counts)
    n_bins = min(30, max(15, n_clusters // 3))  # Adaptive bin count
    bins = np.linspace(cluster_counts.min(), cluster_counts.max(), n_bins + 1)
    
    # Plot histogram with better styling
    ax.hist(cluster_counts, bins=bins, color='skyblue', alpha=0.8, edgecolor='black', 
            linewidth=0.8, density=True, hatch='///')
    ax.axvline(np.mean(cluster_counts), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(cluster_counts):.1f}')
    ax.set_title('Particle Counts per Cluster', fontweight='bold')
    ax.set_xlabel('Number of Particles', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Particle Usage Visualization (simplified and cleaner) ---
    if zs.shape[1] >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Particle Usage Analysis', fontsize=18, y=0.95, fontweight='bold')
        
        # Plot 1: All particles density (hexbin)
        ax = axes[0, 0]
        hb = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=30, cmap='Blues', mincnt=1, alpha=0.6)
        ax.scatter(zs[:, 0], zs[:, 1], c=cluster_indices, 
                  cmap=matplotlib.colors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, min(len(cluster_centers), 20)))), 
                  s=0.5, alpha=0.7, rasterized=True)
        ax.set_title('All Particles Density', fontweight='bold')
        ax.set_xlabel('z₁', fontweight='bold')
        ax.set_ylabel('z₂', fontweight='bold')
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 2: Top 5 clusters - particles used for FSC
        ax = axes[0, 1]
        top_5 = sorted_indices[-5:]
        colors_top5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=30, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(top_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs[used_particles, 0], zs[used_particles, 1], 
                      c=[colors_top5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers[top_5, 0], cluster_centers[top_5, 1], 
                  c=colors_top5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Top 5 Clusters - Particles Used for FSC', fontweight='bold')
        ax.set_xlabel('z₁', fontweight='bold')
        ax.set_ylabel('z₂', fontweight='bold')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 3: Bottom 5 clusters - particles used for FSC
        ax = axes[0, 2]
        bottom_5 = sorted_indices[:5]
        colors_bottom5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=30, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(bottom_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs[used_particles, 0], zs[used_particles, 1], 
                      c=[colors_bottom5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers[bottom_5, 0], cluster_centers[bottom_5, 1], 
                  c=colors_bottom5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Bottom 5 Clusters - Particles Used for FSC', fontweight='bold')
        ax.set_xlabel('z₁', fontweight='bold')
        ax.set_ylabel('z₂', fontweight='bold')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 4: Usage statistics with improved histogram
        ax = axes[1, 0]
        
        # Calculate usage statistics
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        # Create histogram of usage counts with better styling
        unique_counts, count_frequencies = np.unique(usage_counts, return_counts=True)
        
        bars = ax.bar(unique_counts, count_frequencies, color='skyblue', edgecolor='black', 
                     alpha=0.8, linewidth=1)
        ax.set_xlabel('Number of Clusters Using Particle', fontweight='bold')
        ax.set_ylabel('Number of Particles', fontweight='bold')
        ax.set_title('Particle Usage Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Add statistics text with better formatting
        mean_usage = np.mean(usage_counts)
        median_usage = np.median(usage_counts)
        unused_particles = np.sum(usage_counts == 0)
        total_particles = len(usage_counts)
        
        stats_text = f'Mean usage: {mean_usage:.1f}\nMedian usage: {median_usage:.1f}\nUnused particles: {unused_particles}\n({unused_particles/total_particles*100:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10, fontweight='bold')
        
        # Plot 5: FSC Score Distribution with improved histogram
        ax = axes[1, 1]
        
        # Use adaptive binning for better histogram readability
        n_bins_fsc = min(30, max(15, len(halfmap_fscs) // 3))
        bins_fsc = np.linspace(min(halfmap_fscs), max(halfmap_fscs), n_bins_fsc + 1)
        
        ax.hist(halfmap_fscs, bins=bins_fsc, color='blue', alpha=0.8, edgecolor='black', 
                linewidth=0.8, density=True, hatch='///')
        ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {np.mean(halfmap_fscs):.3f}')
        ax.set_xlabel('Half-map FSC Score', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Distribution of Half-map FSC Scores', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 6: FSC vs Mean FSC comparison with improved scatter
        ax = axes[1, 2]
        
        # Create hexbin density plot for background
        # try:
        hb = ax.hexbin(halfmap_fscs, vs_mean_fscs, gridsize=20, cmap='Blues', alpha=0.3, mincnt=1)
        # except:
        #     pass
        
        # Main scatter plot with improved styling
        scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
        ax.set_xlabel('Half-map FSC', fontweight='bold')
        ax.set_ylabel('vs-Mean FSC', fontweight='bold')
        ax.set_title('Half-map FSC vs vs-Mean FSC', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        _safe_savefig(os.path.join(output_folder, f'particle_usage_visualization_{zdim_key}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # --- Create comprehensive analysis plots with improved styling ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FSC Score Analysis', fontsize=16, y=0.95, fontweight='bold')
    
    # Plot 1: Half-map FSC histogram with improved styling
    ax = axes[0, 0]
    n_bins_fsc = min(30, max(15, len(halfmap_fscs) // 3))
    bins_fsc = np.linspace(min(halfmap_fscs), max(halfmap_fscs), n_bins_fsc + 1)
    
    ax.hist(halfmap_fscs, bins=bins_fsc, color='blue', alpha=0.8, edgecolor='black', 
            linewidth=0.8, density=True, hatch='///')
    ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    ax.set_xlabel('Half-map FSC Score', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Distribution of Half-map FSC Scores', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 2: vs-Mean FSC histogram with improved styling
    ax = axes[0, 1]
    n_bins_vs_mean = min(30, max(15, len(vs_mean_fscs) // 3))
    bins_vs_mean = np.linspace(min(vs_mean_fscs), max(vs_mean_fscs), n_bins_vs_mean + 1)
    
    ax.hist(vs_mean_fscs, bins=bins_vs_mean, color='green', alpha=0.8, edgecolor='black', 
            linewidth=0.8, density=True, hatch='///')
    ax.axvline(np.mean(vs_mean_fscs), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(vs_mean_fscs):.3f}')
    ax.set_xlabel('vs-Mean FSC Score', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Distribution of vs-Mean FSC Scores', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 3: Half-map FSC AUC histogram with improved styling
    ax = axes[0, 2]
    n_bins_auc = min(30, max(15, len(halfmap_aucs) // 3))
    bins_auc = np.linspace(min(halfmap_aucs), max(halfmap_aucs), n_bins_auc + 1)
    
    ax.hist(halfmap_aucs, bins=bins_auc, color='orange', alpha=0.8, edgecolor='black', 
            linewidth=0.8, density=True, hatch='///')
    ax.axvline(np.mean(halfmap_aucs), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(halfmap_aucs):.3f}')
    ax.set_xlabel('Half-map FSC AUC', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Distribution of Half-map FSC AUC Scores', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 4: vs-Mean FSC AUC histogram with improved styling
    ax = axes[1, 0]
    n_bins_vs_mean_auc = min(30, max(15, len(vs_mean_aucs) // 3))
    bins_vs_mean_auc = np.linspace(min(vs_mean_aucs), max(vs_mean_aucs), n_bins_vs_mean_auc + 1)
    
    ax.hist(vs_mean_aucs, bins=bins_vs_mean_auc, color='purple', alpha=0.8, edgecolor='black', 
            linewidth=0.8, density=True, hatch='///')
    ax.axvline(np.mean(vs_mean_aucs), color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(vs_mean_aucs):.3f}')
    ax.set_xlabel('vs-Mean FSC AUC', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Distribution of vs-Mean FSC AUC Scores', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 5: Half-map FSC vs Half-map FSC AUC scatter with improved styling
    ax = axes[1, 1]
    
    # Create hexbin density plot for background
    # try:
    hb = ax.hexbin(halfmap_fscs, halfmap_aucs, gridsize=20, cmap='Blues', alpha=0.3, mincnt=1)
    # except:
    #     pass
    
    scatter = ax.scatter(halfmap_fscs, halfmap_aucs, c=halfmap_fscs, cmap='viridis', 
                        alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Half-map FSC Score', fontweight='bold')
    ax.set_ylabel('Half-map FSC AUC', fontweight='bold')
    ax.set_title('Half-map FSC Score vs Half-map FSC AUC', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 6: Half-map FSC vs vs-Mean FSC comparison with improved styling
    ax = axes[1, 2]
    
    # Create hexbin density plot for background
    # try:
    hb = ax.hexbin(halfmap_fscs, vs_mean_fscs, gridsize=20, cmap='Blues', alpha=0.3, mincnt=1)
    # except:
    #     pass
    
    scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', 
                        alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC', fontweight='bold')
    ax.set_ylabel('vs-Mean FSC', fontweight='bold')
    ax.set_title('Half-map FSC vs vs-Mean FSC', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'fsc_analysis_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate cluster sizes
    cluster_sizes = [np.sum(cluster_indices == i) for i in range(len(cluster_centers))]
    
    # Save detailed results
    results = {
        'cluster_centers': cluster_centers,
        'cluster_indices': cluster_indices,
        'fsc_scores': fsc_scores,
        'fsc_auc_scores': fsc_auc_scores,
        'particle_usage': particle_usage,
        'cluster_sizes': cluster_sizes,
        'halfmap_fscs': halfmap_fscs,
        'halfmap_aucs': halfmap_aucs,
        'vs_mean_fscs': vs_mean_fscs,
        'vs_mean_aucs': vs_mean_aucs
    }
    
    with open(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Identify potential junk clusters using both metrics
    # Half-map FSC based outlier detection
    halfmap_fsc_threshold = np.percentile(halfmap_fscs, 25)  # Bottom 25%
    halfmap_auc_threshold = np.percentile(halfmap_aucs, 25)  # Bottom 25%
    
    # vs-Mean FSC based outlier detection
    vs_mean_fsc_threshold = np.percentile(vs_mean_fscs, 25)  # Bottom 25%
    vs_mean_auc_threshold = np.percentile(vs_mean_aucs, 25)  # Bottom 25%
    
    # Find junk clusters based on different criteria
    halfmap_junk_clusters = []
    vs_mean_junk_clusters = []
    
    for i in range(len(cluster_centers)):
        # Half-map FSC based junk detection
        if halfmap_fscs[i] < halfmap_fsc_threshold and halfmap_aucs[i] < halfmap_auc_threshold:
            halfmap_junk_clusters.append(i)
        
        # vs-Mean FSC based junk detection
        if vs_mean_fscs[i] < vs_mean_fsc_threshold and vs_mean_aucs[i] < vs_mean_auc_threshold:
            vs_mean_junk_clusters.append(i)
    
    # Combined junk detection (clusters that are junk by both methods)
    combined_junk_clusters = list(set(halfmap_junk_clusters) & set(vs_mean_junk_clusters))
    
    # Map cluster indices to particle indices
    halfmap_junk_particle_indices = []
    vs_mean_junk_particle_indices = []
    combined_junk_particle_indices = []
    
    for cluster_idx in halfmap_junk_clusters:
        cluster_particles = np.where(cluster_indices == cluster_idx)[0]
        halfmap_junk_particle_indices.extend(cluster_particles)
    
    for cluster_idx in vs_mean_junk_clusters:
        cluster_particles = np.where(cluster_indices == cluster_idx)[0]
        vs_mean_junk_particle_indices.extend(cluster_particles)
    
    for cluster_idx in combined_junk_clusters:
        cluster_particles = np.where(cluster_indices == cluster_idx)[0]
        combined_junk_particle_indices.extend(cluster_particles)
    
    # Convert to numpy arrays and remove duplicates
    halfmap_junk_particle_indices = np.unique(halfmap_junk_particle_indices)
    vs_mean_junk_particle_indices = np.unique(vs_mean_junk_particle_indices)
    combined_junk_particle_indices = np.unique(combined_junk_particle_indices)
    
    # Save detailed junk cluster information
    junk_info = {
        'combined_junk_clusters': combined_junk_clusters,
        'halfmap_junk_clusters': halfmap_junk_clusters,
        'vs_mean_junk_clusters': vs_mean_junk_clusters,
        'combined_junk_particle_indices': combined_junk_particle_indices,
        'halfmap_junk_particle_indices': halfmap_junk_particle_indices,
        'vs_mean_junk_particle_indices': vs_mean_junk_particle_indices,
        'halfmap_fsc_threshold': halfmap_fsc_threshold,
        'halfmap_auc_threshold': halfmap_auc_threshold,
        'vs_mean_fsc_threshold': vs_mean_fsc_threshold,
        'vs_mean_auc_threshold': vs_mean_auc_threshold,
        'total_particles': len(zs),
        'combined_junk_particles': len(combined_junk_particle_indices),
        'halfmap_junk_particles': len(halfmap_junk_particle_indices),
        'vs_mean_junk_particles': len(vs_mean_junk_particle_indices),
        'combined_junk_percentage': len(combined_junk_particle_indices) / len(zs) * 100,
        'halfmap_junk_percentage': len(halfmap_junk_particle_indices) / len(zs) * 100,
        'vs_mean_junk_percentage': len(vs_mean_junk_particle_indices) / len(zs) * 100
    }
    
    with open(os.path.join(output_folder, f'junk_cluster_info_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(junk_info, f)
    
    # Print summary statistics
    logger.info(f"=== Junk Detection Summary ===")
    logger.info(f"Total clusters: {len(cluster_centers)}")
    logger.info(f"Total particles: {len(zs)}")
    logger.info(f"")
    logger.info(f"Half-map FSC based detection:")
    logger.info(f"  - Junk clusters: {len(halfmap_junk_clusters)} ({len(halfmap_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(halfmap_junk_particle_indices)} ({len(halfmap_junk_particle_indices)/len(zs)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"vs-Mean FSC based detection:")
    logger.info(f"  - Junk clusters: {len(vs_mean_junk_clusters)} ({len(vs_mean_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(vs_mean_junk_particle_indices)} ({len(vs_mean_junk_particle_indices)/len(zs)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Combined detection:")
    logger.info(f"  - Junk clusters: {len(combined_junk_clusters)} ({len(combined_junk_clusters)/len(cluster_centers)*100:.1f}%)")
    logger.info(f"  - Junk particles: {len(combined_junk_particle_indices)} ({len(combined_junk_particle_indices)/len(zs)*100:.1f}%)")
    
    return junk_info


def plot_umap_visualization(zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, output_folder, zdim_key):
    """
    Create enhanced UMAP visualization of latent space with cluster analysis including AUC scores.
    
    Parameters:
    - zs: Latent embeddings
    - cluster_centers: K-means cluster centers
    - cluster_indices: Cluster assignments for each particle
    - fsc_scores: Dictionary with FSC scores for each cluster
    - fsc_auc_scores: Dictionary with FSC AUC scores for each cluster
    - output_folder: Output directory for saving plots
    - zdim_key: Dimension key for embeddings
    """
    if zs.shape[1] < 2:
        logger.warning("UMAP visualization requires at least 2 dimensions")
        return
    
    logger.info("Computing UMAP embedding for visualization...")
    
    # Compute UMAP embedding
    n_components = min(zs.shape[1], 2)
    mapper = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1).fit(zs)
    zs_umap = mapper.transform(zs)
    cluster_centers_umap = mapper.transform(cluster_centers)
    
    # Extract FSC scores and AUC scores
    halfmap_fscs = [fsc_scores[i]['halfmap_fsc'] for i in range(len(cluster_centers))]
    vs_mean_fscs = [fsc_scores[i]['vs_mean_fsc'] for i in range(len(cluster_centers))]
    halfmap_aucs = [fsc_auc_scores[i]['halfmap_auc'] for i in range(len(cluster_centers))]
    vs_mean_aucs = [fsc_auc_scores[i]['vs_mean_auc'] for i in range(len(cluster_centers))]
    
    # Create comprehensive UMAP visualization with AUC scores
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Enhanced UMAP Visualization with AUC Scores (zdim={zdim_key})', fontsize=18, y=0.95)
    
    # Plot 1: All particles density (hexbin)
    ax = axes[0, 0]
    hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Blues', mincnt=1)
    ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
               edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Particle Density')
    ax.set_title('Particle Density in UMAP Space')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Particles colored by Half-map FSC AUC score
    ax = axes[0, 1]
    particle_halfmap_aucs = np.array([halfmap_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], C=particle_halfmap_aucs, gridsize=60, 
                  cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
               edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Half-map FSC AUC Score')
    ax.set_title('Particles by Half-map FSC AUC Score')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Particles colored by vs-Mean FSC AUC score
    ax = axes[0, 2]
    particle_vs_mean_aucs = np.array([vs_mean_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], C=particle_vs_mean_aucs, gridsize=60, 
                  cmap='plasma', reduce_C_function=np.mean, mincnt=1)
    ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
               edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('vs-Mean FSC AUC Score')
    ax.set_title('Particles by vs-Mean FSC AUC Score')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Top 10 clusters by Half-map FSC AUC
    ax = axes[1, 0]
    sorted_indices_auc = np.argsort(halfmap_aucs)
    top_10_auc = sorted_indices_auc[-10:]
    colors_top10 = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Create a background hexbin for all particles
    hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
    
    for i, cluster_idx in enumerate(top_10_auc):
        cluster_mask = cluster_indices == cluster_idx
        ax.scatter(zs_umap[cluster_mask, 0], zs_umap[cluster_mask, 1], 
                  c=[colors_top10[i]], alpha=0.8, s=20, 
                  label=f'Cluster {cluster_idx} (AUC={halfmap_aucs[cluster_idx]:.3f})')
    ax.scatter(cluster_centers_umap[top_10_auc, 0], cluster_centers_umap[top_10_auc, 1], 
              c=colors_top10, s=150, marker='x', linewidth=3, zorder=10)
    ax.set_title('Top 10 Clusters by Half-map FSC AUC')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Bottom 10 clusters by Half-map FSC AUC
    ax = axes[1, 1]
    bottom_10_auc = sorted_indices_auc[:10]
    colors_bottom10 = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Create a background hexbin for all particles
    hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
    
    for i, cluster_idx in enumerate(bottom_10_auc):
        cluster_mask = cluster_indices == cluster_idx
        ax.scatter(zs_umap[cluster_mask, 0], zs_umap[cluster_mask, 1], 
                  c=[colors_bottom10[i]], alpha=0.8, s=20, 
                  label=f'Cluster {cluster_idx} (AUC={halfmap_aucs[cluster_idx]:.3f})')
    ax.scatter(cluster_centers_umap[bottom_10_auc, 0], cluster_centers_umap[bottom_10_auc, 1], 
              c=colors_bottom10, s=150, marker='x', linewidth=3, zorder=10)
    ax.set_title('Bottom 10 Clusters by Half-map FSC AUC')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Enhanced FSC score distributions
    ax = axes[1, 2]
    
    # Create a more detailed histogram with both FSC types
    # Use actual data range instead of fixed 0-1 range
    all_fsc_scores = np.concatenate([halfmap_fscs, vs_mean_fscs])
    min_score = np.min(all_fsc_scores)
    max_score = np.max(all_fsc_scores)
    
    # Add some padding to the range for better visualization
    score_range = max_score - min_score
    if score_range == 0:
        # If all scores are the same, use a small range around the value
        min_score = min_score - 0.01
        max_score = max_score + 0.01
    else:
        # Add 5% padding on each side
        padding = score_range * 0.05
        min_score = max(0, min_score - padding)  # Don't go below 0
        max_score = min(1, max_score + padding)  # Don't go above 1
    
    bins = np.linspace(min_score, max_score, 25)  # More bins for better resolution
    
    # Plot both distributions with transparency
    ax.hist(halfmap_fscs, bins=bins, alpha=0.6, color='blue', label=f'Half-map FSC (μ={np.mean(halfmap_fscs):.3f})', 
            edgecolor='black', linewidth=0.5)
    ax.hist(vs_mean_fscs, bins=bins, alpha=0.6, color='red', label=f'vs-Mean FSC (μ={np.mean(vs_mean_fscs):.3f})', 
            edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    ax.axvline(np.mean(halfmap_fscs), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(np.mean(vs_mean_fscs), color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add FSC=1/7 threshold line if it's within the range
    if 1/7 >= min_score and 1/7 <= max_score:
        ax.axvline(1/7, color='black', linestyle=':', linewidth=2, alpha=0.8, label='FSC=1/7 threshold')
    
    ax.set_xlabel('FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'Distribution of FSC Scores\n(Range: {min_score:.3f} - {max_score:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'umap_visualization_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure focused on AUC scores and comparisons
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle(f'AUC Score Analysis and FSC Comparisons (zdim={zdim_key})', fontsize=18, y=0.95)
    
    # Plot 1: Half-map FSC AUC distribution
    ax = axes2[0, 0]
    # Use actual data range for AUC scores
    all_auc_scores = np.concatenate([halfmap_aucs, vs_mean_aucs])
    min_auc = np.min(all_auc_scores)
    max_auc = np.max(all_auc_scores)
    
    # Add some padding to the range for better visualization
    auc_range = max_auc - min_auc
    if auc_range == 0:
        # If all scores are the same, use a small range around the value
        min_auc = min_auc - 0.01
        max_auc = max_auc + 0.01
    else:
        # Add 5% padding on each side
        padding = auc_range * 0.05
        min_auc = max(0, min_auc - padding)  # Don't go below 0
        max_auc = min(1, max_auc + padding)  # Don't go above 1
    
    bins_auc = np.linspace(min_auc, max_auc, 25)
    ax.hist(halfmap_aucs, bins=bins_auc, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(halfmap_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_aucs):.3f}')
    ax.set_xlabel('Half-map FSC AUC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: vs-Mean FSC AUC distribution
    ax = axes2[0, 1]
    ax.hist(vs_mean_aucs, bins=bins_auc, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(vs_mean_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(vs_mean_aucs):.3f}')
    ax.set_xlabel('vs-Mean FSC AUC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of vs-Mean FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Half-map FSC vs Half-map FSC AUC scatter
    ax = axes2[0, 2]
    scatter = ax.scatter(halfmap_fscs, halfmap_aucs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Half-map FSC AUC Score')
    ax.set_title('Half-map FSC vs Half-map FSC AUC')
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: vs-Mean FSC vs vs-Mean FSC AUC scatter
    ax = axes2[1, 0]
    scatter = ax.scatter(vs_mean_fscs, vs_mean_aucs, c=vs_mean_fscs, cmap='plasma', alpha=0.7, s=50)
    ax.set_xlabel('vs-Mean FSC Score')
    ax.set_ylabel('vs-Mean FSC AUC Score')
    ax.set_title('vs-Mean FSC vs vs-Mean FSC AUC')
    plt.colorbar(scatter, ax=ax, label='vs-Mean FSC Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Half-map FSC AUC vs vs-Mean FSC AUC scatter
    ax = axes2[1, 1]
    scatter = ax.scatter(halfmap_aucs, vs_mean_aucs, c=halfmap_aucs, cmap='viridis', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC AUC Score')
    ax.set_ylabel('vs-Mean FSC AUC Score')
    ax.set_title('Half-map FSC AUC vs vs-Mean FSC AUC')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Half-map FSC AUC Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Half-map FSC vs vs-Mean FSC comparison
    ax = axes2[1, 2]
    scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('vs-Mean FSC Score')
    ax.set_title('Half-map FSC vs vs-Mean FSC')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'auc_analysis_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save UMAP coordinates for potential future use
    umap_data = {
        'zs_umap': zs_umap,
        'cluster_centers_umap': cluster_centers_umap,
        'mapper': mapper,
        'halfmap_fscs': halfmap_fscs,
        'vs_mean_fscs': vs_mean_fscs,
        'halfmap_aucs': halfmap_aucs,
        'vs_mean_aucs': vs_mean_aucs
    }
    with open(os.path.join(output_folder, f'umap_coordinates_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(umap_data, f)
    
    logger.info("Enhanced UMAP visualization with AUC scores completed and saved.")
    return umap_data


def detect_junk_clusters(fsc_scores, fsc_auc_scores, output_folder, zdim_key, 
                        method='adaptive_threshold', percentile_threshold=25, 
                        std_threshold=2.0, min_junk_fraction=0.1, max_junk_fraction=0.8):
    """
    Detect junk clusters based on FSC and AUC scores using multiple methods.
    
    Parameters:
    - fsc_scores: Dictionary with FSC scores for each cluster
    - fsc_auc_scores: Dictionary with FSC AUC scores for each cluster
    - output_folder: Output directory for saving plots
    - zdim_key: Dimension key for embeddings
    - method: 'adaptive_threshold', 'percentile', 'std_based', or 'consensus'
    - percentile_threshold: Percentile threshold for percentile method
    - std_threshold: Standard deviation threshold for std_based method
    - min_junk_fraction: Minimum fraction of clusters that can be classified as junk
    - max_junk_fraction: Maximum fraction of clusters that can be classified as junk
    
    Returns:
    - junk_clusters: List of cluster indices classified as junk
    - junk_info: Dictionary with detailed junk detection results
    """
    logger.info("Detecting junk clusters based on FSC and AUC scores...")
    
    # Extract scores
    halfmap_fscs = np.array([fsc_scores[i]['halfmap_fsc'] for i in range(len(fsc_scores))])
    vs_mean_fscs = np.array([fsc_scores[i]['vs_mean_fsc'] for i in range(len(fsc_scores))])
    halfmap_aucs = np.array([fsc_auc_scores[i]['halfmap_auc'] for i in range(len(fsc_auc_scores))])
    vs_mean_aucs = np.array([fsc_auc_scores[i]['vs_mean_auc'] for i in range(len(fsc_auc_scores))])
    
    # Combine scores (geometric mean to be robust to outliers)
    combined_fsc = np.sqrt(halfmap_fscs * vs_mean_fscs)
    combined_auc = np.sqrt(halfmap_aucs * vs_mean_aucs)
    
    # Method 1: Adaptive threshold based on score distribution
    def adaptive_threshold_detection(scores):
        """Detect junk using adaptive threshold based on score distribution."""
        # Sort scores
        sorted_scores = np.sort(scores)
        
        # Find the elbow point (where the slope changes significantly)
        # Use the second derivative to find the elbow
        if len(sorted_scores) > 3:
            # Calculate second derivative
            diff1 = np.diff(sorted_scores)
            diff2 = np.diff(diff1)
            
            # Find the point with maximum second derivative (elbow)
            elbow_idx = np.argmax(np.abs(diff2)) + 1
            
            # Use the score at the elbow as threshold
            threshold = sorted_scores[elbow_idx]
        else:
            # Fallback to percentile method
            threshold = np.percentile(scores, percentile_threshold)
        
        return scores < threshold
    
    # Method 2: Percentile-based detection
    def percentile_detection(scores):
        """Detect junk using percentile threshold."""
        threshold = np.percentile(scores, percentile_threshold)
        return scores < threshold
    
    # Method 3: Standard deviation based detection
    def std_based_detection(scores):
        """Detect junk using standard deviation threshold."""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score - std_threshold * std_score
        return scores < threshold
    
    # Method 4: Gap-based detection
    def gap_based_detection(scores):
        """Detect junk by finding the largest gap in sorted scores."""
        sorted_scores = np.sort(scores)
        gaps = np.diff(sorted_scores)
        
        if len(gaps) > 0:
            # Find the largest gap
            max_gap_idx = np.argmax(gaps)
            threshold = sorted_scores[max_gap_idx]
            return scores < threshold
        else:
            return np.zeros(len(scores), dtype=bool)
    
    # Apply different detection methods
    methods_results = {}
    
    # FSC-based detection
    methods_results['fsc_adaptive'] = adaptive_threshold_detection(combined_fsc)
    methods_results['fsc_percentile'] = percentile_detection(combined_fsc)
    methods_results['fsc_std'] = std_based_detection(combined_fsc)
    methods_results['fsc_gap'] = gap_based_detection(combined_fsc)
    
    # AUC-based detection
    methods_results['auc_adaptive'] = adaptive_threshold_detection(combined_auc)
    methods_results['auc_percentile'] = percentile_detection(combined_auc)
    methods_results['auc_std'] = std_based_detection(combined_auc)
    methods_results['auc_gap'] = gap_based_detection(combined_auc)
    
    # Consensus detection
    def consensus_detection():
        """Combine multiple methods for robust detection."""
        # Count how many methods classify each cluster as junk
        junk_votes = np.zeros(len(combined_fsc), dtype=int)
        
        for method_name, is_junk in methods_results.items():
            junk_votes += is_junk.astype(int)
        
        # A cluster is junk if majority of methods classify it as junk
        consensus_threshold = len(methods_results) // 2
        return junk_votes >= consensus_threshold
    
    methods_results['consensus'] = consensus_detection()
    
    # Apply fraction constraints
    def apply_fraction_constraints(is_junk, scores):
        """Apply min/max junk fraction constraints."""
        n_clusters = len(scores)
        n_junk = np.sum(is_junk)
        junk_fraction = n_junk / n_clusters
        
        if junk_fraction < min_junk_fraction:
            # Need to classify more as junk
            n_target = max(1, int(min_junk_fraction * n_clusters))
            # Take the worst n_target clusters
            worst_indices = np.argsort(scores)[:n_target]
            is_junk = np.zeros(n_clusters, dtype=bool)
            is_junk[worst_indices] = True
        elif junk_fraction > max_junk_fraction:
            # Need to classify fewer as junk
            n_target = int(max_junk_fraction * n_clusters)
            # Take only the worst n_target clusters
            worst_indices = np.argsort(scores)[:n_target]
            is_junk = np.zeros(n_clusters, dtype=bool)
            is_junk[worst_indices] = True
        
        return is_junk
    
    # Apply constraints to each method
    for method_name in methods_results:
        if method_name == 'consensus':
            # For consensus, use combined scores
            methods_results[method_name] = apply_fraction_constraints(
                methods_results[method_name], combined_fsc
            )
        elif 'fsc' in method_name:
            methods_results[method_name] = apply_fraction_constraints(
                methods_results[method_name], combined_fsc
            )
        else:  # auc methods
            methods_results[method_name] = apply_fraction_constraints(
                methods_results[method_name], combined_auc
            )
    
    # Select final junk clusters based on method
    if method == 'consensus':
        final_junk_mask = methods_results['consensus']
    elif method == 'adaptive_threshold':
        # Combine FSC and AUC adaptive thresholds
        fsc_junk = methods_results['fsc_adaptive']
        auc_junk = methods_results['auc_adaptive']
        final_junk_mask = fsc_junk | auc_junk  # Union of both
        final_junk_mask = apply_fraction_constraints(final_junk_mask, combined_fsc)
    elif method == 'percentile':
        # Combine FSC and AUC percentile thresholds
        fsc_junk = methods_results['fsc_percentile']
        auc_junk = methods_results['auc_percentile']
        final_junk_mask = fsc_junk | auc_junk  # Union of both
        final_junk_mask = apply_fraction_constraints(final_junk_mask, combined_fsc)
    elif method == 'std_based':
        # Combine FSC and AUC std-based thresholds
        fsc_junk = methods_results['fsc_std']
        auc_junk = methods_results['auc_std']
        final_junk_mask = fsc_junk | auc_junk  # Union of both
        final_junk_mask = apply_fraction_constraints(final_junk_mask, combined_fsc)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    junk_clusters = np.where(final_junk_mask)[0]
    good_clusters = np.where(~final_junk_mask)[0]
    
    # Create visualizations
    create_junk_detection_visualizations(
        halfmap_fscs, vs_mean_fscs, halfmap_aucs, vs_mean_aucs,
        combined_fsc, combined_auc, methods_results, final_junk_mask,
        junk_clusters, good_clusters, output_folder, zdim_key, method
    )
    
    # Compile results
    junk_info = {
        'junk_clusters': junk_clusters.tolist(),
        'good_clusters': good_clusters.tolist(),
        'junk_fraction': len(junk_clusters) / len(combined_fsc),
        'method': method,
        'methods_results': {k: v.tolist() for k, v in methods_results.items()},
        'scores': {
            'halfmap_fscs': halfmap_fscs.tolist(),
            'vs_mean_fscs': vs_mean_fscs.tolist(),
            'halfmap_aucs': halfmap_aucs.tolist(),
            'vs_mean_aucs': vs_mean_aucs.tolist(),
            'combined_fsc': combined_fsc.tolist(),
            'combined_auc': combined_auc.tolist()
        },
        'detection_parameters': {
            'percentile_threshold': percentile_threshold,
            'std_threshold': std_threshold,
            'min_junk_fraction': min_junk_fraction,
            'max_junk_fraction': max_junk_fraction
        }
    }
    
    logger.info(f"Junk detection complete. Found {len(junk_clusters)} junk clusters "
                f"({len(junk_clusters)/len(combined_fsc)*100:.1f}%) using method '{method}'")
    
    return junk_clusters, junk_info


def create_junk_detection_visualizations(halfmap_fscs, vs_mean_fscs, halfmap_aucs, vs_mean_aucs,
                                       combined_fsc, combined_auc, methods_results, final_junk_mask,
                                       junk_clusters, good_clusters, output_folder, zdim_key, method):
    """
    Create comprehensive visualizations for junk detection results.
    """
    logger.info("Creating junk detection visualizations...")
    
    # Set up plotting style with better defaults
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Use a more professional color scheme
    colors = {
        'all': '#E0E0E0',  # Light gray
        'good': '#2E8B57',  # Sea green
        'junk': '#DC143C',  # Crimson red
        'mean': '#4169E1',  # Royal blue
        'grid': '#F0F0F0'   # Very light gray
    }
    
    # Create main junk detection summary figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Junk Detection Results (Method: {method}, zdim={zdim_key})', 
                 fontsize=18, y=0.95, fontweight='bold')
    
    # Plot 1: FSC score distributions with junk highlighted
    ax = axes[0, 0]
    
    # Use adaptive binning based on data range and number of clusters
    n_clusters = len(combined_fsc)
    n_bins = min(30, max(15, n_clusters // 3))  # Adaptive bin count
    bins = np.linspace(combined_fsc.min(), combined_fsc.max(), n_bins + 1)
    
    # Plot histograms with better styling
    # All clusters (background)
    ax.hist(combined_fsc, bins=bins, alpha=0.3, color=colors['all'], 
            label='All clusters', edgecolor='black', linewidth=0.5, density=True)
    
    # Good clusters
    if len(good_clusters) > 0:
        ax.hist(combined_fsc[good_clusters], bins=bins, alpha=0.8, color=colors['good'], 
                label=f'Good clusters (n={len(good_clusters)})', edgecolor='black', 
                linewidth=0.8, density=True, hatch='///')
    
    # Junk clusters
    if len(junk_clusters) > 0:
        ax.hist(combined_fsc[junk_clusters], bins=bins, alpha=0.8, color=colors['junk'], 
                label=f'Junk clusters (n={len(junk_clusters)})', edgecolor='black', 
                linewidth=0.8, density=True, hatch='\\\\\\')
    
    # Add mean line with better styling
    mean_fsc = np.mean(combined_fsc)
    ax.axvline(mean_fsc, color=colors['mean'], linestyle='--', linewidth=2.5, alpha=0.9,
               label=f'Mean: {mean_fsc:.3f}')
    
    ax.set_xlabel('Combined FSC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('FSC Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')  # Very light background
    
    # Plot 2: AUC score distributions with junk highlighted
    ax = axes[0, 1]
    
    # Use adaptive binning for AUC
    n_bins_auc = min(30, max(15, n_clusters // 3))
    bins_auc = np.linspace(combined_auc.min(), combined_auc.max(), n_bins_auc + 1)
    
    # All clusters (background)
    ax.hist(combined_auc, bins=bins_auc, alpha=0.3, color=colors['all'], 
            label='All clusters', edgecolor='black', linewidth=0.5, density=True)
    
    # Good clusters
    if len(good_clusters) > 0:
        ax.hist(combined_auc[good_clusters], bins=bins_auc, alpha=0.8, color=colors['good'], 
                label=f'Good clusters (n={len(good_clusters)})', edgecolor='black', 
                linewidth=0.8, density=True, hatch='///')
    
    # Junk clusters
    if len(junk_clusters) > 0:
        ax.hist(combined_auc[junk_clusters], bins=bins_auc, alpha=0.8, color=colors['junk'], 
                label=f'Junk clusters (n={len(junk_clusters)})', edgecolor='black', 
                linewidth=0.8, density=True, hatch='\\\\\\')
    
    # Add mean line
    mean_auc = np.mean(combined_auc)
    ax.axvline(mean_auc, color=colors['mean'], linestyle='--', linewidth=2.5, alpha=0.9,
               label=f'Mean: {mean_auc:.3f}')
    
    ax.set_xlabel('Combined AUC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('AUC Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 3: FSC vs AUC scatter with junk highlighted
    ax = axes[0, 2]
    
    # Create hexbin density plot for all clusters
    try:
        # Debug: check data bounds
        if np.any(np.isnan(combined_fsc)) or np.any(np.isnan(combined_auc)):
            logger.warning(f"NaN values found: FSC NaN count: {np.sum(np.isnan(combined_fsc))}, AUC NaN count: {np.sum(np.isnan(combined_auc))}")
            hb = None
        elif np.any(np.isinf(combined_fsc)) or np.any(np.isinf(combined_auc)):
            logger.warning(f"Infinite values found: FSC inf count: {np.sum(np.isinf(combined_fsc))}, AUC inf count: {np.sum(np.isinf(combined_auc))}")
            hb = None
        elif combined_fsc.max() - combined_fsc.min() < 1e-10 or combined_auc.max() - combined_auc.min() < 1e-10:
            logger.warning(f"No range in data: FSC range: {combined_fsc.max() - combined_fsc.min()}, AUC range: {combined_auc.max() - combined_auc.min()}")
            hb = None
        else:
            hb = ax.hexbin(combined_fsc, combined_auc, gridsize=20, cmap='Blues', alpha=0.6, 
                           mincnt=1, reduce_C_function=np.mean)
    except Exception as e:
        logger.warning(f"Hexbin failed: {e}")
        hb = None
    
    # Plot good and junk clusters separately with better styling
    if len(good_clusters) > 0:
        ax.scatter(combined_fsc[good_clusters], combined_auc[good_clusters], 
                  c=colors['good'], alpha=0.8, s=60, label=f'Good clusters (n={len(good_clusters)})',
                  edgecolor='black', linewidth=0.8, zorder=5)
    if len(junk_clusters) > 0:
        ax.scatter(combined_fsc[junk_clusters], combined_auc[junk_clusters], 
                  c=colors['junk'], alpha=0.8, s=60, label=f'Junk clusters (n={len(junk_clusters)})',
                  edgecolor='black', linewidth=0.8, zorder=5)
    
    # Add colorbar for hexbin
    if hb is not None:
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Density', fontsize=10)
    
    ax.set_xlabel('Combined FSC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Combined AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('FSC vs AUC Scores', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 4: Method comparison heatmap
    ax = axes[1, 0]
    method_names = list(methods_results.keys())
    n_methods = len(method_names)
    n_clusters = len(combined_fsc)
    
    # Create heatmap data
    heatmap_data = np.zeros((n_methods, n_clusters))
    for i, method_name in enumerate(method_names):
        heatmap_data[i, :] = methods_results[method_name]
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_xlabel('Cluster Index', fontsize=12, fontweight='bold')
    ax.set_title('Method Comparison\n(Red=Junk, Green=Good)', fontsize=14, fontweight='bold')
    
    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Good', 'Junk'], fontsize=10)
    cbar.set_label('Classification', fontsize=10)
    
    # Plot 5: Score rankings with junk highlighted
    ax = axes[1, 1]
    
    # Sort clusters by combined FSC score
    sorted_indices = np.argsort(combined_fsc)[::-1]  # Best to worst
    sorted_scores = combined_fsc[sorted_indices]
    sorted_junk = final_junk_mask[sorted_indices]
    
    # Create bar plot with better styling
    colors_bars = [colors['junk'] if is_junk else colors['good'] for is_junk in sorted_junk]
    bars = ax.bar(range(len(sorted_scores)), sorted_scores, color=colors_bars, 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Cluster Rank (Best to Worst)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Combined FSC Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Rankings by FSC Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Add legend with better styling
    legend_elements = [matplotlib.patches.Patch(facecolor=colors['good'], alpha=0.8, label='Good clusters'),
                      matplotlib.patches.Patch(facecolor=colors['junk'], alpha=0.8, label='Junk clusters')]
    ax.legend(handles=legend_elements, frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate statistics
    total_clusters = len(combined_fsc)
    n_junk = len(junk_clusters)
    n_good = len(good_clusters)
    junk_fraction = n_junk / total_clusters * 100
    
    # Good cluster statistics
    if n_good > 0:
        good_fsc_mean = np.mean(combined_fsc[good_clusters])
        good_fsc_std = np.std(combined_fsc[good_clusters])
        good_auc_mean = np.mean(combined_auc[good_clusters])
        good_auc_std = np.std(combined_auc[good_clusters])
    else:
        good_fsc_mean = good_fsc_std = good_auc_mean = good_auc_std = 0
    
    # Junk cluster statistics
    if n_junk > 0:
        junk_fsc_mean = np.mean(combined_fsc[junk_clusters])
        junk_fsc_std = np.std(combined_fsc[junk_clusters])
        junk_auc_mean = np.mean(combined_auc[junk_clusters])
        junk_auc_std = np.std(combined_auc[junk_clusters])
    else:
        junk_fsc_mean = junk_fsc_std = junk_auc_mean = junk_auc_std = 0
    
    # Create summary text with better formatting
    summary_text = f"""
Junk Detection Summary

Method: {method}
Total Clusters: {total_clusters}

Junk Clusters: {n_junk} ({junk_fraction:.1f}%)
Good Clusters: {n_good} ({100-junk_fraction:.1f}%)

Good Cluster Statistics:
  FSC: {good_fsc_mean:.3f} ± {good_fsc_std:.3f}
  AUC: {good_auc_mean:.3f} ± {good_auc_std:.3f}

Junk Cluster Statistics:
  FSC: {junk_fsc_mean:.3f} ± {junk_fsc_std:.3f}
  AUC: {junk_auc_mean:.3f} ± {junk_auc_std:.3f}

Score Separation:
  FSC Difference: {good_fsc_mean - junk_fsc_mean:.3f}
  AUC Difference: {good_auc_mean - junk_auc_mean:.3f}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, 
                     edgecolor='black', linewidth=1))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'junk_detection_summary_{zdim_key}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create method comparison figure with improved styling
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Junk Detection Method Comparison (zdim={zdim_key})', 
                  fontsize=16, y=0.95, fontweight='bold')
    
    # Plot 1: Method agreement matrix
    ax = axes2[0, 0]
    method_names = list(methods_results.keys())
    n_methods = len(method_names)
    
    # Calculate agreement matrix
    agreement_matrix = np.zeros((n_methods, n_methods))
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            agreement = np.sum(methods_results[method1] == methods_results[method2]) / len(combined_fsc)
            agreement_matrix[i, j] = agreement
    
    im = ax.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_title('Method Agreement Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations with better styling
    for i in range(n_methods):
        for j in range(n_methods):
            text_color = "white" if agreement_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                   ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Agreement Fraction')
    
    # Plot 2: Junk fraction by method
    ax = axes2[0, 1]
    junk_fractions = [np.sum(methods_results[method]) / len(combined_fsc) * 100 
                     for method in method_names]
    
    bars = ax.bar(method_names, junk_fractions, color='skyblue', edgecolor='black', 
                  alpha=0.8, linewidth=1)
    ax.set_ylabel('Junk Fraction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Junk Fraction by Detection Method', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Add value labels on bars with better styling
    for bar, value in zip(bars, junk_fractions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Score distributions by method
    ax = axes2[1, 0]
    
    # Show score distributions for different methods with better styling
    colors_methods = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    for i, method_name in enumerate(method_names):
        is_junk = methods_results[method_name]
        if np.sum(is_junk) > 0:
            ax.hist(combined_fsc[is_junk], bins=20, alpha=0.7, 
                   label=f'{method_name} (junk)', density=True, 
                   color=colors_methods[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Combined FSC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('FSC Score Distributions (Junk Clusters)', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax.set_facecolor('#FAFAFA')
    
    # Plot 4: Method performance comparison
    ax = axes2[1, 1]
    
    # Calculate performance metrics for each method
    performance_data = []
    for method_name in method_names:
        is_junk = methods_results[method_name]
        if np.sum(is_junk) > 0 and np.sum(~is_junk) > 0:
            junk_mean = np.mean(combined_fsc[is_junk])
            good_mean = np.mean(combined_fsc[~is_junk])
            separation = good_mean - junk_mean
            performance_data.append((method_name, separation))
    
    if performance_data:
        methods, separations = zip(*performance_data)
        bars = ax.bar(methods, separations, color='lightgreen', edgecolor='black', 
                      alpha=0.8, linewidth=1)
        ax.set_ylabel('Score Separation (Good - Junk)', fontsize=12, fontweight='bold')
        ax.set_title('Method Performance (Score Separation)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Add value labels on bars with better styling
        for bar, value in zip(bars, separations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'junk_detection_methods_{zdim_key}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info("Junk detection visualizations completed and saved.")


def junk_particle_detection(recovar_result_dir, output_folder=None, zdim=10, n_clusters=100, 
                           batch_size=100, n_particles_per_cluster=100, no_z_regularization=False,
                           save_reconstructions=False, filter_resolution=None, filter_fourier_shells=10,
                           junk_detection_method="adaptive_threshold", percentile_threshold=25.0,
                           std_threshold=2.0, min_junk_fraction=0.1, max_junk_fraction=0.8,
                           save_pipeline_indices=False, output_format="both"):
    """
    Main function to detect junk particles from latent space using clustering and FSC analysis.
    
    Parameters:
    - recovar_result_dir: Directory containing recovar pipeline results
    - output_folder: Output directory for results
    - zdim: Latent space dimension to use
    - n_clusters: Number of k-means clusters
    - batch_size: Batch size for reconstruction
    - n_particles_per_cluster: Number of particles per halfmap (so 2*n_particles_per_cluster total)
    - no_z_regularization: Whether to use unregularized embeddings
    - save_reconstructions: Whether to save reconstructions to file
    - filter_resolution: Resolution in Angstroms to filter combined reconstructions (if None, no filtering)
    - filter_fourier_shells: Number of Fourier shells to keep when filtering (default: 10)
    - junk_detection_method: Method for junk detection ("adaptive_threshold", "percentile", "std_based", "consensus")
    - percentile_threshold: Percentile threshold for percentile method (default: 25.0)
    - std_threshold: Standard deviation threshold for std_based method (default: 2.0)
    - min_junk_fraction: Minimum fraction of clusters that can be classified as junk (default: 0.1)
    - max_junk_fraction: Maximum fraction of clusters that can be classified as junk (default: 0.8)
    - save_pipeline_indices: Whether to save particle indices in pipeline-compatible format
    - output_format: Which indices to save ("both", "junk_only", "good_only")
    
    Algorithm:
    1. Performs k-means clustering on latent embeddings
    2. For each cluster center, finds the n_particles_per_cluster closest particles in latent space
    3. Splits these particles into two halfsets for halfmap generation
    4. Maps global indices to local indices for each existing dataset halfset
    5. Uses relion-style reconstruction with subset generators from existing dataset splits
    6. Computes FSC scores between halfmaps and against mean reconstruction
    7. Identifies junk clusters based on low FSC scores using specified method
    8. If filter_resolution is provided, applies low-pass filtering to combined reconstructions
    9. Optionally saves pipeline-compatible indices for downstream use
    """
    
    # Set up output directory
    if output_folder is None:
        output_folder = os.path.join(recovar_result_dir, f'junk_detection_{zdim}')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_folder, 'junk_detection.log'))
        ]
    )
    
    logger.info(f"Starting junk particle detection with zdim={zdim}, n_clusters={n_clusters}")
    if save_reconstructions:
        logger.info("Reconstructions will be saved to file")
    if filter_resolution is not None:
        logger.info(f"Combined reconstructions will be filtered to {filter_resolution:.1f} Angstroms with {filter_fourier_shells} Fourier shells")
    if save_pipeline_indices:
        logger.info(f"Will save pipeline-compatible indices in format: {output_format}")
    
    # Load pipeline output
    pipeline_output = output.PipelineOutput(recovar_result_dir)
    
    # Determine zdim key
    zdim_key = f"{zdim}_noreg" if no_z_regularization else zdim
    
    # Get zs data and ensure it's a dictionary
    zs_data = pipeline_output.get('zs')
    if not isinstance(zs_data, dict):
        logger.error(f"Expected 'zs' to be a dictionary, got {type(zs_data)}")
        raise ValueError(f"Invalid 'zs' data type: {type(zs_data)}")
    
    if zdim_key not in zs_data:
        available_dims = list(zs_data.keys())
        logger.error(f"zdim {zdim_key} not found. Available dimensions: {available_dims}")
        raise ValueError(f"zdim {zdim_key} not found")
    
    # Load embeddings
    zs = zs_data[zdim_key]
    logger.info(f"Loaded embeddings with shape: {zs.shape}")
    
    # Perform k-means clustering (cap clusters at sample count)
    n_clusters = min(n_clusters, len(zs))
    logger.info(f"Performing k-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_indices = kmeans.fit_predict(zs)
    cluster_centers = kmeans.cluster_centers_
    
    logger.info(f"Clustering complete. Cluster sizes: {np.bincount(cluster_indices)}")
    
    # Compute FSC scores for each cluster
    logger.info("Computing FSC scores for each cluster...")
    result = compute_cluster_fsc_scores(
        pipeline_output, cluster_centers, cluster_indices, zdim_key, 
        batch_size, n_particles_per_cluster, save_reconstructions, output_folder, filter_resolution, filter_fourier_shells
    )
    
    # Handle the return values based on save_reconstructions flag
    if save_reconstructions:
        if len(result) != 4:
            raise ValueError(f"Expected 4 return values when save_reconstructions=True, got {len(result)}")
        fsc_scores, fsc_auc_scores, particle_usage, reconstructions = result
    else:
        if len(result) != 3:
            raise ValueError(f"Expected 3 return values when save_reconstructions=False, got {len(result)}")
        fsc_scores, fsc_auc_scores, particle_usage = result
        reconstructions = None
    
    # Create plots and identify junk clusters
    logger.info("Creating plots and identifying junk clusters...")
    junk_info = plot_junk_detection_results(
        zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, 
        particle_usage, output_folder, zdim_key
    )
    
    # Create UMAP visualization
    logger.info("Creating UMAP visualization...")
    umap_data = plot_umap_visualization(
        zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, output_folder, zdim_key
    )
    junk_info['umap_data'] = umap_data
    
    # Perform junk detection
    logger.info("Performing junk detection...")
    junk_clusters, junk_detection_info = detect_junk_clusters(
        fsc_scores, fsc_auc_scores, output_folder, zdim_key,
        method=junk_detection_method, percentile_threshold=percentile_threshold,
        std_threshold=std_threshold, min_junk_fraction=min_junk_fraction, max_junk_fraction=max_junk_fraction
    )
    junk_info['junk_detection'] = junk_detection_info
    junk_info['junk_clusters'] = junk_clusters.tolist()
    junk_info['good_clusters'] = junk_detection_info['good_clusters']
    logger.info(f"Junk detection completed. Found {len(junk_clusters)} junk clusters.")
    
    # Map clusters to particles and save particle classifications
    logger.info("Mapping clusters to particles...")
    junk_particles, good_particles, particle_stats = map_clusters_to_particles(
        junk_clusters, cluster_indices, output_folder, zdim_key, junk_detection_method
    )
    # Create particle classification visualizations
    create_particle_classification_visualizations(
        zs, cluster_indices, junk_particles, good_particles, 
        particle_stats, output_folder, zdim_key, junk_detection_method
    )
    # Get original particle indices for mapping
    original_particle_indices = np.concatenate(pipeline_output.get('particles_halfsets'))
    # Save particle classifications
    save_particle_classifications(
        junk_particles, good_particles, particle_stats, 
        cluster_indices, output_folder, zdim_key, junk_detection_method, original_particle_indices
    )
    # Save pipeline-compatible indices if requested
    if save_pipeline_indices:
        logger.info("Saving pipeline-compatible indices...")
        if output_format in ["both", "junk_only"]:
            junk_pipeline_file = os.path.join(output_folder, f'junk_pipeline_indices_{zdim_key}.pkl')
            with open(junk_pipeline_file, 'wb') as f:
                pickle.dump(junk_particles, f)
            logger.info(f"Saved junk indices for pipeline: {junk_pipeline_file}")
        if output_format in ["both", "good_only"]:
            good_pipeline_file = os.path.join(output_folder, f'good_pipeline_indices_{zdim_key}.pkl')
            with open(good_pipeline_file, 'wb') as f:
                pickle.dump(good_particles, f)
            logger.info(f"Saved good indices for pipeline: {good_pipeline_file}")
        # Create a summary file with usage instructions
        summary_file = os.path.join(output_folder, f'pipeline_usage_summary_{zdim_key}.txt')
        with open(summary_file, 'w') as f:
            f.write("Pipeline-Compatible Indices Usage Summary\n")
            f.write("==========================================\n\n")
            f.write(f"Generated from: {recovar_result_dir}\n")
            f.write(f"Method: {junk_detection_method}\n")
            f.write(f"zdim: {zdim_key}\n\n")
            if output_format in ["both", "junk_only"]:
                f.write("Junk particles (exclude these):\n")
                f.write(f"  --ind {os.path.abspath(junk_pipeline_file)} (for regular datasets)\n")
                f.write(f"  --particle-ind {os.path.abspath(junk_pipeline_file)} (for tilt series)\n\n")
            if output_format in ["both", "good_only"]:
                f.write("Good particles (include these):\n")
                f.write(f"  --ind {os.path.abspath(good_pipeline_file)} (for regular datasets)\n")
                f.write(f"  --particle-ind {os.path.abspath(good_pipeline_file)} (for tilt series)\n\n")
            f.write("Example usage:\n")
            f.write("  # Run pipeline excluding junk particles:\n")
            if output_format in ["both", "junk_only"]:
                f.write(f"  python -m recovar.commands.pipeline particles.star --poses poses.pkl --ctf ctf.pkl --ind {os.path.basename(junk_pipeline_file)} -o output_dir\n\n")
            f.write("  # Run pipeline with only good particles:\n")
            if output_format in ["both", "good_only"]:
                f.write(f"  python -m recovar.commands.pipeline particles.star --poses poses.pkl --ctf ctf.pkl --ind {os.path.basename(good_pipeline_file)} -o output_dir\n")
        logger.info(f"Pipeline usage summary saved to: {summary_file}")
    # Save reconstruction info if reconstructions were saved
    if save_reconstructions and reconstructions is not None:
        reconstructions_info_path = os.path.join(output_folder, f'reconstructions_info_{zdim_key}.pkl')
        with open(reconstructions_info_path, 'wb') as f:
            pickle.dump(reconstructions, f)
        logger.info(f"Saved reconstruction info to {reconstructions_info_path}")
    logger.info(f"Particle mapping completed. Found {len(junk_particles)} junk particles and {len(good_particles)} good particles.")
    
    logger.info(f"Junk particle detection complete. Results saved to {output_folder}")
    return junk_info


def add_args(parser):
    """Add command line arguments for junk particle detection."""
    parser.add_argument("recovar_result_dir", type=str, help="Directory containing recovar pipeline results")
    parser.add_argument("--output-folder", "-o", type=str, help="Output directory for results (default: recovar_result_dir/junk_detection_zdim)")
    parser.add_argument("--zdim", type=int, default=10, help="Latent space dimension to use (default: 10)")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of k-means clusters (default: 100)")
    parser.add_argument("--batch-size", type=int, help="Batch size for reconstruction (auto: based on GPU memory and grid size)")
    parser.add_argument("--n-particles-per-cluster", type=int, help="Number of particles per halfmap (auto: min(100, max(10, n_particles/n_clusters)))")
    parser.add_argument("--no-z-regularization", action="store_true", help="Use unregularized embeddings")
    parser.add_argument("--save-reconstructions", action="store_true", help="Save reconstructions to file")
    parser.add_argument("--filter-resolution", type=float, help="Resolution in Angstroms to filter combined reconstructions")
    parser.add_argument("--filter-fourier-shells", type=int, default=10, help="Number of Fourier shells to keep when filtering (default: 10)")
    parser.add_argument("--junk-detection-method", type=str, default="adaptive_threshold", 
                       choices=["adaptive_threshold", "percentile", "std_based", "consensus"],
                       help="Junk detection method (default: adaptive_threshold)")
    parser.add_argument("--percentile-threshold", type=float, default=25.0, help="Percentile threshold for percentile method (default: 25.0)")
    parser.add_argument("--std-threshold", type=float, default=2.0, help="Standard deviation threshold for std_based method (default: 2.0)")
    parser.add_argument("--min-junk-fraction", type=float, default=0.1, help="Minimum fraction of clusters that can be classified as junk (default: 0.1)")
    parser.add_argument("--max-junk-fraction", type=float, default=0.8, help="Maximum fraction of clusters that can be classified as junk (default: 0.8)")
    parser.add_argument("--save-pipeline-indices", action="store_true", 
                       help="Save particle indices in pipeline-compatible format (for --ind or --particle-ind)")
    parser.add_argument("--output-format", type=str, default="both", 
                       choices=["both", "junk_only", "good_only"], 
                       help="Which indices to save (default: both)")
    
    return parser


def main():
    parser = argparse.ArgumentParser(description="Junk Particle Detection")
    parser = add_args(parser)
    args = parser.parse_args()
    
    # Automatically calculate batch_size and n_particles_per_cluster if not provided
    if args.batch_size is None or args.n_particles_per_cluster is None:
        # Load pipeline output to get necessary information
        pipeline_output = output.PipelineOutput(args.recovar_result_dir)
        
        # Get GPU memory and grid size for batch size calculation
        from recovar import utils
        gpu_memory = utils.get_gpu_memory_total()
        cryos = pipeline_output.get('dataset')
        grid_size = cryos[0].grid_size
        
        # Calculate automatic batch size like in pipeline
        auto_batch_size = utils.get_image_batch_size(grid_size, gpu_memory)
        
        # Calculate n_particles_per_cluster as min(100, max(10, n_particles/n_clusters))
        zdim_key = f"{args.zdim}_noreg" if args.no_z_regularization else args.zdim
        n_particles = len(pipeline_output.get('zs')[zdim_key])
        auto_n_particles_per_cluster = min(100, max(10, n_particles // args.n_clusters))
        
        # Use provided values or auto-calculated values
        batch_size = args.batch_size if args.batch_size is not None else auto_batch_size
        n_particles_per_cluster = args.n_particles_per_cluster if args.n_particles_per_cluster is not None else auto_n_particles_per_cluster
        
        logger.info(f"Auto-calculated: batch_size={batch_size}, n_particles_per_cluster={n_particles_per_cluster}")
    else:
        batch_size = args.batch_size
        n_particles_per_cluster = args.n_particles_per_cluster
    
    # Call the unified function with all arguments
    junk_particle_detection(
        args.recovar_result_dir,
        args.output_folder,
        args.zdim,
        args.n_clusters,
        batch_size,
        n_particles_per_cluster,
        args.no_z_regularization,
        args.save_reconstructions,
        args.filter_resolution,
        args.filter_fourier_shells,
        args.junk_detection_method,
        args.percentile_threshold,
        args.std_threshold,
        args.min_junk_fraction,
        args.max_junk_fraction,
        args.save_pipeline_indices,
        args.output_format
    )


def map_clusters_to_particles(junk_clusters, cluster_indices, output_folder, zdim_key, method):
    """
    Map cluster-level junk detection to individual particle classifications.
    
    Parameters:
    - junk_clusters: List of cluster indices classified as junk
    - cluster_indices: Cluster assignments for each particle
    - output_folder: Output directory for saving results
    - zdim_key: Dimension key for embeddings
    - method: Junk detection method used
    
    Returns:
    - junk_particles: Array of particle indices classified as junk
    - good_particles: Array of particle indices classified as good
    - particle_stats: Dictionary with particle-level statistics
    """
    logger.info("Mapping cluster classifications to individual particles...")
    
    # Create boolean masks for junk and good clusters
    n_clusters = len(np.unique(cluster_indices))
    junk_cluster_mask = np.zeros(n_clusters, dtype=bool)
    junk_cluster_mask[junk_clusters] = True
    
    # Map to particles
    junk_particle_mask = junk_cluster_mask[cluster_indices]
    good_particle_mask = ~junk_particle_mask
    
    # Get particle indices
    junk_particles = np.where(junk_particle_mask)[0]
    good_particles = np.where(good_particle_mask)[0]
    
    # Calculate statistics
    total_particles = len(cluster_indices)
    n_junk_particles = len(junk_particles)
    n_good_particles = len(good_particles)
    junk_fraction = n_junk_particles / total_particles
    
    # Cluster-level statistics
    cluster_sizes = np.bincount(cluster_indices)
    junk_cluster_sizes = cluster_sizes[junk_clusters]
    good_cluster_sizes = cluster_sizes[~junk_cluster_mask]
    
    particle_stats = {
        'total_particles': total_particles,
        'junk_particles': n_junk_particles,
        'good_particles': n_good_particles,
        'junk_fraction': junk_fraction,
        'n_junk_clusters': len(junk_clusters),
        'n_good_clusters': n_clusters - len(junk_clusters),
        'avg_junk_cluster_size': np.mean(junk_cluster_sizes) if len(junk_cluster_sizes) > 0 else 0,
        'avg_good_cluster_size': np.mean(good_cluster_sizes) if len(good_cluster_sizes) > 0 else 0,
        'max_junk_cluster_size': np.max(junk_cluster_sizes) if len(junk_cluster_sizes) > 0 else 0,
        'max_good_cluster_size': np.max(good_cluster_sizes) if len(good_cluster_sizes) > 0 else 0,
        'min_junk_cluster_size': np.min(junk_cluster_sizes) if len(junk_cluster_sizes) > 0 else 0,
        'min_good_cluster_size': np.min(good_cluster_sizes) if len(good_cluster_sizes) > 0 else 0,
        'method': method
    }
    
    logger.info(f"Particle mapping complete:")
    logger.info(f"  Total particles: {total_particles}")
    logger.info(f"  Junk particles: {n_junk_particles} ({junk_fraction*100:.1f}%)")
    logger.info(f"  Good particles: {n_good_particles} ({(1-junk_fraction)*100:.1f}%)")
    
    return junk_particles, good_particles, particle_stats


def create_particle_classification_visualizations(zs, cluster_indices, junk_particles, good_particles, 
                                                particle_stats, output_folder, zdim_key, method):
    """
    Create comprehensive visualizations for particle-level classifications.
    
    Parameters:
    - zs: Latent embeddings
    - cluster_indices: Cluster assignments for each particle
    - junk_particles: Array of particle indices classified as junk
    - good_particles: Array of particle indices classified as good
    - particle_stats: Dictionary with particle-level statistics
    - output_folder: Output directory for saving plots
    - zdim_key: Dimension key for embeddings
    - method: Junk detection method used
    """
    logger.info("Creating particle classification visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main particle classification figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Particle Classification Results (Method: {method}, zdim={zdim_key})', fontsize=18, y=0.95)
    
    # Plot 1: Particle distribution in latent space (first 2 dimensions)
    ax = axes[0, 0]
    if zs.shape[1] >= 2:
        # Plot all particles in background
        ax.scatter(zs[:, 0], zs[:, 1], c='lightgray', alpha=0.3, s=1, label='All particles')
        
        # Plot good and junk particles
        if len(good_particles) > 0:
            ax.scatter(zs[good_particles, 0], zs[good_particles, 1], 
                      c='green', alpha=0.7, s=10, label=f'Good particles (n={len(good_particles)})')
        if len(junk_particles) > 0:
            ax.scatter(zs[junk_particles, 0], zs[junk_particles, 1], 
                      c='red', alpha=0.7, s=10, label=f'Junk particles (n={len(junk_particles)})')
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('Particle Classification in Latent Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient dimensions for 2D plot', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Particle Classification in Latent Space')
    
    # Plot 2: Particle classification histogram
    ax = axes[0, 1]
    categories = ['Good', 'Junk']
    counts = [len(good_particles), len(junk_particles)]
    colors = ['green', 'red']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Particles')
    ax.set_title('Particle Classification Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Cluster size distribution by classification
    ax = axes[0, 2]
    cluster_sizes = np.bincount(cluster_indices)
    
    # Get cluster indices for junk and good particles
    junk_cluster_indices = cluster_indices[junk_particles]
    good_cluster_indices = cluster_indices[good_particles]
    
    # Get unique cluster indices and their sizes
    unique_junk_clusters = np.unique(junk_cluster_indices)
    unique_good_clusters = np.unique(good_cluster_indices)
    
    junk_cluster_sizes = cluster_sizes[unique_junk_clusters]
    good_cluster_sizes = cluster_sizes[unique_good_clusters]
    
    if len(junk_cluster_sizes) > 0:
        ax.hist(junk_cluster_sizes, bins=20, alpha=0.7, color='red', 
                label=f'Junk clusters (n={len(junk_cluster_sizes)})', density=True)
    if len(good_cluster_sizes) > 0:
        ax.hist(good_cluster_sizes, bins=20, alpha=0.7, color='green', 
                label=f'Good clusters (n={len(good_cluster_sizes)})', density=True)
    
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Density')
    ax.set_title('Cluster Size Distribution by Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: UMAP visualization if available
    ax = axes[1, 0]
    try:
        import umap
        # Compute UMAP embedding
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        zs_umap = reducer.fit_transform(zs)
        
        # Plot all particles in background
        ax.scatter(zs_umap[:, 0], zs_umap[:, 1], c='lightgray', alpha=0.3, s=1, label='All particles')
        
        # Plot good and junk particles
        if len(good_particles) > 0:
            ax.scatter(zs_umap[good_particles, 0], zs_umap[good_particles, 1], 
                      c='green', alpha=0.7, s=10, label=f'Good particles (n={len(good_particles)})')
        if len(junk_particles) > 0:
            ax.scatter(zs_umap[junk_particles, 0], zs_umap[junk_particles, 1], 
                      c='red', alpha=0.7, s=10, label=f'Junk particles (n={len(junk_particles)})')
        
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Particle Classification in UMAP Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except ImportError:
        ax.text(0.5, 0.5, 'UMAP not available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Particle Classification in UMAP Space')
    
    # Plot 5: Latent dimension distributions
    ax = axes[1, 1]
    if zs.shape[1] >= 1:
        # Plot distribution of first latent dimension
        ax.hist(zs[good_particles, 0], bins=30, alpha=0.7, color='green', 
                label=f'Good particles (n={len(good_particles)})', density=True)
        ax.hist(zs[junk_particles, 0], bins=30, alpha=0.7, color='red', 
                label=f'Junk particles (n={len(junk_particles)})', density=True)
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Density')
        ax.set_title('Latent Dimension 1 Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No latent dimensions available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Latent Dimension Distribution')
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create detailed summary text
    stats = particle_stats
    summary_text = f"""
Particle Classification Summary

Method: {method}
Total Particles: {stats['total_particles']:,}

Junk Particles: {stats['junk_particles']:,} ({stats['junk_fraction']*100:.1f}%)
Good Particles: {stats['good_particles']:,} ({(1-stats['junk_fraction'])*100:.1f}%)

Cluster Statistics:
  Junk Clusters: {stats['n_junk_clusters']}
  Good Clusters: {stats['n_good_clusters']}
  
  Avg Junk Cluster Size: {stats['avg_junk_cluster_size']:.1f}
  Avg Good Cluster Size: {stats['avg_good_cluster_size']:.1f}
  
  Max Junk Cluster Size: {stats['max_junk_cluster_size']}
  Max Good Cluster Size: {stats['max_good_cluster_size']}
  
  Min Junk Cluster Size: {stats['min_junk_cluster_size']}
  Min Good Cluster Size: {stats['min_good_cluster_size']}

Classification Quality:
  Particles per Junk Cluster: {stats['junk_particles']/stats['n_junk_clusters']:.1f}
  Particles per Good Cluster: {stats['good_particles']/stats['n_good_clusters']:.1f}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'particle_classification_{zdim_key}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed analysis figure
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Detailed Particle Analysis (Method: {method}, zdim={zdim_key})', fontsize=16, y=0.95)
    
    # Plot 1: Particle classification by cluster
    ax = axes2[0, 0]
    unique_clusters = np.unique(cluster_indices)
    cluster_junk_counts = []
    cluster_good_counts = []
    
    for cluster_idx in unique_clusters:
        cluster_particles = cluster_indices == cluster_idx
        cluster_junk_count = np.sum(np.isin(np.where(cluster_particles)[0], junk_particles))
        cluster_good_count = np.sum(cluster_particles) - cluster_junk_count
        cluster_junk_counts.append(cluster_junk_count)
        cluster_good_counts.append(cluster_good_count)
    
    # Create stacked bar chart
    x_pos = np.arange(len(unique_clusters))
    ax.bar(x_pos, cluster_good_counts, color='green', alpha=0.7, label='Good particles')
    ax.bar(x_pos, cluster_junk_counts, bottom=cluster_good_counts, color='red', alpha=0.7, label='Junk particles')
    
    ax.set_xlabel('Cluster Index')
    ax.set_ylabel('Number of Particles')
    ax.set_title('Particle Classification by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distribution of particle classifications
    ax = axes2[0, 1]
    sorted_clusters = unique_clusters[np.argsort(cluster_sizes[unique_clusters])[::-1]]
    cumulative_junk = []
    cumulative_good = []
    
    for cluster_idx in sorted_clusters:
        cluster_particles = cluster_indices == cluster_idx
        cluster_junk_count = np.sum(np.isin(np.where(cluster_particles)[0], junk_particles))
        cluster_good_count = np.sum(cluster_particles) - cluster_junk_count
        
        if len(cumulative_junk) == 0:
            cumulative_junk.append(cluster_junk_count)
            cumulative_good.append(cluster_good_count)
        else:
            cumulative_junk.append(cumulative_junk[-1] + cluster_junk_count)
            cumulative_good.append(cumulative_good[-1] + cluster_good_count)
    
    x_pos = np.arange(len(sorted_clusters))
    ax.plot(x_pos, cumulative_good, 'g-', linewidth=2, label='Cumulative good particles')
    ax.plot(x_pos, cumulative_junk, 'r-', linewidth=2, label='Cumulative junk particles')
    ax.plot(x_pos, np.array(cumulative_good) + np.array(cumulative_junk), 'b-', linewidth=2, label='Total particles')
    
    ax.set_xlabel('Cluster Rank (by size)')
    ax.set_ylabel('Cumulative Number of Particles')
    ax.set_title('Cumulative Particle Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Latent space density with classification
    ax = axes2[1, 0]
    if zs.shape[1] >= 2:
        # Create hexbin plot for all particles
        hb = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=30, cmap='Blues', alpha=0.3, mincnt=1)
        
        # Overlay good and junk particles
        if len(good_particles) > 0:
            ax.scatter(zs[good_particles, 0], zs[good_particles, 1], 
                      c='green', alpha=0.6, s=5, label=f'Good particles')
        if len(junk_particles) > 0:
            ax.scatter(zs[junk_particles, 0], zs[junk_particles, 1], 
                      c='red', alpha=0.6, s=5, label=f'Junk particles')
        
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('Particle Density with Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient dimensions for 2D plot', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Particle Density with Classification')
    
    # Plot 4: Classification confidence analysis
    ax = axes2[1, 1]
    # Calculate classification confidence based on cluster purity
    cluster_purities = []
    cluster_sizes_plot = []
    
    for cluster_idx in unique_clusters:
        cluster_particles = cluster_indices == cluster_idx
        cluster_size = np.sum(cluster_particles)
        cluster_junk_count = np.sum(np.isin(np.where(cluster_particles)[0], junk_particles))
        cluster_good_count = cluster_size - cluster_junk_count
        
        # Purity is the fraction of the majority class
        purity = max(cluster_junk_count, cluster_good_count) / cluster_size
        cluster_purities.append(purity)
        cluster_sizes_plot.append(cluster_size)
    
    # Scatter plot of cluster size vs purity
    colors = ['red' if purity > 0.5 else 'green' for purity in cluster_purities]
    ax.scatter(cluster_sizes_plot, cluster_purities, c=colors, alpha=0.7, s=50)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Classification Purity')
    ax.set_title('Cluster Classification Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    _safe_savefig(os.path.join(output_folder, f'particle_analysis_detailed_{zdim_key}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Particle classification visualizations completed and saved.")


def save_particle_classifications(junk_particles, good_particles, particle_stats, 
                                cluster_indices, output_folder, zdim_key, method, original_indices):
    """
    Save particle-level classifications to pickle files with comprehensive metadata.
    
    Parameters:
    - junk_particles: Array of particle indices classified as junk - IN THE INTERNAL ORDERING
    - good_particles: Array of particle indices classified as good - IN THE INTERNAL ORDERING
    - particle_stats: Dictionary with particle-level statistics
    - cluster_indices: Cluster assignments for each particle
    - output_folder: Output directory for saving files
    - zdim_key: Dimension key for embeddings
    - method: Junk detection method used
    """
    logger.info("Saving particle classifications...")
    
    # Create comprehensive results dictionary

    # Map particles back to original ordering
    junk_particles_original = original_indices[junk_particles]
    good_particles_original = original_indices[good_particles]

    junk_particles = junk_particles_original
    good_particles = good_particles_original
    
    # Update results dictionary with original indices
    results = {
        'junk_particles': junk_particles.tolist(),
        'good_particles': good_particles.tolist(),
        'particle_stats': particle_stats,
        'cluster_indices': cluster_indices.tolist(),
        'classification_method': method,
        'zdim_key': zdim_key,
        'timestamp': str(pd.Timestamp.now()),
        'metadata': {
            'total_particles': len(cluster_indices),
            'n_clusters': len(np.unique(cluster_indices)),
            'junk_fraction': particle_stats['junk_fraction'],
            'good_fraction': 1 - particle_stats['junk_fraction']
        }
    }
    

    # Save main results file
    results_file = os.path.join(output_folder, f'particle_classifications_{zdim_key}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save indices in pipeline-compatible format (for --ind or --particle-ind)
    # These are just the raw numpy arrays that can be directly used
    junk_indices_file = os.path.join(output_folder, f'junk_indices_{zdim_key}.pkl')
    with open(junk_indices_file, 'wb') as f:
        pickle.dump(junk_particles, f)
    
    good_indices_file = os.path.join(output_folder, f'good_indices_{zdim_key}.pkl')
    with open(good_indices_file, 'wb') as f:
        pickle.dump(good_particles, f)
    
    logger.info(f"Particle classifications saved:")
    logger.info(f"  Main results: {results_file}")
    logger.info(f"  Junk indices (pipeline format): {junk_indices_file}")
    logger.info(f"  Good indices (pipeline format): {good_indices_file}")
    
    return results


if __name__ == "__main__":
    main()
