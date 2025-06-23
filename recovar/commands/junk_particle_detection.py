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
import sys
import argparse
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import logging
from recovar import output, relion_functions, plot_utils
from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
import seaborn as sns
import mrcfile
import umap

matplotlib.rcParams["contour.negative_linestyle"] = "solid"
ftu = fourier_transform_utils(jnp)

logger = logging.getLogger(__name__)


def compute_fsc_auc(fsc_curve, grid_size, voxel_size, threshold=1/7):
    """
    Compute the Area Under Curve (AUC) of the FSC curve above threshold.
    Higher AUC indicates better quality.
    """
    freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size=0.5*voxel_size, scaled=True)
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
        return np.trapz(fsc_curve, freq)

    if np.all(~above_threshold):
        return 0.0

    # Find first index below threshold
    idx = np.argmin(above_threshold)
    if idx == 0:
        return 0.0

    # Compute AUC up to that point
    auc = np.trapz(fsc_curve[:idx], freq[:idx])
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
    mean_real = ftu.get_idft3(mean_volume)
    
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
        halfmap1_real = ftu.get_idft3(halfmaps[0].reshape(volume_shape))
        halfmap2_real = ftu.get_idft3(halfmaps[1].reshape(volume_shape))
        
        # Compute combined reconstruction
        combined_recon = (halfmaps[0] + halfmaps[1]) / 2
        combined_real = ftu.get_idft3(combined_recon.reshape(volume_shape))
        
        # Apply low-pass filtering if requested
        if filter_resolution is not None:
            from recovar import regularization
            
            # Convert resolution to frequency
            freq_threshold = 1.0 / filter_resolution  # 1/Angstrom
            
            # Get frequency grid
            freq_grid = ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size=voxel_size, scaled=True)
            freq_magnitude = jnp.linalg.norm(freq_grid, axis=-1)
            
            # Create low-pass filter
            # Keep only the first filter_fourier_shells shells
            max_freq = freq_threshold * (filter_fourier_shells / 10.0)  # Scale based on requested shells
            low_pass_filter = (freq_magnitude <= max_freq).astype(np.float32)
            
            # Apply filter in Fourier space
            combined_recon_filtered = combined_recon * low_pass_filter.reshape(-1)
            combined_real_filtered = ftu.get_idft3(combined_recon_filtered.reshape(volume_shape))
            
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
    
    # Set up seaborn styling for better-looking plots
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
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
    
    # Color code curves by FSC score with better transparency
    colors = plt.cm.viridis(np.array(halfmap_fscs))
    for i, curve in enumerate(all_fsc_curves):
        ax.plot(freq_axis, curve, color=colors[i], alpha=0.4, linewidth=0.8)
    
    # Mean and IQR with better styling
    all_fsc_array = np.array(all_fsc_curves)
    mean_curve = np.mean(all_fsc_array, axis=0)
    q25 = np.percentile(all_fsc_array, 25, axis=0)
    q75 = np.percentile(all_fsc_array, 75, axis=0)
    ax.plot(freq_axis, mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25, q75, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('Fourier Shell Correlation')
    ax.set_title(f'Half-map FSC Curves for All Clusters (n={len(cluster_centers)})')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(halfmap_fscs), vmax=max(halfmap_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Half-map FSC Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'all_halfmap_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot all vs-mean FSC curves with improved styling ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color code curves by vs-mean FSC score
    colors = plt.cm.viridis(np.array(vs_mean_fscs))
    for i, curve in enumerate(all_vs_mean_curves):
        ax.plot(freq_axis, curve, color=colors[i], alpha=0.4, linewidth=0.8)
    
    # Mean and IQR
    all_vs_mean_array = np.array(all_vs_mean_curves)
    mean_vs_mean_curve = np.mean(all_vs_mean_array, axis=0)
    q25_vs_mean = np.percentile(all_vs_mean_array, 25, axis=0)
    q75_vs_mean = np.percentile(all_vs_mean_array, 75, axis=0)
    ax.plot(freq_axis, mean_vs_mean_curve, color='red', linewidth=3, label='Mean FSC', zorder=10)
    ax.fill_between(freq_axis, q25_vs_mean, q75_vs_mean, color='orange', alpha=0.2, label='IQR (25-75%)')
    
    # Add threshold line
    ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.8, label='FSC=1/7 threshold', linewidth=2)
    
    ax.set_xlabel('Spatial Frequency (1/Å)')
    ax.set_ylabel('Fourier Shell Correlation')
    ax.set_title(f'vs-Mean FSC Curves for All Clusters (n={len(cluster_centers)})')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, freq_axis[-1])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(vs_mean_fscs), vmax=max(vs_mean_fscs)))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('vs-Mean FSC Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'all_vs_mean_fsc_curves_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Individual cluster FSC plots (top 10 and bottom 10) with better layout ---
    sorted_indices = np.argsort(halfmap_fscs)
    top_10 = sorted_indices[-10:]
    bottom_10 = sorted_indices[:10]
    
    # Plot top 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Top 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98)
    
    for i, cluster_idx in enumerate(top_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {len(sorted_indices)-i})', fontsize=10)
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'top_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot bottom 10 clusters with improved layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Bottom 10 Clusters by Half-map FSC Score', fontsize=16, y=0.98)
    
    for i, cluster_idx in enumerate(bottom_10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        ax.plot(freq_axis, all_fsc_curves[cluster_idx], 'b-', linewidth=2, label=f'FSC={halfmap_fscs[cluster_idx]:.3f}')
        ax.plot(freq_axis, all_vs_mean_curves[cluster_idx], 'r-', linewidth=2, label=f'vs-Mean={vs_mean_fscs[cluster_idx]:.3f}')
        ax.axhline(y=1/7, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(f'Cluster {cluster_idx}\n(Rank {i+1})', fontsize=10)
        ax.set_ylabel('FSC' if col == 0 else '')
        ax.set_xlabel('Freq (1/Å)' if row == 1 else '')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, freq_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'bottom_10_clusters_fsc_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Create main summary plot with hexbin visualizations ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f'Junk Particle Detection Summary (zdim={zdim_key}, n_clusters={len(cluster_centers)})', 
                 fontsize=20, y=0.95)

    # 1. Latent space colored by cluster (scatter plot with small points)
    ax = axes[0, 0]
    # Use a color palette that works well for many categories
    n_clusters = len(cluster_centers)
    colors_cluster = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    if n_clusters > 20:
        # Repeat colors if we have more clusters
        colors_cluster = np.tile(colors_cluster, (int(np.ceil(n_clusters/20)), 1))[:n_clusters]
    
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=cluster_indices, 
                        cmap=matplotlib.colors.ListedColormap(colors_cluster), 
                        s=0.5, alpha=0.7, rasterized=True)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100, 
               linewidth=2, label='Cluster Centers', zorder=10)
    ax.set_title('Latent Space by Cluster ID')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Latent space colored by Half-map FSC AUC (hexbin)
    ax = axes[0, 1]
    particle_halfmap_aucs = np.array([halfmap_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs[:, 0], zs[:, 1], C=particle_halfmap_aucs, gridsize=50, 
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean Half-map FSC AUC')
    ax.set_title('Latent Space by Half-map FSC AUC')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.grid(True, alpha=0.3)

    # 3. Latent space colored by FSC vs Mean AUC (hexbin)
    ax = axes[1, 0]
    particle_vs_mean_aucs = np.array([vs_mean_aucs[i] for i in cluster_indices])
    hb = ax.hexbin(zs[:, 0], zs[:, 1], C=particle_vs_mean_aucs, gridsize=50, 
                   cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Mean FSC vs Mean AUC')
    ax.set_title('Latent Space by FSC vs Mean AUC')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram of particle counts per cluster
    ax = axes[1, 1]
    cluster_counts = np.bincount(cluster_indices)
    sns.histplot(data=cluster_counts, bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(cluster_counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cluster_counts):.1f}')
    ax.set_title('Particle Counts per Cluster')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Number of Clusters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Particle Usage Visualization (simplified and cleaner) ---
    if zs.shape[1] >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Particle Usage Analysis', fontsize=18, y=0.95)
        
        # Plot 1: All particles density (hexbin)
        ax = axes[0, 0]
        hb = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=60, cmap='Blues', mincnt=1)
        ax.scatter(zs[:, 0], zs[:, 1], c=cluster_indices, 
                   cmap=matplotlib.colors.ListedColormap(colors_cluster), 
                   s=0.5, alpha=0.7, rasterized=True)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Particle Density')
        ax.set_title('Particle Density in Latent Space')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Particles used for FSC calculation (colored by cluster FSC score)
        ax = axes[0, 1]
        all_used_particles = []
        for cluster_idx in range(len(cluster_centers)):
            all_used_particles.extend(particle_usage[cluster_idx]['all_particles'])
        all_used_particles = np.array(all_used_particles)
        
        if len(all_used_particles) > 0:
            used_particle_fsc_scores = []
            for particle_idx in all_used_particles:
                cluster_idx = cluster_indices[particle_idx]
                used_particle_fsc_scores.append(halfmap_fscs[cluster_idx])
            
            hb = ax.hexbin(zs[all_used_particles, 0], zs[all_used_particles, 1], 
                          C=used_particle_fsc_scores, gridsize=50, cmap='viridis', 
                          reduce_C_function=np.mean, mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label('Mean Cluster FSC Score')
        
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        ax.set_title('Particles Used for FSC Calculation\n(colored by cluster FSC score)')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Particle usage frequency
        ax = axes[0, 2]
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        hb = ax.hexbin(zs[:, 0], zs[:, 1], C=usage_counts, gridsize=60, 
                      cmap='plasma', reduce_C_function=np.mean, mincnt=1)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Usage Count')
        ax.set_title('Particle Usage Frequency\n(how many clusters use each particle)')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Top 5 clusters - particles used for FSC
        ax = axes[1, 0]
        top_5 = sorted_indices[-5:]
        colors_top5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(top_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs[used_particles, 0], zs[used_particles, 1], 
                      c=[colors_top5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers[top_5, 0], cluster_centers[top_5, 1], 
                  c=colors_top5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Top 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Bottom 5 clusters - particles used for FSC
        ax = axes[1, 1]
        bottom_5 = sorted_indices[:5]
        colors_bottom5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(bottom_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs[used_particles, 0], zs[used_particles, 1], 
                      c=[colors_bottom5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers[bottom_5, 0], cluster_centers[bottom_5, 1], 
                  c=colors_bottom5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Bottom 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Usage statistics
        ax = axes[1, 2]
        
        # Calculate usage statistics
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        # Create histogram of usage counts
        unique_counts, count_frequencies = np.unique(usage_counts, return_counts=True)
        
        bars = ax.bar(unique_counts, count_frequencies, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Clusters Using Particle')
        ax.set_ylabel('Number of Particles')
        ax.set_title('Particle Usage Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_usage = np.mean(usage_counts)
        median_usage = np.median(usage_counts)
        unused_particles = np.sum(usage_counts == 0)
        total_particles = len(usage_counts)
        
        stats_text = f'Mean usage: {mean_usage:.1f}\nMedian usage: {median_usage:.1f}\nUnused particles: {unused_particles}\n({unused_particles/total_particles*100:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(os.path.join(output_folder, f'particle_usage_visualization_{zdim_key}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # --- UMAP Visualization of Particle Usage Analysis ---
    if zs.shape[1] >= 2:
        logger.info("Computing UMAP embedding for visualization...")
        
        # Compute UMAP embedding
        n_components = min(zs.shape[1], 2)
        mapper = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1).fit(zs)
        zs_umap = mapper.transform(zs)
        cluster_centers_umap = mapper.transform(cluster_centers)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Particle Usage Analysis (UMAP)', fontsize=18, y=0.95)
        
        # Plot 1: All particles density (hexbin) in UMAP space
        ax = axes[0, 0]
        hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Blues', mincnt=1)
        ax.scatter(zs_umap[:, 0], zs_umap[:, 1], c=cluster_indices, 
                   cmap=matplotlib.colors.ListedColormap(colors_cluster), 
                   s=0.5, alpha=0.7, rasterized=True)
        ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Particle Density')
        ax.set_title('Particle Density in UMAP Space')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Particles used for FSC calculation (colored by cluster FSC score) in UMAP space
        ax = axes[0, 1]
        all_used_particles = []
        for cluster_idx in range(len(cluster_centers)):
            all_used_particles.extend(particle_usage[cluster_idx]['all_particles'])
        all_used_particles = np.array(all_used_particles)
        
        if len(all_used_particles) > 0:
            used_particle_fsc_scores = []
            for particle_idx in all_used_particles:
                cluster_idx = cluster_indices[particle_idx]
                used_particle_fsc_scores.append(halfmap_fscs[cluster_idx])
            
            hb = ax.hexbin(zs_umap[all_used_particles, 0], zs_umap[all_used_particles, 1], 
                          C=used_particle_fsc_scores, gridsize=50, cmap='viridis', 
                          reduce_C_function=np.mean, mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label('Mean Cluster FSC Score')
        
        ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        ax.set_title('Particles Used for FSC Calculation\n(colored by cluster FSC score)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Particle usage frequency in UMAP space
        ax = axes[0, 2]
        usage_counts = np.zeros(len(zs))
        for cluster_idx in range(len(cluster_centers)):
            used_particles = particle_usage[cluster_idx]['all_particles']
            usage_counts[used_particles] += 1
        
        hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], C=usage_counts, gridsize=60, 
                      cmap='plasma', reduce_C_function=np.mean, mincnt=1)
        ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Usage Count')
        ax.set_title('Particle Usage Frequency\n(how many clusters use each particle)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Top 5 clusters - particles used for FSC in UMAP space
        ax = axes[1, 0]
        top_5 = sorted_indices[-5:]
        colors_top5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(top_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs_umap[used_particles, 0], zs_umap[used_particles, 1], 
                      c=[colors_top5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers_umap[top_5, 0], cluster_centers_umap[top_5, 1], 
                  c=colors_top5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Top 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Bottom 5 clusters - particles used for FSC in UMAP space
        ax = axes[1, 1]
        bottom_5 = sorted_indices[:5]
        colors_bottom5 = plt.cm.viridis(np.linspace(0, 1, 5))
        
        # Create a background hexbin for all particles
        hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
        
        for i, cluster_idx in enumerate(bottom_5):
            used_particles = particle_usage[cluster_idx]['all_particles']
            ax.scatter(zs_umap[used_particles, 0], zs_umap[used_particles, 1], 
                      c=[colors_bottom5[i]], alpha=0.8, s=30, 
                      label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
        ax.scatter(cluster_centers_umap[bottom_5, 0], cluster_centers_umap[bottom_5, 1], 
                  c=colors_bottom5, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
        ax.set_title('Bottom 5 Clusters - Particles Used for FSC')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, 1))
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Cluster quality distribution in UMAP space
        ax = axes[1, 2]
        
        # Color particles by their cluster's FSC score
        particle_fsc_scores = np.array([halfmap_fscs[i] for i in cluster_indices])
        
        hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], C=particle_fsc_scores, gridsize=60, 
                      cmap='viridis', reduce_C_function=np.mean, mincnt=1)
        ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
                   edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Mean Cluster FSC Score')
        ax.set_title('Cluster Quality Distribution\n(colored by cluster FSC score)')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(os.path.join(output_folder, f'particle_usage_visualization_umap_{zdim_key}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save UMAP coordinates for potential future use
        umap_data = {
            'zs_umap': zs_umap,
            'cluster_centers_umap': cluster_centers_umap,
            'mapper': mapper
        }
        with open(os.path.join(output_folder, f'umap_coordinates_{zdim_key}.pkl'), 'wb') as f:
            pickle.dump(umap_data, f)
        
        logger.info("UMAP visualization completed and saved.")
    
    # --- Create comprehensive analysis plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FSC Score Analysis', fontsize=16, y=0.95)
    
    # Plot 1: Half-map FSC histogram
    ax = axes[0, 0]
    sns.histplot(data=halfmap_fscs, bins=20, ax=ax, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: vs-Mean FSC histogram
    ax = axes[0, 1]
    sns.histplot(data=vs_mean_fscs, bins=20, ax=ax, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(vs_mean_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(vs_mean_fscs):.3f}')
    ax.set_xlabel('vs-Mean FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of vs-Mean FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Half-map FSC AUC histogram
    ax = axes[0, 2]
    sns.histplot(data=halfmap_aucs, bins=20, ax=ax, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_aucs):.3f}')
    ax.set_xlabel('Half-map FSC AUC')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of Half-map FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: vs-Mean FSC AUC histogram
    ax = axes[1, 0]
    sns.histplot(data=vs_mean_aucs, bins=20, ax=ax, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(vs_mean_aucs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(vs_mean_aucs):.3f}')
    ax.set_xlabel('vs-Mean FSC AUC')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of vs-Mean FSC AUC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Half-map FSC vs Half-map FSC AUC scatter
    ax = axes[1, 1]
    scatter = ax.scatter(halfmap_fscs, halfmap_aucs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Half-map FSC AUC')
    ax.set_title('Half-map FSC Score vs Half-map FSC AUC')
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Half-map FSC vs vs-Mean FSC comparison
    ax = axes[1, 2]
    scatter = ax.scatter(halfmap_fscs, vs_mean_fscs, c=halfmap_fscs, cmap='viridis', alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Half-map FSC')
    ax.set_ylabel('vs-Mean FSC')
    ax.set_title('Half-map FSC vs vs-Mean FSC')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Half-map FSC Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(output_folder, f'fsc_analysis_{zdim_key}.png'), dpi=300, bbox_inches='tight')
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
    
    # Combined outlier detection (clusters that are outliers in either metric)
    combined_junk_clusters = []
    for i in range(len(cluster_centers)):
        if (halfmap_fscs[i] < halfmap_fsc_threshold or halfmap_aucs[i] < halfmap_auc_threshold or
            vs_mean_fscs[i] < vs_mean_fsc_threshold or vs_mean_aucs[i] < vs_mean_auc_threshold):
            combined_junk_clusters.append(i)
    
    # Half-map FSC only outlier detection
    halfmap_junk_clusters = []
    for i in range(len(cluster_centers)):
        if halfmap_fscs[i] < halfmap_fsc_threshold or halfmap_aucs[i] < halfmap_auc_threshold:
            halfmap_junk_clusters.append(i)
    
    # vs-Mean FSC only outlier detection
    vs_mean_junk_clusters = []
    for i in range(len(cluster_centers)):
        if vs_mean_fscs[i] < vs_mean_fsc_threshold or vs_mean_aucs[i] < vs_mean_auc_threshold:
            vs_mean_junk_clusters.append(i)
    
    # Save junk cluster indices for each method
    combined_junk_particle_indices = np.where(np.isin(cluster_indices, combined_junk_clusters))[0]
    halfmap_junk_particle_indices = np.where(np.isin(cluster_indices, halfmap_junk_clusters))[0]
    vs_mean_junk_particle_indices = np.where(np.isin(cluster_indices, vs_mean_junk_clusters))[0]
    
    with open(os.path.join(output_folder, f'combined_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(combined_junk_particle_indices, f)
    with open(os.path.join(output_folder, f'halfmap_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(halfmap_junk_particle_indices, f)
    with open(os.path.join(output_folder, f'vs_mean_junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(vs_mean_junk_particle_indices, f)
    
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


def plot_umap_visualization(zs, cluster_centers, cluster_indices, fsc_scores, output_folder, zdim_key):
    """
    Create UMAP visualization of latent space with cluster analysis.
    
    Parameters:
    - zs: Latent embeddings
    - cluster_centers: K-means cluster centers
    - cluster_indices: Cluster assignments for each particle
    - fsc_scores: Dictionary with FSC scores for each cluster
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
    
    # Extract FSC scores
    halfmap_fscs = [fsc_scores[i]['halfmap_fsc'] for i in range(len(cluster_centers))]
    vs_mean_fscs = [fsc_scores[i]['vs_mean_fsc'] for i in range(len(cluster_centers))]
    
    # Create comprehensive UMAP visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'UMAP Visualization of Latent Space (zdim={zdim_key})', fontsize=18, y=0.95)
    
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
    
    # Plot 2: Particles colored by cluster ID
    ax = axes[0, 1]
    n_clusters = len(cluster_centers)
    colors_cluster = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
    if n_clusters > 20:
        colors_cluster = np.tile(colors_cluster, (int(np.ceil(n_clusters/20)), 1))[:n_clusters]
    
    scatter = ax.scatter(zs_umap[:, 0], zs_umap[:, 1], c=cluster_indices, 
                        cmap=matplotlib.colors.ListedColormap(colors_cluster), 
                        s=0.5, alpha=0.7, rasterized=True)
    ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
               edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
    ax.set_title('Particles by Cluster ID')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Particles colored by cluster FSC score
    ax = axes[0, 2]
    particle_fsc_scores = np.array([halfmap_fscs[i] for i in cluster_indices])
    hb = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], C=particle_fsc_scores, gridsize=60, 
                  cmap='viridis', reduce_C_function=np.mean, mincnt=1)
    ax.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c='red', s=100, 
               edgecolors='black', marker='x', linewidth=2, zorder=10, label='Cluster Centers')
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Cluster FSC Score')
    ax.set_title('Particles by Cluster FSC Score')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Top 10 clusters
    ax = axes[1, 0]
    sorted_indices = np.argsort(halfmap_fscs)
    top_10 = sorted_indices[-10:]
    colors_top10 = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Create a background hexbin for all particles
    hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
    
    for i, cluster_idx in enumerate(top_10):
        cluster_mask = cluster_indices == cluster_idx
        ax.scatter(zs_umap[cluster_mask, 0], zs_umap[cluster_mask, 1], 
                  c=[colors_top10[i]], alpha=0.8, s=20, 
                  label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
    ax.scatter(cluster_centers_umap[top_10, 0], cluster_centers_umap[top_10, 1], 
              c=colors_top10, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
    ax.set_title('Top 10 Clusters by FSC Score')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Bottom 10 clusters
    ax = axes[1, 1]
    bottom_10 = sorted_indices[:10]
    colors_bottom10 = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Create a background hexbin for all particles
    hb_bg = ax.hexbin(zs_umap[:, 0], zs_umap[:, 1], gridsize=60, cmap='Greys', alpha=0.3, mincnt=1)
    
    for i, cluster_idx in enumerate(bottom_10):
        cluster_mask = cluster_indices == cluster_idx
        ax.scatter(zs_umap[cluster_mask, 0], zs_umap[cluster_mask, 1], 
                  c=[colors_bottom10[i]], alpha=0.8, s=20, 
                  label=f'Cluster {cluster_idx} (FSC={halfmap_fscs[cluster_idx]:.3f})')
    ax.scatter(cluster_centers_umap[bottom_10, 0], cluster_centers_umap[bottom_10, 1], 
              c=colors_bottom10, s=150, edgecolors='black', marker='x', linewidth=3, zorder=10)
    ax.set_title('Bottom 10 Clusters by FSC Score')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    # Plot 6: FSC score distribution
    ax = axes[1, 2]
    sns.histplot(data=halfmap_fscs, bins=20, ax=ax, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(halfmap_fscs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    ax.set_xlabel('Half-map FSC Score')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Distribution of FSC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(output_folder, f'umap_visualization_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save UMAP coordinates for potential future use
    umap_data = {
        'zs_umap': zs_umap,
        'cluster_centers_umap': cluster_centers_umap,
        'mapper': mapper
    }
    with open(os.path.join(output_folder, f'umap_coordinates_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(umap_data, f)
    
    logger.info("UMAP visualization completed and saved.")
    return umap_data


def junk_particle_detection(recovar_result_dir, output_folder=None, zdim=10, n_clusters=100, 
                           batch_size=100, n_particles_per_cluster=10, no_z_regularization=False,
                           save_reconstructions=False, filter_resolution=None, filter_fourier_shells=10):
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
    
    Algorithm:
    1. Performs k-means clustering on latent embeddings
    2. For each cluster center, finds the n_particles_per_cluster closest particles in latent space
    3. Splits these particles into two halfsets for halfmap generation
    4. Maps global indices to local indices for each existing dataset halfset
    5. Uses relion-style reconstruction with subset generators from existing dataset splits
    6. Computes FSC scores between halfmaps and against mean reconstruction
    7. Identifies junk clusters based on low FSC scores
    8. If filter_resolution is provided, applies low-pass filtering to combined reconstructions
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
    
    # Load pipeline output
    pipeline_output = output.PipelineOutput(recovar_result_dir)
    
    
    # Determine zdim key
    zdim_key = f"{zdim}_noreg" if no_z_regularization else zdim
    
    if zdim_key not in pipeline_output.get('zs'):
        available_dims = list(pipeline_output.get('zs').keys())
        logger.error(f"zdim {zdim_key} not found. Available dimensions: {available_dims}")
        raise ValueError(f"zdim {zdim_key} not found")
    
    # Load embeddings
    zs = pipeline_output.get('zs')[zdim_key]
    logger.info(f"Loaded embeddings with shape: {zs.shape}")
    
    # Perform k-means clustering
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
    
    if save_reconstructions:
        fsc_scores, fsc_auc_scores, particle_usage, reconstructions = result
    else:
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
    try:
        umap_data = plot_umap_visualization(
            zs, cluster_centers, cluster_indices, fsc_scores, output_folder, zdim_key
        )
        # Add UMAP data to junk_info for potential future use
        junk_info['umap_data'] = umap_data
    except Exception as e:
        logger.warning(f"UMAP visualization failed: {e}")
    
    # Save reconstruction info if reconstructions were saved
    if save_reconstructions and reconstructions is not None:
        reconstructions_info_path = os.path.join(output_folder, f'reconstructions_info_{zdim_key}.pkl')
        with open(reconstructions_info_path, 'wb') as f:
            pickle.dump(reconstructions, f)
        logger.info(f"Saved reconstruction info to {reconstructions_info_path}")
    
    logger.info(f"Junk particle detection complete. Results saved to {output_folder}")
    return junk_info


def add_args(parser):
    """Add command line arguments for junk particle detection."""
    parser.add_argument("input_dir", type=str, help="Directory where the recovar results are stored.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for results.")
    parser.add_argument("--zdim", type=int, default=4, help="Dimension of the zs array to use.")
    parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of k-means clusters.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for reconstruction.")
    parser.add_argument("--n-particles-per-cluster", type=int, default=10, 
                       help="Number of particles per halfmap (so 2*n_particles_per_cluster total for reconstruction).")
    parser.add_argument("--save-reconstructions", action="store_true", 
                       help="Save reconstructions (halfmaps and combined) to MRC files.")
    parser.add_argument("--filter-resolution", type=float, default=None,
                       help="Resolution in Angstroms to filter combined reconstructions (if None, no filtering).")
    parser.add_argument("--filter-fourier-shells", type=int, default=10,
                       help="Number of Fourier shells to keep when filtering (default: 10).")
    
    return parser


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Detect junk particles from latent space using clustering and FSC analysis.")
    args = add_args(parser).parse_args()
    
    junk_particle_detection(
        args.input_dir,
        args.output_dir,
        args.zdim,
        args.n_clusters,
        args.batch_size,
        args.n_particles_per_cluster,
        args.no_z_regularization,
        args.save_reconstructions,
        args.filter_resolution,
        args.filter_fourier_shells
    )


if __name__ == "__main__":
    main() 