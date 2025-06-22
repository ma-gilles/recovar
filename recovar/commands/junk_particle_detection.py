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
    freq = freq[1:]
    fsc_curve = fsc_curve[1:]
    
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
                              zdim_key, batch_size=100, n_particles_per_cluster=100):
    """
    Compute FSC scores for each cluster by generating halfmaps and comparing them.
    
    Parameters:
    - pipeline_output: PipelineOutput object
    - cluster_centers: K-means cluster centers
    - cluster_indices: Cluster assignments for each particle
    - zdim_key: Dimension key for embeddings
    - batch_size: Batch size for reconstruction
    - n_particles_per_cluster: Number of particles to use per halfmap (so 2*n_particles_per_cluster total)
    
    Returns:
    - fsc_scores: Dictionary with FSC scores for each cluster
    - fsc_auc_scores: Dictionary with FSC AUC scores for each cluster
    
    Note:
    - Uses the n_particles_per_cluster closest particles in latent space to each cluster center
    - Particles can come from any cluster if they're closer in latent space
    - Creates two halfmaps with n_particles_per_cluster particles each
    - Uses relion-style reconstruction with existing dataset splits from pipeline output
    - Maps global particle indices to local indices for each halfset dataset
    """
    cryos = pipeline_output.get('dataset')  # This returns [cryo1, cryo2] for the two halfsets
    zs = pipeline_output.get('zs')[zdim_key]
    volume_shape = pipeline_output.get('volume_shape')
    voxel_size = pipeline_output.get('voxel_size')
    
    # Get mean reconstruction for comparison
    mean_volume = pipeline_output.get('mean')
    
    fsc_scores = {}
    fsc_auc_scores = {}
    
    for cluster_idx in range(len(cluster_centers)):
        logger.info(f"Processing cluster {cluster_idx + 1}/{len(cluster_centers)}")
        
        # Find the n_particles_per_cluster closest particles to this cluster center in latent space
        cluster_center = cluster_centers[cluster_idx]
        distances = np.linalg.norm(zs - cluster_center, axis=1)
        closest_indices = np.argsort(distances)[:n_particles_per_cluster]
        
        logger.info(f"Cluster {cluster_idx}: Using {len(closest_indices)} closest particles (min distance: {distances[closest_indices[0]]:.3f}, max distance: {distances[closest_indices[-1]]:.3f})")
        
        # Create temporary directory for this cluster
        temp_dir = f"/tmp/cluster_{cluster_idx}_reconstruction"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Split the closest particles into two halfsets for halfmap generation
            # We need 2 * n_particles_per_cluster total particles (n_particles_per_cluster per halfmap)
            total_needed = 2 * n_particles_per_cluster
            
            if len(closest_indices) < total_needed:
                logger.warning(f"Cluster {cluster_idx}: Only {len(closest_indices)} particles available, need {total_needed}. Using all available particles.")
                # Use all available particles, split as evenly as possible
                n_half = len(closest_indices) // 2
                halfset1_indices = closest_indices[:n_half]
                halfset2_indices = closest_indices[n_half:2*n_half]
            else:
                # Use exactly n_particles_per_cluster per halfmap
                halfset1_indices = closest_indices[:n_particles_per_cluster]
                halfset2_indices = closest_indices[n_particles_per_cluster:2*n_particles_per_cluster]
            
            logger.info(f"Cluster {cluster_idx}: Using {len(halfset1_indices)} particles for halfmap1, {len(halfset2_indices)} particles for halfmap2")
            
            # Map global indices to local indices for each halfset
            # cryos[0].dataset_indices maps local indices to global indices
            # We need to find which local indices correspond to our global indices
            halfset1_local_indices = []
            halfset2_local_indices = []
            
            for global_idx in halfset1_indices:
                local_idx = np.where(cryos[0].dataset_indices == global_idx)[0]
                if len(local_idx) > 0:
                    halfset1_local_indices.append(local_idx[0])
            
            for global_idx in halfset2_indices:
                local_idx = np.where(cryos[1].dataset_indices == global_idx)[0]
                if len(local_idx) > 0:
                    halfset2_local_indices.append(local_idx[0])
            
            halfset1_local_indices = np.array(halfset1_local_indices)
            halfset2_local_indices = np.array(halfset2_local_indices)
            
            logger.info(f"Cluster {cluster_idx}: Mapped to {len(halfset1_local_indices)} local indices for halfmap1, {len(halfset2_local_indices)} for halfmap2")
            
            # Get noise variance (use mean if available, otherwise estimate)
            if hasattr(cryos[0], 'noise') and cryos[0].noise is not None:
                noise_variance1 = cryos[0].noise.get(halfset1_local_indices)
                noise_variance2 = cryos[1].noise.get(halfset2_local_indices)
            else:
                # Use a default noise variance if not available
                noise_variance1 = np.ones(len(halfset1_local_indices), dtype=np.float32)
                noise_variance2 = np.ones(len(halfset2_local_indices), dtype=np.float32)
            
            # Generate halfmap 1 using subset generator from first halfset dataset
            cryos[0].update_volume_upsampling_factor(2)  # Use 2x upsampling
            Ft_ctf1, F_ty1 = relion_functions.relion_style_triangular_kernel(
                cryos[0], noise_variance1, batch_size=None, 
                disc_type='linear_interp', 
                data_generator=cryos[0].get_dataset_subset_generator(batch_size, halfset1_local_indices)
            )
            halfmap1 = relion_functions.post_process_from_filter(
                cryos[0], Ft_ctf1, F_ty1, 
                disc_type='linear_interp', use_spherical_mask=True, 
                grid_correct=True, gridding_correct="square"
            )
            
            # Generate halfmap 2 using subset generator from second halfset dataset
            cryos[1].update_volume_upsampling_factor(2)  # Use 2x upsampling
            Ft_ctf2, F_ty2 = relion_functions.relion_style_triangular_kernel(
                cryos[1], noise_variance2, batch_size=None, 
                disc_type='linear_interp',
                data_generator=cryos[1].get_dataset_subset_generator(batch_size, halfset2_local_indices)
            )
            halfmap2 = relion_functions.post_process_from_filter(
                cryos[1], Ft_ctf2, F_ty2, 
                disc_type='linear_interp', use_spherical_mask=True, 
                grid_correct=True, gridding_correct="square"
            )
            
            # Convert to real space for FSC computation
            halfmap1_real = ftu.get_idft3(halfmap1.reshape(volume_shape))
            halfmap2_real = ftu.get_idft3(halfmap2.reshape(volume_shape))
            
            # Compute FSC between halfmaps
            fsc_curve = plot_utils.FSC(halfmap1_real, halfmap2_real)
            fsc_score = plot_utils.fsc_score(fsc_curve, volume_shape[0], voxel_size, threshold=1/7)
            fsc_auc = compute_fsc_auc(fsc_curve, volume_shape[0], voxel_size, threshold=1/7)
            
            # Compute FSC between cluster reconstruction and mean
            cluster_recon = (halfmap1_real + halfmap2_real) / 2
            mean_real = ftu.get_idft3(mean_volume.reshape(volume_shape))
            fsc_vs_mean_curve = plot_utils.FSC(cluster_recon, mean_real)
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
            
            logger.info(f"Cluster {cluster_idx}: FSC={fsc_score:.3f}, AUC={fsc_auc:.3f}, vs_mean_FSC={fsc_vs_mean_score:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing cluster {cluster_idx}: {e}")
            fsc_scores[cluster_idx] = {'halfmap_fsc': 0.0, 'vs_mean_fsc': 0.0}
            fsc_auc_scores[cluster_idx] = {'halfmap_auc': 0.0, 'vs_mean_auc': 0.0}
        
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    return fsc_scores, fsc_auc_scores


def plot_junk_detection_results(zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, 
                               output_folder, zdim_key):
    """
    Create plots for junk particle detection results.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract FSC scores
    halfmap_fscs = [fsc_scores[i]['halfmap_fsc'] for i in range(len(cluster_centers))]
    vs_mean_fscs = [fsc_scores[i]['vs_mean_fsc'] for i in range(len(cluster_centers))]
    halfmap_aucs = [fsc_auc_scores[i]['halfmap_auc'] for i in range(len(cluster_centers))]
    vs_mean_aucs = [fsc_auc_scores[i]['vs_mean_auc'] for i in range(len(cluster_centers))]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: FSC histogram
    axes[0, 0].hist(halfmap_fscs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('FSC Score')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('Distribution of FSC Scores')
    axes[0, 0].axvline(np.mean(halfmap_fscs), color='red', linestyle='--', label=f'Mean: {np.mean(halfmap_fscs):.3f}')
    axes[0, 0].legend()
    
    # Plot 2: FSC AUC histogram
    axes[0, 1].hist(halfmap_aucs, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('FSC AUC')
    axes[0, 1].set_ylabel('Number of Clusters')
    axes[0, 1].set_title('Distribution of FSC AUC Scores')
    axes[0, 1].axvline(np.mean(halfmap_aucs), color='red', linestyle='--', label=f'Mean: {np.mean(halfmap_aucs):.3f}')
    axes[0, 1].legend()
    
    # Plot 3: FSC vs FSC AUC scatter
    axes[0, 2].scatter(halfmap_fscs, halfmap_aucs, alpha=0.7)
    axes[0, 2].set_xlabel('FSC Score')
    axes[0, 2].set_ylabel('FSC AUC')
    axes[0, 2].set_title('FSC Score vs FSC AUC')
    
    # Plot 4: Latent space with cluster centers colored by FSC
    if zs.shape[1] >= 2:
        scatter = axes[1, 0].scatter(zs[:, 0], zs[:, 1], c=cluster_indices, cmap='tab20', alpha=0.3, s=1)
        axes[1, 0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                          c=halfmap_fscs, cmap='viridis', s=100, edgecolors='black')
        axes[1, 0].set_xlabel('Latent Dimension 1')
        axes[1, 0].set_ylabel('Latent Dimension 2')
        axes[1, 0].set_title('Latent Space (colored by FSC)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')
    
    # Plot 5: FSC vs Mean FSC comparison
    axes[1, 1].scatter(halfmap_fscs, vs_mean_fscs, alpha=0.7)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Halfmap FSC')
    axes[1, 1].set_ylabel('vs Mean FSC')
    axes[1, 1].set_title('FSC Comparison')
    
    # Plot 6: Cluster sizes vs FSC
    cluster_sizes = [np.sum(cluster_indices == i) for i in range(len(cluster_centers))]
    axes[1, 2].scatter(cluster_sizes, halfmap_fscs, alpha=0.7)
    axes[1, 2].set_xlabel('Cluster Size')
    axes[1, 2].set_ylabel('FSC Score')
    axes[1, 2].set_title('Cluster Size vs FSC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'cluster_centers': cluster_centers,
        'cluster_indices': cluster_indices,
        'fsc_scores': fsc_scores,
        'fsc_auc_scores': fsc_auc_scores,
        'cluster_sizes': cluster_sizes,
        'halfmap_fscs': halfmap_fscs,
        'halfmap_aucs': halfmap_aucs,
        'vs_mean_fscs': vs_mean_fscs,
        'vs_mean_aucs': vs_mean_aucs
    }
    
    with open(os.path.join(output_folder, f'junk_detection_results_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Identify potential junk clusters (low FSC scores)
    fsc_threshold = np.percentile(halfmap_fscs, 25)  # Bottom 25%
    auc_threshold = np.percentile(halfmap_aucs, 25)  # Bottom 25%
    
    junk_clusters = []
    for i in range(len(cluster_centers)):
        if halfmap_fscs[i] < fsc_threshold or halfmap_aucs[i] < auc_threshold:
            junk_clusters.append(i)
    
    # Save junk cluster indices
    junk_particle_indices = np.where(np.isin(cluster_indices, junk_clusters))[0]
    with open(os.path.join(output_folder, f'junk_particle_indices_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(junk_particle_indices, f)
    
    # Save junk cluster information
    junk_info = {
        'junk_clusters': junk_clusters,
        'junk_particle_indices': junk_particle_indices,
        'fsc_threshold': fsc_threshold,
        'auc_threshold': auc_threshold,
        'total_particles': len(zs),
        'junk_particles': len(junk_particle_indices),
        'junk_percentage': len(junk_particle_indices) / len(zs) * 100
    }
    
    with open(os.path.join(output_folder, f'junk_cluster_info_{zdim_key}.pkl'), 'wb') as f:
        pickle.dump(junk_info, f)
    
    logger.info(f"Identified {len(junk_clusters)} junk clusters out of {len(cluster_centers)} total clusters")
    logger.info(f"Identified {len(junk_particle_indices)} junk particles ({junk_info['junk_percentage']:.1f}%)")
    
    return junk_info


def junk_particle_detection(recovar_result_dir, output_folder=None, zdim=10, n_clusters=100, 
                           batch_size=100, n_particles_per_cluster=100, no_z_regularization=False):
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
    
    Algorithm:
    1. Performs k-means clustering on latent embeddings
    2. For each cluster center, finds the n_particles_per_cluster closest particles in latent space
    3. Splits these particles into two halfsets for halfmap generation
    4. Maps global indices to local indices for each existing dataset halfset
    5. Uses relion-style reconstruction with subset generators from existing dataset splits
    6. Computes FSC scores between halfmaps and against mean reconstruction
    7. Identifies junk clusters based on low FSC scores
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
    fsc_scores, fsc_auc_scores = compute_cluster_fsc_scores(
        pipeline_output, cluster_centers, cluster_indices, zdim_key, 
        batch_size, n_particles_per_cluster
    )
    
    # Create plots and identify junk clusters
    logger.info("Creating plots and identifying junk clusters...")
    junk_info = plot_junk_detection_results(
        zs, cluster_centers, cluster_indices, fsc_scores, fsc_auc_scores, 
        output_folder, zdim_key
    )
    
    logger.info(f"Junk particle detection complete. Results saved to {output_folder}")
    return junk_info


def add_args(parser):
    """Add command line arguments for junk particle detection."""
    parser.add_argument("input_dir", type=str, help="Directory where the recovar results are stored.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for results.")
    parser.add_argument("--zdim", type=int, default=10, help="Dimension of the zs array to use.")
    parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of k-means clusters.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for reconstruction.")
    parser.add_argument("--n-particles-per-cluster", type=int, default=100, 
                       help="Number of particles per halfmap (so 2*n_particles_per_cluster total for reconstruction).")
    
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
        args.no_z_regularization
    )


if __name__ == "__main__":
    main() 