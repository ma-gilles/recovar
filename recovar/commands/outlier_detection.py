import argparse
import logging
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from recovar.data_io import cryo_dataset
from recovar.data_io._index_utils import TiltSeriesOriginalIndexMap
from recovar.output import output

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

# Set up logger
logger = logging.getLogger(__name__)

def plot_anomaly_detection_results(zs, original_indices, folder_name):
    """
    Plots anomaly detection results for given data and saves the plots and inlier/outlier indices.

    Parameters:
    - zs: numpy array
        In the local ordering, loaded from pipeline_output.get(zs)
    - original_indices: numpy array
        The original indices of the particles, loaded from np.concatenate(pipeline_output.get(particles_halfsets))
    - folder_name: str
        The folder name where all files (plots and indices) will be saved in the original ordering
    """
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    n_features = zs.shape[1]
    total_samples = zs.shape[0]

    # If too few samples, anomaly detection is not meaningful — mark all as inliers
    min_samples = max(5, n_features + 2)
    if total_samples < min_samples:
        logger.warning(
            f"Too few samples ({total_samples}) for anomaly detection "
            f"(need >= {min_samples}). Marking all particles as inliers."
        )
        all_inliers = np.arange(total_samples)
        all_inliers_orig = original_indices
        with open(os.path.join(folder_name, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(all_inliers_orig, f)
        with open(os.path.join(folder_name, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([], dtype=int), f)
        return

    # Compute UMAP on valid zs
    umapper = output.umap_latent_space(zs)
    umap_valid = umapper.embedding_

    # Cap LOF n_neighbors to avoid crash when n_samples is small
    lof_n_neighbors = min(35, total_samples - 1)

    # Define anomaly detection algorithms
    anomaly_algorithms = [
        # ("Robust covariance", EllipticEnvelope(random_state=42)),  # Will be added later after contamination is calculated
        ("Isolation Forest", IsolationForest(random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=lof_n_neighbors)),
    ]

    # Initialize lists to store predictions, algorithm names, and outlier percentages
    predictions = []
    algorithm_names = []
    outlier_percentages = []

    # Fit each algorithm (excluding Robust Covariance for now) and store predictions
    for name, algorithm in anomaly_algorithms:
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(zs)
        else:
            y_pred = algorithm.fit(zs).predict(zs)
        predictions.append(y_pred)
        algorithm_names.append(name)

        # Calculate percentage of outliers
        num_outliers = np.sum(y_pred == -1)
        percentage_outliers = (num_outliers / total_samples)
        outlier_percentages.append(percentage_outliers)

        inliers_indices = np.where(y_pred == 1)[0]
        outliers_indices = np.where(y_pred == -1)[0]

        # Sanitize algorithm name for filename
        safe_name = name.replace(" ", "_").lower()

        # Save indices to pickle files in the specified folder
        with open(os.path.join(folder_name, f"inliers_{safe_name}.pkl"), "wb") as f:
            pickle.dump(inliers_indices, f)
        with open(os.path.join(folder_name, f"outliers_{safe_name}.pkl"), "wb") as f:
            pickle.dump(outliers_indices, f)

    # Calculate average outlier percentage from other algorithms
    avg_contamination = np.mean(outlier_percentages)
    # Ensure contamination is within (0, 0.5)
    avg_contamination = min(max(avg_contamination, 0.001), 0.5)

    # Now add Robust Covariance (EllipticEnvelope uses MCD which requires n_samples > n_features + 1)
    if total_samples > n_features + 1:
        robust_covariance = ("Robust covariance", EllipticEnvelope(random_state=42, contamination=avg_contamination))
        anomaly_algorithms.insert(0, robust_covariance)  # Insert at the beginning

        # Fit Robust Covariance
        name, algorithm = robust_covariance
        y_pred = algorithm.fit(zs).predict(zs)
        predictions.insert(0, y_pred)
        algorithm_names.insert(0, name)

        # Save indices of inliers and outliers in original indexing
        inliers_indices = np.where(y_pred == 1)[0]
        outliers_indices = np.where(y_pred == -1)[0]

        # Sanitize algorithm name for filename
        safe_name = name.replace(" ", "_").lower()

        # Save indices to pickle files in the specified folder
        with open(os.path.join(folder_name, f"inliers_{safe_name}.pkl"), "wb") as f:
            pickle.dump(inliers_indices, f)
        with open(os.path.join(folder_name, f"outliers_{safe_name}.pkl"), "wb") as f:
            pickle.dump(outliers_indices, f)

        # Update outlier percentages and total samples
        num_outliers = np.sum(y_pred == -1)
        percentage_outliers = (num_outliers / total_samples)
        outlier_percentages.insert(0, percentage_outliers)
    else:
        logger.warning(
            f"Skipping EllipticEnvelope (Robust covariance): too few samples "
            f"({total_samples}) for n_features={n_features}."
        )

    # Compute consensus
    # Convert predictions to numpy array
    predictions_array = np.array(predictions)
    # Convert outlier labels (-1) to 1 for counting
    outlier_flags = (predictions_array == -1).astype(int)
    # Sum across algorithms
    consensus_scores = outlier_flags.sum(axis=0)
    # Set consensus threshold (majority vote)
    consensus_threshold = len(anomaly_algorithms) // 2 + 1
    # Determine consensus outliers and inliers
    consensus_outliers = consensus_scores >= consensus_threshold
    consensus_inliers = ~consensus_outliers

    # Save consensus indices in original indexing

    inliers_mapped_back_to_original_indices = original_indices[consensus_inliers]
    outliers_mapped_back_to_original_indices = original_indices[consensus_outliers]


    with open(os.path.join(folder_name, "inliers_consensus.pkl"), "wb") as f:
        pickle.dump(inliers_mapped_back_to_original_indices, f)
    with open(os.path.join(folder_name, "outliers_consensus.pkl"), "wb") as f:
        pickle.dump(outliers_mapped_back_to_original_indices, f)

    # Add consensus results to the algorithms for plotting
    algorithm_names.append("Consensus")
    predictions.append(np.where(consensus_outliers, -1, 1))
    outlier_percentages.append(np.mean(consensus_outliers))

    # Prepare for plotting
    n_algorithms = len(algorithm_names)
    n_rows = 3  # zs[:, 0] vs zs[:, 1], zs[:, 2] vs zs[:, 3], umap[:, 0] vs umap[:, 1]
    n_cols = n_algorithms
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False
    )
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.07, top=0.90, wspace=0.3, hspace=0.4
    )

    for i_algo, (name, y_pred) in enumerate(zip(algorithm_names, predictions)):
        # Identify inliers and outliers
        inliers = y_pred == 1
        outliers = y_pred == -1

        # Compute number and percentage of outliers
        num_outliers = np.sum(outliers)
        percentage_outliers = (num_outliers / total_samples) * 100

        # --- Plot zs[:, 0] vs zs[:, 1] ---
        if zs.shape[1] >= 2:
            ax = axes[0, i_algo]
            if np.sum(inliers) > 1:
                # Compute axis limits based on inliers
                x_min, x_max = np.percentile(zs[inliers, 0], [0.1, 99.9])
                y_min, y_max = np.percentile(zs[inliers, 1], [0.1, 99.9])

                hb = ax.hexbin(
                    zs[inliers, 0],
                    zs[inliers, 1],
                    gridsize=50,
                    cmap='Blues',
                    bins='log',
                    mincnt=1
                )
                # Add colorbar
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label('log$_{10}$(N)')
            else:
                ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
                x_min, x_max = zs[:, 0].min(), zs[:, 0].max()
                y_min, y_max = zs[:, 1].min(), zs[:, 1].max()

            # Overlay outliers
            if np.sum(outliers) > 0:
                ax.scatter(
                    zs[outliers, 0],
                    zs[outliers, 1],
                    s=10,
                    c="red",
                    alpha=0.8,
                    label="Outliers",
                    edgecolor="k",
                    linewidth=0.5,
                )
            # Set axis limits for zs plots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Add title with number and percentage of outliers
            ax.set_title(
                f"{name}\n(zs[0] vs zs[1])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
                fontsize=12,
            )
            if i_algo == 0:
                ax.set_ylabel("zs[1]", fontsize=10)
            ax.set_xlabel("zs[0]", fontsize=10)
            ax.legend(loc="upper right")
        else:
            # If zs has less than 2 dimensions, hide the subplot
            axes[0, i_algo].axis('off')

        # --- Plot zs[:, 2] vs zs[:, 3] if available ---
        if zs.shape[1] >= 4:
            ax = axes[1, i_algo]
            if np.sum(inliers) > 1:
                # Compute axis limits based on inliers
                x_min, x_max = np.percentile(zs[inliers, 2], [0.1, 99.9])
                y_min, y_max = np.percentile(zs[inliers, 3], [0.1, 99.9])

                hb = ax.hexbin(
                    zs[inliers, 2],
                    zs[inliers, 3],
                    gridsize=50,
                    cmap='Blues',
                    bins='log',
                    mincnt=1
                )
                # Add colorbar
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label('log$_{10}$(N)')
            else:
                ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
                x_min, x_max = zs[:, 2].min(), zs[:, 2].max()
                y_min, y_max = zs[:, 3].min(), zs[:, 3].max()
            # Overlay outliers
            if np.sum(outliers) > 0:
                ax.scatter(
                    zs[outliers, 2],
                    zs[outliers, 3],
                    s=10,
                    c="red",
                    alpha=0.8,
                    label="Outliers",
                    edgecolor="k",
                    linewidth=0.5,
                )
            # Set axis limits for zs plots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Add title with number and percentage of outliers
            ax.set_title(
                f"{name}\n(zs[2] vs zs[3])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
                fontsize=12,
            )
            if i_algo == 0:
                ax.set_ylabel("zs[3]", fontsize=10)
            ax.set_xlabel("zs[2]", fontsize=10)
            ax.legend(loc="upper right")
        else:
            # If zs has less than 4 dimensions, hide the subplot
            axes[1, i_algo].axis('off')

        # --- Plot umap[:, 0] vs umap[:, 1] ---
        if umap_valid.shape[1] >= 2:
            ax = axes[2, i_algo]
            if np.sum(inliers) > 1:
                hb = ax.hexbin(
                    umap_valid[inliers, 0],
                    umap_valid[inliers, 1],
                    gridsize=50,
                    cmap='Blues',
                    bins='log',
                    mincnt=1
                )
                # Add colorbar
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label('log$_{10}$(N)')
            else:
                ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
            # Overlay outliers
            if np.sum(outliers) > 0:
                ax.scatter(
                    umap_valid[outliers, 0],
                    umap_valid[outliers, 1],
                    s=10,
                    c="red",
                    alpha=0.8,
                    label="Outliers",
                    edgecolor="k",
                    linewidth=0.5,
                )
            # Do not set axis limits for umap plots

            # Add title with number and percentage of outliers
            ax.set_title(
                f"{name}\n(umap[0] vs umap[1])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
                fontsize=12,
            )
            if i_algo == 0:
                ax.set_ylabel("umap[1]", fontsize=10)
            ax.set_xlabel("umap[0]", fontsize=10)
            ax.legend(loc="upper right")
        else:
            # If umap has less than 2 dimensions, hide the subplot
            axes[2, i_algo].axis('off')

    plt.suptitle("Anomaly Detection Results Including Consensus", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to the specified folder
    plot_filename = os.path.join(folder_name, "anomaly_detection_results.png")
    plt.savefig(plot_filename)
    plt.close()
    return inliers_mapped_back_to_original_indices, outliers_mapped_back_to_original_indices



def outlier_detection_from_contrast(pipeline_output, zdim_key=4,
                                   low_contrast_threshold=0.1,
                                   high_contrast_threshold=3.5,
                                   max_contrast=4.0,
                                   particle_bad_fraction_threshold=0.7,
                                   micrograph_bad_fraction_threshold=0.7,
                                   output_dir=None,
                                   noreg=False):
    """
    Perform outlier detection based on contrast values.

    Parameters:
    - pipeline_output: Pipeline output object
    - zdim_key: Dimension key for embeddings (int)
    - low_contrast_threshold: Threshold for low contrast outliers
    - high_contrast_threshold: Threshold for high contrast outliers
    - max_contrast: Maximum contrast value for normalization
    - particle_bad_fraction_threshold: Threshold for particle-based outlier detection
    - micrograph_bad_fraction_threshold: Threshold for micrograph-based outlier detection
    - output_dir: Output directory for saving results
    - noreg: Whether to use unregularized embeddings

    Returns:
    - image_outliers: Array of image-level outlier indices
    - image_inliers: Array of image-level inlier indices
    - particle_outliers: Array of particle-level outlier indices (None if not tilt series)
    - particle_inliers: Array of particle-level inlier indices (None if not tilt series)
    """
    logger.info("Contrast-based outlier detection for zdim=%s", zdim_key)

    contrast_entry = 'contrasts_noreg' if noreg else 'contrasts'
    input_args = pipeline_output.get('input_args')
    starfile = getattr(input_args, 'particles', None)
    contrasts = pipeline_output.get(contrast_entry)[zdim_key]
    contrast_array = np.asarray(contrasts, dtype=np.float32)
    
    # Parse starfile for grouping information
    # Check if this is a tilt-series dataset by looking at pipeline input arguments
    is_tilt_series = getattr(input_args, 'tilt_series', False)
    shared_contrast_across_tilts = getattr(input_args, 'shared_contrast_across_tilts', False)
    
    # Individual image outlier detection
    low_contrast_outliers = contrast_array < low_contrast_threshold
    high_contrast_outliers = contrast_array > high_contrast_threshold
    individual_outliers = low_contrast_outliers | high_contrast_outliers
    
    n_low_contrast = np.sum(low_contrast_outliers)
    n_high_contrast = np.sum(high_contrast_outliers)
    n_individual_outliers = np.sum(individual_outliers)
    
    original_image_indices = np.concatenate(pipeline_output.get('halfsets'))
    original_particle_indices = np.concatenate(pipeline_output.get('particles_halfsets'))
    n_images = len(original_image_indices)

    logger.info("Final contrast array shape: %s", contrast_array.shape)
    logger.info("Contrast-based outlier detection for %d images", n_images)
    logger.info("Low contrast threshold: %s", low_contrast_threshold)
    logger.info("High contrast threshold: %s", high_contrast_threshold)
    logger.info("Particle bad fraction threshold: %s", particle_bad_fraction_threshold)
    logger.info("Micrograph bad fraction threshold: %s", micrograph_bad_fraction_threshold)


    logger.info("Individual image outlier detection:")
    logger.info("  Low contrast outliers (< %s): %d (%.1f%%)",
                low_contrast_threshold, n_low_contrast, n_low_contrast / n_images * 100)
    logger.info("  High contrast outliers (> %s): %d (%.1f%%)",
                high_contrast_threshold, n_high_contrast, n_high_contrast / n_images * 100)
    logger.info("  Total individual outliers: %d (%.1f%%)",
                n_individual_outliers, n_individual_outliers / n_images * 100)

    outliers_ind = np.where(individual_outliers)[0]
    inliers_ind = np.where(~individual_outliers)[0]

    # Initialize return values
    image_outliers = original_image_indices[outliers_ind]
    image_inliers = original_image_indices[inliers_ind]
    particle_outliers = None
    particle_inliers = None

    if not is_tilt_series:
        return image_outliers, image_inliers, particle_outliers, particle_inliers

    # If the dataset is a tilt-series or the contrast is shared across tilts, skip the grouping based on particle or micrograph
    if is_tilt_series and shared_contrast_across_tilts:
        outliers_ind_mapped_back_to_original_indices = original_particle_indices[outliers_ind]
        inliers_ind_mapped_back_to_original_indices = original_particle_indices[inliers_ind]

        return outliers_ind_mapped_back_to_original_indices, inliers_ind_mapped_back_to_original_indices, None, None

    # Only try to parse starfile if it actually has a .star extension
    particle_to_tilts = None
    tilts_to_particle = None
    tilt_index_map = None
    micrographtilt_to_tilts = None
    tilts_to_micrographtilt = None
    
    if starfile is not None and starfile.endswith('.star'):
        tilt_index_map = TiltSeriesOriginalIndexMap.from_particles_file(starfile)
        particle_to_tilts = tilt_index_map.particle_to_images
        tilts_to_particle = tilt_index_map.image_to_particle
        micrographtilt_to_tilts, tilts_to_micrographtilt = cryo_dataset.TiltSeriesDataset.parse_micrograph_tilt_mapping(starfile)
    else:
        logger.info("Starfile %s is not a .star file, skipping particle and micrograph-based outlier detection", starfile)

    outliers_image_identified_by_particle = np.zeros(n_images, dtype=bool)
    # Particle-based outlier detection
    if particle_to_tilts is not None:
        logger.info("\nParticle-based outlier detection:")
        logger.info("Total of particles: %s, of which %s were used in pipeline", len(particle_to_tilts), original_particle_indices.size)
        
        particle_outliers = []
        particle_inliers = []
        tilts_mapped_to_particles = [tilts_to_particle[tilt] for tilt in original_image_indices]

        # Get unique particles and their inverse mapping
        unique_particles, inverse_particles = np.unique(tilts_mapped_to_particles, return_inverse=True)
        
        particle_median_contrasts = {}

        for i, particle in enumerate(unique_particles):
            particle_indices = inverse_particles == i
            contrast_values = contrast_array[particle_indices]
            particle_median_contrasts[particle] = np.median(contrast_values)
            bad_fraction = np.mean(individual_outliers[particle_indices])
            
            if bad_fraction >= particle_bad_fraction_threshold:
                particle_outliers.append(particle)
                outliers_image_identified_by_particle[particle_indices] = True
            else:
                particle_inliers.append(particle)

        logger.info("  Total particle-based outliers: %s (%.1f%% of particles)", len(particle_outliers), len(particle_outliers)/(len(particle_outliers) + len(particle_inliers))*100)
        logger.info("  Corresponding to: %s (%.1f%% of images)", np.sum(outliers_image_identified_by_particle), np.sum(outliers_image_identified_by_particle)/n_images*100)


    outliers_image_identified_by_micrograph = np.zeros(n_images, dtype=bool)

    # Micrograph-based outlier detection
    if micrographtilt_to_tilts is not None:
        logger.info("\nMicrograph-based outlier detection:")
        logger.info("Number of micrographs: %s", len(micrographtilt_to_tilts))
        if len(micrographtilt_to_tilts) > 0:
            micrograph_sizes = [micrograph.size for micrograph in micrographtilt_to_tilts]
            logger.info("Average number of images per micrograph: %.1f", np.mean(micrograph_sizes))
            logger.info("Max number of images per micrograph: %s", np.max(micrograph_sizes))
            logger.info("Min number of images per micrograph: %s", np.min(micrograph_sizes))
            logger.info("Median number of images per micrograph: %.1f", np.median(micrograph_sizes))
            logger.info("Number of micrographs with less than 10 images: %s", np.sum([size < 10 for size in micrograph_sizes]))
            logger.info("Number of micrographs with 1 images: %s", np.sum([size ==1 for size in micrograph_sizes]))

        else:
            logger.info("No micrographs found in the dataset")
        
        micrograph_outliers = []
        micrograph_inliers = []
        
        # Map tilts to micrographs
        tilts_mapped_to_micrographs = [tilts_to_micrographtilt[tilt] for tilt in original_image_indices]
        
        # Get unique micrographs and their inverse mapping
        unique_micrographs, inverse_micrographs = np.unique(tilts_mapped_to_micrographs, return_inverse=True)
        
        micrograph_median_contrasts = {}
        
        for i, micrograph in enumerate(unique_micrographs):
            micrograph_indices = inverse_micrographs == i
            contrast_values = contrast_array[micrograph_indices]
            micrograph_median_contrasts[micrograph] = np.median(contrast_values)
            bad_fraction = np.mean(individual_outliers[micrograph_indices])
            
            # Get the actual tilt indices for this micrograph
            micrograph_tilt_indices = np.where(micrograph_indices)[0]
            
            if bad_fraction >= micrograph_bad_fraction_threshold:
                micrograph_outliers.append(micrograph)
                outliers_image_identified_by_micrograph[micrograph_indices] = True
            else:
                micrograph_inliers.append(micrograph)

        logger.info("  Total micrograph-based outliers: %s (%.1f%%) of micrographs", len(micrograph_outliers), len(micrograph_outliers)/(len(micrograph_outliers) + len(micrograph_inliers))*100)
        logger.info("  Corresponding to: %s (%.1f%% of images)", np.sum(outliers_image_identified_by_micrograph), np.sum(outliers_image_identified_by_micrograph)/n_images*100)
    # Print overlap statistics between methods
    if (particle_to_tilts is not None) or (micrographtilt_to_tilts is not None):
        logger.info("\nOutlier detection method overlap:")
        individual_count = np.sum(individual_outliers)
        particle_count = np.sum(outliers_image_identified_by_particle)
        micrograph_count = np.sum(outliers_image_identified_by_micrograph)
        
        logger.info("  Individual outliers: %s (%.1f%%)", individual_count, individual_count/n_images*100)
        logger.info("  Particle outliers: %s (%.1f%%)", particle_count, particle_count/n_images*100)
        logger.info("  Micrograph outliers: %s (%.1f%%)", micrograph_count, micrograph_count/n_images*100)
        
        # Calculate overlaps
        if individual_count > 0 and particle_count > 0:
            overlap = np.sum(individual_outliers & outliers_image_identified_by_particle)
            logger.info("  Individual-Particle overlap: %s (%.1f%% of individual)", overlap, overlap/individual_count*100)
        if individual_count > 0 and micrograph_count > 0:
            overlap = np.sum(individual_outliers & outliers_image_identified_by_micrograph)
            logger.info("  Individual-Micrograph overlap: %s (%.1f%% of individual)", overlap, overlap/individual_count*100)
        if particle_count > 0 and micrograph_count > 0:
            overlap = np.sum(outliers_image_identified_by_particle & outliers_image_identified_by_micrograph)
            logger.info("  Particle-Micrograph overlap: %s (%.1f%% of micrographs)", overlap, overlap/np.sum(outliers_image_identified_by_micrograph)*100)
    
    # Create plots if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create basic contrast histogram
        plt.figure(figsize=(10, 6))
        plt.hist(contrast_array, bins=50, alpha=0.7, color='blue', label='All images')
        plt.hist(contrast_array[individual_outliers], bins=50, alpha=0.8, color='red', label='Individual outliers')
        plt.axvline(x=low_contrast_threshold, color='orange', linestyle='--', linewidth=2, label=f'Low threshold ({low_contrast_threshold})')
        plt.axvline(x=high_contrast_threshold, color='orange', linestyle='--', linewidth=2, label=f'High threshold ({high_contrast_threshold})')
        plt.xlabel('Contrast')
        plt.ylabel('Number of images')
        plt.title(f'Contrast Distribution and Outlier Detection (zdim={zdim_key})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Total images: {n_images}
Individual outliers: {n_individual_outliers} ({n_individual_outliers/n_images*100:.1f}%)"""
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'contrast_outlier_detection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create median contrast distributions if grouping data available
        if particle_to_tilts is not None or micrographtilt_to_tilts is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot median contrast per particle (reuse computed values)
            if particle_to_tilts is not None and 'particle_median_contrasts' in locals():
                particle_contrast_values = list(particle_median_contrasts.values())
                if len(particle_contrast_values) > 0:
                    axes[0].hist(particle_contrast_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    axes[0].axvline(x=np.median(particle_contrast_values), color='red', linestyle='--', linewidth=2)
                    axes[0].set_xlabel('Median Contrast per Particle')
                    axes[0].set_ylabel('Number of Particles')
                    axes[0].set_title(f'Median Contrast per Particle\n(n={len(particle_contrast_values)} particles)')
                    axes[0].grid(True, alpha=0.3)
                else:
                    axes[0].text(0.5, 0.5, 'No particle contrast data available', transform=axes[0].transAxes, ha='center', va='center')
                    axes[0].set_title('Median Contrast per Particle')
            else:
                axes[0].text(0.5, 0.5, 'No particle grouping available', transform=axes[0].transAxes, ha='center', va='center')
                axes[0].set_title('Median Contrast per Particle')
            
            # Plot median contrast per micrograph (reuse computed values)
            if micrographtilt_to_tilts is not None and 'micrograph_median_contrasts' in locals():
                micrograph_contrast_values = list(micrograph_median_contrasts.values())
                if len(micrograph_contrast_values) > 0:
                    axes[1].hist(micrograph_contrast_values, bins=30, alpha=0.7, color='green', edgecolor='black')
                    axes[1].axvline(x=np.median(micrograph_contrast_values), color='red', linestyle='--', linewidth=2)
                    axes[1].set_xlabel('Median Contrast per Micrograph')
                    axes[1].set_ylabel('Number of Micrographs')
                    axes[1].set_title(f'Median Contrast per Micrograph\n(n={len(micrograph_contrast_values)} micrographs)')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No micrograph contrast data available', transform=axes[1].transAxes, ha='center', va='center')
                    axes[1].set_title('Median Contrast per Micrograph')
            else:
                axes[1].text(0.5, 0.5, 'No micrograph grouping available', transform=axes[1].transAxes, ha='center', va='center')
                axes[1].set_title('Median Contrast per Micrograph')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'median_contrast_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("  Image-level outliers: %s images", len(image_outliers))
        logger.info("  Image-level inliers: %s images", len(image_inliers))
        if particle_outliers is not None:
            logger.info("  Particle-level outliers: %s particles", len(particle_outliers))
        if particle_inliers is not None:
            logger.info("  Particle-level inliers: %s particles", len(particle_inliers))
    
    logger.info("\nResults saved to: %s", output_dir)
    

    combined_image_outliers = (individual_outliers | outliers_image_identified_by_particle | outliers_image_identified_by_micrograph)
    outliers_ind = np.where(combined_image_outliers)[0]
    inliers_ind = np.where(~combined_image_outliers)[0]

    # Initialize return values
    image_outliers = original_image_indices[outliers_ind]
    image_inliers = original_image_indices[inliers_ind]


    return image_outliers, image_inliers, particle_outliers, particle_inliers


def create_particle_outlier_visualization(all_particle_outliers, method_names, output_dir, zdim_key, total_particles):
    """
    Create visualization of particle-level outliers from different methods.
    
    Parameters:
    - all_particle_outliers: List of particle outlier arrays from different methods
    - method_names: List of method names
    - output_dir: Output directory for saving plots
    - zdim_key: Dimension key for embeddings
    - total_particles: Total number of particles
    """
    if len(all_particle_outliers) == 0:
        return
    
    logger.info("Creating particle outlier visualization...")
    
    # Create overlap matrix visualization for particle-level outliers
    n_methods = len(method_names)
    overlap_matrix = np.zeros((n_methods, n_methods), dtype=int)
    
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: total number of outliers for each method
                overlap_matrix[i, j] = len(all_particle_outliers[i])
            else:
                # Off-diagonal: overlap between methods
                overlap_matrix[i, j] = len(np.intersect1d(all_particle_outliers[i], all_particle_outliers[j]))
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overlap matrix as heatmap
    im = ax1.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(n_methods))
    ax1.set_yticks(range(n_methods))
    ax1.set_xticklabels(method_names)
    ax1.set_yticklabels(method_names)
    ax1.set_title('Particle-Level Outlier Detection Method Overlap Matrix\n(Number of particles)')
    
    # Add text annotations to the heatmap
    for i in range(n_methods):
        for j in range(n_methods):
            text = ax1.text(j, i, str(overlap_matrix[i, j]), 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Number of Particles')
    
    # Plot 2: Percentage overlap matrix
    percentage_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: percentage of total particles
                percentage_matrix[i, j] = (overlap_matrix[i, j] / total_particles) * 100
            else:
                # Off-diagonal: percentage overlap relative to method i
                if overlap_matrix[i, i] > 0:
                    percentage_matrix[i, j] = (overlap_matrix[i, j] / overlap_matrix[i, i]) * 100
                else:
                    percentage_matrix[i, j] = 0
            
    im2 = ax2.imshow(percentage_matrix, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(n_methods))
    ax2.set_yticks(range(n_methods))
    ax2.set_xticklabels(method_names)
    ax2.set_yticklabels(method_names)
    ax2.set_title('Particle-Level Outlier Detection Method Overlap Matrix\n(Percentage)')
    
    # Add text annotations to the percentage heatmap
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: percentage of total particles
                text = ax2.text(j, i, f'{percentage_matrix[i, j]:.1f}%\n({overlap_matrix[i, j]})', 
                              ha="center", va="center", color="black", fontweight='bold')
            else:
                # Off-diagonal: percentage overlap
                if percentage_matrix[i, j] > 0:
                    text = ax2.text(j, i, f'{percentage_matrix[i, j]:.1f}%\n({overlap_matrix[i, j]})', 
                                  ha="center", va="center", color="black", fontweight='bold')
                else:
                    text = ax2.text(j, i, '0%\n(0)', 
                                  ha="center", va="center", color="black", fontweight='bold')
            
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Percentage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_outlier_method_overlap_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed overlap analysis for particles
    overlap_analysis_file = os.path.join(output_dir, 'particle_outlier_method_overlap_analysis.txt')
    with open(overlap_analysis_file, 'w') as f:
        f.write("Particle-Level Outlier Detection Method Overlap Analysis\n")
        f.write("======================================================\n\n")
        f.write(f"Total particles analyzed: {total_particles}\n\n")
        
        for i, method in enumerate(method_names):
            f.write(f"{method} Particle Outliers:\n")
            f.write(f"  Total: {overlap_matrix[i, i]} ({percentage_matrix[i, i]:.1f}% of total)\n")
            
            for j, other_method in enumerate(method_names):
                if i != j:
                    if overlap_matrix[i, i] > 0:
                        overlap_pct = (overlap_matrix[i, j] / overlap_matrix[i, i]) * 100
                        f.write(f"  Overlap with {other_method}: {overlap_matrix[i, j]} ({overlap_pct:.1f}% of {method} outliers)\n")
                    else:
                        f.write(f"  Overlap with {other_method}: 0 (0% of {method} outliers)\n")
            f.write("\n")
    
    logger.info("Particle outlier visualization saved to: %s", output_dir)

def create_overlap_matrix_visualization(all_outliers, method_names, output_dir, zdim_key, total_particles):
    """
    Create overlap matrix visualization for different outlier detection methods.
    
    Parameters:
    - all_outliers: List of outlier arrays from different methods
    - method_names: List of method names
    - output_dir: Output directory for saving plots
    - zdim_key: Dimension key for embeddings
    - total_particles: Total number of particles
    """
    n_methods = len(method_names)
    overlap_matrix = np.zeros((n_methods, n_methods), dtype=int)
    
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: total number of outliers for each method
                overlap_matrix[i, j] = len(all_outliers[i])
            else:
                # Off-diagonal: overlap between methods
                overlap_matrix[i, j] = len(np.intersect1d(all_outliers[i], all_outliers[j]))
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overlap matrix as heatmap
    im = ax1.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(n_methods))
    ax1.set_yticks(range(n_methods))
    ax1.set_xticklabels(method_names)
    ax1.set_yticklabels(method_names)
    ax1.set_title('Outlier Detection Method Overlap Matrix\n(Number of particles)')
    
    # Add text annotations to the heatmap
    for i in range(n_methods):
        for j in range(n_methods):
            text = ax1.text(j, i, str(overlap_matrix[i, j]), 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Number of Particles')
    
    # Plot 2: Percentage overlap matrix
    percentage_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: percentage of total particles
                percentage_matrix[i, j] = (overlap_matrix[i, j] / total_particles) * 100
            else:
                # Off-diagonal: percentage overlap relative to method i
                if overlap_matrix[i, i] > 0:
                    percentage_matrix[i, j] = (overlap_matrix[i, j] / overlap_matrix[i, i]) * 100
                else:
                    percentage_matrix[i, j] = 0
            
    im2 = ax2.imshow(percentage_matrix, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(n_methods))
    ax2.set_yticks(range(n_methods))
    ax2.set_xticklabels(method_names)
    ax2.set_yticklabels(method_names)
    ax2.set_title('Outlier Detection Method Overlap Matrix\n(Percentage)')
    
    # Add text annotations to the percentage heatmap
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal: percentage of total particles
                text = ax2.text(j, i, f'{percentage_matrix[i, j]:.1f}%\n({overlap_matrix[i, j]})', 
                              ha="center", va="center", color="black", fontweight='bold')
            else:
                # Off-diagonal: percentage overlap
                if percentage_matrix[i, j] > 0:
                    text = ax2.text(j, i, f'{percentage_matrix[i, j]:.1f}%\n({overlap_matrix[i, j]})', 
                                  ha="center", va="center", color="black", fontweight='bold')
                else:
                    text = ax2.text(j, i, '0%\n(0)', 
                                  ha="center", va="center", color="black", fontweight='bold')
            
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Percentage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_method_overlap_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Method overlap matrix saved to: %s", os.path.join(output_dir, 'outlier_method_overlap_matrix.png'))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Outlier Detection from Pipeline Results")
    parser = add_args(parser)
    args = parser.parse_args()
    
    # Set up main output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.pipeline_output_dir, 'outlier_detection')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            RobustFileHandler(os.path.join(args.output_dir, 'outlier_detection.log')),
            RobustStreamHandler()
        ]
    )
    
    logger.info("Starting outlier detection from pipeline output: %s", args.pipeline_output_dir)
    logger.info("Output directory: %s", args.output_dir)
    
    # Load pipeline output
    pipeline_output = output.PipelineOutput(args.pipeline_output_dir)
    
    # Select reg vs noreg entry
    coords_entry = 'latent_coords_noreg' if args.no_z_regularization else 'latent_coords'
    contrast_entry = 'contrasts_noreg' if args.no_z_regularization else 'contrasts'
    zdim_key = args.zdim_key

    # Check if embeddings exist
    zs_dict = pipeline_output.get(coords_entry)
    if zs_dict is None or zdim_key not in zs_dict:
        available_dims = list(zs_dict.keys()) if zs_dict is not None else []
        logger.error("zdim %s not found. Available dimensions: %s", zdim_key, available_dims)
        sys.exit(1)

    # Load embeddings
    zs = zs_dict[zdim_key]
    logger.info("Loaded embeddings with shape: %s", zs.shape)

    # Get the actual number of particles from the pipeline output
    # For tilt series, this might be different from the number of embeddings
    input_args = pipeline_output.get('input_args')
    is_tilt_series = getattr(input_args, 'tilt_series', False)
    is_shared_contrast = getattr(input_args, 'shared_contrast', False)
    starfile = getattr(input_args, 'particles', None)

    # Get both types of halfsets
    particles_halfsets = pipeline_output.get('particles_halfsets')
    image_halfsets = pipeline_output.get('halfsets')
    
    if is_tilt_series:
        # For tilt series, get the number of particles (not images)
        total_particles = len(particles_halfsets[0]) + len(particles_halfsets[1])
        logger.info("Tilt series dataset: %s particles, %s total images", total_particles, len(image_halfsets[0]) + len(image_halfsets[1]))
    else:
        # For regular datasets, embeddings count = particle count
        total_particles = len(zs)
        logger.info("Cryo-EM dataset: %s particles and images ", total_particles)
    
    # Log halfset information
    logger.info("Particles halfsets: %s + %s = %s particles", len(particles_halfsets[0]), len(particles_halfsets[1]), len(particles_halfsets[0]) + len(particles_halfsets[1]))
    logger.info("Image halfsets: %s + %s = %s images", len(image_halfsets[0]), len(image_halfsets[1]), len(image_halfsets[0]) + len(image_halfsets[1]))
    
    
    # --- Method 1: Anomaly Detection (UMAP-based) ---
    logger.info("Running anomaly detection...")
    anomaly_output_dir = os.path.join(args.output_dir, 'anomaly_detection')
    os.makedirs(anomaly_output_dir, exist_ok=True)
    
    plot_anomaly_detection_results(zs, np.concatenate(particles_halfsets), anomaly_output_dir)
    
    # Load consensus results
    consensus_inliers_file = os.path.join(anomaly_output_dir, 'inliers_consensus.pkl')
    consensus_outliers_file = os.path.join(anomaly_output_dir, 'outliers_consensus.pkl')
    
    if not os.path.exists(consensus_inliers_file) or not os.path.exists(consensus_outliers_file):
        logger.error("Consensus results not found. Anomaly detection may have failed.")
        sys.exit(1)
    
    with open(consensus_inliers_file, 'rb') as f:
        anomaly_inliers = pickle.load(f)
    with open(consensus_outliers_file, 'rb') as f:
        anomaly_outliers = pickle.load(f)
    
    # Validate anomaly detection results
    
    logger.info("Anomaly detection completed. Found %s inliers and %s particle outliers.", len(anomaly_inliers), len(anomaly_outliers))
    
    # --- Method 2: Contrast-based Outlier Detection ---
    contrasts = pipeline_output.get(contrast_entry)
    contrast_image_inliers = None
    contrast_image_outliers = None
    contrast_particle_inliers = None
    contrast_particle_outliers = None

    if contrasts is not None and zdim_key in contrasts:
        # Extract contrast values for the specific zdim_key
        contrast_values = contrasts[zdim_key]
        logger.info("Found contrast values for zdim=%s, shape: %s", zdim_key, contrast_values.shape)
        
        # Load starfile path and options from pipeline input_args
        contrast_output_dir = os.path.join(args.output_dir, 'contrast_based')
        os.makedirs(contrast_output_dir, exist_ok=True)
        
        image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_detection_from_contrast(
            pipeline_output=pipeline_output,
            zdim_key=zdim_key,
            low_contrast_threshold=args.low_contrast_threshold,
            high_contrast_threshold=args.high_contrast_threshold,
            max_contrast=args.max_contrast,
            particle_bad_fraction_threshold=args.particle_bad_fraction_threshold,
            micrograph_bad_fraction_threshold=args.micrograph_bad_fraction_threshold,
            output_dir=contrast_output_dir,
            noreg=args.no_z_regularization,
        )
        
        # Store results
        contrast_image_inliers = image_inliers
        contrast_image_outliers = image_outliers
        contrast_particle_inliers = particle_inliers
        contrast_particle_outliers = particle_outliers

        logger.info("Contrast-based outlier detection completed. Found %s image inliers and %s image outliers.", len(contrast_image_inliers), len(contrast_image_outliers))
        
    else:
        contrast_image_inliers = None
        contrast_image_outliers = None
        contrast_particle_inliers = None
        contrast_particle_outliers = None

        logger.warning("No contrast data available for contrast-based outlier detection.")
    
    # --- Method 3: Junk Particle Detection ---
    junk_inliers = None
    junk_outliers = None
    
    if args.use_junk_detection:
        logger.info("Running junk particle detection...")

        from recovar.commands import junk_particle_detection
        
        if args.use_junk_detection:
            junk_output_dir = os.path.join(args.output_dir, 'junk_detection')
            os.makedirs(junk_output_dir, exist_ok=True)
            
            # Automatically set batch_size and n_particles_per_cluster
            # Get GPU memory and grid size for batch size calculation
            from recovar import utils
            gpu_memory = utils.get_gpu_memory_total()
            pipeline_output = output.PipelineOutput(args.pipeline_output_dir)
            cryos = pipeline_output.get('dataset')
            grid_size = cryos.grid_size
            
            # Calculate automatic batch size like in pipeline
            batch_size = utils.get_image_batch_size(grid_size, gpu_memory)
            
            # Calculate n_particles_per_cluster as min(100, max(10, n_particles/n_clusters))
            n_clusters = 100
            n_particles = len(pipeline_output.get(coords_entry)[zdim_key])
            n_particles_per_cluster = min(100, max(10, n_particles // n_clusters))
            
            # Override with user-provided value if specified
            if hasattr(args, 'particles_per_cluster') and args.particles_per_cluster is not None:
                n_particles_per_cluster = args.particles_per_cluster
            
            logger.info("Junk detection: auto batch_size=%s, auto n_particles_per_cluster=%s", batch_size, n_particles_per_cluster)
            
            # Run junk detection
            junk_particle_detection.junk_particle_detection(
                recovar_result_dir=args.pipeline_output_dir,
                output_folder=junk_output_dir,
                zdim=args.zdim_key,
                n_clusters=n_clusters,
                batch_size=batch_size,
                n_particles_per_cluster=n_particles_per_cluster,
                no_z_regularization=args.no_z_regularization,
                save_reconstructions=False,
                filter_resolution=None,
                filter_fourier_shells=10,
                junk_detection_method="adaptive_threshold",
                percentile_threshold=25.0,
                std_threshold=args.junk_threshold,
                min_junk_fraction=0.1,
                max_junk_fraction=0.8,
                save_pipeline_indices=args.save_pipeline_indices,
                output_format=args.output_format
            )
            
            # Load junk detection results
            junk_outliers_file = os.path.join(junk_output_dir, f'junk_indices_{zdim_key}.pkl')
            junk_inliers_file = os.path.join(junk_output_dir, f'good_indices_{zdim_key}.pkl')
            
            if os.path.exists(junk_outliers_file):
                with open(junk_outliers_file, 'rb') as f:
                    junk_outliers = pickle.load(f)
            else:
                logger.warning("Junk outliers file not found: %s", junk_outliers_file)
                raise FileNotFoundError(f"Junk outliers file not found: {junk_outliers_file}")
                
            if os.path.exists(junk_inliers_file):
                with open(junk_inliers_file, 'rb') as f:
                    junk_inliers = pickle.load(f)
            else:
                logger.warning("Junk inliers file not found: %s", junk_inliers_file)
                raise FileNotFoundError(f"Junk inliers file not found: {junk_inliers_file}")

    
    # --- Combine Results from All Methods ---
    logger.info("Combining results from all methods...")
    combined_output_dir = os.path.join(args.output_dir, 'combined_results')
    os.makedirs(combined_output_dir, exist_ok=True)

    # Collect all particle-level outlier indices (for visualization)
    all_particle_outliers = []
    particle_method_names = []
    
    if anomaly_outliers is not None:
        all_particle_outliers.append(anomaly_outliers)
        particle_method_names.append("Anomaly Detection")
    
    if contrast_particle_outliers is not None:
        all_particle_outliers.append(contrast_particle_outliers)
        particle_method_names.append("Contrast-based")
    
    if junk_outliers is not None:
        all_particle_outliers.append(junk_outliers)
        particle_method_names.append("Junk Detection")
    
    original_particle_indices = np.concatenate(pipeline_output.get('particles_halfsets'))
    original_image_indices = np.concatenate(pipeline_output.get('halfsets'))
    
    # Handle particle outlier combination safely
    if len(all_particle_outliers) > 0:
        combined_particle_outliers = all_particle_outliers[0]
        for outliers in all_particle_outliers[1:]:
            combined_particle_outliers = np.union1d(combined_particle_outliers, outliers)
        # Convert back to integers since np.union1d can return floats
        combined_particle_outliers = combined_particle_outliers.astype(np.int64)
        combined_particle_inliers = np.setdiff1d(original_particle_indices, combined_particle_outliers)
    else:
        # If no particle outliers, all particles are inliers
        combined_particle_outliers = np.array([], dtype=int)
        combined_particle_inliers = original_particle_indices


    # Create particle-level visualization if we have particle outliers
    if len(all_particle_outliers) > 0:
        create_particle_outlier_visualization(all_particle_outliers, particle_method_names, combined_output_dir, zdim_key, total_particles)
    
    # Collect all image-level outlier indices (for final combination)
    all_image_outliers = []
    image_method_names = []
    
    if anomaly_outliers is not None:
        # Convert particle outliers to image outliers for anomaly detection
        if is_tilt_series:
            # For tilt series, map particle outliers to image outliers
            anomaly_image_outliers = map_particle_original_indexing_to_images_original_indexing(anomaly_outliers, original_image_indices, starfile)
        else:
            # For regular datasets, particle outliers = image outliers
            anomaly_image_outliers = anomaly_outliers
        all_image_outliers.append(anomaly_image_outliers)
        image_method_names.append("Anomaly Detection")
    
    if contrast_image_outliers is not None:
        all_image_outliers.append(contrast_image_outliers)
        image_method_names.append("Contrast-based")
    


    if junk_outliers is not None:
        # Convert particle outliers to image outliers for junk detection
        if is_tilt_series:
            # For tilt series, map particle outliers to image outliers
            junk_image_outliers = map_particle_original_indexing_to_images_original_indexing(junk_outliers, original_image_indices, starfile)
        else:
            # For regular datasets, particle outliers = image outliers
            junk_image_outliers = junk_outliers
        all_image_outliers.append(junk_image_outliers)
        image_method_names.append("Junk Detection")


    # Always save combined results, even if some methods don't produce outliers
    if len(all_image_outliers) > 0:
        # Combine results: images are considered outliers if they are outliers in ANY method
        combined_image_outliers = all_image_outliers[0]
        for outliers in all_image_outliers[1:]:
            combined_image_outliers = np.union1d(combined_image_outliers, outliers)
        # Convert back to integers since np.union1d can return floats
        combined_image_outliers = combined_image_outliers.astype(np.int64)
    else:
        # If no methods produced image outliers, all images are inliers
        combined_image_outliers = np.array([], dtype=int)

    # Combine particle outliers with image outliers
    if is_tilt_series:
        particle_outliers_to_image_outliers = map_particle_original_indexing_to_images_original_indexing(combined_particle_outliers, original_image_indices, starfile)
        combined_image_outliers = np.union1d(combined_image_outliers, particle_outliers_to_image_outliers)

    combined_image_inliers = np.setdiff1d(original_image_indices , combined_image_outliers)

    # Save image-level indices (for --ind) - ALWAYS save these
    image_outliers_file = os.path.join(combined_output_dir, f'combined_image_outliers_{zdim_key}.pkl')
    image_inliers_file = os.path.join(combined_output_dir, f'combined_image_inliers_{zdim_key}.pkl')
    
    with open(image_outliers_file, 'wb') as f:
        pickle.dump(combined_image_outliers, f)
    with open(image_inliers_file, 'wb') as f:
        pickle.dump(combined_image_inliers, f)
    
    # Save particle-level indices (for --particle-ind) if this is a tilt series or if we have particle outliers
    if is_tilt_series or len(all_particle_outliers) > 0:
        particle_outliers_file = os.path.join(combined_output_dir, f'combined_particle_outliers_{zdim_key}.pkl')
        particle_inliers_file = os.path.join(combined_output_dir, f'combined_particle_inliers_{zdim_key}.pkl')
        
        with open(particle_outliers_file, 'wb') as f:
            pickle.dump(combined_particle_outliers, f)
        with open(particle_inliers_file, 'wb') as f:
            pickle.dump(combined_particle_inliers, f)
    
    # Save simple breakdown
    breakdown_file = os.path.join(combined_output_dir, 'detection_breakdown.txt')
    with open(breakdown_file, 'w') as f:
        f.write(f"Combined Outlier Detection Results\n")
        f.write(f"Total particles: {total_particles}\n")
        f.write(f"Combined image outliers: {len(combined_image_outliers)} ({len(combined_image_outliers)/original_image_indices.size*100:.1f}%)\n")
        f.write(f"Combined image inliers: {len(combined_image_inliers)} ({len(combined_image_inliers)/original_image_indices.size*100:.1f}%)\n\n")
        if len(all_image_outliers) > 0:
            for i, (outliers, method) in enumerate(zip(all_image_outliers, image_method_names)):
                f.write(f"{method}: {len(outliers)} outliers ({len(outliers)/original_image_indices.size*100:.1f}%)\n")
        else:
            f.write("No outlier detection methods produced results.\n")
    



    # Create overlap matrix visualization
    if len(all_image_outliers) > 1:
        create_overlap_matrix_visualization(all_image_outliers, image_method_names, combined_output_dir, zdim_key, original_image_indices.size)
    
    # Create outlier visualizations (UMAP, latent space, contrast histograms)
    if len(all_particle_outliers) > 0:
        create_outlier_visualizations(pipeline_output, all_particle_outliers, particle_method_names,
                                    combined_particle_outliers, combined_particle_inliers,
                                    args.output_dir, zdim_key, total_particles, is_tilt_series, starfile,
                                    noreg=args.no_z_regularization)
    
    logger.info("Combined results saved to: %s", combined_output_dir)
    logger.info("Combined image outliers: %s (%.1f%%)", len(combined_image_outliers), len(combined_image_outliers)/original_image_indices.size*100)
    logger.info("Combined particle outliers: %s (%.1f%%)", len(combined_particle_outliers), len(combined_particle_outliers)/original_particle_indices.size*100)
    
    # Debug information about the confusion
    logger.info("Debug: Total images in dataset: %s", original_image_indices.size)
    logger.info("Debug: Total particles in dataset: %s", original_particle_indices.size)
    logger.info("Debug: Number of particle outlier methods: %s", len(all_particle_outliers))
    logger.info("Debug: Number of image outlier methods: %s", len(all_image_outliers))
    if len(all_particle_outliers) > 0:
        for i, (outliers, method) in enumerate(zip(all_particle_outliers, particle_method_names)):
            logger.info("Debug: %s particle outliers: %s", method, len(outliers))
    if len(all_image_outliers) > 0:
        for i, (outliers, method) in enumerate(zip(all_image_outliers, image_method_names)):
            logger.info("Debug: %s image outliers: %s", method, len(outliers))
    


    # Create summary file
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Outlier Detection Summary\n")
        f.write(f"Pipeline output: {args.pipeline_output_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"zdim key: {zdim_key}\n")
        f.write(f"Total particles: {total_particles}\n")
        f.write("Methods: Anomaly Detection, Contrast-based, Junk Detection\n")
    
    logger.info("Outlier detection completed. Results saved to %s", args.output_dir)



def map_particle_original_indexing_to_images_original_indexing(particle_indices_in_original_ordering, image_subset, starfile):
    index_map = TiltSeriesOriginalIndexMap.from_particles_file(starfile)
    particle_indices = index_map.sanitize_particle_indices(
        particle_indices_in_original_ordering,
        name="particle_indices_in_original_ordering",
    )
    return index_map.image_indices_from_particles(
        particle_indices,
        allowed_images=image_subset,
    )


def add_args(parser):
    """Add command line arguments for outlier detection."""
    parser.add_argument("pipeline_output_dir", type=str, help="Directory containing recovar pipeline results")
    parser.add_argument("--zdim-key", type=int, default=4, help="Dimension key for embeddings (default: 4)")
    parser.add_argument("--no-z-regularization", action="store_true", help="Use unregularized embeddings")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for results (default: pipeline_output_dir/outlier_detection)")
    parser.add_argument("--save-pipeline-indices", action="store_true", 
                       help="Save indices in pipeline-compatible format (--ind for images, --particle-ind for particles in tilt series)")
    parser.add_argument("--output-format", type=str, default="both", 
                       choices=["both", "outliers_only", "inliers_only"], 
                       help="Which indices to save (default: both)")
    
    # Contrast-based outlier detection arguments
    parser.add_argument("--low-contrast-threshold", type=float, default=0.1, 
                       help="Low contrast threshold for outlier detection (default: 0.1)")
    parser.add_argument("--high-contrast-threshold", type=float, default=3.5, 
                       help="High contrast threshold for outlier detection (default: 3.5)")
    parser.add_argument("--max-contrast", type=float, default=4.0, 
                       help="Maximum contrast value to consider (default: 4.0)")
    parser.add_argument("--particle-bad-fraction-threshold", type=float, default=0.7, 
                       help="Threshold for bad fraction in particle (default: 0.7)")
    parser.add_argument("--micrograph-bad-fraction-threshold", type=float, default=0.7,
                       help="If this fraction of a micrograph's images are bad, reject entire micrograph (default: 0.7)")
    
    # Junk detection arguments
    parser.add_argument("--use-junk-detection", action="store_true", 
                       help="Run junk particle detection in addition to outlier detection")
    parser.add_argument("--junk-threshold", type=float, default=0.5, 
                       help="Threshold for junk particle detection (default: 0.5)")
    parser.add_argument("--particles-per-cluster", type=int, 
                       help="Number of particles per cluster for junk detection (auto: min(100, max(10, n_particles/n_clusters)))")
    
    return parser



def create_outlier_visualizations(pipeline_output, all_particle_outliers, method_names, combined_particle_outliers,
                                 combined_particle_inliers, output_dir, zdim_key, total_particles, is_tilt_series=False, starfile=None, noreg=False):
    """
    Create UMAP and latent space visualizations with contrast histograms for each outlier group.

    Parameters:
    - pipeline_output: Pipeline output containing zs and other data
    - all_particle_outliers: List of particle outlier arrays for each method
    - method_names: List of method names
    - combined_particle_outliers: Combined particle outliers
    - combined_particle_inliers: Combined particle inliers
    - output_dir: Output directory for saving visualizations
    - zdim_key: Dimension key for embeddings (int)
    - total_particles: Total number of particles
    - is_tilt_series: Whether this is a tilt series dataset
    - starfile: Star file path for tilt series mapping
    - noreg: Whether to use unregularized embeddings
    """
    logger.info("Creating outlier visualizations...")

    coords_entry = 'latent_coords_noreg' if noreg else 'latent_coords'
    contrast_entry = 'contrasts_noreg' if noreg else 'contrasts'

    # Set up improved plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Use professional color scheme
    colors = {
        'background': '#E0E0E0',  # Light gray
        'inliers': '#2E8B57',     # Sea green
        'outliers': '#DC143C',    # Crimson red
        'mean': '#4169E1',        # Royal blue
        'grid': '#F0F0F0',        # Very light gray
        'scatter': 'cornflowerblue'
    }

    # Get zs embeddings and original particle indices
    zs = pipeline_output.get(coords_entry)[zdim_key]
    original_particle_indices = np.concatenate(pipeline_output.get('particles_halfsets'))
    original_image_indices = np.concatenate(pipeline_output.get('halfsets'))

    # Get contrast values if available
    contrast_values = None
    contrast_values = pipeline_output.get(contrast_entry)[zdim_key]
    umapper = output.umap_latent_space(zs)
    umap_coords = umapper.embedding_
    has_umap = True
    # Pad to at least 4 dimensions so all 2D scatter/hexbin plots work
    n_pad = max(0, 4 - zs.shape[1])
    if n_pad > 0:
        zs = np.column_stack([zs, np.zeros((len(zs), n_pad))])
    if umap_coords.shape[1] < 2:
        umap_coords = np.column_stack([umap_coords, np.zeros(len(umap_coords))])

    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'outlier_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    contrast_tilt_index_map = None
    if is_tilt_series and starfile is not None and str(starfile).endswith(".star"):
        contrast_tilt_index_map = TiltSeriesOriginalIndexMap.from_particles_file(starfile)

    # Function to map particle indices to image indices for contrast plotting
    def get_contrast_indices_for_particles(particle_indices):
        """Map particle indices to image indices for contrast plotting."""
        if contrast_values is None:
            return None
        particle_indices = np.asarray(particle_indices, dtype=np.int64)
        if particle_indices.size == 0:
            return np.array([], dtype=original_image_indices.dtype)
        
        if is_tilt_series:
            if contrast_tilt_index_map is None:
                return np.array([], dtype=original_image_indices.dtype)
            valid_particle_indices = contrast_tilt_index_map.sanitize_particle_indices(
                particle_indices,
                name="particle_indices",
            )
            if valid_particle_indices.size == 0:
                return np.array([], dtype=original_image_indices.dtype)
            return contrast_tilt_index_map.image_indices_from_particles(
                valid_particle_indices,
                allowed_images=original_image_indices,
            )
        else:
            # For regular datasets, particle indices = image indices
            return particle_indices
    
    # Function to create plots for a set of outliers/inliers
    def create_method_plots(outlier_indices, inlier_indices, method_name, save_prefix):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{method_name} Outlier Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Create boolean masks for particle-level plotting (UMAP and latent space)
        outlier_mask = np.isin(original_particle_indices, outlier_indices)
        inlier_mask = np.isin(original_particle_indices, inlier_indices)
        
        # Function to downsample large datasets for better visualization
        def downsample_points(points, mask, max_points=10000):
            """Downsample points to max_points for better visualization."""
            if np.sum(mask) <= max_points:
                return points[mask]
            else:
                # Randomly sample max_points from the masked data
                indices = np.where(mask)[0]
                sampled_indices = np.random.choice(indices, size=max_points, replace=False)
                return points[sampled_indices]
        
        # Plot 1: UMAP visualization with improved styling for large datasets
        if has_umap:
            ax = axes[0, 0]
            
            # Create hexbin density plot for background with larger gridsize for big datasets
            try:
                # Use larger gridsize for better density representation with many points
                gridsize = min(50, max(20, int(np.sqrt(len(umap_coords) / 100))))
                hb = ax.hexbin(umap_coords[:, 0], umap_coords[:, 1], gridsize=gridsize, 
                              cmap='Blues', alpha=0.4, mincnt=1, reduce_C_function=np.mean)
            except (ValueError, TypeError):
                pass
            
            # Downsample points for scatter plot to avoid overcrowding
            if np.sum(inlier_mask) > 0:
                inlier_points = downsample_points(umap_coords, inlier_mask, max_points=5000)
                ax.scatter(inlier_points[:, 0], inlier_points[:, 1], 
                          c=colors['inliers'], alpha=0.7, s=6, label=f'Inliers (n={np.sum(inlier_mask):,})', 
                          rasterized=True, edgecolors='none')
            
            if np.sum(outlier_mask) > 0:
                outlier_points = downsample_points(umap_coords, outlier_mask, max_points=2000)
                ax.scatter(outlier_points[:, 0], outlier_points[:, 1], 
                          c=colors['outliers'], alpha=0.9, s=10, label=f'Outliers (n={np.sum(outlier_mask):,})', 
                          rasterized=True, edgecolors='none')
            
            ax.set_xlabel('UMAP 1', fontweight='bold')
            ax.set_ylabel('UMAP 2', fontweight='bold')
            ax.set_title('UMAP Embedding', fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, color=colors['grid'])
            ax.set_facecolor('#FAFAFA')
        
        # Plot 2: Latent space (zs[:, 0] vs zs[:, 1]) with improved styling for large datasets
        ax = axes[0, 1]
        
        # Create hexbin density plot for background
        try:
            gridsize = min(50, max(20, int(np.sqrt(len(zs) / 100))))
            hb = ax.hexbin(zs[:, 0], zs[:, 1], gridsize=gridsize, 
                          cmap='Blues', alpha=0.4, mincnt=1, reduce_C_function=np.mean)
        except (ValueError, TypeError):
            pass
        
        # Downsample points for scatter plot
        if np.sum(inlier_mask) > 0:
            inlier_points = downsample_points(zs, inlier_mask, max_points=5000)
            ax.scatter(inlier_points[:, 0], inlier_points[:, 1], 
                      c=colors['inliers'], alpha=0.7, s=6, label=f'Inliers (n={np.sum(inlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        if np.sum(outlier_mask) > 0:
            outlier_points = downsample_points(zs, outlier_mask, max_points=2000)
            ax.scatter(outlier_points[:, 0], outlier_points[:, 1], 
                      c=colors['outliers'], alpha=0.9, s=10, label=f'Outliers (n={np.sum(outlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        ax.set_xlabel('Latent Dimension 1', fontweight='bold')
        ax.set_ylabel('Latent Dimension 2', fontweight='bold')
        ax.set_title('Latent Space (Dim 1 vs Dim 2)', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 3: Latent space (zs[:, 2] vs zs[:, 3]) with improved styling for large datasets
        ax = axes[0, 2]
        
        # Create hexbin density plot for background
        try:
            gridsize = min(50, max(20, int(np.sqrt(len(zs) / 100))))
            hb = ax.hexbin(zs[:, 2], zs[:, 3], gridsize=gridsize, 
                          cmap='Blues', alpha=0.4, mincnt=1, reduce_C_function=np.mean)
        except (ValueError, TypeError):
            pass
        
        # Downsample points for scatter plot
        if np.sum(inlier_mask) > 0:
            inlier_points = downsample_points(zs, inlier_mask, max_points=5000)
            ax.scatter(inlier_points[:, 2], inlier_points[:, 3], 
                      c=colors['inliers'], alpha=0.7, s=6, label=f'Inliers (n={np.sum(inlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        if np.sum(outlier_mask) > 0:
            outlier_points = downsample_points(zs, outlier_mask, max_points=2000)
            ax.scatter(outlier_points[:, 2], outlier_points[:, 3], 
                      c=colors['outliers'], alpha=0.9, s=10, label=f'Outliers (n={np.sum(outlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        ax.set_xlabel('Latent Dimension 3', fontweight='bold')
        ax.set_ylabel('Latent Dimension 4', fontweight='bold')
        ax.set_title('Latent Space (Dim 3 vs Dim 4)', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 4: Contrast histogram with improved styling for large datasets
        if contrast_values is not None:
            ax = axes[1, 0]
            
            # Get image indices for contrast plotting
            outlier_image_indices = get_contrast_indices_for_particles(outlier_indices)
            inlier_image_indices = get_contrast_indices_for_particles(inlier_indices)

            if outlier_image_indices.size + inlier_image_indices.size != original_image_indices.size:
                logger.info("Number of images in outlier indices: %s", len(outlier_image_indices))
                logger.info("Number of images in inlier indices: %s", len(inlier_image_indices))
                logger.info("Number of images in original image indices: %s", original_image_indices.size)
                logger.info("Number of images in outlier indices + inlier indices: %s", outlier_image_indices.size + inlier_image_indices.size)
                raise ValueError("Number of images in outlier indices + inlier indices does not match original image indices")

            # Use adaptive binning for better histogram readability with large datasets
            n_bins = min(50, max(20, len(contrast_values) // 200))
            bins = np.linspace(contrast_values.min(), contrast_values.max(), n_bins + 1)
            
            if outlier_image_indices is not None and len(outlier_image_indices) > 0:
                # Create boolean mask for image-level contrast values
                outlier_contrast_mask = np.isin(original_image_indices, outlier_image_indices)
                outlier_contrast = contrast_values[outlier_contrast_mask]
                ax.hist(outlier_contrast, bins=bins, alpha=0.5, 
                       color=colors['outliers'], label=f'Outliers (n={len(outlier_contrast):,})', 
                       density=True, edgecolor='black', linewidth=0.8, hatch='///')
            
            if inlier_image_indices is not None and len(inlier_image_indices) > 0:
                # Create boolean mask for image-level contrast values
                inlier_contrast_mask = np.isin(original_image_indices, inlier_image_indices)
                inlier_contrast = contrast_values[inlier_contrast_mask]
                ax.hist(inlier_contrast, bins=bins, alpha=0.5, 
                       color=colors['inliers'], label=f'Inliers (n={len(inlier_contrast):,})', 
                       density=True, edgecolor='black', linewidth=0.8, hatch='\\\\\\')
            
            # Add mean lines with better styling
            if outlier_image_indices is not None and len(outlier_image_indices) > 0:
                outlier_contrast_mask = np.isin(original_image_indices, outlier_image_indices)
                outlier_contrast = contrast_values[outlier_contrast_mask]
                ax.axvline(np.mean(outlier_contrast), color=colors['outliers'], 
                          linestyle='--', linewidth=2.5, alpha=0.9,
                          label=f'Outlier mean: {np.mean(outlier_contrast):.3f}')
            
            if inlier_image_indices is not None and len(inlier_image_indices) > 0:
                inlier_contrast_mask = np.isin(original_image_indices, inlier_image_indices)
                inlier_contrast = contrast_values[inlier_contrast_mask]
                ax.axvline(np.mean(inlier_contrast), color=colors['inliers'], 
                          linestyle='--', linewidth=2.5, alpha=0.9,
                          label=f'Inlier mean: {np.mean(inlier_contrast):.3f}')
            
            ax.set_xlabel('Contrast Value', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title('Contrast Distribution', fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, color=colors['grid'])
            ax.set_facecolor('#FAFAFA')
        
        # Plot 5: Latent space (zs[:, 0] vs zs[:, 2]) with improved styling for large datasets
        ax = axes[1, 1]
        
        # Create hexbin density plot for background
        try:
            gridsize = min(50, max(20, int(np.sqrt(len(zs) / 100))))
            hb = ax.hexbin(zs[:, 0], zs[:, 2], gridsize=gridsize, 
                          cmap='Blues', alpha=0.4, mincnt=1, reduce_C_function=np.mean)
        except (ValueError, TypeError):
            pass
        
        # Downsample points for scatter plot
        if np.sum(inlier_mask) > 0:
            inlier_points = downsample_points(zs, inlier_mask, max_points=5000)
            ax.scatter(inlier_points[:, 0], inlier_points[:, 2], 
                      c=colors['inliers'], alpha=0.7, s=6, label=f'Inliers (n={np.sum(inlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        if np.sum(outlier_mask) > 0:
            outlier_points = downsample_points(zs, outlier_mask, max_points=2000)
            ax.scatter(outlier_points[:, 0], outlier_points[:, 2], 
                      c=colors['outliers'], alpha=0.9, s=10, label=f'Outliers (n={np.sum(outlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        ax.set_xlabel('Latent Dimension 1', fontweight='bold')
        ax.set_ylabel('Latent Dimension 3', fontweight='bold')
        ax.set_title('Latent Space (Dim 1 vs Dim 3)', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        # Plot 6: Latent space (zs[:, 1] vs zs[:, 3]) with improved styling for large datasets
        ax = axes[1, 2]
        
        # Create hexbin density plot for background
        try:
            gridsize = min(50, max(20, int(np.sqrt(len(zs) / 100))))
            hb = ax.hexbin(zs[:, 1], zs[:, 3], gridsize=gridsize, 
                          cmap='Blues', alpha=0.4, mincnt=1, reduce_C_function=np.mean)
        except (ValueError, TypeError):
            pass
        
        # Downsample points for scatter plot
        if np.sum(inlier_mask) > 0:
            inlier_points = downsample_points(zs, inlier_mask, max_points=5000)
            ax.scatter(inlier_points[:, 1], inlier_points[:, 3], 
                      c=colors['inliers'], alpha=0.7, s=6, label=f'Inliers (n={np.sum(inlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        if np.sum(outlier_mask) > 0:
            outlier_points = downsample_points(zs, outlier_mask, max_points=2000)
            ax.scatter(outlier_points[:, 1], outlier_points[:, 3], 
                      c=colors['outliers'], alpha=0.9, s=10, label=f'Outliers (n={np.sum(outlier_mask):,})', 
                      rasterized=True, edgecolors='none')
        ax.set_xlabel('Latent Dimension 2', fontweight='bold')
        ax.set_ylabel('Latent Dimension 4', fontweight='bold')
        ax.set_title('Latent Space (Dim 2 vs Dim 4)', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(os.path.join(viz_dir, f'{save_prefix}.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save statistics
        stats_file = os.path.join(viz_dir, f'{save_prefix}_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f'{method_name} Statistics\n')
            f.write(f'Total particles: {total_particles}\n')
            f.write(f'Outliers: {len(outlier_indices)} ({len(outlier_indices)/total_particles*100:.1f}%)\n')
            f.write(f'Inliers: {len(inlier_indices)} ({len(inlier_indices)/total_particles*100:.1f}%)\n')
            if contrast_values is not None:
                outlier_image_indices = get_contrast_indices_for_particles(outlier_indices)
                inlier_image_indices = get_contrast_indices_for_particles(inlier_indices)
                
                if outlier_image_indices is not None and len(outlier_image_indices) > 0:
                    outlier_contrast_mask = np.isin(original_image_indices, outlier_image_indices)
                    outlier_contrast = contrast_values[outlier_contrast_mask]
                    f.write(f'Outlier contrast - Mean: {np.mean(outlier_contrast):.3f}, Std: {np.std(outlier_contrast):.3f}\n')
                
                if inlier_image_indices is not None and len(inlier_image_indices) > 0:
                    inlier_contrast_mask = np.isin(original_image_indices, inlier_image_indices)
                    inlier_contrast = contrast_values[inlier_contrast_mask]
                    f.write(f'Inlier contrast - Mean: {np.mean(inlier_contrast):.3f}, Std: {np.std(inlier_contrast):.3f}\n')
    
    # Create plots for each method
    for i, (outliers, method) in enumerate(zip(all_particle_outliers, method_names)):
        if len(outliers) > 0:
            # Get inliers for this method
            method_inliers = np.setdiff1d(original_particle_indices, outliers)
            create_method_plots(outliers, method_inliers, method, f'{method.lower().replace(" ", "_")}_{zdim_key}')
    
    # Create plots for combined results
    if len(combined_particle_outliers) > 0:
        create_method_plots(combined_particle_outliers, combined_particle_inliers, 
                          'Combined', f'combined_{zdim_key}')
    
    logger.info("Outlier visualizations saved to: %s", viz_dir)



if __name__ == "__main__":
    main()
