import os
import sys
import argparse
import pickle  # For saving indices
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from recovar import output
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from recovar import tilt_dataset

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

# Set up logger
logger = logging.getLogger(__name__)

def plot_anomaly_detection_results(zs, original_indices, folder_name):
    """
    Plots anomaly detection results for given data and saves the plots and inlier/outlier indices.

    Parameters:
    - zs: numpy array
        The original dataset (may contain NaN values).
    - folder_name: str
        The folder name where all files (plots and indices) will be saved.
    """
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Identify valid entries (rows without NaNs)
    valid_mask = np.all(np.isfinite(zs), axis=1)
    valid_indices = np.where(valid_mask)[0]
    zs_valid = zs[valid_mask]

    if zs_valid.shape[0] == 0:
        print("No valid entries in zs after removing NaN values.")
        return

    # Compute UMAP on valid zs
    umapper = output.umap_latent_space(zs_valid)
    umap_valid = umapper.embedding_

    # Define anomaly detection algorithms
    anomaly_algorithms = [
        # ("Robust covariance", EllipticEnvelope(random_state=42)),  # Will be added later after contamination is calculated
        ("Isolation Forest", IsolationForest(random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35)),
    ]

    # Initialize lists to store predictions, algorithm names, and outlier percentages
    predictions = []
    algorithm_names = []
    outlier_percentages = []

    # Fit each algorithm (excluding Robust Covariance for now) and store predictions
    total_samples = zs_valid.shape[0]
    for name, algorithm in anomaly_algorithms:
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(zs_valid)
        else:
            y_pred = algorithm.fit(zs_valid).predict(zs_valid)
        predictions.append(y_pred)
        algorithm_names.append(name)

        # Calculate percentage of outliers
        num_outliers = np.sum(y_pred == -1)
        percentage_outliers = (num_outliers / total_samples)
        outlier_percentages.append(percentage_outliers)

        # Save indices of inliers and outliers in original indexing
        inliers_indices_valid = np.where(y_pred == 1)[0]
        outliers_indices_valid = np.where(y_pred == -1)[0]
        inliers_indices = valid_indices[inliers_indices_valid]
        outliers_indices = valid_indices[outliers_indices_valid]

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

    # Now add Robust Covariance
    robust_covariance = ("Robust covariance", EllipticEnvelope(random_state=42, contamination=avg_contamination))
    anomaly_algorithms.insert(0, robust_covariance)  # Insert at the beginning

    # Fit Robust Covariance
    name, algorithm = robust_covariance
    y_pred = algorithm.fit(zs_valid).predict(zs_valid)
    predictions.insert(0, y_pred)
    algorithm_names.insert(0, name)

    # Save indices of inliers and outliers in original indexing
    inliers_indices_valid = np.where(y_pred == 1)[0]
    outliers_indices_valid = np.where(y_pred == -1)[0]
    inliers_indices = valid_indices[inliers_indices_valid]
    outliers_indices = valid_indices[outliers_indices_valid]

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
    # consensus_inliers_indices_valid = np.where(consensus_inliers)[0]
    # consensus_outliers_indices_valid = np.where(consensus_outliers)[0]
    # consensus_inliers_indices = valid_indices[consensus_inliers_indices_valid]
    # consensus_outliers_indices = valid_indices[consensus_outliers_indices_valid]

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
    n_rows = 3  # zs[0] vs zs[1], zs[2] vs zs[3], umap[0] vs umap[1]
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
        ax = axes[0, i_algo]
        if np.sum(inliers) > 1:
            # Compute axis limits based on inliers
            x_min, x_max = np.percentile(zs_valid[inliers, 0], [0.1, 99.9])
            y_min, y_max = np.percentile(zs_valid[inliers, 1], [0.1, 99.9])

            hb = ax.hexbin(
                zs_valid[inliers, 0],
                zs_valid[inliers, 1],
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
            x_min, x_max = zs_valid[:, 0].min(), zs_valid[:, 0].max()
            y_min, y_max = zs_valid[:, 1].min(), zs_valid[:, 1].max()

        # Overlay outliers
        if np.sum(outliers) > 0:
            ax.scatter(
                zs_valid[outliers, 0],
                zs_valid[outliers, 1],
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

        # --- Plot zs[:, 2] vs zs[:, 3] if available ---
        if zs_valid.shape[1] >= 4:
            ax = axes[1, i_algo]
            if np.sum(inliers) > 1:
                # Compute axis limits based on inliers
                x_min, x_max = np.percentile(zs_valid[inliers, 2], [0.1, 99.9])
                y_min, y_max = np.percentile(zs_valid[inliers, 3], [0.1, 99.9])

                hb = ax.hexbin(
                    zs_valid[inliers, 2],
                    zs_valid[inliers, 3],
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
                x_min, x_max = zs_valid[:, 2].min(), zs_valid[:, 2].max()
                y_min, y_max = zs_valid[:, 3].min(), zs_valid[:, 3].max()
            # Overlay outliers
            if np.sum(outliers) > 0:
                ax.scatter(
                    zs_valid[outliers, 2],
                    zs_valid[outliers, 3],
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

    plt.suptitle("Anomaly Detection Results Including Consensus", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to the specified folder
    plot_filename = os.path.join(folder_name, "anomaly_detection_results.png")
    plt.savefig(plot_filename)
    plt.close()
    return inliers_mapped_back_to_original_indices, outliers_mapped_back_to_original_indices




from recovar import tilt_dataset    
def outlier_detection_from_contrast(pipeline_output, zdim_key=4, 
                                   low_contrast_threshold=0.1, 
                                   high_contrast_threshold=3.5,
                                   max_contrast=4.0,
                                   particle_bad_fraction_threshold=0.7,
                                   micrograph_bad_fraction_threshold=0.7,
                                   output_dir=None):
    """
    Perform outlier detection based on contrast values.
    
    Parameters:
    - contrasts: Array of contrast values for each image (1D array)
    - starfile: Path to starfile for grouping information
    - pipeline_output: Pipeline output object
    - zdim_key: Dimension key for embeddings
    - low_contrast_threshold: Threshold for low contrast outliers
    - high_contrast_threshold: Threshold for high contrast outliers
    - max_contrast: Maximum contrast value for normalization
    - particle_bad_fraction_threshold: Threshold for particle-based outlier detection
    - micrograph_bad_fraction_threshold: Threshold for micrograph-based outlier detection
    - output_dir: Output directory for saving results
    """
    print(f"Contrast-based outlier detection for zdim={zdim_key}")
    # print(f"Input contrasts type: {type(contrasts)}, shape: {contrasts.shape if hasattr(contrasts, 'shape') else len(contrasts)}")
    
    input_args = pipeline_output.get('input_args')
    starfile = getattr(input_args, 'particles', None)
    datadir = getattr(input_args, 'datadir', None)
    strip_prefix = getattr(input_args, 'strip_prefix', None)

    contrasts = pipeline_output.get('contrasts')[zdim_key]

    # Ensure contrasts is a numpy array
    if not isinstance(contrasts, np.ndarray):
        contrasts = np.array(contrasts, dtype=float)
    
    n_images = len(contrasts)
    contrast_array = contrasts
    
    print(f"Final contrast array shape: {contrast_array.shape}")
    print(f"Number of images: {n_images}")
    
    
    print(f"Contrast-based outlier detection for {n_images} images")
    print(f"Low contrast threshold: {low_contrast_threshold}")
    print(f"High contrast threshold: {high_contrast_threshold}")
    print(f"Particle bad fraction threshold: {particle_bad_fraction_threshold}")
    print(f"Micrograph bad fraction threshold: {micrograph_bad_fraction_threshold}")
    
    # Parse starfile for grouping information
    # Check if this is a tilt-series dataset by looking at pipeline input arguments
    input_args = pipeline_output.get('input_args')
    is_tilt_series = getattr(input_args, 'tilt_series', False)
    shared_contrast_across_tilts = getattr(input_args, 'shared_contrast_across_tilts', False)
    
    # Individual image outlier detection
    low_contrast_outliers = contrast_array < low_contrast_threshold
    high_contrast_outliers = contrast_array > high_contrast_threshold
    individual_outliers = low_contrast_outliers | high_contrast_outliers
    
    n_low_contrast = np.sum(low_contrast_outliers)
    n_high_contrast = np.sum(high_contrast_outliers)
    n_individual_outliers = np.sum(individual_outliers)
    
    print(f"\nIndividual image outlier detection:")
    print(f"  Low contrast outliers (< {low_contrast_threshold}): {n_low_contrast} ({n_low_contrast/n_images*100:.1f}%)")
    print(f"  High contrast outliers (> {high_contrast_threshold}): {n_high_contrast} ({n_high_contrast/n_images*100:.1f}%)")
    print(f"  Total individual outliers: {n_individual_outliers} ({n_individual_outliers/n_images*100:.1f}%)")
    
    halfsets = np.concatenate(pipeline_output.get('halfsets'))
    particles_halfsets = np.concatenate(pipeline_output.get('particles_halfsets'))


    outliers_ind = np.where(individual_outliers)[0]
    inliers_ind = np.where(~individual_outliers)[0]


    if not is_tilt_series:
        outliers_ind_mapped_back_to_original_indices = halfsets[outliers_ind]
        inliers_ind_mapped_back_to_original_indices = halfsets[inliers_ind]

        return outliers_ind_mapped_back_to_original_indices, inliers_ind_mapped_back_to_original_indices, None, None

    # If the dataset is a tilt-series or the contrast is shared across tilts, skip the grouping based on particle or micrograph
    if is_tilt_series and shared_contrast_across_tilts:
        outliers_ind_mapped_back_to_original_indices = particles_halfsets[outliers_ind]
        inliers_ind_mapped_back_to_original_indices = particles_halfsets[inliers_ind]

        return outliers_ind_mapped_back_to_original_indices, inliers_ind_mapped_back_to_original_indices, None, None

    particle_to_tilts, tilts_to_particle = tilt_dataset.TiltSeriesData.parse_particle_tilt(starfile)
    micrographtilt_to_tilts, tilts_to_micrographtilt = tilt_dataset.TiltSeriesData.parse_micrograph_tilt(starfile)

    outliers_image_identified_by_particle = np.zeros(n_images, dtype=bool)
    # Particle-based outlier detection
    if particle_to_tilts is not None:
        print(f"\nParticle-based outlier detection:")
        print(f"Contrast array size: {len(contrast_array)}")
        print(f"Total of particles: {len(particle_to_tilts)}, of which {particles_halfsets.size} were used in pipeline")
        
        particle_outliers = []
        particle_inliers = []
        tilts_mapped_to_particles = [tilts_to_particle[tilt] for tilt in halfsets]

        # Get unique particles and their inverse mapping
        unique_particles, inverse_particles = np.unique(tilts_mapped_to_particles, return_inverse=True)
        
        particle_median_contrasts = {}

        for i, particle in enumerate(unique_particles):
            particle_indices = inverse_particles == i
            contrast_values = contrast_array[particle_indices]
            particle_median_contrasts[particle] = np.median(contrast_values)
            bad_fraction = np.mean(individual_outliers[particle_indices])
            
            # # Get the actual tilt indices for this particle
            particle_tilt_indices = np.where(particle_indices)[0]
            
            if bad_fraction >= particle_bad_fraction_threshold:
                print(f"  Particle {particle}: {len(particle_tilt_indices)} images, {bad_fraction*100:.1f}% bad -> REJECTING ENTIRE PARTICLE")
                particle_outliers.append(particle)
                outliers_image_identified_by_particle[particle_indices] = True
            else:
                print(f"  Particle {particle}: {len(particle_tilt_indices)} images, {bad_fraction*100:.1f}% bad -> KEEPING")
                particle_inliers.append(particle)

        print(f"  Total particle-based outliers: {len(particle_outliers)} ({len(particle_outliers)/n_images*100:.1f}%)")


    outliers_image_identified_by_micrograph = np.zeros(n_images, dtype=bool)

    # Micrograph-based outlier detection
    if micrographtilt_to_tilts is not None:
        print(f"\nMicrograph-based outlier detection:")
        print(f"Contrast array size: {len(contrast_array)}")
        print(f"Number of micrographs: {len(micrographtilt_to_tilts)}")
        
        micrograph_outliers = []
        micrograph_inliers = []
        
        # Map tilts to micrographs
        tilts_mapped_to_micrographs = [tilts_to_micrographtilt[tilt] for tilt in halfsets]
        
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
                print(f"  Micrograph {micrograph}: {len(micrograph_tilt_indices)} images, {bad_fraction*100:.1f}% bad -> REJECTING ENTIRE MICROGRAPH")
                micrograph_outliers.extend(micrograph_tilt_indices)
                outliers_image_identified_by_micrograph[micrograph_indices] = True
            else:
                print(f"  Micrograph {micrograph}: {len(micrograph_tilt_indices)} images, {bad_fraction*100:.1f}% bad -> KEEPING")
                micrograph_inliers.append(micrograph)
        
        print(f"  Total micrograph-based outliers: {len(micrograph_outliers)} ({len(micrograph_outliers)/n_images*100:.1f}%)")


    # Print overlap statistics between methods
    if (particle_to_tilts is not None) or (micrographtilt_to_tilts is not None):
        print(f"\nOutlier detection method overlap:")
        individual_count = np.sum(individual_outliers)
        particle_count = np.sum(outliers_image_identified_by_particle)
        micrograph_count = np.sum(outliers_image_identified_by_micrograph)
        
        print(f"  Individual outliers: {individual_count} ({individual_count/n_images*100:.1f}%)")
        print(f"  Particle outliers: {particle_count} ({particle_count/n_images*100:.1f}%)")
        print(f"  Micrograph outliers: {micrograph_count} ({micrograph_count/n_images*100:.1f}%)")
        
        # Calculate overlaps
        if individual_count > 0 and particle_count > 0:
            overlap = np.sum(individual_outliers & outliers_image_identified_by_particle)
            print(f"  Individual-Particle overlap: {overlap} ({overlap/individual_count*100:.1f}% of individual)")
        if individual_count > 0 and micrograph_count > 0:
            overlap = np.sum(individual_outliers & outliers_image_identified_by_micrograph)
            print(f"  Individual-Micrograph overlap: {overlap} ({overlap/individual_count*100:.1f}% of individual)")
        if particle_count > 0 and micrograph_count > 0:
            overlap = np.sum(outliers_image_identified_by_particle & outliers_image_identified_by_micrograph)
            print(f"  Particle-Micrograph overlap: {overlap} ({overlap/particle_count*100:.1f}% of particle)")
    
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
            
            # Plot median contrast per particle
            if particle_to_tilts is not None:
                particle_median_contrasts = [np.median(contrast_array[particle_tilts]) for particle_tilts in particle_to_tilts if len(particle_tilts) > 0]
                axes[0].hist(particle_median_contrasts, bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[0].axvline(x=np.median(particle_median_contrasts), color='red', linestyle='--', linewidth=2)
                axes[0].set_xlabel('Median Contrast per Particle')
                axes[0].set_ylabel('Number of Particles')
                axes[0].set_title(f'Median Contrast per Particle\n(n={len(particle_median_contrasts)} particles)')
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, 'No particle grouping available', transform=axes[0].transAxes, ha='center', va='center')
                axes[0].set_title('Median Contrast per Particle')
            
            # Plot median contrast per micrograph
            if micrographtilt_to_tilts is not None:
                micrograph_median_contrasts = [np.median(contrast_array[micrograph_tilts]) for micrograph_tilts in micrographtilt_to_tilts if len(micrograph_tilts) > 0]
                axes[1].hist(micrograph_median_contrasts, bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1].axvline(x=np.median(micrograph_median_contrasts), color='red', linestyle='--', linewidth=2)
                axes[1].set_xlabel('Median Contrast per Micrograph')
                axes[1].set_ylabel('Number of Micrographs')
                axes[1].set_title(f'Median Contrast per Micrograph\n(n={len(micrograph_median_contrasts)} micrographs)')
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No micrograph grouping available', transform=axes[1].transAxes, ha='center', va='center')
                axes[1].set_title('Median Contrast per Micrograph')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'median_contrast_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save outlier indices
        outlier_indices = np.where(individual_outliers)[0]
        inlier_indices = np.where(~individual_outliers)[0]
        
        # Get halfsets from pipeline output for proper index mapping
        particles_halfsets = pipeline_output.get('particles_halfsets')
        image_halfsets = pipeline_output.get('halfsets')
        is_tilt_series = pipeline_output.get('input_args').tilt_series
        
        # Save indices using the unified function
        print(f"  Image-level outliers: {len(outlier_indices)} images")
        print(f"  Image-level inliers: {len(inlier_indices)} images")
    
    print(f"\nResults saved to: {output_dir}")
    
    combined_outliers = individual_outliers | outliers_image_identified_by_particle | outliers_image_identified_by_micrograph
    save_outlier_image_indices(combined_outliers, halfsets, False, output_dir, 'contrast_based')

    # Create results dictionary for return
    results = {
        'individual_outliers': individual_outliers,
        'outliers_image_identified_by_particle': outliers_image_identified_by_particle,
        'outliers_image_identified_by_micrograph': outliers_image_identified_by_micrograph,
        'final_outliers': individual_outliers,  # Use individual outliers as final result
        'statistics': {
            'n_images': n_images,
            'n_individual_outliers': n_individual_outliers,
            'n_final_outliers': n_individual_outliers,
            'n_particle_outliers': len(particle_outliers),
            'n_micrograph_outliers': len(micrograph_outliers)
        }

    }

    return results


def save_outlier_image_indices(classification, original_indices, is_tilt_series, output_dir, name):
    outlier_indices = np.where(classification)[0]
    inlier_indices = np.where(~classification)[0]

    outlier_indices_mapped_back_to_original_indices = original_indices[outlier_indices]
    inlier_indices_mapped_back_to_original_indices = original_indices[inlier_indices]

    if is_tilt_series:
        logger.info(f"Saving {name} tilt indices outliers and inliers for tilt series to {output_dir}")
        pickle.dump(outlier_indices_mapped_back_to_original_indices, open(os.path.join(output_dir, f"{name}_tilt_series_outliers.pkl"), "wb"))
        pickle.dump(inlier_indices_mapped_back_to_original_indices, open(os.path.join(output_dir, f"{name}_tilt_series_inliers.pkl"), "wb"))
    else:
        logger.info(f"Saving {name} image indices outliers and inliers for regular dataset to {output_dir}")
        pickle.dump(outlier_indices_mapped_back_to_original_indices, open(os.path.join(output_dir, f"{name}_outliers.pkl"), "wb"))
        pickle.dump(inlier_indices_mapped_back_to_original_indices, open(os.path.join(output_dir, f"{name}_inliers.pkl"), "wb"))
    return

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
    
    logger.info(f"Method overlap matrix saved to: {os.path.join(output_dir, 'outlier_method_overlap_matrix.png')}")

    
import logging
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
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'outlier_detection.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting outlier detection from pipeline output: {args.pipeline_output_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load pipeline output
    pipeline_output = output.PipelineOutput(args.pipeline_output_dir)
    
    # Determine zdim key
    zdim_key = f"{args.zdim_key}_noreg" if args.no_z_regularization else args.zdim_key
    
    # Check if embeddings exist
    zs_dict = pipeline_output.get('zs')
    if zs_dict is None or zdim_key not in zs_dict:
        available_dims = list(zs_dict.keys()) if zs_dict is not None else []
        logger.error(f"zdim {zdim_key} not found. Available dimensions: {available_dims}")
        sys.exit(1)
    
    # Load embeddings
    zs = zs_dict[zdim_key]
    logger.info(f"Loaded embeddings with shape: {zs.shape}")

    # Get the actual number of particles from the pipeline output
    # For tilt series, this might be different from the number of embeddings
    input_args = pipeline_output.get('input_args')
    is_tilt_series = getattr(input_args, 'tilt_series', False)
    is_shared_contrast = getattr(input_args, 'shared_contrast', False)

    # Get both types of halfsets
    particles_halfsets = pipeline_output.get('particles_halfsets')
    image_halfsets = pipeline_output.get('halfsets')
    
    if is_tilt_series:
        # For tilt series, get the number of particles (not images)
        total_particles = len(particles_halfsets[0]) + len(particles_halfsets[1])
        logger.info(f"Tilt series dataset: {total_particles} particles, {len(image_halfsets[0]) + len(image_halfsets[1])} total images")
    else:
        # For regular datasets, embeddings count = particle count
        total_particles = len(zs)
        logger.info(f"Cryo-EM dataset: {total_particles} particles and images ")
    
    # Log halfset information
    logger.info(f"Particles halfsets: {len(particles_halfsets[0])} + {len(particles_halfsets[1])} = {len(particles_halfsets[0]) + len(particles_halfsets[1])} particles")
    logger.info(f"Image halfsets: {len(image_halfsets[0])} + {len(image_halfsets[1])} = {len(image_halfsets[0]) + len(image_halfsets[1])} images")
    
    # Validate that outlier indices are within bounds
    def validate_outlier_indices(outliers, method_name):
        if outliers is not None and len(outliers) > 0:
            max_index = np.max(outliers)
            if max_index >= total_particles:
                error_msg = f"{method_name}: Maximum outlier index ({max_index}) exceeds total particles ({total_particles}). This indicates a bug in the outlier detection logic."
                logger.error(error_msg)
                raise ValueError(error_msg)
        return outliers
    
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
    anomaly_inliers = validate_outlier_indices(anomaly_inliers, "Anomaly Detection (inliers)")
    anomaly_outliers = validate_outlier_indices(anomaly_outliers, "Anomaly Detection (outliers)")
    
    logger.info(f"Anomaly detection completed. Found {len(anomaly_inliers)} inliers and {len(anomaly_outliers)} particle outliers.")
    
    # # Save pipeline-compatible indices if requested
    # if args.save_pipeline_indices:
    #     logger.info("Saving anomaly detection pipeline-compatible indices...")
    #     save_outlier_indices(anomaly_inliers, anomaly_outliers, anomaly_output_dir, zdim_key, "anomaly_detection",
    #                        particles_halfsets, image_halfsets, is_tilt_series)
    
    # --- Method 2: Contrast-based Outlier Detection ---
    contrasts = pipeline_output.get('contrasts')
    contrast_inliers = None
    contrast_outliers = None
    
        # Extract contrast values for the specific zdim_key

    contrast_values = contrasts[zdim_key]
    logger.info(f"Found contrast values for zdim={zdim_key}, shape: {contrast_values.shape}")
    
    # Load starfile path and options from pipeline input_args
    # logger.info(f"Using starfile: {starfile} (datadir={datadir}, strip_prefix={strip_prefix})")
    contrast_output_dir = os.path.join(args.output_dir, 'contrast_based')
    os.makedirs(contrast_output_dir, exist_ok=True)
    
    outlier_detection_from_contrast(
        pipeline_output=pipeline_output,
        zdim_key=zdim_key,
        low_contrast_threshold=args.low_contrast_threshold,
        high_contrast_threshold=args.high_contrast_threshold,
        max_contrast=args.max_contrast,
        particle_bad_fraction_threshold=args.particle_bad_fraction_threshold,
        micrograph_bad_fraction_threshold=args.micrograph_bad_fraction_threshold,
        output_dir=contrast_output_dir,
    )
    
    # Load contrast-based results
    contrast_inliers_file = os.path.join(contrast_output_dir, 'inlier_indices.pkl')
    contrast_outliers_file = os.path.join(contrast_output_dir, 'outlier_indices.pkl')
    
    if os.path.exists(contrast_inliers_file) and os.path.exists(contrast_outliers_file):
        with open(contrast_inliers_file, 'rb') as f:
            contrast_image_inliers = pickle.load(f)
        with open(contrast_outliers_file, 'rb') as f:
            contrast_image_outliers = pickle.load(f)
        
        # Validate contrast detection results (image-level)
        contrast_image_inliers = validate_outlier_indices(contrast_image_inliers, "Contrast-based (image inliers)")
        contrast_image_outliers = validate_outlier_indices(contrast_image_outliers, "Contrast-based (image outliers)")
        
        logger.info(f"Contrast-based outlier detection completed. Found {len(contrast_image_inliers)} image inliers and {len(contrast_image_outliers)} image outliers.")
        
        # For tilt series, also load particle-level indices if they exist
        if is_tilt_series:
            particle_inliers_file = os.path.join(contrast_output_dir, 'particle_inlier_indices.pkl')
            particle_outliers_file = os.path.join(contrast_output_dir, 'particle_outlier_indices.pkl')
            
            if os.path.exists(particle_inliers_file) and os.path.exists(particle_outliers_file):
                with open(particle_inliers_file, 'rb') as f:
                    contrast_particle_inliers = pickle.load(f)
                with open(particle_outliers_file, 'rb') as f:
                    contrast_particle_outliers = pickle.load(f)
                
                # Validate particle-level results
                contrast_particle_inliers = validate_outlier_indices(contrast_particle_inliers, "Contrast-based (particle inliers)")
                contrast_particle_outliers = validate_outlier_indices(contrast_particle_outliers, "Contrast-based (particle outliers)")
                
                logger.info(f"Loaded particle-level contrast results: {len(contrast_particle_inliers)} particle inliers and {len(contrast_particle_outliers)} particle outliers")
                
                # Use particle-level indices for combined results
                contrast_inliers = contrast_particle_inliers
                contrast_outliers = contrast_particle_outliers
            else:
                logger.error(f"Particle-level contrast indices not found for tilt-series dataset. Expected files:")
                logger.error(f"  {particle_inliers_file}")
                logger.error(f"  {particle_outliers_file}")
                logger.error("Contrast detection may have failed or the dataset may not be properly configured for tilt-series.")
                sys.exit(1)
        else:
            # For regular datasets, image indices = particle indices
            contrast_inliers = contrast_image_inliers
            contrast_outliers = contrast_image_outliers
        
        # Save both image and particle indices if requested
        if args.save_pipeline_indices:
            logger.info("Saving contrast-based pipeline indices...")
            save_outlier_indices(contrast_image_inliers, contrast_image_outliers, contrast_output_dir, zdim_key, "contrast_based",
                               particles_halfsets, image_halfsets, is_tilt_series)
    else:
        logger.error(f"Contrast-based results not found. Expected files:")
        logger.error(f"  {contrast_inliers_file}")
        logger.error(f"  {contrast_outliers_file}")
        logger.error("Contrast detection may have failed.")
        sys.exit(1)
    
    # --- Method 3: Junk Particle Detection ---
    junk_inliers = None
    junk_outliers = None
    
    if args.use_junk_detection:
        logger.info("Running junk particle detection...")
        
        # Import junk detection module only when needed
        try:
            from recovar.commands import junk_particle_detection
        except ImportError as e:
            logger.error(f"Failed to import junk_particle_detection: {e}")
            logger.error("Junk detection requires UMAP and other dependencies.")
            sys.exit(1)
        
        if args.use_junk_detection:
            junk_output_dir = os.path.join(args.output_dir, 'junk_detection')
            os.makedirs(junk_output_dir, exist_ok=True)
            
            # Run junk detection
            junk_particle_detection.junk_particle_detection_with_args(
                recovar_result_dir=args.pipeline_output_dir,
                output_folder=junk_output_dir,
                zdim=args.zdim_key,
                n_clusters=100,  # Default number of clusters
                batch_size=100,
                n_particles_per_cluster=args.particles_per_cluster,
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
            junk_inliers_file = os.path.join(junk_output_dir, 'inliers_pipeline_indices.pkl')
            junk_outliers_file = os.path.join(junk_output_dir, 'outliers_pipeline_indices.pkl')
            
            if os.path.exists(junk_inliers_file) and os.path.exists(junk_outliers_file):
                with open(junk_inliers_file, 'rb') as f:
                    junk_inliers = pickle.load(f)
                with open(junk_outliers_file, 'rb') as f:
                    junk_outliers = pickle.load(f)
                
                # Validate junk detection results
                junk_inliers = validate_outlier_indices(junk_inliers, "Junk Detection (inliers)")
                junk_outliers = validate_outlier_indices(junk_outliers, "Junk Detection (outliers)")
                
                logger.info(f"Junk particle detection completed. Found {len(junk_inliers)} inliers and {len(junk_outliers)} outliers.")
            else:
                logger.error(f"Junk detection results not found. Expected files:")
                logger.error(f"  {junk_inliers_file}")
                logger.error(f"  {junk_outliers_file}")
                logger.error("Junk detection may have failed.")
                sys.exit(1)
    
    # --- Combine Results from All Methods ---
    logger.info("Combining results from all methods...")
    combined_output_dir = os.path.join(args.output_dir, 'combined_results')
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # Collect all outlier indices
    all_outliers = []
    method_names = []
    
    if anomaly_outliers is not None:
        all_outliers.append(anomaly_outliers)
        method_names.append("Anomaly Detection")
    
    if contrast_outliers is not None:
        all_outliers.append(contrast_outliers)
        method_names.append("Contrast-based")
    
    if junk_outliers is not None:
        all_outliers.append(junk_outliers)
        method_names.append("Junk Detection")
    
    if len(all_outliers) > 0:
        # Combine results: particles are considered outliers if they are outliers in ANY method
        combined_outliers = all_outliers[0]
        for outliers in all_outliers[1:]:
            combined_outliers = np.union1d(combined_outliers, outliers)
        combined_inliers = np.setdiff1d(np.arange(total_particles), combined_outliers)
        
        # Save combined results
        if args.save_pipeline_indices:
            save_outlier_indices(combined_inliers, combined_outliers, combined_output_dir, zdim_key, "combined",
                               particles_halfsets, image_halfsets, is_tilt_series)
        
        # Save detailed breakdown
        breakdown_file = os.path.join(combined_output_dir, 'detection_breakdown.txt')
        with open(breakdown_file, 'w') as f:
            f.write("Combined Outlier Detection Results\n")
            f.write("==================================\n\n")
            f.write(f"Total particles: {total_particles}\n")
            f.write(f"Combined outliers: {len(combined_outliers)} ({len(combined_outliers)/total_particles*100:.1f}%)\n")
            f.write(f"Combined inliers: {len(combined_inliers)} ({len(combined_inliers)/total_particles*100:.1f}%)\n\n")
            
            f.write("Individual Method Results:\n")
            f.write("-------------------------\n")
            
            for i, (outliers, method) in enumerate(zip(all_outliers, method_names)):
                f.write(f"{method}:\n")
                f.write(f"  Outliers: {len(outliers)} ({len(outliers)/total_particles*100:.1f}%)\n")
                f.write(f"  Inliers: {total_particles - len(outliers)} ({(total_particles - len(outliers))/total_particles*100:.1f}%)\n\n")
            
            # Find particles detected by multiple methods
            f.write("Method Overlap Analysis:\n")
            f.write("-----------------------\n")
            
            for i in range(len(all_outliers)):
                for j in range(i + 1, len(all_outliers)):
                    overlap = np.intersect1d(all_outliers[i], all_outliers[j])
                    f.write(f"{method_names[i]} + {method_names[j]} overlap: {len(overlap)} ({len(overlap)/total_particles*100:.1f}%)\n")
            
            if len(all_outliers) == 3:
                # All three methods overlap
                all_three_overlap = np.intersect1d.reduce(all_outliers)
                f.write(f"All three methods overlap: {len(all_three_overlap)} ({len(all_three_overlap)/total_particles*100:.1f}%)\n")
            
            # Find unique outliers for each method
            f.write("\nUnique Outliers Analysis:\n")
            f.write("------------------------\n")
            
            for i, (outliers, method) in enumerate(zip(all_outliers, method_names)):
                # Find outliers unique to this method
                other_outliers = []
                for j, other_outliers_list in enumerate(all_outliers):
                    if i != j:
                        other_outliers.extend(other_outliers_list)
                
                if other_outliers:
                    other_outliers = np.unique(other_outliers)
                    unique_outliers = np.setdiff1d(outliers, other_outliers)
                    f.write(f"{method} unique outliers: {len(unique_outliers)} ({len(unique_outliers)/total_particles*100:.1f}%)\n")
                else:
                    f.write(f"{method} unique outliers: {len(outliers)} ({len(outliers)/total_particles*100:.1f}%)\n")
        
        # Create overlap matrix visualization
        if len(all_outliers) > 1:
            create_overlap_matrix_visualization(all_outliers, method_names, combined_output_dir, zdim_key, total_particles)
        
        # Create contrast distribution plots if contrast data is available
        if contrasts is not None:
            # Collect all inlier arrays
            all_inliers = []
            for outliers in all_outliers:
                inliers = np.setdiff1d(np.arange(total_particles), outliers)
                all_inliers.append(inliers)
            
            # Get starfile path from pipeline input_args
            input_args = pipeline_output.get('input_args')
            starfile = getattr(input_args, 'particles', None)
            
            create_contrast_distribution_plots(all_outliers, all_inliers, method_names, contrasts,
                                             particles_halfsets, image_halfsets, is_tilt_series,
                                             combined_output_dir, zdim_key, total_particles, starfile)
        
        logger.info(f"Combined results saved to: {combined_output_dir}")
        logger.info(f"Combined outliers: {len(combined_outliers)} ({len(combined_outliers)/total_particles*100:.1f}%)")
    else:
        logger.warning("No outlier detection methods produced results. Combined analysis skipped.")
    
    # Create summary file
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Outlier Detection Summary\n")
        f.write("========================\n\n")
        f.write(f"Pipeline output directory: {args.pipeline_output_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"zdim key: {zdim_key}\n")
        f.write(f"Total particles: {total_particles}\n\n")
        
        f.write("Methods Run:\n")
        f.write("------------\n")
        f.write("✓ Anomaly Detection (UMAP-based)\n")
        if contrasts is not None:
            f.write("✓ Contrast-based Outlier Detection\n")
        else:
            f.write("✗ Contrast-based Outlier Detection (no contrast data)\n")
        if args.use_junk_detection:
            f.write("✓ Junk Particle Detection\n")
        else:
            f.write("✗ Junk Particle Detection (not enabled)\n")
        
        f.write("\nOutput Structure:\n")
        f.write("-----------------\n")
        f.write("anomaly_detection/     - UMAP-based anomaly detection results\n")
        f.write("contrast_based/        - Contrast-based outlier detection results\n")
        f.write("junk_detection/        - Junk particle detection results\n")
        f.write("combined_results/      - Combined results from all methods\n")
        f.write("outlier_detection.log  - Log file\n")
        f.write("summary.txt           - This summary file\n")
    
    logger.info(f"Outlier detection completed. Results saved to {args.output_dir}")
    logger.info(f"Summary file: {summary_file}")

def add_args(parser):
    """Add command line arguments for outlier detection."""
    parser.add_argument("pipeline_output_dir", type=str, help="Directory containing recovar pipeline results")
    parser.add_argument("--zdim-key", type=int, default=4, help="Dimension key for embeddings (default: 4)")
    parser.add_argument("--no-z-regularization", action="store_true", help="Use unregularized embeddings")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for results (default: pipeline_output_dir/outlier_detection)")
    parser.add_argument("--save-pipeline-indices", action="store_true", 
                       help="Save indices in pipeline-compatible format (--ind for images, --tilt-ind for particles in tilt series)")
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
    parser.add_argument("--particles-per-cluster", type=int, default=100, 
                       help="Number of particles per cluster for junk detection (default: 100)")
    
    return parser


# def save_outlier_indices(inliers_indices, outliers_indices, output_dir, zdim_key, method="consensus", 
#                         particles_halfsets=None, image_halfsets=None, is_tilt_series=False):
#     """
#     Save outlier indices in pipeline-compatible format using proper halfsets mapping.
    
#     Parameters:
#     - inliers_indices: Array of inlier indices (in the order of embeddings/contrasts)
#     - outliers_indices: Array of outlier indices (in the order of embeddings/contrasts)
#     - output_dir: Output directory for saving files
#     - zdim_key: Dimension key for embeddings
#     - method: Outlier detection method used
#     - particles_halfsets: Particle-level halfsets (for --tilt-ind)
#     - image_halfsets: Image-level halfsets (for --ind)
#     - is_tilt_series: Whether this is a tilt series dataset
#     """
#     logger.info("Saving outlier indices in pipeline-compatible format...")
    
#     # Save image-level indices (for --ind)
#     inliers_pipeline_file = os.path.join(output_dir, f'inliers_pipeline_indices_{zdim_key}.pkl')
#     with open(inliers_pipeline_file, 'wb') as f:
#         pickle.dump(inliers_indices, f)
    
#     outliers_pipeline_file = os.path.join(output_dir, f'outliers_pipeline_indices_{zdim_key}.pkl')
#     with open(outliers_pipeline_file, 'wb') as f:
#         pickle.dump(outliers_indices, f)
    
#     # For tilt-series, also save particle-level indices (for --tilt-ind)
#     if is_tilt_series and particles_halfsets is not None:
#         # Convert image indices to particle indices using halfsets mapping
#         def image_to_particle_indices(image_indices):
#             """Convert image indices to particle indices using halfsets mapping."""
#             particle_indices = set()
#             for image_idx in image_indices:
#                 # Find which halfset this image belongs to
#                 if image_idx in image_halfsets[0]:
#                     image_pos_in_halfset = np.where(image_halfsets[0] == image_idx)[0][0]
#                     particle_indices.add(particles_halfsets[0][image_pos_in_halfset])
#                 elif image_idx in image_halfsets[1]:
#                     image_pos_in_halfset = np.where(image_halfsets[1] == image_idx)[0][0]
#                     particle_indices.add(particles_halfsets[1][image_pos_in_halfset])
#             return np.array(sorted(list(particle_indices)))
        
#         particle_inliers = image_to_particle_indices(inliers_indices)
#         particle_outliers = image_to_particle_indices(outliers_indices)
        
#         # Save particle-level indices with different naming
#         particle_inliers_file = os.path.join(output_dir, f'particle_inliers_pipeline_indices_{zdim_key}.pkl')
#         particle_outliers_file = os.path.join(output_dir, f'particle_outliers_pipeline_indices_{zdim_key}.pkl')
        
#         with open(particle_inliers_file, 'wb') as f:
#             pickle.dump(particle_inliers, f)
#         with open(particle_outliers_file, 'wb') as f:
#             pickle.dump(particle_outliers, f)
        
#         logger.info(f"Saved particle-level indices: {len(particle_inliers)} inliers, {len(particle_outliers)} outliers")
    
#     # Create usage summary
#     summary_file = os.path.join(output_dir, f'pipeline_usage_summary_{zdim_key}.txt')
#     with open(summary_file, 'w') as f:
#         f.write("Pipeline-Compatible Outlier Indices Usage Summary\n")
#         f.write("==================================================\n\n")
#         f.write(f"Method: {method}\n")
#         f.write(f"zdim: {zdim_key}\n")
#         f.write(f"Dataset type: {'Tilt series' if is_tilt_series else 'Regular'}\n\n")
        
#         f.write("Image-level indices (for --ind):\n")
#         f.write(f"  Inlier images: {os.path.abspath(inliers_pipeline_file)}\n")
#         f.write(f"  Outlier images: {os.path.abspath(outliers_pipeline_file)}\n\n")
        
#         if is_tilt_series and particles_halfsets is not None:
#             f.write("Particle-level indices (for --tilt-ind):\n")
#             f.write(f"  Inlier particles: {os.path.abspath(particle_inliers_file)}\n")
#             f.write(f"  Outlier particles: {os.path.abspath(particle_outliers_file)}\n\n")
        
#         f.write("Example usage:\n")
#         f.write("  # Run pipeline with only inlier images:\n")
#         f.write(f"  python -m recovar.commands.pipeline particles.star --poses poses.pkl --ctf ctf.pkl --ind {os.path.basename(inliers_pipeline_file)} -o output_dir\n")
        
#         if is_tilt_series and particles_halfsets is not None:
#             f.write("  # Run pipeline with only inlier particles (tilt series):\n")
#             f.write(f"  python -m recovar.commands.pipeline particles.star --poses poses.pkl --ctf ctf.pkl --tilt-ind {os.path.basename(particle_inliers_file)} -o output_dir\n")
    
#     logger.info(f"Pipeline-compatible indices saved:")
#     logger.info(f"  Image inliers: {inliers_pipeline_file}")
#     logger.info(f"  Image outliers: {outliers_pipeline_file}")
#     if is_tilt_series and particles_halfsets is not None:
#         logger.info(f"  Particle inliers: {particle_inliers_file}")
#         logger.info(f"  Particle outliers: {particle_outliers_file}")
#     logger.info(f"  Usage summary: {summary_file}")
    
#     return inliers_pipeline_file, outliers_pipeline_file, summary_file

# def create_contrast_distribution_plots(all_outliers, all_inliers, method_names, contrasts, 
#                                      particles_halfsets, image_halfsets, is_tilt_series, 
#                                      output_dir, zdim_key, total_particles, starfile=None):
#     """
#     Create contrast distribution histograms for outliers vs inliers from different detection methods.
    
#     Parameters:
#     - all_outliers: List of outlier arrays from different methods
#     - all_inliers: List of inlier arrays from different methods  
#     - method_names: List of method names
#     - contrasts: Array of contrast values (one per image)
#     - particles_halfsets: Particle-level halfsets (for --tilt-ind)
#     - image_halfsets: Image-level halfsets (for --ind)
#     - is_tilt_series: Whether this is a tilt series dataset
#     - output_dir: Output directory for saving plots
#     - zdim_key: Dimension key for embeddings
#     - total_particles: Total number of particles
#     - starfile: Path to starfile (needed for tilt series mapping)
#     """
#     if contrasts is None:
#         logger.warning("No contrast data available. Skipping contrast distribution plots.")
#         return
    
#     logger.info("Creating contrast distribution plots...")
    
#     # Convert particle indices to image indices for contrast lookup
#     def particle_to_image_indices(particle_indices):
#         """Convert particle indices to image indices for tilt series."""
#         if not is_tilt_series:
#             # For regular datasets, particle indices = image indices
#             return particle_indices
        
#         if starfile is None:
#             logger.error("Starfile is required for tilt series but not provided.")
#             sys.exit(1)
        
#         # Use proper tilt dataset functions
#         try:
#             particle_to_tilts, tilts_to_particle = tilt_dataset.TiltSeriesData.parse_particle_tilt(starfile)
            
#             # Convert particle indices to tilt indices
#             image_indices = set()
#             for particle_idx in particle_indices:
#                 if particle_idx in particle_to_tilts:
#                     # Get all tilt indices for this particle
#                     tilt_indices = particle_to_tilts[particle_idx]
#                     image_indices.update(tilt_indices)
            
#             return np.array(list(image_indices))
#         except Exception as e:
#             logger.error(f"Failed to use tilt dataset functions: {e}")
#             sys.exit(1)
    
#     # Create subplots for each method
#     n_methods = len(method_names)
#     fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
#     if n_methods == 1:
#         axes = axes.reshape(2, 1)
    
#     for i, (outliers, inliers, method) in enumerate(zip(all_outliers, all_inliers, method_names)):
#         # Convert particle indices to image indices
#         outlier_image_indices = particle_to_image_indices(outliers)
#         inlier_image_indices = particle_to_image_indices(inliers)
        
#         # Get contrast values for outliers and inliers
#         outlier_contrasts = contrasts[outlier_image_indices]
#         inlier_contrasts = contrasts[inlier_image_indices]
        
#         # Plot 1: Histogram of contrast distributions
#         ax1 = axes[0, i]
#         ax1.hist(inlier_contrasts, bins=50, alpha=0.7, label='Inliers', color='blue', density=True)
#         ax1.hist(outlier_contrasts, bins=50, alpha=0.7, label='Outliers', color='red', density=True)
#         ax1.set_xlabel('Contrast')
#         ax1.set_ylabel('Density')
#         ax1.set_title(f'{method}\nContrast Distribution')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Add statistics
#         inlier_mean = np.mean(inlier_contrasts)
#         outlier_mean = np.mean(outlier_contrasts)
#         inlier_std = np.std(inlier_contrasts)
#         outlier_std = np.std(outlier_contrasts)
        
#         stats_text = f'Inliers: μ={inlier_mean:.3f}, σ={inlier_std:.3f}\nOutliers: μ={outlier_mean:.3f}, σ={outlier_std:.3f}'
#         ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
#         # Plot 2: Box plot comparison
#         ax2 = axes[1, i]
#         data = [inlier_contrasts, outlier_contrasts]
#         labels = ['Inliers', 'Outliers']
#         colors = ['lightblue', 'lightcoral']
        
#         bp = ax2.boxplot(data, labels=labels, patch_artist=True)
#         for patch, color in zip(bp['boxes'], colors):
#             patch.set_facecolor(color)
        
#         ax2.set_ylabel('Contrast')
#         ax2.set_title(f'{method}\nContrast Box Plot')
#         ax2.grid(True, alpha=0.3)
        
#         # Add sample sizes
#         ax2.text(0.02, 0.98, f'n_inliers={len(inlier_contrasts)}\nn_outliers={len(outlier_contrasts)}', 
#                 transform=ax2.transAxes, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'contrast_distributions.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Create a summary plot showing all methods together
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Combined histogram
#     for i, (outliers, inliers, method) in enumerate(zip(all_outliers, all_inliers, method_names)):
#         outlier_image_indices = particle_to_image_indices(outliers)
#         inlier_image_indices = particle_to_image_indices(inliers)
        
#         outlier_contrasts = contrasts[outlier_image_indices]
#         inlier_contrasts = contrasts[inlier_image_indices]
        
#         # Use different colors for each method
#         colors = ['red', 'orange', 'green', 'purple', 'brown']
#         color = colors[i % len(colors)]
        
#         ax1.hist(outlier_contrasts, bins=30, alpha=0.6, label=f'{method} Outliers', 
#                 color=color, density=True, histtype='step', linewidth=2)
    
#     ax1.set_xlabel('Contrast')
#     ax1.set_ylabel('Density')
#     ax1.set_title('Contrast Distribution: All Outlier Methods')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Combined box plot
#     all_outlier_contrasts = []
#     all_inlier_contrasts = []
#     all_labels = []
    
#     for i, (outliers, inliers, method) in enumerate(zip(all_outliers, all_inliers, method_names)):
#         outlier_image_indices = particle_to_image_indices(outliers)
#         inlier_image_indices = particle_to_image_indices(inliers)
        
#         outlier_contrasts = contrasts[outlier_image_indices]
#         inlier_contrasts = contrasts[inlier_image_indices]
        
#         all_outlier_contrasts.append(outlier_contrasts)
#         all_inlier_contrasts.append(inlier_contrasts)
#         all_labels.extend([f'{method} Inliers', f'{method} Outliers'])
    
#     # Combine all data for box plot
#     all_data = []
#     for inlier_contrasts, outlier_contrasts in zip(all_inlier_contrasts, all_outlier_contrasts):
#         all_data.extend([inlier_contrasts, outlier_contrasts])
    
#     bp = ax2.boxplot(all_data, labels=all_labels, patch_artist=True)
#     colors = ['lightblue', 'lightcoral'] * len(method_names)
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
    
#     ax2.set_ylabel('Contrast')
#     ax2.set_title('Contrast Comparison: All Methods')
#     ax2.tick_params(axis='x', rotation=45)
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'contrast_distributions_combined.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     logger.info(f"Contrast distribution plots saved to: {os.path.join(output_dir, 'contrast_distributions.png')}")
#     logger.info(f"Combined contrast distribution plot saved to: {os.path.join(output_dir, 'contrast_distributions_combined.png')}")


if __name__ == "__main__":
    main()
