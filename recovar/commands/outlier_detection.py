import os
import sys
import argparse
import pickle  # For saving indices
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from recovar import output
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

matplotlib.rcParams["contour.negative_linestyle"] = "solid"

# import os
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import pickle  # For saving indices
# from recovar import output
# from sklearn.covariance import EllipticEnvelope
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor

# matplotlib.rcParams["contour.negative_linestyle"] = "solid"

def plot_anomaly_detection_results(zs, folder_name):
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

    # Now add Robust Covariance with calculated contamination
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
    consensus_inliers_indices_valid = np.where(consensus_inliers)[0]
    consensus_outliers_indices_valid = np.where(consensus_outliers)[0]
    consensus_inliers_indices = valid_indices[consensus_inliers_indices_valid]
    consensus_outliers_indices = valid_indices[consensus_outliers_indices_valid]
    with open(os.path.join(folder_name, "inliers_consensus.pkl"), "wb") as f:
        pickle.dump(consensus_inliers_indices, f)
    with open(os.path.join(folder_name, "outliers_consensus.pkl"), "wb") as f:
        pickle.dump(consensus_outliers_indices, f)

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



from recovar import tilt_dataset    
def outlier_detection_from_contrast(contrasts, starfile, zdim_key=4, 
                                   low_contrast_threshold=0.1, 
                                   high_contrast_threshold=3.5,
                                   max_contrast=4.0,
                                   tomogram_bad_fraction_threshold=0.7,
                                   tilt_series_bad_fraction_threshold=0.7,
                                   output_dir=None):
    """
    Perform outlier detection based on estimated contrast values.
    
    Args:
        contrasts: Dictionary of contrast arrays from pipeline output
        starfile: Path to the starfile containing particle metadata
        zdim_key: Key for the contrast array to use (e.g., 4, 10, '4_noreg')
        low_contrast_threshold: Contrast values below this are considered outliers (default: 0.1)
        high_contrast_threshold: Contrast values above this are considered outliers (default: 3.5)
        max_contrast: Maximum possible contrast value (default: 4.0)
        tomogram_bad_fraction_threshold: If this fraction of a tomogram's images are bad, reject entire tomogram (default: 0.7)
        tilt_series_bad_fraction_threshold: If this fraction of a tilt series' images are bad, reject entire tilt series (default: 0.7)
        output_dir: Directory to save results (optional)
    
    Returns:
        dict: Dictionary containing outlier detection results
    """
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get contrast array for the specified zdim
    if zdim_key not in contrasts:
        raise ValueError(f"zdim_key {zdim_key} not found in contrasts. Available keys: {list(contrasts.keys())}")
    
    contrast_array = contrasts[zdim_key]
    n_images = len(contrast_array)
    
    print(f"Analyzing {n_images} images with contrast values from zdim={zdim_key}")
    print(f"Contrast range: {contrast_array.min():.3f} to {contrast_array.max():.3f}")
    print(f"Contrast mean: {contrast_array.mean():.3f}, std: {contrast_array.std():.3f}")
    
    # Parse grouping information from starfile
    try:
        tomogram_to_tilts, tilts_to_tomogram = tilt_dataset.TiltSeriesData.parse_tomogram_tilt(starfile)
        tomogramtilt_to_tilts, tilts_to_tomogramtilt = tilt_dataset.TiltSeriesData.parse_tomogramtilt_tilt(starfile)
        print(f"Found {len(tomogram_to_tilts)} tomograms and {len(tomogramtilt_to_tilts)} tomogram tilts")
    except Exception as e:
        print(f"Warning: Could not parse starfile for grouping: {e}")
        print("Proceeding with individual image outlier detection only")
        tomogram_to_tilts = None
        tilts_to_tomogram = None
        tomogramtilt_to_tilts = None
        tilts_to_tomogramtilt = None
    
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
    
    # Initialize results
    results = {
        'individual_outliers': individual_outliers,
        'low_contrast_outliers': low_contrast_outliers,
        'high_contrast_outliers': high_contrast_outliers,
        'tomogram_outliers': np.zeros(n_images, dtype=bool),
        'tilt_series_outliers': np.zeros(n_images, dtype=bool),
        'final_outliers': individual_outliers.copy(),
        'statistics': {
            'n_images': n_images,
            'n_individual_outliers': n_individual_outliers,
            'n_tomogram_outliers': 0,
            'n_tilt_series_outliers': 0,
            'n_final_outliers': n_individual_outliers
        }
    }
    
    # Tomogram-based outlier detection
    if tomogram_to_tilts is not None:
        print(f"\nTomogram-based outlier detection:")
        tomogram_outliers = []
        
        for i, tomogram_tilts in enumerate(tomogram_to_tilts):
            if len(tomogram_tilts) == 0:
                continue
                
            # Get contrast values for this tomogram
            tomogram_contrasts = contrast_array[tomogram_tilts]
            tomogram_individual_outliers = individual_outliers[tomogram_tilts]
            
            # Calculate fraction of bad images in this tomogram
            bad_fraction = np.mean(tomogram_individual_outliers)
            
            if bad_fraction >= tomogram_bad_fraction_threshold:
                print(f"  Tomogram {i}: {len(tomogram_tilts)} images, {bad_fraction*100:.1f}% bad -> REJECTING ENTIRE TOMOGRAM")
                tomogram_outliers.extend(tomogram_tilts)
                results['tomogram_outliers'][tomogram_tilts] = True
            else:
                print(f"  Tomogram {i}: {len(tomogram_tilts)} images, {bad_fraction*100:.1f}% bad -> KEEPING")
        
        results['statistics']['n_tomogram_outliers'] = len(tomogram_outliers)
        results['final_outliers'] |= results['tomogram_outliers']
        print(f"  Total tomogram-based outliers: {len(tomogram_outliers)} ({len(tomogram_outliers)/n_images*100:.1f}%)")
    
    # Tilt series-based outlier detection
    if tomogramtilt_to_tilts is not None:
        print(f"\nTilt series-based outlier detection:")
        tilt_series_outliers = []
        
        for i, tilt_series_tilts in enumerate(tomogramtilt_to_tilts):
            if len(tilt_series_tilts) == 0:
                continue
                
            # Get contrast values for this tilt series
            tilt_series_contrasts = contrast_array[tilt_series_tilts]
            tilt_series_individual_outliers = individual_outliers[tilt_series_tilts]
            
            # Calculate fraction of bad images in this tilt series
            bad_fraction = np.mean(tilt_series_individual_outliers)
            
            if bad_fraction >= tilt_series_bad_fraction_threshold:
                print(f"  Tilt series {i}: {len(tilt_series_tilts)} images, {bad_fraction*100:.1f}% bad -> REJECTING ENTIRE TILT SERIES")
                tilt_series_outliers.extend(tilt_series_tilts)
                results['tilt_series_outliers'][tilt_series_tilts] = True
            else:
                print(f"  Tilt series {i}: {len(tilt_series_tilts)} images, {bad_fraction*100:.1f}% bad -> KEEPING")
        
        results['statistics']['n_tilt_series_outliers'] = len(tilt_series_outliers)
        results['final_outliers'] |= results['tilt_series_outliers']
        print(f"  Total tilt series-based outliers: {len(tilt_series_outliers)} ({len(tilt_series_outliers)/n_images*100:.1f}%)")
    
    # Update final statistics
    results['statistics']['n_final_outliers'] = np.sum(results['final_outliers'])
    
    print(f"\nFinal outlier detection summary:")
    print(f"  Individual outliers: {results['statistics']['n_individual_outliers']} ({results['statistics']['n_individual_outliers']/n_images*100:.1f}%)")
    print(f"  Tomogram-based outliers: {results['statistics']['n_tomogram_outliers']} ({results['statistics']['n_tomogram_outliers']/n_images*100:.1f}%)")
    print(f"  Tilt series-based outliers: {results['statistics']['n_tilt_series_outliers']} ({results['statistics']['n_tilt_series_outliers']/n_images*100:.1f}%)")
    print(f"  TOTAL FINAL OUTLIERS: {results['statistics']['n_final_outliers']} ({results['statistics']['n_final_outliers']/n_images*100:.1f}%)")
    
    # Create plots if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, 'contrast_outlier_detection_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Create contrast histogram with outlier regions highlighted
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.hist(contrast_array, bins=50, alpha=0.7, color='blue', label='All images')
        plt.hist(contrast_array[individual_outliers], bins=50, alpha=0.8, color='red', label='Individual outliers')
        
        # Add vertical lines for thresholds
        plt.axvline(x=low_contrast_threshold, color='orange', linestyle='--', linewidth=2, label=f'Low threshold ({low_contrast_threshold})')
        plt.axvline(x=high_contrast_threshold, color='orange', linestyle='--', linewidth=2, label=f'High threshold ({high_contrast_threshold})')
        
        plt.xlabel('Contrast')
        plt.ylabel('Number of images')
        plt.title(f'Contrast Distribution and Outlier Detection (zdim={zdim_key})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f"""Statistics:
Total images: {n_images}
Individual outliers: {n_individual_outliers} ({n_individual_outliers/n_images*100:.1f}%)
Final outliers: {results['statistics']['n_final_outliers']} ({results['statistics']['n_final_outliers']/n_images*100:.1f}%)"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'contrast_outlier_detection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot showing outlier progression
        if tomogram_to_tilts is not None or tomogramtilt_to_tilts is not None:
            plt.figure(figsize=(12, 8))
            
            # Create a simple 2D representation (using image index vs contrast)
            x = np.arange(n_images)
            
            # Plot all points
            plt.scatter(x, contrast_array, alpha=0.6, s=10, color='blue', label='All images')
            
            # Highlight different types of outliers
            if np.any(individual_outliers):
                plt.scatter(x[individual_outliers], contrast_array[individual_outliers], 
                           alpha=0.8, s=20, color='red', label='Individual outliers')
            
            if np.any(results['tomogram_outliers']):
                plt.scatter(x[results['tomogram_outliers']], contrast_array[results['tomogram_outliers']], 
                           alpha=0.8, s=30, color='purple', marker='s', label='Tomogram outliers')
            
            if np.any(results['tilt_series_outliers']):
                plt.scatter(x[results['tilt_series_outliers']], contrast_array[results['tilt_series_outliers']], 
                           alpha=0.8, s=30, color='green', marker='^', label='Tilt series outliers')
            
            # Add threshold lines
            plt.axhline(y=low_contrast_threshold, color='orange', linestyle='--', alpha=0.7)
            plt.axhline(y=high_contrast_threshold, color='orange', linestyle='--', alpha=0.7)
            
            plt.xlabel('Image Index')
            plt.ylabel('Contrast')
            plt.title(f'Contrast Outlier Detection Progression (zdim={zdim_key})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'contrast_outlier_progression.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save outlier indices
        outlier_indices = np.where(results['final_outliers'])[0]
        inlier_indices = np.where(~results['final_outliers'])[0]
        
        with open(os.path.join(output_dir, 'outlier_indices.pkl'), 'wb') as f:
            pickle.dump(outlier_indices, f)
        
        with open(os.path.join(output_dir, 'inlier_indices.pkl'), 'wb') as f:
            pickle.dump(inlier_indices, f)
        
        print(f"\nResults saved to: {output_dir}")
    
    return results

    
import logging
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run outlier detection and plot results.")
    parser.add_argument("input_dir", type=str, help="Directory where the recovar results are stored.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Folder where the plots and indices will be saved.")
    parser.add_argument("--zdim", type=int, required=True, help="Dimension of the zs array to use.")
    parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
    parser.add_argument("--starfile", type=str, help="Path to starfile for contrast-based outlier detection.")
    parser.add_argument("--contrast-outliers", action="store_true", help="Run contrast-based outlier detection.")
    parser.add_argument("--low-contrast-threshold", type=float, default=0.1, help="Low contrast threshold for outlier detection.")
    parser.add_argument("--high-contrast-threshold", type=float, default=3.5, help="High contrast threshold for outlier detection.")
    parser.add_argument("--tomogram-threshold", type=float, default=0.7, help="Fraction threshold for rejecting entire tomograms.")
    parser.add_argument("--tilt-series-threshold", type=float, default=0.7, help="Fraction threshold for rejecting entire tilt series.")

    args = parser.parse_args()

    # Use the parsed arguments
    recovar_result_dir = args.input_dir
    zdim = args.zdim
    no_z_regularization = args.no_z_regularization
    if no_z_regularization:
        zdim_key = f"{zdim}_noreg"
    else:
        zdim_key = zdim

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    pipeline_output = output.PipelineOutput(recovar_result_dir)

    # Run contrast-based outlier detection if requested
    if args.contrast_outliers:
        if args.starfile is None:
            logger.error("--starfile is required for contrast-based outlier detection")
            return
        
        logger.info("Running contrast-based outlier detection...")
        
        # Get contrasts from pipeline output
        unsorted_embedding = pipeline_output.get('unsorted_embedding')
        contrasts = unsorted_embedding['contrasts']
        
        # Run contrast-based outlier detection
        contrast_results = outlier_detection_from_contrast(
            contrasts=contrasts,
            starfile=args.starfile,
            zdim_key=zdim_key,
            low_contrast_threshold=args.low_contrast_threshold,
            high_contrast_threshold=args.high_contrast_threshold,
            tomogram_bad_fraction_threshold=args.tomogram_threshold,
            tilt_series_bad_fraction_threshold=args.tilt_series_threshold,
            output_dir=args.output_dir
        )
        
        logger.info("Contrast-based outlier detection completed.")
        return

    # Run traditional anomaly detection
    logger.info("Running traditional anomaly detection...")
    
    # Get zs from pipeline output
    unsorted_embedding = pipeline_output.get('unsorted_embedding')
    zs = unsorted_embedding['zs'][zdim_key]

    # Call the function with the parsed arguments
    plot_anomaly_detection_results(zs, args.output_dir)
    logger.info("Traditional anomaly detection completed.")

if __name__ == "__main__":
    main()
