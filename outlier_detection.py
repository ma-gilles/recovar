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
            x_min, x_max = np.percentile(zs_valid[inliers, 0], [1, 99])
            y_min, y_max = np.percentile(zs_valid[inliers, 1], [1, 99])

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
                x_min, x_max = np.percentile(zs_valid[inliers, 2], [1, 99])
                y_min, y_max = np.percentile(zs_valid[inliers, 3], [1, 99])

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


# def plot_anomaly_detection_results(zs, folder_name):
#     """
#     Plots anomaly detection results for given data and saves the plots and inlier/outlier indices.

#     Parameters:
#     - zs: numpy array
#         The original dataset (may contain NaN values).
#     - folder_name: str
#         The folder name where all files (plots and indices) will be saved.
#     """
#     # Ensure the folder exists
#     os.makedirs(folder_name, exist_ok=True)

#     # Identify valid entries (rows without NaNs)
#     valid_mask = np.all(np.isfinite(zs), axis=1)
#     valid_indices = np.where(valid_mask)[0]
#     zs_valid = zs[valid_mask]

#     if zs_valid.shape[0] == 0:
#         print("No valid entries in zs after removing NaN values.")
#         return

#     # Compute UMAP on valid zs
#     umapper = output.umap_latent_space(zs_valid)
#     umap_valid = umapper.embedding_

#     # Define anomaly detection algorithms
#     anomaly_algorithms = [
#         ("Robust covariance", EllipticEnvelope(random_state=42)),
#         ("Isolation Forest", IsolationForest(random_state=42)),
#         ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35)),
#     ]

#     # Initialize lists to store predictions and algorithm names
#     predictions = []
#     algorithm_names = []

#     # Fit each algorithm and store predictions
#     total_samples = zs_valid.shape[0]
#     for name, algorithm in anomaly_algorithms:
#         if name == "Local Outlier Factor":
#             y_pred = algorithm.fit_predict(zs_valid)
#         else:
#             y_pred = algorithm.fit(zs_valid).predict(zs_valid)
#         predictions.append(y_pred)
#         algorithm_names.append(name)

#         # Save indices of inliers and outliers in original indexing
#         inliers_indices_valid = np.where(y_pred == 1)[0]
#         outliers_indices_valid = np.where(y_pred == -1)[0]
#         inliers_indices = valid_indices[inliers_indices_valid]
#         outliers_indices = valid_indices[outliers_indices_valid]

#         # Sanitize algorithm name for filename
#         safe_name = name.replace(" ", "_").lower()

#         # Save indices to pickle files in the specified folder
#         with open(os.path.join(folder_name, f"inliers_{safe_name}.pkl"), "wb") as f:
#             pickle.dump(inliers_indices, f)
#         with open(os.path.join(folder_name, f"outliers_{safe_name}.pkl"), "wb") as f:
#             pickle.dump(outliers_indices, f)

#     # Compute consensus
#     # Convert predictions to numpy array
#     predictions_array = np.array(predictions)
#     # Convert outlier labels (-1) to 1 for counting
#     outlier_flags = (predictions_array == -1).astype(int)
#     # Sum across algorithms
#     consensus_scores = outlier_flags.sum(axis=0)
#     # Set consensus threshold (majority vote)
#     consensus_threshold = len(anomaly_algorithms) // 2 + 1
#     # Determine consensus outliers and inliers
#     consensus_outliers = consensus_scores >= consensus_threshold
#     consensus_inliers = ~consensus_outliers

#     # Save consensus indices in original indexing
#     consensus_inliers_indices_valid = np.where(consensus_inliers)[0]
#     consensus_outliers_indices_valid = np.where(consensus_outliers)[0]
#     consensus_inliers_indices = valid_indices[consensus_inliers_indices_valid]
#     consensus_outliers_indices = valid_indices[consensus_outliers_indices_valid]
#     with open(os.path.join(folder_name, "inliers_consensus.pkl"), "wb") as f:
#         pickle.dump(consensus_inliers_indices, f)
#     with open(os.path.join(folder_name, "outliers_consensus.pkl"), "wb") as f:
#         pickle.dump(consensus_outliers_indices, f)

#     # Add consensus results to the algorithms for plotting
#     algorithm_names.append("Consensus")
#     predictions.append(np.where(consensus_outliers, -1, 1))

#     # Prepare for plotting
#     n_algorithms = len(algorithm_names)
#     n_rows = 3  # zs[0] vs zs[1], zs[2] vs zs[3], umap[0] vs umap[1]
#     n_cols = n_algorithms
#     fig, axes = plt.subplots(
#         nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False
#     )
#     plt.subplots_adjust(
#         left=0.05, right=0.95, bottom=0.07, top=0.90, wspace=0.3, hspace=0.4
#     )

#     for i_algo, (name, y_pred) in enumerate(zip(algorithm_names, predictions)):
#         # Identify inliers and outliers
#         inliers = y_pred == 1
#         outliers = y_pred == -1

#         # Compute number and percentage of outliers
#         num_outliers = np.sum(outliers)
#         percentage_outliers = (num_outliers / total_samples) * 100

#         # --- Plot zs[:, 0] vs zs[:, 1] ---
#         ax = axes[0, i_algo]
#         if np.sum(inliers) > 1:
#             # Compute axis limits based on inliers
#             x_min, x_max = np.percentile(zs_valid[inliers, 0], [1, 99])
#             y_min, y_max = np.percentile(zs_valid[inliers, 1], [1, 99])

#             hb = ax.hexbin(
#                 zs_valid[inliers, 0],
#                 zs_valid[inliers, 1],
#                 gridsize=50,
#                 cmap='Blues',
#                 bins='log',
#                 mincnt=1
#             )
#             # Add colorbar
#             cb = fig.colorbar(hb, ax=ax)
#             cb.set_label('log$_{10}$(N)')
#         else:
#             ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
#                     ha='center', va='center', fontsize=12)
#             x_min, x_max = zs_valid[:, 0].min(), zs_valid[:, 0].max()
#             y_min, y_max = zs_valid[:, 1].min(), zs_valid[:, 1].max()

#         # Overlay outliers
#         if np.sum(outliers) > 0:
#             ax.scatter(
#                 zs_valid[outliers, 0],
#                 zs_valid[outliers, 1],
#                 s=10,
#                 c="red",
#                 alpha=0.8,
#                 label="Outliers",
#                 edgecolor="k",
#                 linewidth=0.5,
#             )
#         # Set axis limits for zs plots
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)

#         # Add title with number and percentage of outliers
#         ax.set_title(
#             f"{name}\n(zs[0] vs zs[1])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
#             fontsize=12,
#         )
#         if i_algo == 0:
#             ax.set_ylabel("zs[1]", fontsize=10)
#         ax.set_xlabel("zs[0]", fontsize=10)
#         ax.legend(loc="upper right")

#         # --- Plot zs[:, 2] vs zs[:, 3] if available ---
#         if zs_valid.shape[1] >= 4:
#             ax = axes[1, i_algo]
#             if np.sum(inliers) > 1:
#                 # Compute axis limits based on inliers
#                 x_min, x_max = np.percentile(zs_valid[inliers, 2], [1, 99])
#                 y_min, y_max = np.percentile(zs_valid[inliers, 3], [1, 99])

#                 hb = ax.hexbin(
#                     zs_valid[inliers, 2],
#                     zs_valid[inliers, 3],
#                     gridsize=50,
#                     cmap='Blues',
#                     bins='log',
#                     mincnt=1
#                 )
#                 # Add colorbar
#                 cb = fig.colorbar(hb, ax=ax)
#                 cb.set_label('log$_{10}$(N)')
#             else:
#                 ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
#                         ha='center', va='center', fontsize=12)
#                 x_min, x_max = zs_valid[:, 2].min(), zs_valid[:, 2].max()
#                 y_min, y_max = zs_valid[:, 3].min(), zs_valid[:, 3].max()
#             # Overlay outliers
#             if np.sum(outliers) > 0:
#                 ax.scatter(
#                     zs_valid[outliers, 2],
#                     zs_valid[outliers, 3],
#                     s=10,
#                     c="red",
#                     alpha=0.8,
#                     label="Outliers",
#                     edgecolor="k",
#                     linewidth=0.5,
#                 )
#             # Set axis limits for zs plots
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # Add title with number and percentage of outliers
#             ax.set_title(
#                 f"{name}\n(zs[2] vs zs[3])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
#                 fontsize=12,
#             )
#             if i_algo == 0:
#                 ax.set_ylabel("zs[3]", fontsize=10)
#             ax.set_xlabel("zs[2]", fontsize=10)
#             ax.legend(loc="upper right")
#         else:
#             # If zs has less than 4 dimensions, hide the subplot
#             axes[1, i_algo].axis('off')

#         # --- Plot umap[:, 0] vs umap[:, 1] ---
#         ax = axes[2, i_algo]
#         if np.sum(inliers) > 1:
#             hb = ax.hexbin(
#                 umap_valid[inliers, 0],
#                 umap_valid[inliers, 1],
#                 gridsize=50,
#                 cmap='Blues',
#                 bins='log',
#                 mincnt=1
#             )
#             # Add colorbar
#             cb = fig.colorbar(hb, ax=ax)
#             cb.set_label('log$_{10}$(N)')
#         else:
#             ax.text(0.5, 0.5, "Insufficient inliers for hexbin", transform=ax.transAxes,
#                     ha='center', va='center', fontsize=12)
#         # Overlay outliers
#         if np.sum(outliers) > 0:
#             ax.scatter(
#                 umap_valid[outliers, 0],
#                 umap_valid[outliers, 1],
#                 s=10,
#                 c="red",
#                 alpha=0.8,
#                 label="Outliers",
#                 edgecolor="k",
#                 linewidth=0.5,
#             )
#         # Do not set axis limits for umap plots

#         # Add title with number and percentage of outliers
#         ax.set_title(
#             f"{name}\n(umap[0] vs umap[1])\nOutliers: {num_outliers} ({percentage_outliers:.2f}%)",
#             fontsize=12,
#         )
#         if i_algo == 0:
#             ax.set_ylabel("umap[1]", fontsize=10)
#         ax.set_xlabel("umap[0]", fontsize=10)
#         ax.legend(loc="upper right")

#     plt.suptitle("Anomaly Detection Results Including Consensus", fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # Save the plot to the specified folder
#     plot_filename = os.path.join(folder_name, "anomaly_detection_results.png")
#     plt.savefig(plot_filename)
#     plt.close()

    
import logging
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run anomaly detection and plot results.")
    parser.add_argument("input_dir", type=str, help="Directory where the recovar results are stored.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Folder where the plots and indices will be saved.")
    parser.add_argument("--zdim", type=int, required=True, help="Dimension of the zs array to use.")
    parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")

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

    zs = pipeline_output.get('unsorted_embedding')['zs'][zdim_key]

    # Call the function with the parsed arguments
    plot_anomaly_detection_results(zs, args.output_dir)

if __name__ == "__main__":
    main()
