#!/usr/bin/env python

import os
import sys
import argparse
import pickle
import shutil
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Import necessary functions from pipeline.py and output module
from pipeline import add_args, standard_recovar_pipeline
from recovar import output
from outlier_detection import plot_anomaly_detection_results
matplotlib.rcParams["contour.negative_linestyle"] = "solid"

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

def run_pipeline_with_outlier_removal():
    """
    Runs the pipeline and outlier detection iteratively for K rounds.
    Accepts command line arguments and keeps all the rounds of inliers.
    Provides an option to delete the results of all rounds except the inliers/outliers.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pipeline with Outlier Removal")
    add_args(parser)  # Add pipeline.py arguments

    # Add additional argument for number of rounds K
    parser.add_argument("--k-rounds", type=int, default=1, help="Number of rounds to run")
    # Add additional argument for no-z-regularization
    parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
    # Add option to delete round results
    parser.add_argument("--cleanup", action="store_true", help="Delete results of all rounds except the inliers/outliers")

    args = parser.parse_args()

    # Ensure the output directory exists
    output.mkdir_safe(args.outdir)

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.outdir, 'run.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Original output directory
    original_outdir = args.outdir
    os.makedirs(original_outdir, exist_ok=True)  # Ensure the original output directory exists

    # Handle zdim
    if args.zdim:
        zdim_list = args.zdim
    else:
        logger.error("Please specify --zdim")
        sys.exit(1)
    zdim = zdim_list[-1]  # Use the last zdim for outlier detection

    # Initialize indices (start with None to include all particles)
    current_indices = None

    # Keep track of inliers from all rounds
    all_rounds_inliers = {}

    # Keep track of directories to delete if cleanup is enabled
    round_dirs = []

    for k in range(args.k_rounds):
        round_number = k + 1
        logger.info(f"Starting round {round_number}/{args.k_rounds}")

        # Update output directory to avoid overwriting
        args.outdir = os.path.join(original_outdir, f"round_{round_number}")
        os.makedirs(args.outdir, exist_ok=True)
        round_dirs.append(args.outdir)  # Keep track of the round directory

        # If not the first round, update the indices file
        if current_indices is not None:
            # Save current_indices to a pickle file
            indices_filename = os.path.join(args.outdir, f"inliers_round_{k}.pkl")
            with open(indices_filename, "wb") as f:
                pickle.dump(current_indices, f)
            # Update args to use the indices file
            args.ind = indices_filename
            logger.info(f"Using inliers from round {k} as input indices for round {round_number}")
        else:
            args.ind = None  # Use all particles in the first round

        # Run the pipeline
        standard_recovar_pipeline(args)

        # Load the embeddings from the pipeline output
        pipeline_output_dir = os.path.join(args.outdir, 'model')
        embeddings_file = os.path.join(pipeline_output_dir, 'embeddings.pkl')
        if not os.path.exists(embeddings_file):
            logger.error(f"Embeddings file not found: {embeddings_file}")
            sys.exit(1)
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)

        # Select the zs to use for outlier detection
        if args.no_z_regularization:
            zdim_key = f"{zdim}_noreg"
        else:
            zdim_key = zdim
        if zdim_key not in embeddings['zs']:
            logger.error(f"zdim {zdim_key} not found in embeddings")
            sys.exit(1)
        zs = embeddings['zs'][zdim_key]

        # Run outlier detection
        outlier_output_dir = os.path.join(args.outdir, 'outlier_detection')
        os.makedirs(outlier_output_dir, exist_ok=True)
        plot_anomaly_detection_results(zs, outlier_output_dir)

        # Load the consensus inliers indices for the next round
        consensus_inliers_file = os.path.join(outlier_output_dir, 'inliers_consensus.pkl')
        if not os.path.exists(consensus_inliers_file):
            logger.error(f"Consensus inliers file not found: {consensus_inliers_file}")
            sys.exit(1)
        with open(consensus_inliers_file, 'rb') as f:
            current_indices = pickle.load(f)

        # Save the inliers and outliers for this round in the original output directory
        inliers_save_path = os.path.join(original_outdir, f"inliers_round_{round_number}.pkl")
        outliers_save_path = os.path.join(original_outdir, f"outliers_round_{round_number}.pkl")
        shutil.copy(consensus_inliers_file, inliers_save_path)
        shutil.copy(os.path.join(outlier_output_dir, 'outliers_consensus.pkl'), outliers_save_path)
        logger.info(f"Saved inliers of round {round_number} to {inliers_save_path}")
        logger.info(f"Saved outliers of round {round_number} to {outliers_save_path}")

        # Keep track of inliers for all rounds
        all_rounds_inliers[round_number] = current_indices

        # Check if there are enough inliers to continue
        if len(current_indices) == 0:
            logger.warning(f"No inliers left after round {round_number}. Stopping iterations.")
            break

        logger.info(f"Round {round_number} completed. Number of inliers: {len(current_indices)}")

    # Save all rounds inliers to a file
    all_inliers_file = os.path.join(original_outdir, "all_rounds_inliers.pkl")
    with open(all_inliers_file, 'wb') as f:
        pickle.dump(all_rounds_inliers, f)
    logger.info(f"Saved inliers from all rounds to {all_inliers_file}")

    # Cleanup: delete the results of all rounds except the inliers/outliers if --cleanup is specified
    if args.cleanup:
        logger.info("Cleanup enabled. Deleting intermediate round results.")
        for dir_path in round_dirs:
            if os.path.exists(dir_path):
                try:
                    # Remove all contents except the inliers/outliers files
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        elif os.path.isfile(item_path):
                            if not item.endswith('.pkl') or not item.startswith('inliers_round_') and not item.startswith('outliers_round_'):
                                os.remove(item_path)
                    # Optionally, remove the round directory itself
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted directory {dir_path}")
                except Exception as e:
                    logger.error(f"Error deleting directory {dir_path}: {e}")
    else:
        logger.info("Cleanup not enabled. Intermediate round results are kept.")

    logger.info("Pipeline with outlier removal completed.")

if __name__ == "__main__":
    run_pipeline_with_outlier_removal()


# # import os
# # import sys
# # import argparse
# # import pickle
# # import shutil
# # import logging

# # # Import necessary functions from pipeline.py and outlier_detection script
# # from pipeline import add_args, standard_recovar_pipeline
# # from outlier_detection import plot_anomaly_detection_results
# # from recovar import output

# # def run_pipeline_with_outlier_removal():
# #     """
# #     Runs the pipeline and outlier detection iteratively for K rounds.
# #     Accepts command line arguments and keeps all the rounds of inliers.
# #     Provides an option to delete the results of all rounds except the inliers/outliers.
# #     """
# #     # Set up argument parser
# #     parser = argparse.ArgumentParser(description="Pipeline with Outlier Removal")
# #     add_args(parser)  # Add pipeline.py arguments

# #     # Add additional argument for number of rounds K
# #     parser.add_argument("--k-rounds", type=int, default=1, help="Number of rounds to run")
# #     # Add additional argument for no-z-regularization
# #     parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
# #     # Ensure output directory is specified
# #     # parser.add_argument("-o", "--outdir", type=os.path.abspath, required=True, help="Output directory to save results")
# #     # Add option to delete round results
# #     parser.add_argument("--cleanup", action="store_true", help="Delete results of all rounds except the inliers/outliers")

# #     args = parser.parse_args()

# #     # Ensure the output directory exists
# #     output.mkdir_safe(args.outdir)

# #     # Set up logging
# #     logging.basicConfig(
# #         format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
# #         level=logging.INFO,
# #         handlers=[
# #             logging.FileHandler(os.path.join(args.outdir, 'run.log')),
# #             logging.StreamHandler()
# #         ]
# #     )
# #     logger = logging.getLogger(__name__)

# #     # Original output directory
# #     original_outdir = args.outdir
# #     os.makedirs(original_outdir, exist_ok=True)  # Ensure the original output directory exists

# #     # Handle zdim
# #     if args.zdim:
# #         zdim_list = args.zdim
# #     else:
# #         logger.error("Please specify --zdim")
# #         sys.exit(1)
# #     zdim = zdim_list[0]  # Use the first zdim for outlier detection

# #     # Initialize indices (start with None to include all particles)
# #     current_indices = None

# #     # Keep track of inliers from all rounds
# #     all_rounds_inliers = {}

# #     # Keep track of directories to delete if cleanup is enabled
# #     round_dirs = []

# #     for k in range(args.k_rounds):
# #         round_number = k + 1
# #         logger.info(f"Starting round {round_number}/{args.k_rounds}")

# #         # Update output directory to avoid overwriting
# #         args.outdir = os.path.join(original_outdir, f"round_{round_number}")
# #         os.makedirs(args.outdir, exist_ok=True)
# #         round_dirs.append(args.outdir)  # Keep track of the round directory

# #         # If not the first round, update the indices file
# #         if current_indices is not None:
# #             # Save current_indices to a pickle file
# #             indices_filename = os.path.join(args.outdir, f"inliers_round_{k}.pkl")
# #             with open(indices_filename, "wb") as f:
# #                 pickle.dump(current_indices, f)
# #             # Update args to use the indices file
# #             args.ind = indices_filename
# #             logger.info(f"Using inliers from round {k} as input indices for round {round_number}")
# #         else:
# #             args.ind = None  # Use all particles in the first round

# #         # Run the pipeline
# #         standard_recovar_pipeline(args)


# #         # Load the embeddings from the pipeline output
# #         pipeline_output_dir = os.path.join(args.outdir, 'model')
# #         embeddings_file = os.path.join(pipeline_output_dir, 'embeddings.pkl')
# #         if not os.path.exists(embeddings_file):
# #             logger.error(f"Embeddings file not found: {embeddings_file}")
# #             sys.exit(1)
# #         with open(embeddings_file, 'rb') as f:
# #             embeddings = pickle.load(f)

# #         # Select the zs to use for outlier detection
# #         if args.no_z_regularization:
# #             zdim_key = f"{zdim}_noreg"
# #         else:
# #             zdim_key = zdim
# #         if zdim_key not in embeddings['zs']:
# #             logger.error(f"zdim {zdim_key} not found in embeddings")
# #             sys.exit(1)
# #         zs = embeddings['zs'][zdim_key]

# #         # Run UMAP on zs
# #         umapper = output.umap_latent_space(zs)
# #         umap = umapper.embedding_

# #         # Run outlier detection
# #         outlier_output_dir = os.path.join(args.outdir, 'outlier_detection')
# #         os.makedirs(outlier_output_dir, exist_ok=True)
# #         plot_anomaly_detection_results(zs, umap, outlier_output_dir)

# #         # Load the consensus inliers indices for the next round
# #         consensus_inliers_file = os.path.join(outlier_output_dir, 'inliers_consensus.pkl')
# #         if not os.path.exists(consensus_inliers_file):
# #             logger.error(f"Consensus inliers file not found: {consensus_inliers_file}")
# #             sys.exit(1)
# #         with open(consensus_inliers_file, 'rb') as f:
# #             current_indices = pickle.load(f)

# #         # Save the inliers and outliers for this round in the original output directory
# #         inliers_save_path = os.path.join(original_outdir, f"inliers_round_{round_number}.pkl")
# #         outliers_save_path = os.path.join(original_outdir, f"outliers_round_{round_number}.pkl")
# #         shutil.copy(consensus_inliers_file, inliers_save_path)
# #         shutil.copy(os.path.join(outlier_output_dir, 'outliers_consensus.pkl'), outliers_save_path)
# #         logger.info(f"Saved inliers of round {round_number} to {inliers_save_path}")
# #         logger.info(f"Saved outliers of round {round_number} to {outliers_save_path}")

# #         # Keep track of inliers for all rounds
# #         all_rounds_inliers[round_number] = current_indices

# #         # Check if there are enough inliers to continue
# #         if len(current_indices) == 0:
# #             logger.warning(f"No inliers left after round {round_number}. Stopping iterations.")
# #             break

# #         logger.info(f"Round {round_number} completed. Number of inliers: {len(current_indices)}")

# #     # Save all rounds inliers to a file
# #     all_inliers_file = os.path.join(original_outdir, "all_rounds_inliers.pkl")
# #     with open(all_inliers_file, 'wb') as f:
# #         pickle.dump(all_rounds_inliers, f)
# #     logger.info(f"Saved inliers from all rounds to {all_inliers_file}")

# #     # Cleanup: delete the results of all rounds except the inliers/outliers if --cleanup is specified
# #     if args.cleanup:
# #         logger.info("Cleanup enabled. Deleting intermediate round results.")
# #         for dir_path in round_dirs:
# #             if os.path.exists(dir_path):
# #                 try:
# #                     # Remove all contents except the inliers/outliers files
# #                     for item in os.listdir(dir_path):
# #                         item_path = os.path.join(dir_path, item)
# #                         if os.path.isdir(item_path):
# #                             shutil.rmtree(item_path)
# #                         elif os.path.isfile(item_path):
# #                             if not item.endswith('.pkl') or not item.startswith('inliers_round_') and not item.startswith('outliers_round_'):
# #                                 os.remove(item_path)
# #                     # Optionally, remove the round directory itself
# #                     shutil.rmtree(dir_path)
# #                     logger.info(f"Deleted directory {dir_path}")
# #                 except Exception as e:
# #                     logger.error(f"Error deleting directory {dir_path}: {e}")
# #     else:
# #         logger.info("Cleanup not enabled. Intermediate round results are kept.")

# #     logger.info("Pipeline with outlier removal completed.")

# # if __name__ == "__main__":
# #     run_pipeline_with_outlier_removal()

# #!/usr/bin/env python

# import os
# import sys
# import argparse
# import pickle
# import shutil
# import logging
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.covariance import EllipticEnvelope
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor

# # Import necessary functions from pipeline.py and output module
# from pipeline import add_args, standard_recovar_pipeline
# from recovar import output

# matplotlib.rcParams["contour.negative_linestyle"] = "solid"

# def plot_anomaly_detection_results(zs, umap_valid, valid_indices, folder_name):
#     """
#     Plots anomaly detection results for given data and saves the plots and inlier/outlier indices.

#     Parameters:
#     - zs: numpy array
#         The original dataset (may contain NaN values).
#     - umap_valid: numpy array
#         The dataset transformed using UMAP, only for valid entries.
#     - valid_indices: numpy array
#         Indices of valid entries (non-NaN rows in zs).
#     - folder_name: str
#         The folder name where all files (plots and indices) will be saved.
#     """
#     # Ensure the folder exists
#     os.makedirs(folder_name, exist_ok=True)

#     # Extract valid zs
#     zs_valid = zs[valid_indices]

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

# def run_pipeline_with_outlier_removal():
#     """
#     Runs the pipeline and outlier detection iteratively for K rounds.
#     Accepts command line arguments and keeps all the rounds of inliers.
#     Provides an option to delete the results of all rounds except the inliers/outliers.
#     """
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Pipeline with Outlier Removal")
#     add_args(parser)  # Add pipeline.py arguments

#     # Add additional argument for number of rounds K
#     parser.add_argument("--k-rounds", type=int, default=1, help="Number of rounds to run")
#     # Add additional argument for no-z-regularization
#     parser.add_argument("--no-z-regularization", action="store_true", help="Disable z regularization.")
#     # Ensure output directory is specified
#     # parser.add_argument("-o", "--outdir", type=os.path.abspath, required=True, help="Output directory to save results")
#     # Add option to delete round results
#     parser.add_argument("--cleanup", action="store_true", help="Delete results of all rounds except the inliers/outliers")

#     args = parser.parse_args()

#     # Ensure the output directory exists
#     output.mkdir_safe(args.outdir)

#     # Set up logging
#     logging.basicConfig(
#         format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
#         level=logging.INFO,
#         handlers=[
#             logging.FileHandler(os.path.join(args.outdir, 'run.log')),
#             logging.StreamHandler()
#         ]
#     )
#     logger = logging.getLogger(__name__)

#     # Original output directory
#     original_outdir = args.outdir
#     os.makedirs(original_outdir, exist_ok=True)  # Ensure the original output directory exists

#     # Handle zdim
#     if args.zdim:
#         zdim_list = args.zdim
#     else:
#         logger.error("Please specify --zdim")
#         sys.exit(1)
#     zdim = zdim_list[0]  # Use the first zdim for outlier detection

#     # Initialize indices (start with None to include all particles)
#     current_indices = None

#     # Keep track of inliers from all rounds
#     all_rounds_inliers = {}

#     # Keep track of directories to delete if cleanup is enabled
#     round_dirs = []

#     for k in range(args.k_rounds):
#         round_number = k + 1
#         logger.info(f"Starting round {round_number}/{args.k_rounds}")

#         # Update output directory to avoid overwriting
#         args.outdir = os.path.join(original_outdir, f"round_{round_number}")
#         os.makedirs(args.outdir, exist_ok=True)
#         round_dirs.append(args.outdir)  # Keep track of the round directory

#         # If not the first round, update the indices file
#         if current_indices is not None:
#             # Save current_indices to a pickle file
#             indices_filename = os.path.join(args.outdir, f"inliers_round_{k}.pkl")
#             with open(indices_filename, "wb") as f:
#                 pickle.dump(current_indices, f)
#             # Update args to use the indices file
#             args.ind = indices_filename
#             logger.info(f"Using inliers from round {k} as input indices for round {round_number}")
#         else:
#             args.ind = None  # Use all particles in the first round

#         # Run the pipeline
#         standard_recovar_pipeline(args)

#         # Load the embeddings from the pipeline output
#         pipeline_output_dir = os.path.join(args.outdir, 'model')
#         embeddings_file = os.path.join(pipeline_output_dir, 'embeddings.pkl')
#         if not os.path.exists(embeddings_file):
#             logger.error(f"Embeddings file not found: {embeddings_file}")
#             sys.exit(1)
#         with open(embeddings_file, 'rb') as f:
#             embeddings = pickle.load(f)

#         # Select the zs to use for outlier detection
#         if args.no_z_regularization:
#             zdim_key = f"{zdim}_noreg"
#         else:
#             zdim_key = zdim
#         if zdim_key not in embeddings['zs']:
#             logger.error(f"zdim {zdim_key} not found in embeddings")
#             sys.exit(1)
#         zs = embeddings['zs'][zdim_key]

#         # Identify valid entries (rows without NaNs)
#         valid_mask = np.all(np.isfinite(zs), axis=1)
#         valid_indices = np.where(valid_mask)[0]
#         zs_valid = zs[valid_mask]

#         if zs_valid.shape[0] == 0:
#             logger.error("No valid entries in zs after removing NaN values.")
#             sys.exit(1)

#         # Compute UMAP on valid zs
#         umapper = output.umap_latent_space(zs_valid)
#         umap_valid = umapper.embedding_

#         # Run outlier detection
#         outlier_output_dir = os.path.join(args.outdir, 'outlier_detection')
#         os.makedirs(outlier_output_dir, exist_ok=True)
#         plot_anomaly_detection_results(zs, umap_valid, valid_indices, outlier_output_dir)

#         # Load the consensus inliers indices for the next round
#         consensus_inliers_file = os.path.join(outlier_output_dir, 'inliers_consensus.pkl')
#         if not os.path.exists(consensus_inliers_file):
#             logger.error(f"Consensus inliers file not found: {consensus_inliers_file}")
#             sys.exit(1)
#         with open(consensus_inliers_file, 'rb') as f:
#             current_indices = pickle.load(f)

#         # Save the inliers and outliers for this round in the original output directory
#         inliers_save_path = os.path.join(original_outdir, f"inliers_round_{round_number}.pkl")
#         outliers_save_path = os.path.join(original_outdir, f"outliers_round_{round_number}.pkl")
#         shutil.copy(consensus_inliers_file, inliers_save_path)
#         shutil.copy(os.path.join(outlier_output_dir, 'outliers_consensus.pkl'), outliers_save_path)
#         logger.info(f"Saved inliers of round {round_number} to {inliers_save_path}")
#         logger.info(f"Saved outliers of round {round_number} to {outliers_save_path}")

#         # Keep track of inliers for all rounds
#         all_rounds_inliers[round_number] = current_indices

#         # Check if there are enough inliers to continue
#         if len(current_indices) == 0:
#             logger.warning(f"No inliers left after round {round_number}. Stopping iterations.")
#             break

#         logger.info(f"Round {round_number} completed. Number of inliers: {len(current_indices)}")

#     # Save all rounds inliers to a file
#     all_inliers_file = os.path.join(original_outdir, "all_rounds_inliers.pkl")
#     with open(all_inliers_file, 'wb') as f:
#         pickle.dump(all_rounds_inliers, f)
#     logger.info(f"Saved inliers from all rounds to {all_inliers_file}")

#     # Cleanup: delete the results of all rounds except the inliers/outliers if --cleanup is specified
#     if args.cleanup:
#         logger.info("Cleanup enabled. Deleting intermediate round results.")
#         for dir_path in round_dirs:
#             if os.path.exists(dir_path):
#                 try:
#                     # Remove all contents except the inliers/outliers files
#                     for item in os.listdir(dir_path):
#                         item_path = os.path.join(dir_path, item)
#                         if os.path.isdir(item_path):
#                             shutil.rmtree(item_path)
#                         elif os.path.isfile(item_path):
#                             if not item.endswith('.pkl') or not item.startswith('inliers_round_') and not item.startswith('outliers_round_'):
#                                 os.remove(item_path)
#                     # Optionally, remove the round directory itself
#                     shutil.rmtree(dir_path)
#                     logger.info(f"Deleted directory {dir_path}")
#                 except Exception as e:
#                     logger.error(f"Error deleting directory {dir_path}: {e}")
#     else:
#         logger.info("Cleanup not enabled. Intermediate round results are kept.")

#     logger.info("Pipeline with outlier removal completed.")

# if __name__ == "__main__":
#     run_pipeline_with_outlier_removal()


