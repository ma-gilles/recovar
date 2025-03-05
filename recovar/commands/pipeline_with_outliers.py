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
from recovar import output

# Import necessary functions from pipeline.py and output module
from recovar.commands.pipeline import add_args, standard_recovar_pipeline
from recovar import output
from recovar.commands.outlier_detection import plot_anomaly_detection_results
matplotlib.rcParams["contour.negative_linestyle"] = "solid"

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
            if args.tilt_series:
                args.tilt_ind = indices_filename
            else:
                args.ind = indices_filename
            logger.info(f"Using inliers from round {k} as input indices for round {round_number}")
        # else:
        #     if args.ind 
        #     args.ind = None  # Use all particles in the first round

        # Run the pipeline
        standard_recovar_pipeline(args)

        # Add plot
        po = output.PipelineOutput(args.outdir + '/')
        output.standard_pipeline_plots(po, zdim, args.outdir + '/output/plots/')
        del po

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


def main():
    run_pipeline_with_outlier_removal()

if __name__ == "__main__":
    main()

