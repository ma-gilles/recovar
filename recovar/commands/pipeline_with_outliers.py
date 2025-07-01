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
from recovar import output

# Import necessary functions from pipeline.py and output module
from recovar.commands.pipeline import add_args, standard_recovar_pipeline
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
    
    # Add comprehensive outlier detection arguments
    parser.add_argument("--use-contrast-detection", action="store_true", 
                       help="Use contrast-based outlier detection")
    parser.add_argument("--use-junk-detection", action="store_true", 
                       help="Use junk particle detection in addition to outlier detection")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip plotting and visualization in outlier detection")
    
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
    parser.add_argument("--junk-threshold", type=float, default=0.5, 
                       help="Threshold for junk particle detection (default: 0.5)")
    parser.add_argument("--particles-per-cluster", type=int, 
                       help="Number of particles per cluster for junk detection (auto: min(100, max(10, n_particles/n_clusters)))")
    
    # Output format arguments
    parser.add_argument("--save-pipeline-indices", action="store_true", 
                       help="Save indices in pipeline-compatible format (--ind for images, --particle-ind for particles in tilt series)")
    parser.add_argument("--output-format", type=str, default="both", 
                       choices=["both", "outliers_only", "inliers_only"], 
                       help="Which indices to save (default: both)")

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
    current_particle_indices = None

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
            # Update args to use the indices file - always use image indices after first round
            args.ind = indices_filename
            args.tilt_ind = None  # Clear particle indices since we're using image indices
            logger.info(f"Using image inliers ({len(current_indices)} images) from round {k} for round {round_number}")
        else:
            # First round - store the original index arguments for future rounds
            if args.tilt_series:
                args.original_ind = args.ind
                args.original_tilt_ind = args.tilt_ind

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

        # Run comprehensive outlier detection
        logger.info("Running comprehensive outlier detection...")
        
        # Create command line arguments for outlier detection
        original_argv = sys.argv
        
        # Build argument list for outlier detection
        outlier_argv = [
            'outlier_detection',  # script name
            args.outdir,  # pipeline_output_dir
            '--zdim-key', str(zdim_key),
            '--output-dir', os.path.join(args.outdir, 'outlier_detection')
        ]
        
        if args.no_z_regularization:
            outlier_argv.append('--no-z-regularization')
        
        if args.save_pipeline_indices:
            outlier_argv.append('--save-pipeline-indices')
        
        outlier_argv.extend(['--output-format', args.output_format])
        
        if args.no_plots:
            outlier_argv.append('--no-plots')
        
        # Contrast-based detection
        if args.use_contrast_detection:
            outlier_argv.extend([
                '--low-contrast-threshold', str(args.low_contrast_threshold),
                '--high-contrast-threshold', str(args.high_contrast_threshold),
                '--max-contrast', str(args.max_contrast),
                '--particle-bad-fraction-threshold', str(args.particle_bad_fraction_threshold),
                '--micrograph-bad-fraction-threshold', str(args.micrograph_bad_fraction_threshold)
            ])
        
        # Junk detection
        if args.use_junk_detection:
            outlier_argv.extend([
                '--use-junk-detection',
                '--junk-threshold', str(args.junk_threshold)
            ])
            # Only add particles-per-cluster if explicitly provided
            if hasattr(args, 'particles_per_cluster') and args.particles_per_cluster is not None:
                outlier_argv.extend(['--particles-per-cluster', str(args.particles_per_cluster)])
        
        # Temporarily replace sys.argv and run outlier detection
        sys.argv = outlier_argv
        from recovar.commands.outlier_detection import main as outlier_main
        outlier_main()
        logger.info("Comprehensive outlier detection completed successfully.")
        # except Exception as e:
        #     logger.error(f"Comprehensive outlier detection failed: {e}")
        #     sys.exit(1)
        # finally:
        #     # Restore original argv
        #     sys.argv = original_argv
        
        # Load the combined inliers indices for the next round
        outlier_output_dir = os.path.join(args.outdir, 'outlier_detection')
        combined_inliers_file = os.path.join(outlier_output_dir, 'combined_results', f'combined_image_inliers_{zdim_key}.pkl')
        if not os.path.exists(combined_inliers_file):
            logger.error(f"Combined inliers file not found: {combined_inliers_file}")
            sys.exit(1)
        with open(combined_inliers_file, 'rb') as f:
            current_indices = pickle.load(f)
        
        # For tilt series, also load particle indices
        current_particle_indices = None
        if args.tilt_series:
            combined_particle_inliers_file = os.path.join(outlier_output_dir, 'combined_results', f'combined_particle_inliers_{zdim_key}.pkl')
            if os.path.exists(combined_particle_inliers_file):
                with open(combined_particle_inliers_file, 'rb') as f:
                    current_particle_indices = pickle.load(f)
                logger.info(f"Loaded particle inliers: {len(current_particle_indices)} particles")
            else:
                logger.warning(f"Particle inliers file not found: {combined_particle_inliers_file}")
        
        # Save the inliers and outliers for this round in the original output directory
        inliers_save_path = os.path.join(original_outdir, f"inliers_round_{round_number}.pkl")
        outliers_save_path = os.path.join(original_outdir, f"outliers_round_{round_number}.pkl")
        
        # Copy combined results
        shutil.copy(combined_inliers_file, inliers_save_path)
        combined_outliers_file = os.path.join(outlier_output_dir, 'combined_results', f'combined_image_outliers_{zdim_key}.pkl')
        if os.path.exists(combined_outliers_file):
            shutil.copy(combined_outliers_file, outliers_save_path)
        
        # For tilt series, also save particle indices
        if args.tilt_series and current_particle_indices is not None:
            particle_inliers_save_path = os.path.join(original_outdir, f"particle_inliers_round_{round_number}.pkl")
            particle_outliers_save_path = os.path.join(original_outdir, f"particle_outliers_round_{round_number}.pkl")
            
            shutil.copy(combined_particle_inliers_file, particle_inliers_save_path)
            combined_particle_outliers_file = os.path.join(outlier_output_dir, 'combined_results', f'combined_particle_outliers_{zdim_key}.pkl')
            if os.path.exists(combined_particle_outliers_file):
                shutil.copy(combined_particle_outliers_file, particle_outliers_save_path)
            
            logger.info(f"Saved particle inliers of round {round_number} to {particle_inliers_save_path}")
            logger.info(f"Saved particle outliers of round {round_number} to {particle_outliers_save_path}")
        
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

