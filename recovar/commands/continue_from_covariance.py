#!/usr/bin/env python3
"""
Continue RECOVAR pipeline from concatenated multi-node covariance results.

This script loads concatenated covariance results and continues the pipeline
with PCA, embedding, and volume generation.

Usage:
    python -m recovar.commands.continue_from_covariance \
        --covariance-dir output/covariance_parts \
        --output-dir output \
        [--original-args-file output/pipeline_args.pkl]

Note: This is a framework implementation. Full integration requires refactoring
principal_components.py to cleanly separate covariance computation from PCA.
"""

import argparse
import logging
import pickle
import os
import sys
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def load_concatenated_covariance(covariance_dir):
    """
    Load concatenated covariance results.
    
    Returns:
        Dictionary with covariance data
    """
    covariance_file = os.path.join(covariance_dir, "covariance_complete.pkl")
    
    if not os.path.exists(covariance_file):
        raise FileNotFoundError(
            f"Concatenated covariance not found: {covariance_file}\n"
            f"Run concatenate_covariance first:\n"
            f"  python -m recovar.commands.concatenate_covariance {covariance_dir}"
        )
    
    logger.info(f"Loading concatenated covariance from: {covariance_file}")
    
    with open(covariance_file, 'rb') as f:
        data = pickle.load(f)
    
    required_keys = ['covariance_cols', 'picked_frequencies', 'column_fscs']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in covariance file")
    
    return data


def load_original_pipeline_args(args_file):
    """Load original pipeline arguments."""
    if not os.path.exists(args_file):
        logger.warning(f"Pipeline args file not found: {args_file}")
        return None
    
    logger.info(f"Loading original pipeline args from: {args_file}")
    
    with open(args_file, 'rb') as f:
        args = pickle.load(f)
    
    return args


def continue_pipeline(covariance_data, output_dir, original_args):
    """
    Continue pipeline from covariance onwards.
    
    This is a FRAMEWORK implementation. The actual continuation logic
    requires refactoring principal_components.py to separate:
      1. Covariance computation (already done in multi-node)
      2. PCA computation (needs to be callable separately)
      3. Downstream processing (embedding, volumes, etc.)
    
    Steps to implement:
    1. Load covariance columns from covariance_data
    2. Perform SVD/PCA on covariance matrix
    3. Compute per-image embeddings
    4. Generate output volumes and plots
    5. Save results
    
    Current status: Framework only - prints what needs to be done.
    """
    logger.info("=" * 70)
    logger.info("CONTINUING PIPELINE FROM MULTI-NODE COVARIANCE")
    logger.info("=" * 70)
    
    covariance_cols = covariance_data['covariance_cols']
    picked_frequencies = covariance_data['picked_frequencies']
    column_fscs = covariance_data['column_fscs']
    
    logger.info(f"Covariance shape: {covariance_cols['est_mask'].shape}")
    logger.info(f"Number of frequencies: {len(picked_frequencies)}")
    logger.info(f"Number of FSCs: {len(column_fscs)}")
    
    # Display original pipeline arguments if available
    if original_args:
        logger.info("\nOriginal pipeline parameters:")
        logger.info(f"  Output directory: {original_args.outdir}")
        logger.info(f"  Particles file: {original_args.particles}")
        logger.info(f"  Image size: {getattr(original_args, 'n_images', 'N/A')}")
        logger.info(f"  Latent dims: {getattr(original_args, 'zdim', 'N/A')}")
    
    # Framework: Document what needs to be implemented
    logger.info("\n" + "=" * 70)
    logger.info("IMPLEMENTATION ROADMAP")
    logger.info("=" * 70)
    
    logger.info("\nStep 1: Perform PCA on covariance matrix")
    logger.info("  - Input: covariance_cols['est_mask'] (shape: %s)" % str(covariance_cols['est_mask'].shape))
    logger.info("  - Method: Randomized SVD or standard SVD")
    logger.info("  - Output: U (principal components), S (singular values)")
    logger.info("  - Status: ⚠️  TODO - Requires refactoring principal_components.get_cov_svds()")
    
    logger.info("\nStep 2: Compute per-image embeddings")
    logger.info("  - Input: Mean volumes, principal components U, S")
    logger.info("  - Method: Project images onto PC basis")
    logger.info("  - Output: Per-image latent coordinates (zs)")
    logger.info("  - Status: ⚠️  TODO - Call embedding.get_per_image_embedding()")
    
    logger.info("\nStep 3: Generate output volumes")
    logger.info("  - Input: Mean, PCs, embeddings")
    logger.info("  - Method: Reconstruct volumes along PCs")
    logger.info("  - Output: Volume series along each PC")
    logger.info("  - Status: ⚠️  TODO - Call output.save_covar_output_volumes()")
    
    logger.info("\nStep 4: Create plots and visualizations")
    logger.info("  - Input: Embeddings, latent space")
    logger.info("  - Method: UMAP, PCA plots, trajectories")
    logger.info("  - Output: PNG/PDF plots")
    logger.info("  - Status: ⚠️  TODO - Call output.standard_pipeline_plots()")
    
    logger.info("\nStep 5: Save final results")
    logger.info("  - Input: All computed results")
    logger.info("  - Method: Pickle dump")
    logger.info("  - Output: params.pkl, embeddings.pkl, etc.")
    logger.info("  - Status: ⚠️  TODO - Organize and save results")
    
    logger.info("\n" + "=" * 70)
    logger.info("REFACTORING REQUIRED")
    logger.info("=" * 70)
    
    logger.info("\nTo complete this implementation, refactor principal_components.py:")
    logger.info("  1. Extract get_cov_svds() to be callable with pre-computed covariance")
    logger.info("  2. Separate covariance computation from PCA computation")
    logger.info("  3. Make downstream processing (embedding, volumes) callable independently")
    logger.info("  4. Update standard_recovar_pipeline() to optionally skip covariance")
    
    logger.info("\nAlternative approach (simpler):")
    logger.info("  - Modify estimate_principal_components() to accept pre-computed covariance")
    logger.info("  - Add a 'covariance_cols' parameter that bypasses computation if provided")
    logger.info("  - This requires minimal changes to existing code")
    
    # Placeholder: Show what a simple continuation might look like
    logger.info("\n" + "=" * 70)
    logger.info("FRAMEWORK STATUS")
    logger.info("=" * 70)
    logger.info("✓ Covariance loaded successfully")
    logger.info("✓ Structure validated")
    logger.info("⚠️  PCA computation: NOT IMPLEMENTED")
    logger.info("⚠️  Embedding computation: NOT IMPLEMENTED")
    logger.info("⚠️  Volume generation: NOT IMPLEMENTED")
    logger.info("")
    logger.info("This framework provides the structure for continuation.")
    logger.info("Actual implementation requires refactoring principal_components.py")
    logger.info("to cleanly separate covariance from downstream processing.")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Continue pipeline from multi-node covariance"
    )
    
    parser.add_argument(
        "--covariance-dir",
        type=str,
        required=True,
        help="Directory containing concatenated covariance (covariance_complete.pkl)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for continued pipeline results"
    )
    
    parser.add_argument(
        "--original-args-file",
        type=str,
        default=None,
        help="Path to saved original pipeline arguments (optional)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Load concatenated covariance
        covariance_data = load_concatenated_covariance(args.covariance_dir)
        
        # Load original parameters if provided
        original_args = None
        if args.original_args_file:
            original_args = load_original_pipeline_args(args.original_args_file)
        
        # Continue pipeline (framework only)
        continue_pipeline(covariance_data, args.output_dir, original_args)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
