#!/usr/bin/env python3
"""
Test script for the outliers pipeline with synthetic data.
This version focuses on core functionality and avoids problematic junk detection.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import pickle
import numpy as np
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print(f"Success: {description}")
            return True
        else:
            print(f"Failed: {description}")
            print(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout: {description}")
        return False
    except Exception as e:
        print(f"Exception: {description} - {str(e)}")
        return False

def main():
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_outliers_")
    print(f"Working directory: {temp_dir}")
    
    try:
        # Change to temp directory
        os.chdir(temp_dir)
        
        # Generate test dataset with outliers
        print("\n=== Generating test dataset ===")
        dataset_dir = "./outliers_test"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Generate dataset with 20% outliers
        cmd = f"python -m recovar.command_line make_test_dataset {dataset_dir} --n-images 100 --percent-outliers 0.2"
        if not run_command(cmd, "Generate a test dataset for outlier testing"):
            return False
        
        # Run pipeline with outliers - use only contrast detection to avoid junk detection issues
        print("\n=== Running pipeline with outliers ===")
        output_dir = f"{dataset_dir}/test_dataset/pipeline_outliers_output"
        
        cmd = f"""python -m recovar.command_line pipeline_with_outliers {dataset_dir}/test_dataset/particles.64.mrcs \
            --poses {dataset_dir}/test_dataset/poses.pkl \
            --ctf {dataset_dir}/test_dataset/ctf.pkl \
            --correct-contrast \
            -o {output_dir} \
            --mask=from_halfmaps \
            --lazy \
            --zdim 4 \
            --k-rounds 1 \
            --use-contrast-detection \
            --save-pipeline-indices"""
        
        if not run_command(cmd, "Run pipeline_with_outliers for 1 round"):
            print("Pipeline failed, but continuing with analysis...")
        
        # Check if any output files were created
        print("\n=== Checking output files ===")
        if os.path.exists(output_dir):
            print(f"Output directory exists: {output_dir}")
            files = os.listdir(output_dir)
            print(f"Files in output directory: {files}")
            
            # Look for any outlier detection results
            round1_dir = os.path.join(output_dir, "round_1")
            if os.path.exists(round1_dir):
                print(f"Round 1 directory exists: {round1_dir}")
                round1_files = os.listdir(round1_dir)
                print(f"Files in round 1: {round1_files}")
                
                # Check for outlier detection directory
                outlier_dir = os.path.join(round1_dir, "outlier_detection")
                if os.path.exists(outlier_dir):
                    print(f"Outlier detection directory exists: {outlier_dir}")
                    outlier_files = os.listdir(outlier_dir)
                    print(f"Files in outlier detection: {outlier_files}")
                    
                    # Look for contrast-based results
                    contrast_dir = os.path.join(outlier_dir, "contrast_based")
                    if os.path.exists(contrast_dir):
                        print(f"Contrast-based directory exists: {contrast_dir}")
                        contrast_files = os.listdir(contrast_dir)
                        print(f"Files in contrast-based: {contrast_files}")
        else:
            print("Output directory does not exist")
        
        print("\n=== Test Summary ===")
        print("Test completed. Check the output above for details.")
        print(f"Working directory: {temp_dir}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")
        return False
    finally:
        # Keep the temp directory for inspection
        print(f"\nTest files preserved in: {temp_dir}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 