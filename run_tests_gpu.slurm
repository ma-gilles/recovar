#!/bin/bash
#SBATCH --job-name=recovar-tests
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=test_results_%j.out

cd /scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
export PYTHONNOUSERSITE=1

echo "=== Unit tests ==="
/home/mg6942/.conda/envs/recovar_dev_2/bin/python -m pytest tests/unit/ --run-gpu --run-slow -v --tb=short 2>&1

echo ""
echo "=== Integration smoke tests ==="
/home/mg6942/.conda/envs/recovar_dev_2/bin/python -m pytest tests/integration/test_pipeline_smoke.py --run-integration --run-slow -v --tb=short 2>&1

echo ""
echo "=== Done ==="
