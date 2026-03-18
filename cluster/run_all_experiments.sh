#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores (for data loading)
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # Run on 1 GPU card
#$ -N lazimpa_all                # Job name

# Load required modules (adjust based on your cluster's available modules)
module load python/3.9           # Or whatever Python version is available
module load cuda/11.8            # Or appropriate CUDA version
module load pytorch/2.0          # Or load PyTorch module if available

# If PyTorch isn't a module, activate your conda environment:
# source activate lazimpa_env

# Set environment variables
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA EXPERIMENTS - ALL 40 RUNS"
echo "========================================================================"
echo "Job ID: $JOB_ID"
echo "Started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

# Change to project directory (ADJUST THIS PATH!)
cd $HOME/Lazimpa  # Change to wherever you cloned the repository

# Run all experiments using the experiment runner
# This will run all 4 experiment types × 10 seeds = 40 total runs
# Estimated time: 40-80 hours
python run_experiments.py \
  --config experiments_config.json \
  --gpu 0

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished at $(date)"
echo "========================================================================"
