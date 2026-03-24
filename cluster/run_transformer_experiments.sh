#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores (for data loading)
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card
#$ -N lazimpa_transformer        # Job name
#$ -l h_rt=48:00:00              # Max runtime 48 hours

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
echo "LAZIMPA - TRANSFORMER EXPERIMENTS ONLY"
echo "========================================================================"
echo "Job ID: $JOB_ID"
echo "Started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

# Change to project directory (ADJUST THIS PATH!)
cd $HOME/Lazimpa

# Run ONLY transformer experiments (baseline + lazimpa) × 10 seeds = 20 runs
# This runs both transformer_baseline and transformer_lazimpa
python run_experiments.py \
  --config experiments_config.json \
  --experiments transformer_baseline transformer_lazimpa \
  --gpu 0

echo ""
echo "========================================================================"
echo "TRANSFORMER EXPERIMENTS COMPLETE"
echo "Finished at $(date)"
echo "========================================================================"
