#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores (for data loading)
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card
#$ -N lazimpa_transformer        # Job name
#$ -l h_rt=72:00:00              # Max runtime 72 hours

# ============================================================
# OPTION 1: Using Conda Environment (RECOMMENDED)
# ============================================================
module load conda
source activate lazimpa

# ============================================================
# OPTION 2: Using System Modules (uncomment if no conda)
# ============================================================
# module load python/3.10
# module load cuda/11.8
# module load pytorch/2.1

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
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================================================"
echo ""

# Change to project directory (ADJUST THIS PATH!)
cd $HOME/Lazimpa

# Run ONLY transformer experiments (baseline + lazimpa) × 10 seeds = 20 runs
python run_experiments.py \
  --config experiments_config.json \
  --experiments transformer_baseline transformer_lazimpa \
  --gpu 0

echo ""
echo "========================================================================"
echo "TRANSFORMER EXPERIMENTS COMPLETE"
echo "Finished at $(date)"
echo "========================================================================"
