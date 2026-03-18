#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card
#$ -N lazimpa_single             # Job name

# Usage: qsub -v EXPERIMENT=lstm_lazimpa,SEED=42 run_single_experiment.sh

# Load required modules
module load python/3.9
module load cuda/11.8
module load pytorch/2.0

# Set environment
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Default values if not provided
EXPERIMENT=${EXPERIMENT:-"transformer_lazimpa"}
SEED=${SEED:-42}

# Print job info
echo "========================================================================"
echo "LAZIMPA SINGLE EXPERIMENT"
echo "========================================================================"
echo "Job ID: $JOB_ID"
echo "Experiment: $EXPERIMENT"
echo "Seed: $SEED"
echo "Started on $(hostname) at $(date)"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Run the experiment
python run_experiments.py \
  --config experiments_config.json \
  --experiments $EXPERIMENT \
  --gpu 0

echo ""
echo "========================================================================"
echo "Experiment finished at $(date)"
echo "========================================================================"
