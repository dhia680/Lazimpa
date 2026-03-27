#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-40                       # Array job: 40 tasks (4 experiments × 10 seeds)
#$ -N lazimpa_array              # Job name
#$ -o logs/job_$TASK_ID.out      # Output file
#$ -e logs/job_$TASK_ID.err      # Error file

# Load required modules
module load python/3.9
module load cuda/11.8
module load pytorch/2.0

# Set environment
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA ARRAY JOB - Task $SGE_TASK_ID of $SGE_TASK_LAST"
echo "========================================================================"
echo "Job ID: $JOB_ID.$SGE_TASK_ID"
echo "Started on $(hostname) at $(date)"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Create logs directory if needed
mkdir -p logs

# Configuration
declare -a EXPERIMENTS=("lstm_baseline" "lstm_lazimpa" "transformer_baseline" "transformer_lazimpa")
declare -a SEEDS=(42 123 456 789 1011 1337 2048 3141 4242 5555)

# Calculate which experiment and seed to run based on task ID
TASK_INDEX=$((SGE_TASK_ID - 1))  # SGE tasks are 1-indexed
NUM_SEEDS=${#SEEDS[@]}
EXP_INDEX=$((TASK_INDEX / NUM_SEEDS))
SEED_INDEX=$((TASK_INDEX % NUM_SEEDS))

EXPERIMENT=${EXPERIMENTS[$EXP_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running: $EXPERIMENT with seed $SEED"
echo ""

# Run the specific experiment
python run_experiments.py \
  --config experiments_config.json \
  --experiments $EXPERIMENT \
  --gpu 0 2>&1 | grep -A 1 "seed=$SEED"

# Note: The grep filters to only show output for this specific seed
# If you want full output, remove the grep filter

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID finished at $(date)"
echo "========================================================================"
