#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-2                        # Array job: 2 tasks (2 LSTM experiments × 1 seed)
#$ -N lstm_fixed                 # Job name
#$ -l h_rt=24:00:00              # Max 24 hours per job
#$ -o logs/lstm_$TASK_ID.out
#$ -e logs/lstm_$TASK_ID.err

# ============================================================
# Load Conda Environment: lazimpa
# ============================================================
module load conda
conda activate lazimpa

# Set environment
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA LSTM FIXED - Task $SGE_TASK_ID of $SGE_TASK_LAST"
echo "========================================================================"
echo "Job ID: $JOB_ID.$SGE_TASK_ID"
echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Create logs directory if needed
mkdir -p logs

# Configuration - ONLY LSTM EXPERIMENTS (1 seed only)
declare -a EXPERIMENTS=("lstm_baseline" "lstm_lazimpa")
declare -a SEEDS=(42)

# Calculate which experiment and seed to run based on task ID
TASK_INDEX=$((SGE_TASK_ID - 1))  # SGE tasks are 1-indexed
NUM_SEEDS=${#SEEDS[@]}
EXP_INDEX=$((TASK_INDEX / NUM_SEEDS))
SEED_INDEX=$((TASK_INDEX % NUM_SEEDS))

EXPERIMENT=${EXPERIMENTS[$EXP_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running: $EXPERIMENT with seed $SEED"
echo ""

# Create output directory
mkdir -p results/${EXPERIMENT}/seed_${SEED}/{logs,sender,receiver,messages,accuracy}

# Set impatient/reg based on experiment type
if [[ "$EXPERIMENT" == *"lazimpa"* ]]; then
    LAZIMPA_PARAMS="--impatient=True --reg=True"
else
    LAZIMPA_PARAMS=""
fi

# Run the experiment - MATCHES ORIGINAL NOTEBOOK EXACTLY
# Key parameters from original:
#   --sender_entropy_coeff=2.0 (CRITICAL - was 0.1 in broken script)
#   --batch_size=512 (was 32)
#   --sender_embedding=10 (was 50)
#   --receiver_embedding=100 (was 50)
#   --receiver_entropy_coeff uses default 0.1 (not specified)
python -m egg.zoo.channel.train \
  --dir_save=results/${EXPERIMENT}/seed_${SEED} \
  --vocab_size=40 \
  --max_len=30 \
  --n_features=1000 \
  --probs=powerlaw \
  --n_epochs=500 \
  --batch_size=512 \
  --length_cost=0.0 \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --sender_hidden=250 \
  --receiver_hidden=600 \
  --receiver_embedding=100 \
  --sender_embedding=10 \
  --batches_per_epoch=1000 \
  --lr=0.001 \
  --sender_entropy_coeff=2.0 \
  --sender_num_layers=1 \
  --receiver_num_layers=1 \
  --early_stopping_thr=0.99 \
  $LAZIMPA_PARAMS \
  --random_seed=${SEED} \
  --name=${EXPERIMENT}_seed${SEED}

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID ($EXPERIMENT seed $SEED) finished at $(date)"
echo "========================================================================"
