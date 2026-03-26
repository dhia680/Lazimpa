#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-40                       # Array job: 40 tasks (4 experiments × 10 seeds)
#$ -N lazimpa_fixed              # Job name
#$ -l h_rt=24:00:00              # Max 24 hours per job
#$ -o logs/exp_$TASK_ID.out
#$ -e logs/exp_$TASK_ID.err

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
echo "LAZIMPA FIXED EXPERIMENT ARRAY - Task $SGE_TASK_ID of $SGE_TASK_LAST"
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

# Configuration - ALL 4 EXPERIMENTS (LSTM + Transformer)
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

# Create output directory
mkdir -p results/${EXPERIMENT}/seed_${SEED}/{logs,sender,receiver,messages,accuracy}

# Set architecture-specific parameters
if [[ "$EXPERIMENT" == *"transformer"* ]]; then
    # Transformer architecture
    # NOTE: For transformers, we keep entropy at 0.1 as they behave differently
    ARCH_PARAMS="--sender_cell=transformer \
      --receiver_cell=transformer \
      --sender_hidden=512 \
      --receiver_hidden=512 \
      --sender_embedding=256 \
      --receiver_embedding=256 \
      --sender_num_layers=2 \
      --receiver_num_layers=2 \
      --sender_num_heads=8 \
      --receiver_num_heads=8 \
      --causal_sender \
      --causal_receiver \
      --sender_generate_style=in-place"
    # Transformer entropy coefficient (may need tuning)
    SENDER_ENTROPY=0.5
else
    # LSTM architecture - ORIGINAL PAPER SETTINGS
    # CRITICAL: sender_embedding=10, receiver_embedding=100 (as in original)
    ARCH_PARAMS="--sender_cell=lstm \
      --receiver_cell=lstm \
      --sender_hidden=250 \
      --receiver_hidden=600 \
      --sender_embedding=10 \
      --receiver_embedding=100 \
      --sender_num_layers=1 \
      --receiver_num_layers=1"
    # CRITICAL: sender_entropy_coeff=2.0 (as in original notebook!)
    SENDER_ENTROPY=2.0
fi

# Set impatient/reg/length_cost based on experiment type
if [[ "$EXPERIMENT" == *"lazimpa"* ]]; then
    LAZIMPA_PARAMS="--impatient=True --reg=True --length_cost=0.0"
else
    LAZIMPA_PARAMS="--length_cost=0.0"
    # impatient and reg default to False
fi

# Run the experiment
# CRITICAL FIXES to match original notebook:
# 1. sender_entropy_coeff=2.0 for LSTM (was 0.1) - THIS IS THE KEY FIX!
# 2. batch_size=512 (was 32)
# 3. sender_embedding=10, receiver_embedding=100 for LSTM (was 50/50)
# 4. receiver_entropy_coeff NOT specified (uses default 0.1)
# 5. early_stopping_thr=0.99 (original notebook value)
python -m egg.zoo.channel.train \
  --dir_save=results/${EXPERIMENT}/seed_${SEED} \
  --n_features=1000 \
  --vocab_size=40 \
  --max_len=30 \
  --batch_size=512 \
  --batches_per_epoch=1000 \
  --n_epochs=500 \
  --lr=0.001 \
  --probs=powerlaw \
  --early_stopping_thr=0.99 \
  --sender_entropy_coeff=${SENDER_ENTROPY} \
  $ARCH_PARAMS \
  $LAZIMPA_PARAMS \
  --random_seed=${SEED} \
  --name=${EXPERIMENT}_seed${SEED}

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID ($EXPERIMENT seed $SEED) finished at $(date)"
echo "========================================================================"
