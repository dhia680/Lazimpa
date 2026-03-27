#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 8                     # 8 CPU cores (faster data loading)
#$ -q gpu@@crc_a10               # Request A10 GPUs (24GB, fast) - or use @crc_rtx6k
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-2                        # Array job: 2 tasks (2 Transformer experiments × 1 seed)
#$ -N transformer_fixed          # Job name
#$ -l h_rt=24:00:00              # Max 24 hours per job
#$ -o logs/transformer_$TASK_ID.out
#$ -e logs/transformer_$TASK_ID.err

# ============================================================
# Load Conda Environment: lazimpa
# ============================================================
module load conda
conda activate lazimpa

# Set environment
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA TRANSFORMER FIXED - Task $SGE_TASK_ID of $SGE_TASK_LAST"
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

# Configuration - ONLY TRANSFORMER EXPERIMENTS (1 seed only)
declare -a EXPERIMENTS=("transformer_baseline" "transformer_lazimpa")
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

# Run the experiment
# Transformer settings adapted from LSTM:
#   --sender_entropy_coeff=2.0 (same as LSTM to encourage exploration)
#   --batch_size=512
#   --embed_dim must be divisible by num_heads (256 / 8 = 32) ✓
python -m egg.zoo.channel.train \
  --dir_save=results/${EXPERIMENT}/seed_${SEED} \
  --vocab_size=40 \
  --max_len=30 \
  --n_features=1000 \
  --probs=powerlaw \
  --n_epochs=500 \
  --batch_size=512 \
  --length_cost=0.0 \
  --sender_cell=transformer \
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
  --sender_generate_style=in-place \
  --batches_per_epoch=1000 \
  --lr=0.001 \
  --sender_entropy_coeff=2.0 \
  --early_stopping_thr=0.99 \
  $LAZIMPA_PARAMS \
  --random_seed=${SEED} \
  --name=${EXPERIMENT}_seed${SEED}

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID ($EXPERIMENT seed $SEED) finished at $(date)"
echo "========================================================================"
