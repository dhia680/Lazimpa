#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-40                       # Array job: 40 tasks (4 experiments × 10 seeds)
#$ -N lazimpa_full               # Job name
#$ -l h_rt=8:00:00               # Max 8 hours per job
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
echo "LAZIMPA FULL EXPERIMENT ARRAY - Task $SGE_TASK_ID of $SGE_TASK_LAST"
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
else
    # LSTM architecture (original paper settings)
    ARCH_PARAMS="--sender_cell=lstm \
      --receiver_cell=lstm \
      --sender_hidden=250 \
      --receiver_hidden=600 \
      --sender_embedding=50 \
      --receiver_embedding=50 \
      --sender_num_layers=1 \
      --receiver_num_layers=1"
fi

# Set impatient/reg/length_cost based on experiment type
if [[ "$EXPERIMENT" == *"lazimpa"* ]]; then
    LAZIMPA_PARAMS="--impatient=True --reg=True --length_cost=0.01"
else
    LAZIMPA_PARAMS="--length_cost=0.0"
    # impatient and reg default to False
fi

# Run the experiment (FULL PAPER SETTINGS)
python -m egg.zoo.channel.train \
  --dir_save=results/${EXPERIMENT}/seed_${SEED} \
  --n_features=1000 \
  --vocab_size=40 \
  --max_len=30 \
  --batch_size=32 \
  --batches_per_epoch=1000 \
  --n_epochs=500 \
  --lr=0.001 \
  --probs=powerlaw \
  --early_stopping_thr=0.9999 \
  --force_eos=0 \
  --sender_entropy_coeff=0.1 \
  --receiver_entropy_coeff=0.1 \
  $ARCH_PARAMS \
  $LAZIMPA_PARAMS \
  --random_seed=${SEED} \
  --name=${EXPERIMENT}_seed${SEED}

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID ($EXPERIMENT seed $SEED) finished at $(date)"
echo "========================================================================"
