#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -t 1-20                       # Array job: 20 tasks (2 transformer experiments × 10 seeds)
#$ -N lazimpa_tf_array           # Job name
#$ -l h_rt=6:00:00               # Max 6 hours per job
#$ -o logs/transformer_$TASK_ID.out
#$ -e logs/transformer_$TASK_ID.err

# Load required modules
module load python/3.9
module load cuda/11.8
module load pytorch/2.0

# Set environment
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA TRANSFORMER ARRAY - Task $SGE_TASK_ID of $SGE_TASK_LAST"
echo "========================================================================"
echo "Job ID: $JOB_ID.$SGE_TASK_ID"
echo "Started on $(hostname) at $(date)"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Create logs directory if needed
mkdir -p logs

# Configuration - ONLY TRANSFORMER EXPERIMENTS
declare -a EXPERIMENTS=("transformer_baseline" "transformer_lazimpa")
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
mkdir -p results/${EXPERIMENT}/seed_${SEED}/logs
mkdir -p results/${EXPERIMENT}/seed_${SEED}/sender
mkdir -p results/${EXPERIMENT}/seed_${SEED}/receiver
mkdir -p results/${EXPERIMENT}/seed_${SEED}/messages
mkdir -p results/${EXPERIMENT}/seed_${SEED}/accuracy

# Run the specific experiment with this seed directly
python -m egg.zoo.channel.train \
  --n_features=1000 \
  --vocab_size=40 \
  --max_len=30 \
  --batch_size=32 \
  --batches_per_epoch=1000 \
  --n_epochs=500 \
  --lr=0.001 \
  --sender_entropy_coeff=0.1 \
  --receiver_entropy_coeff=0.1 \
  --probs=powerlaw \
  --early_stopping_thr=0.9999 \
  --force_eos=0 \
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
  $(if [ "$EXPERIMENT" == "transformer_lazimpa" ]; then echo "--impatient=1 --reg=1 --length_cost=0.01"; else echo "--impatient=0 --reg=0 --length_cost=0.0"; fi) \
  --random_seed=${SEED} \
  --dir_save=results/${EXPERIMENT}/seed_${SEED} \
  --name=${EXPERIMENT}_seed${SEED}

echo ""
echo "========================================================================"
echo "Task $SGE_TASK_ID finished at $(date)"
echo "========================================================================"
