#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 2                     # 2 CPU cores
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # 1 GPU card
#$ -N tf_baseline_test           # Job name
#$ -l h_rt=00:30:00              # Max 30 minutes for small test
#$ -o logs/test_tf_baseline.out
#$ -e logs/test_tf_baseline.err

# ============================================================
# Load Conda Environment
# ============================================================
module load conda
conda activate lazimpa

# Set environment variables
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "LAZIMPA - TRANSFORMER BASELINE SMALL TEST"
echo "========================================================================"
echo "Job ID: $JOB_ID"
echo "Started on $(hostname) at $(date)"
echo "Working directory: $PWD"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Create output directories
mkdir -p results/test_transformer_baseline/{sender,receiver,messages,accuracy,logs}

# Run small test: TRANSFORMER BASELINE (no impatient, no reg)
echo "Running Transformer BASELINE test (no impatient, no reg)..."
python -m egg.zoo.channel.train \
  --dir_save=results/test_transformer_baseline \
  --n_features=100 \
  --vocab_size=20 \
  --max_len=10 \
  --batch_size=32 \
  --batches_per_epoch=100 \
  --n_epochs=10 \
  --lr=0.001 \
  --probs=powerlaw \
  --early_stopping_thr=0.9999 \
  --force_eos=0 \
  --sender_entropy_coeff=0.1 \
  --receiver_entropy_coeff=0.1 \
  --sender_cell=transformer \
  --receiver_cell=transformer \
  --sender_hidden=256 \
  --receiver_hidden=256 \
  --sender_embedding=128 \
  --receiver_embedding=128 \
  --sender_num_layers=2 \
  --receiver_num_layers=2 \
  --sender_num_heads=8 \
  --receiver_num_heads=8 \
  --causal_sender \
  --causal_receiver \
  --sender_generate_style=in-place \
  --length_cost=0.0 \
  --random_seed=42

# Note: impatient and reg are NOT passed, so they default to False (baseline mode)

echo ""
echo "========================================================================"
echo "TRANSFORMER BASELINE TEST COMPLETE"
echo "Finished at $(date)"
echo "Results saved to: results/test_transformer_baseline/"
echo "========================================================================"
