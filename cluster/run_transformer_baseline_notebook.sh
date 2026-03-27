#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 20                     # 8 CPU cores (faster data loading)
#$ -q gpu@@crc_a10               # Request A10 GPUs
#$ -l gpu_card=1                 # 1 GPU card per job
#$ -N tf_baseline_nb             # Job name
#$ -l h_rt=6:00:00               # 6 hours (should be ~3hr like notebook)
#$ -o logs/tf_baseline_nb.out
#$ -e logs/tf_baseline_nb.err

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
echo "TRANSFORMER BASELINE - NOTEBOOK SETTINGS"
echo "========================================================================"
echo "Job ID: $JOB_ID"
echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa

# Create logs and output directories
mkdir -p logs
mkdir -p results/transformer_baseline_notebook/seed_42/{logs,sender,receiver,messages,accuracy}

# Run TRANSFORMER BASELINE with EXACT notebook settings
# Only changes from original:
#   - sender_cell=transformer (was lstm)
#   - receiver_cell=transformer (was lstm)
#   - transformer-specific params (hidden, embedding, heads, layers)
#   - NO impatient, NO reg (baseline)
python -m egg.zoo.channel.train \
  --dir_save=results/transformer_baseline_notebook/seed_42 \
  --vocab_size=40 \
  --max_len=30 \
  --n_features=100 \
  --probs=powerlaw \
  --n_epochs=501 \
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
  --batches_per_epoch=100 \
  --lr=0.001 \
  --sender_entropy_coeff=2.0 \
  --early_stopping_thr=0.99 \
  --random_seed=42 \
  --name=transformer_baseline_notebook

echo ""
echo "========================================================================"
echo "TRANSFORMER BASELINE NOTEBOOK finished at $(date)"
echo "========================================================================"
