#!/bin/bash

#$ -M anzony.quispe@gmail.com   # Email address for job notification
#$ -m abe                        # Send mail when job begins, ends and aborts
#$ -pe smp 4                     # 4 CPU cores (for data loading)
#$ -q gpu                        # Run on the GPU cluster
#$ -l gpu_card=1                 # Run on 1 GPU card
#$ -N lazimpa_test               # Job name

# Load required modules (adjust based on your cluster's available modules)
module load python/3.9           # Or whatever Python version is available
module load cuda/11.8            # Or appropriate CUDA version
module load pytorch/2.0          # Or load PyTorch module if available

# If PyTorch isn't a module, you might need to activate a conda environment:
# source activate lazimpa_env

# Set environment variables
export OMP_NUM_THREADS=$NSLOTS
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $JOB_ID"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

# Change to project directory
cd $HOME/Lazimpa  # Adjust this path to where your code is located

# Run a single quick test experiment
python -m egg.zoo.channel.train \
  --sender_cell=transformer \
  --receiver_cell=transformer \
  --causal_receiver \
  --impatient=1 \
  --reg=1 \
  --length_cost=0.01 \
  --n_epochs=10 \
  --batch_size=32 \
  --batches_per_epoch=100 \
  --n_features=1000 \
  --vocab_size=40 \
  --max_len=30 \
  --sender_hidden=512 \
  --receiver_hidden=512 \
  --sender_embedding=256 \
  --receiver_embedding=256 \
  --sender_num_heads=8 \
  --receiver_num_heads=8 \
  --sender_num_layers=2 \
  --receiver_num_layers=2 \
  --dir_save=cluster_test/transformer_lazimpa \
  --random_seed=42

echo ""
echo "========================================================================"
echo "Job finished at $(date)"
echo "========================================================================"
