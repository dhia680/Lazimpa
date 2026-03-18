#!/bin/bash
# Simple test to verify MPS (Apple GPU) works

echo "Testing MPS (Apple GPU) support..."
echo ""

# Create test directory
mkdir -p test_mps/sender test_mps/receiver test_mps/messages test_mps/accuracy

# Run WITHOUT --no_cuda flag to enable GPU
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --n_epochs=2 \
  --batch_size=16 \
  --batches_per_epoch=5 \
  --n_features=50 \
  --vocab_size=10 \
  --max_len=5 \
  --sender_hidden=64 \
  --receiver_hidden=64 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --dir_save=test_mps \
  --random_seed=42

echo ""
echo "Check above output - should show device='mps' if working"
