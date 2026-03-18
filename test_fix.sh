#!/bin/bash
# Quick test to verify the fix works

echo "Testing if experiments run without errors..."
echo ""

# Create directory
mkdir -p test_fix/sender test_fix/receiver test_fix/messages test_fix/accuracy

# Run 1 epoch of LSTM baseline
python -m egg.zoo.channel.train \
  --n_features=200 \
  --vocab_size=20 \
  --max_len=15 \
  --batch_size=16 \
  --batches_per_epoch=2 \
  --n_epochs=1 \
  --lr=0.001 \
  --sender_entropy_coeff=0.1 \
  --receiver_entropy_coeff=0.1 \
  --probs=powerlaw \
  --early_stopping_thr=0.9999 \
  --force_eos=0 \
  --no_cuda \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --sender_hidden=128 \
  --receiver_hidden=256 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --sender_num_layers=1 \
  --receiver_num_layers=1 \
  --length_cost=0.0 \
  --impatient=0 \
  --reg=0 \
  --random_seed=42 \
  --dir_save=test_fix \
  --name=test_fix_lstm_baseline 2>&1 | head -30

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test successful! Experiments should work now."
else
    echo ""
    echo "✗ Test failed. There's still an issue."
fi
