#!/bin/bash
# Quick validation test for MacBook (CPU-friendly, ~5-10 minutes)

echo "========================================================================"
echo "Quick Test: Transformer LazImpa on MacBook"
echo "Expected runtime: 5-10 minutes on CPU"
echo "========================================================================"
echo ""

# Create output directories
for exp in lstm_baseline lstm_lazimpa transformer_baseline transformer_lazimpa; do
  mkdir -p quick_test/$exp/sender
  mkdir -p quick_test/$exp/receiver
  mkdir -p quick_test/$exp/messages
  mkdir -p quick_test/$exp/accuracy
done

# Test 1: LSTM baseline (known to work) - 2 minutes
# NOTE: --impatient and --reg have type=bool bug, so we OMIT them for False
echo "Test 1/4: LSTM Baseline (2 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --n_epochs=3 \
  --batch_size=16 \
  --batches_per_epoch=10 \
  --n_features=50 \
  --vocab_size=10 \
  --max_len=5 \
  --sender_hidden=64 \
  --receiver_hidden=64 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --dir_save=quick_test/lstm_baseline \
  --random_seed=42

echo ""
echo "✓ Test 1 complete"

# Test 2: LSTM LazImpa - 2 minutes
# NOTE: Include --impatient and --reg flags for True (no value needed)
echo ""
echo "Test 2/4: LSTM LazImpa (2 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --impatient=1 \
  --reg=1 \
  --length_cost=0.01 \
  --n_epochs=3 \
  --batch_size=16 \
  --batches_per_epoch=10 \
  --n_features=50 \
  --vocab_size=10 \
  --max_len=5 \
  --sender_hidden=64 \
  --receiver_hidden=64 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --dir_save=quick_test/lstm_lazimpa \
  --random_seed=42

echo ""
echo "✓ Test 2 complete"

# Test 3: Transformer baseline - 3 minutes
# NOTE: --causal_receiver is action='store_true', so just include flag
echo ""
echo "Test 3/4: Transformer Baseline (3 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=transformer \
  --causal_receiver \
  --n_epochs=3 \
  --batch_size=16 \
  --batches_per_epoch=10 \
  --n_features=50 \
  --vocab_size=10 \
  --max_len=5 \
  --sender_hidden=64 \
  --receiver_hidden=128 \
  --receiver_embedding=64 \
  --receiver_num_heads=4 \
  --receiver_num_layers=1 \
  --dir_save=quick_test/transformer_baseline \
  --random_seed=42

echo ""
echo "✓ Test 3 complete"

# Test 4: Transformer LazImpa (THE NEW IMPLEMENTATION) - 3 minutes
echo ""
echo "Test 4/4: Transformer LazImpa - NEW IMPLEMENTATION (3 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=transformer \
  --causal_receiver \
  --impatient=1 \
  --reg=1 \
  --length_cost=0.01 \
  --n_epochs=3 \
  --batch_size=16 \
  --batches_per_epoch=10 \
  --n_features=50 \
  --vocab_size=10 \
  --max_len=5 \
  --sender_hidden=64 \
  --receiver_hidden=128 \
  --receiver_embedding=64 \
  --receiver_num_heads=4 \
  --receiver_num_layers=1 \
  --dir_save=quick_test/transformer_lazimpa \
  --random_seed=42

echo ""
echo "✓ Test 4 complete"

echo ""
echo "========================================================================"
echo "ALL TESTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: quick_test/"
echo ""
echo "Quick check - message lengths should be:"
echo "  LSTM baseline: ~4-5 symbols (long, no pressure)"
echo "  LSTM LazImpa: ~2-3 symbols (short, efficient)"
echo "  Transformer baseline: ~4-5 symbols (long, no pressure)"
echo "  Transformer LazImpa: ~2-3 symbols (short, efficient) <- THIS IS NEW!"
echo ""

# Show final message lengths
echo "Checking final message lengths..."
echo ""
for exp in lstm_baseline lstm_lazimpa transformer_baseline transformer_lazimpa; do
  if [ -f "quick_test/$exp/messages/messages_2.npy" ]; then
    echo "$exp: messages saved ✓"
  else
    echo "$exp: WARNING - messages file not found"
  fi
done
