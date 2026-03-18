#!/bin/bash
# Medium-sized test for MacBook (CPU-friendly, ~30-60 minutes)
# More realistic parameters to see actual learning

echo "========================================================================"
echo "Medium Test: Transformer LazImpa on MacBook"
echo "Expected runtime: 30-60 minutes on CPU"
echo "========================================================================"
echo ""

# Create output directories
for exp in medium_lstm_baseline medium_lstm_lazimpa medium_transformer_baseline medium_transformer_lazimpa; do
  mkdir -p medium_test/$exp/sender
  mkdir -p medium_test/$exp/receiver
  mkdir -p medium_test/$exp/messages
  mkdir -p medium_test/$exp/accuracy
done

# Test 1: LSTM Baseline (~10 min)
echo "Test 1/4: LSTM Baseline (~10 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --n_epochs=50 \
  --batch_size=32 \
  --batches_per_epoch=50 \
  --n_features=200 \
  --vocab_size=20 \
  --max_len=15 \
  --sender_hidden=128 \
  --receiver_hidden=256 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --dir_save=medium_test/medium_lstm_baseline \
  --random_seed=42

echo ""
echo "✓ Test 1 complete"
echo ""

# Test 2: LSTM LazImpa (~10 min)
echo "Test 2/4: LSTM LazImpa (~10 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --impatient=1 \
  --reg=1 \
  --length_cost=0.01 \
  --n_epochs=50 \
  --batch_size=32 \
  --batches_per_epoch=50 \
  --n_features=200 \
  --vocab_size=20 \
  --max_len=15 \
  --sender_hidden=128 \
  --receiver_hidden=256 \
  --sender_embedding=32 \
  --receiver_embedding=32 \
  --dir_save=medium_test/medium_lstm_lazimpa \
  --random_seed=42

echo ""
echo "✓ Test 2 complete"
echo ""

# Test 3: Transformer Baseline (~15 min)
echo "Test 3/4: Transformer Baseline (~15 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=transformer \
  --causal_receiver \
  --n_epochs=50 \
  --batch_size=32 \
  --batches_per_epoch=50 \
  --n_features=200 \
  --vocab_size=20 \
  --max_len=15 \
  --sender_hidden=128 \
  --receiver_hidden=256 \
  --receiver_embedding=128 \
  --receiver_num_heads=4 \
  --receiver_num_layers=2 \
  --dir_save=medium_test/medium_transformer_baseline \
  --random_seed=42

echo ""
echo "✓ Test 3 complete"
echo ""

# Test 4: Transformer LazImpa (~15 min) - THE NEW IMPLEMENTATION
echo "Test 4/4: Transformer LazImpa - NEW IMPLEMENTATION (~15 min)..."
python -m egg.zoo.channel.train \
  --sender_cell=lstm \
  --receiver_cell=transformer \
  --causal_receiver \
  --impatient=1 \
  --reg=1 \
  --length_cost=0.01 \
  --n_epochs=50 \
  --batch_size=32 \
  --batches_per_epoch=50 \
  --n_features=200 \
  --vocab_size=20 \
  --max_len=15 \
  --sender_hidden=128 \
  --receiver_hidden=256 \
  --receiver_embedding=128 \
  --receiver_num_heads=4 \
  --receiver_num_layers=2 \
  --dir_save=medium_test/medium_transformer_lazimpa \
  --random_seed=42

echo ""
echo "✓ Test 4 complete"
echo ""

echo "========================================================================"
echo "ALL MEDIUM TESTS COMPLETE!"
echo "========================================================================"
echo ""

# Analyze results
python - << 'EOF'
import numpy as np
from pathlib import Path

experiments = [
    'medium_lstm_baseline',
    'medium_lstm_lazimpa',
    'medium_transformer_baseline',
    'medium_transformer_lazimpa'
]

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

for exp in experiments:
    msg_dir = Path(f'medium_test/{exp}/messages')
    if msg_dir.exists():
        msg_files = sorted(msg_dir.glob('messages_*.npy'))
        if msg_files:
            # Load final messages
            final_msgs = np.load(msg_files[-1])

            # Calculate average message length (count non-zero symbols)
            lengths = (final_msgs != 0).sum(axis=1)
            avg_len = lengths.mean()
            std_len = lengths.std()

            print(f"\n{exp}:")
            print(f"  Final epoch: {len(msg_files) - 1}")
            print(f"  Avg message length: {avg_len:.2f} ± {std_len:.2f}")
            print(f"  Min/Max length: {lengths.min()}/{lengths.max()}")

            # Show a few example messages
            print(f"  Sample messages:")
            for i in range(min(3, len(final_msgs))):
                msg = final_msgs[i]
                msg_str = ','.join(map(str, msg[msg != 0]))
                print(f"    input {i}: [{msg_str}]")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("Baseline: Should use full message length (~15 symbols)")
print("LazImpa: Should learn shorter messages (~5-8 symbols)")
print("Compare LSTM vs Transformer efficiency!")
print("="*70 + "\n")
EOF

echo ""
echo "Results saved to: medium_test/"
