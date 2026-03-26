#!/bin/bash
# Quick test to verify the entropy coefficient fix
# This runs a short experiment to check if message lengths decrease

echo "=============================================="
echo "Testing LSTM Baseline with CORRECT parameters"
echo "Key fix: sender_entropy_coeff=2.0 (was 0.1)"
echo "=============================================="

mkdir -p test_fix/lstm_baseline/{messages,accuracy,sender,receiver}

# Matches original notebook EXACTLY (except n_features=100 for speed)
python -m egg.zoo.channel.train \
  --dir_save=test_fix/lstm_baseline \
  --vocab_size=40 \
  --max_len=30 \
  --n_features=100 \
  --probs=powerlaw \
  --n_epochs=100 \
  --batch_size=512 \
  --length_cost=0.0 \
  --sender_cell=lstm \
  --receiver_cell=lstm \
  --sender_hidden=250 \
  --receiver_hidden=600 \
  --receiver_embedding=100 \
  --sender_embedding=10 \
  --batches_per_epoch=100 \
  --lr=0.001 \
  --sender_entropy_coeff=2.0 \
  --sender_num_layers=1 \
  --receiver_num_layers=1 \
  --early_stopping_thr=0.99 \
  --random_seed=42 \
  --name=test_fix

echo ""
echo "=============================================="
echo "Checking message length evolution..."
echo "=============================================="

python -c "
import numpy as np
from pathlib import Path

msg_dir = Path('test_fix/lstm_baseline/messages')
msg_files = sorted(msg_dir.glob('messages_*.npy'), key=lambda x: int(x.stem.split('_')[1]))

print('Epoch | Mean Length')
print('-' * 25)
for mf in msg_files[::10]:  # Every 10 epochs
    epoch = int(mf.stem.split('_')[1])
    messages = np.load(mf, allow_pickle=True)
    lengths = []
    for msg in messages:
        eos_pos = np.where(msg == 0)[0]
        if len(eos_pos) > 0:
            lengths.append(eos_pos[0] + 1)
        else:
            lengths.append(len(msg))
    print(f'{epoch:5d} | {np.mean(lengths):.2f}')

print()
print('If mean length DECREASES over epochs, the fix is working!')
print('Expected: Start ~30, end ~5-10 for baseline')
"
