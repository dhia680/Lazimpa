# Running Full Experiments Locally on MacBook

This guide shows how to run the full LazImpa experiments on your MacBook Pro (CPU mode).

## Quick Start

```bash
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa

# Run all experiments (12 runs total)
python run_experiments_local.py --config experiments_config_local.json
```

## What Gets Run

**Local configuration** (`experiments_config_local.json`):
- **4 experiment types:**
  - LSTM Baseline
  - LSTM LazImpa
  - Transformer Baseline
  - Transformer LazImpa
- **3 seeds** (vs 10 for cluster)
- **100 epochs** (vs 500 for cluster)
- **Smaller models** (CPU-friendly)
- **Total: 12 runs** (vs 40 for cluster)

## Runtime Estimate

On MacBook Pro CPU:
- **Per run:** ~10-15 minutes
- **Total:** ~2-3 hours for all 12 runs

## Step-by-Step

### 1. Preview the Plan (Optional)

```bash
python run_experiments_local.py \
  --config experiments_config_local.json \
  --dry-run
```

Shows what will run without executing.

### 2. Run All Experiments

```bash
python run_experiments_local.py \
  --config experiments_config_local.json
```

**What happens:**
1. Shows experiment plan
2. Asks for confirmation
3. Runs 12 experiments sequentially
4. Shows progress and ETA
5. Saves results to `results_local/`

**Example output:**
```
======================================================================
LOCAL EXPERIMENT QUEUE (MacBook CPU)
======================================================================
Total runs: 12 (4 experiments × 3 seeds)
Device: CPU (no CUDA)

Experiments:
  • lstm_baseline: 3 seeds
  • lstm_lazimpa: 3 seeds
  • transformer_baseline: 3 seeds
  • transformer_lazimpa: 3 seeds

Estimated runtime: ~2.4 hours (~12 min per run)

Proceed with experiments? [y/N]: y
```

### 3. Monitor Progress

The script shows:
- Current run number (e.g., "Run 5/12")
- Experiment name and seed
- Completion time for each run
- Updated ETA after every 3 runs

**Example:**
```
======================================================================
[2025-03-18 15:30:00] Run 5/12
Experiment: transformer_lazimpa | Seed: 42
Log: results_local/transformer_lazimpa/seed_42/logs/training.log
======================================================================

✓ Completed successfully in 12.3 min
  ETA for remaining 7 runs: ~1.4 hours
```

### 4. Check Results

When complete, results are saved to:
```
results_local/
├── lstm_baseline/
│   ├── seed_42/
│   ├── seed_123/
│   └── seed_456/
├── lstm_lazimpa/
│   ├── seed_42/
│   ├── seed_123/
│   └── seed_456/
├── transformer_baseline/
│   ├── seed_42/
│   ├── seed_123/
│   └── seed_456/
├── transformer_lazimpa/
│   ├── seed_42/
│   ├── seed_123/
│   └── seed_456/
└── experiment_summary.json
```

View summary:
```bash
cat results_local/experiment_summary.json
```

## Run Specific Experiments

### Run Only One Experiment Type

```bash
# Just Transformer LazImpa (3 seeds)
python run_experiments_local.py \
  --config experiments_config_local.json \
  --experiments transformer_lazimpa
```

### Run Multiple Types

```bash
# Just the LazImpa variants
python run_experiments_local.py \
  --config experiments_config_local.json \
  --experiments lstm_lazimpa transformer_lazimpa
```

### Run Just Baselines

```bash
python run_experiments_local.py \
  --config experiments_config_local.json \
  --experiments lstm_baseline transformer_baseline
```

## Resume After Interruption

If experiments get interrupted (laptop sleep, etc.):

```bash
python run_experiments_local.py \
  --config experiments_config_local.json \
  --resume
```

This skips already completed experiments and continues from where it stopped.

## Analyze Results

Use the Jupyter notebook to analyze:

```bash
jupyter notebook LazImpa_notebook.ipynb
```

Or quick analysis in Python:

```python
import numpy as np
from pathlib import Path

experiments = ['lstm_baseline', 'lstm_lazimpa',
               'transformer_baseline', 'transformer_lazimpa']

for exp in experiments:
    msg_dir = Path(f'results_local/{exp}')
    seeds = list(msg_dir.glob('seed_*/messages/messages_*.npy'))

    if seeds:
        # Load final messages from first seed
        final = sorted(seeds)[-1]
        msgs = np.load(final)
        lengths = (msgs != 0).sum(axis=1)

        print(f"{exp}:")
        print(f"  Files: {len(seeds)}")
        print(f"  Avg length: {lengths.mean():.1f}")
        print()
```

## Customization

Edit `experiments_config_local.json` to adjust:

### Make It Faster (Less Accurate)
```json
{
  "n_epochs": 50,          // 50 instead of 100
  "batches_per_epoch": 50, // 50 instead of 100
  "seeds": [42]            // 1 seed instead of 3
}
```
**Runtime:** ~30 min for 4 experiments

### Make It More Thorough (Slower)
```json
{
  "n_epochs": 200,         // More training
  "batches_per_epoch": 200,
  "seeds": [42, 123, 456, 789, 1011]  // More seeds
}
```
**Runtime:** ~10 hours for 20 experiments

### Bigger Models (Much Slower)
```json
{
  "n_features": 500,       // Closer to full size
  "vocab_size": 30,
  "max_len": 20
}
```
**Warning:** May be very slow on CPU!

## Troubleshooting

### Experiments are too slow
- Reduce `n_epochs` to 50
- Reduce `batches_per_epoch` to 50
- Use only 1 seed: `"seeds": [42]`

### Out of memory
- Reduce `batch_size` from 16 to 8
- Reduce `n_features` from 200 to 100
- Reduce model sizes

### Interrupt and resume
```bash
# Stop: Ctrl+C
# Resume later:
python run_experiments_local.py \
  --config experiments_config_local.json \
  --resume
```

### Check progress while running
```bash
# In another terminal
tail -f results_local/*/seed_*/logs/training.log
```

## Comparison: Local vs Cluster

| Aspect | Local (MacBook) | Cluster (GPU) |
|--------|-----------------|---------------|
| **Device** | CPU | GPU (CUDA) |
| **Config** | `experiments_config_local.json` | `experiments_config.json` |
| **Runs** | 12 (4 types × 3 seeds) | 40 (4 types × 10 seeds) |
| **Epochs** | 100 | 500 |
| **Model size** | Smaller (128/256 hidden) | Larger (250-600 hidden) |
| **Problem size** | Medium (200 features) | Large (1000 features) |
| **Runtime** | ~2-3 hours | ~2-4 hours (parallel) |
| **Results dir** | `results_local/` | `results/` |
| **Use case** | Testing, development | Publication results |

## Tips

1. **Run overnight:** Let it run while you sleep
2. **Keep laptop plugged in:** Prevent battery drain
3. **Prevent sleep:**
   ```bash
   caffeinate -i python run_experiments_local.py --config experiments_config_local.json
   ```
4. **Monitor CPU temp:** MacBooks can get hot during long runs
5. **Use resume:** If interrupted, just resume instead of restarting

## Next Steps

After local experiments complete:
1. Analyze results with Jupyter notebook
2. Compare LSTM vs Transformer
3. Verify LazImpa learns shorter messages
4. If results look good, run full experiments on cluster for publication

## Summary Commands

```bash
# Preview
python run_experiments_local.py --config experiments_config_local.json --dry-run

# Run all
python run_experiments_local.py --config experiments_config_local.json

# Run specific
python run_experiments_local.py --config experiments_config_local.json --experiments transformer_lazimpa

# Resume after interrupt
python run_experiments_local.py --config experiments_config_local.json --resume

# Prevent sleep during run
caffeinate -i python run_experiments_local.py --config experiments_config_local.json
```

## Expected Results

After ~2-3 hours, you should see:

**Baseline experiments:**
- Messages use full length (~14-15 symbols)
- Similar accuracy for LSTM and Transformer

**LazImpa experiments:**
- Messages compressed (~5-8 symbols)
- Similar accuracy despite shorter messages
- Both LSTM and Transformer should compress

This validates that TransformerReceiverImpatient works correctly!
