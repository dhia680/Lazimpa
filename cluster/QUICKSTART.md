# Quick Start Guide: Running LazImpa on Notre Dame Cluster

## Step 1: Upload Code to Cluster

On your **local MacBook**:

```bash
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa

# Upload to cluster (replace 'netid' with your actual NetID)
rsync -avz \
  --exclude 'results/' \
  --exclude 'quick_test/' \
  --exclude 'medium_test/' \
  --exclude '.git/' \
  --exclude '*.pyc' \
  --exclude '__pycache__/' \
  . netid@crc.nd.edu:~/Lazimpa/
```

## Step 2: SSH to Cluster

```bash
ssh netid@crc.nd.edu
```

## Step 3: Set Up Environment (First Time Only)

```bash
cd ~/Lazimpa

# Load Python and CUDA
module load python/3.9 cuda/11.8

# Check what modules are available if these don't work
# module avail python
# module avail cuda

# Install PyTorch (if not already installed)
pip install --user torch torchvision torchaudio

# Install other dependencies
pip install --user numpy matplotlib jupyter

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Step 4: Update Script Paths

Edit the scripts to match your setup:

```bash
cd ~/Lazimpa/cluster

# Open each .sh file and update:
# 1. Change 'netid@nd.edu' to your email
# 2. Change '$HOME/Lazimpa' if your path is different
# 3. Update module names to match your cluster

# Quick check of module names:
module avail python
module avail cuda
module avail pytorch
```

## Step 5: Run Test Job

**ALWAYS test first!**

```bash
cd ~/Lazimpa/cluster

# Submit test job
qsub test_single.sh

# Check status (wait ~20-30 minutes)
qstat -u $USER

# Check output when done
cat lazimpa_test.o*    # Output
cat lazimpa_test.e*    # Errors
```

**Expected output**: Should show training for 10 epochs, messages generated, no errors.

## Step 6: Run Full Experiments

### Option A: Sequential (Safe, Slower)

One GPU, runs all 40 experiments sequentially (~3 days):

```bash
cd ~/Lazimpa/cluster
qsub run_all_experiments.sh

# Monitor progress
qstat -u $USER
tail -f lazimpa_all.o*
```

### Option B: Parallel Array (Recommended, Faster)

40 GPUs, all experiments in parallel (~2-4 hours):

```bash
cd ~/Lazimpa/cluster
mkdir -p logs
qsub run_parallel_array.sh

# Check status
qstat -u $USER

# Monitor specific task
tail -f logs/job_1.out

# Count completed
ls -la ../results/*/seed_*/messages/
```

### Option C: Use Helper Script

```bash
cd ~/Lazimpa/cluster
chmod +x manage_jobs.sh

# Interactive menu
./manage_jobs.sh

# Or direct commands
./manage_jobs.sh test          # Test job
./manage_jobs.sh parallel      # Run all in parallel
./manage_jobs.sh status        # Check status
./manage_jobs.sh results       # Check completed experiments
./manage_jobs.sh logs          # View recent logs
```

## Step 7: Monitor Progress

```bash
# Check job status
qstat -u $USER

# Check completed experiments
cd ~/Lazimpa
find results/ -name "messages_*.npy" | wc -l
# Should show 40 when all complete

# Check specific experiment
ls -la results/transformer_lazimpa/seed_*/messages/

# View logs
tail -f cluster/logs/job_*.out
```

## Step 8: Download Results

On your **local MacBook**:

```bash
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa

# Download results (replace 'netid')
rsync -avz netid@crc.nd.edu:~/Lazimpa/results/ ./results/

# Verify
ls -la results/*/seed_*/messages/
```

## Step 9: Analyze Results

On your **local MacBook**:

```bash
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa

# Open Jupyter notebook
jupyter notebook LazImpa_notebook.ipynb

# Or run analysis script
python - << 'EOF'
import numpy as np
from pathlib import Path

experiments = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']

for exp in experiments:
    msg_dir = Path(f'results/{exp}')
    seeds = list(msg_dir.glob('seed_*/messages/messages_*.npy'))
    print(f"{exp}: {len(seeds)} files found")

    if seeds:
        # Load final epoch from first seed
        final_msg = np.load(sorted(seeds)[0])
        lengths = (final_msg != 0).sum(axis=1)
        print(f"  Avg message length: {lengths.mean():.2f}")
EOF
```

## Common Issues

### Job stays in "qw" (queued) state
- Cluster is busy, wait for resources
- Check queue: `qstat -g c`
- Check your priority: `qstat -u $USER`

### Module not found
```bash
# Find correct module names
module avail python
module avail cuda

# Update scripts with correct names
nano test_single.sh
```

### GPU not available
```bash
# Check GPU queue
qstat -F gpu_card

# Your job output will show
cat lazimpa_test.e*
```

### Out of memory
Edit scripts to reduce batch size:
```bash
nano test_single.sh
# Change: --batch_size=32
# To:     --batch_size=16
```

### Permission denied
```bash
chmod +x cluster/*.sh
```

## Timeline Expectations

| Method | GPUs | Time | When to Use |
|--------|------|------|-------------|
| Test | 1 | 30 min | Always run first |
| Sequential | 1 | 40-80 hrs | Limited GPU access |
| Parallel | 40 | 2-4 hrs | Ample GPUs available |

## Quick Reference Commands

```bash
# Submit jobs
qsub test_single.sh                                    # Test
qsub run_all_experiments.sh                            # All sequential
qsub run_parallel_array.sh                             # All parallel
qsub -v EXPERIMENT=transformer_lazimpa,SEED=42 run_single_experiment.sh  # One specific

# Monitor
qstat -u $USER                                         # Your jobs
qstat -j <job_id>                                      # Job details
watch -n 10 'qstat -u $USER'                          # Auto-refresh

# Results
find results/ -name "messages_*.npy" | wc -l          # Count completed
ls -la results/*/seed_*/                              # Check structure
tail -f logs/job_*.out                                # Watch progress

# Cancel
qdel <job_id>                                          # Cancel one job
qdel {100..140}                                        # Cancel range
qstat -u $USER | awk '{print $1}' | xargs qdel       # Cancel all
```

## Need Help?

1. **Cluster issues**: crcsupport@nd.edu
2. **Module problems**: Check `module avail` and update script
3. **Code errors**: Check logs in `lazimpa_*.e*` files
4. **Results issues**: Verify paths in scripts match your setup

## Success Checklist

- [ ] Code uploaded to cluster
- [ ] Modules load without errors
- [ ] PyTorch imports and detects CUDA
- [ ] Test job completes successfully
- [ ] Full experiments running or complete
- [ ] Results downloaded to local machine
- [ ] Analysis notebook runs

Good luck! 🚀
