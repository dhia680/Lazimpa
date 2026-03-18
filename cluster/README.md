# Cluster Job Scripts for LazImpa Experiments

This directory contains SGE (Sun Grid Engine) job scripts for running LazImpa experiments on Notre Dame's GPU cluster.

## Prerequisites

Before submitting jobs, ensure:

1. **Code is uploaded to cluster**:
   ```bash
   # On your local machine
   cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa
   rsync -avz --exclude 'results/' --exclude '.git/' . netid@crc.nd.edu:~/Lazimpa/
   ```

2. **Python environment is set up** on the cluster:
   ```bash
   # SSH to cluster
   ssh netid@crc.nd.edu

   # Load modules
   module load python/3.9 cuda/11.8

   # Install dependencies (if not already done)
   pip install --user torch torchvision torchaudio
   pip install --user numpy matplotlib jupyter
   ```

3. **Update paths** in the scripts:
   - Replace `$HOME/Lazimpa` with your actual path
   - Replace `netid@nd.edu` with your actual NetID
   - Verify module names match your cluster's available modules

## Available Scripts

### 1. Test Single Job (`test_single.sh`)

**Purpose**: Quick test to verify everything works on the cluster.

**Runtime**: ~20-30 minutes
**Resources**: 1 GPU, 4 CPUs

**Submit**:
```bash
cd ~/Lazimpa/cluster
qsub test_single.sh
```

**What it does**: Runs a single transformer_lazimpa experiment for 10 epochs to verify GPU, PyTorch, and code are working.

### 2. Run All Experiments Sequentially (`run_all_experiments.sh`)

**Purpose**: Run all 40 experiments (4 types × 10 seeds) on a single GPU.

**Runtime**: ~40-80 hours
**Resources**: 1 GPU, 4 CPUs

**Submit**:
```bash
cd ~/Lazimpa/cluster
qsub run_all_experiments.sh
```

**What it does**: Runs experiments sequentially using `run_experiments.py`. Good if you have limited GPU access.

### 3. Run Parallel Array Job (`run_parallel_array.sh`) - **RECOMMENDED**

**Purpose**: Run all 40 experiments in parallel (if cluster allows).

**Runtime**: ~2-4 hours per job (40 jobs run simultaneously if resources available)
**Resources**: 40 GPU cards total (1 per job)

**Submit**:
```bash
cd ~/Lazimpa/cluster
mkdir -p logs  # Create logs directory
qsub run_parallel_array.sh
```

**What it does**: Submits an SGE array job with 40 tasks. Each task runs one experiment+seed combination. Much faster if your cluster has available GPUs.

**Check status**:
```bash
qstat -u $USER              # Check job status
qstat -j <job_id>           # Detailed job info
cat logs/job_*.out          # Check outputs
```

### 4. Run Single Experiment (`run_single_experiment.sh`)

**Purpose**: Run a specific experiment type with a specific seed.

**Runtime**: ~2-4 hours
**Resources**: 1 GPU, 4 CPUs

**Submit**:
```bash
# Run transformer_lazimpa with seed 42
qsub -v EXPERIMENT=transformer_lazimpa,SEED=42 run_single_experiment.sh

# Run LSTM baseline with seed 123
qsub -v EXPERIMENT=lstm_baseline,SEED=123 run_single_experiment.sh
```

**Available experiments**:
- `lstm_baseline`
- `lstm_lazimpa`
- `transformer_baseline`
- `transformer_lazimpa`

## Monitoring Jobs

### Check job status
```bash
# List your jobs
qstat -u $USER

# Detailed info for specific job
qstat -j <job_id>

# Watch job queue in real-time
watch -n 10 'qstat -u $USER'
```

### Check output
```bash
# For test_single.sh
cat lazimpa_test.o<job_id>     # Standard output
cat lazimpa_test.e<job_id>     # Error output

# For array jobs
cat logs/job_1.out             # Task 1 output
cat logs/job_1.err             # Task 1 errors

# Monitor progress
tail -f logs/job_*.out
```

### Check results
```bash
# List completed experiments
ls -la results/*/seed_*/messages/

# Count completed runs
find results/ -name "messages_*.npy" | wc -l
```

## Workflow Recommendations

### Option 1: Conservative (Limited GPU Access)
1. Submit test job first: `qsub test_single.sh`
2. Verify it completes successfully
3. Submit all experiments: `qsub run_all_experiments.sh`
4. Wait ~2-3 days for completion

### Option 2: Fast (Ample GPU Access) - **RECOMMENDED**
1. Submit test job first: `qsub test_single.sh`
2. Verify it completes successfully
3. Submit array job: `qsub run_parallel_array.sh`
4. Complete in ~2-4 hours (if GPUs available)

### Option 3: Selective
1. Run specific experiments individually:
   ```bash
   qsub -v EXPERIMENT=transformer_lazimpa,SEED=42 run_single_experiment.sh
   qsub -v EXPERIMENT=transformer_lazimpa,SEED=123 run_single_experiment.sh
   # etc.
   ```

## Troubleshooting

### Job won't start
- Check queue status: `qstat -g c`
- Check cluster status: `qstat -f`
- Verify GPU availability: `qstat -F gpu_card`

### Module not found
```bash
# List available modules
module avail

# Common alternatives
module load python/3.8    # or python/3.10
module load cuda/11.7     # or cuda/12.0
```

### Out of memory
- Reduce `--batch_size` in the scripts (change from 32 to 16)
- Reduce model size (fewer layers/heads)

### Job killed unexpectedly
- Check time limits: Some queues have max walltime
- Check memory usage: May need more RAM
- Add to script: `#$ -l h_rt=72:00:00` (72 hour limit)

## After Experiments Complete

### Download results
```bash
# On your local machine
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa
rsync -avz netid@crc.nd.edu:~/Lazimpa/results/ ./results/
```

### Analyze results
```bash
# On local machine or cluster
cd ~/Lazimpa
jupyter notebook LazImpa_notebook.ipynb
```

## Estimated Costs

**Computational resources** (assuming 2h per run):
- Sequential: 1 GPU × 80 hours = 80 GPU-hours
- Parallel: 40 GPUs × 2 hours = 80 GPU-hours

**Timeline**:
- Sequential: ~3.5 days
- Parallel: ~2-4 hours (if 40 GPUs available)

## Script Customization

### Change email address
Replace `anzony.quispe@gmail.com` with your email in all scripts.

### Change project path
Replace `$HOME/Lazimpa` with your actual installation path.

### Adjust resources
```bash
#$ -pe smp 8          # Use 8 CPUs instead of 4
#$ -l gpu_card=2      # Use 2 GPUs instead of 1
#$ -l h_rt=48:00:00   # Set 48 hour time limit
```

### Modify experiments
Edit `experiments_config.json` to change:
- Number of epochs
- Batch size
- Model architecture
- Seeds

## Contact

For cluster-specific issues, contact Notre Dame CRC support:
- Email: crcsupport@nd.edu
- Docs: https://docs.crc.nd.edu/

For LazImpa code issues:
- Check the main README.md
- Review the paper: literature/lazimpa.latex
