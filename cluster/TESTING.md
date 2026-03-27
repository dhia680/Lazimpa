# Testing Guide: Verify Before Running Full Experiments

**IMPORTANT:** Always test before committing to 40-80 hours of compute time!

## Quick Testing Steps

### 1. Upload Code to Cluster
```bash
# On your MacBook
cd /Users/anzony.quisperojas/Documents/GitHub/python/Lazimpa

rsync -avz \
  --exclude 'results/' \
  --exclude 'quick_test/' \
  --exclude 'medium_test/' \
  --exclude '.git/' \
  . your_netid@crc.nd.edu:~/Lazimpa/
```

### 2. SSH to Cluster
```bash
ssh your_netid@crc.nd.edu
cd ~/Lazimpa/cluster
```

### 3. Verify Environment
```bash
# Load modules
module load python/3.9 cuda/11.8

# Test PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected:**
```
PyTorch: 2.x.x
CUDA: True
```

### 4. Submit Test Job
```bash
qsub test_single.sh
```

**Example output:**
```
Your job 12345 ("lazimpa_test") has been submitted
```

### 5. Wait for Completion (~30 minutes)
```bash
# Check status
qstat -u $USER

# When it disappears from qstat, it's done
```

### 6. Verify Results (Automated!)
```bash
./verify_test.sh
```

**If you see:**
```
🎉 ALL TESTS PASSED!
Passed: 7/7
```
**→ You're ready for full experiments!**

**If you see:**
```
⚠️ SOME TESTS FAILED
Passed: 4/7
Failed: 3/7
```
**→ Fix issues before proceeding**

---

## Manual Verification (If verify_test.sh Doesn't Work)

```bash
# 1. Find your test output files
ls -lh lazimpa_test.*

# 2. Check standard output
cat lazimpa_test.o12345  # Replace 12345 with your job ID

# 3. Look for these signs:
grep "Job finished" lazimpa_test.o12345      # Should find it
grep "Epoch:" lazimpa_test.o12345 | wc -l    # Should show 10
grep "Impatient score=" lazimpa_test.o12345  # Should find it
grep "message:" lazimpa_test.o12345          # Should find many

# 4. Check for errors
cat lazimpa_test.e12345  # Should be empty or just warnings

# 5. Check output files were created
ls -la ~/Lazimpa/cluster_test/transformer_lazimpa/
# Should show: sender/, receiver/, messages/, accuracy/

ls ~/Lazimpa/cluster_test/transformer_lazimpa/messages/
# Should show: messages_0.npy, messages_1.npy, ... messages_9.npy
```

---

## Common Issues and Fixes

### Issue 1: Module not found
```
Module 'python/3.9' not found
```

**Fix:**
```bash
# Find correct module name
module avail python

# Update test_single.sh
nano test_single.sh
# Change: module load python/3.9
# To:     module load python/3.10  # or whatever is available
```

### Issue 2: PyTorch not installed
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:**
```bash
pip install --user torch torchvision torchaudio
```

### Issue 3: CUDA not available
```
CUDA: False
```

**Fix:**
```bash
# Check if CUDA module loaded
module list

# Check GPU queue status
qstat -F gpu_card

# Verify you requested GPU in script
grep "gpu" test_single.sh
# Should show: #$ -q gpu and #$ -l gpu_card=1
```

### Issue 4: Job stays in queue forever
```bash
qstat -u $USER
# Shows "qw" for hours
```

**Fix:**
- Cluster is busy, wait longer
- Check queue status: `qstat -g c`
- Check your priority: `qstat -explain E`
- Try different queue time: submit during off-hours

### Issue 5: Job killed immediately
```
Job terminated
```

**Fix:**
```bash
# Check error file
cat lazimpa_test.e12345

# Common causes:
# - Wrong path: Update path in test_single.sh
# - Module conflicts: module purge, then reload
# - Permissions: chmod +x test_single.sh
```

---

## What Each Test Checks

| Check | What It Means | If Failed |
|-------|---------------|-----------|
| Job completed | Script ran to completion | Check error log |
| GPU environment | GPU allocated correctly | Check GPU queue |
| Training ran | PyTorch and model working | Check imports |
| Messages generated | Forward pass working | Check model code |
| Impatient receiver | Your new code working! | Check implementation |
| No errors | Clean execution | Review error file |
| Files saved | I/O and paths correct | Check permissions |

---

## Step-by-Step Test Workflow

```bash
# 1. SSH to cluster
ssh netid@crc.nd.edu

# 2. Navigate to code
cd ~/Lazimpa/cluster

# 3. Test environment
module load python/3.9 cuda/11.8
python -c "import torch; print(torch.cuda.is_available())"

# 4. Submit test
qsub test_single.sh

# 5. Monitor (optional)
watch -n 30 'qstat -u $USER'
# Wait until job disappears (~30 min)

# 6. Verify results
./verify_test.sh

# 7. If all passed, run full experiment
qsub run_parallel_array.sh
```

---

## Test Timeline

| Step | Time | Action |
|------|------|--------|
| Upload code | 1-5 min | rsync to cluster |
| Setup check | 2 min | module load, python test |
| Submit test | 1 min | qsub test_single.sh |
| **Wait in queue** | 0-60 min | Cluster assigns GPU |
| **Test runs** | 20-30 min | Actual training |
| Verify | 1 min | ./verify_test.sh |
| **TOTAL** | **25-100 min** | Depends on queue |

---

## After Successful Test

Once `verify_test.sh` shows all passed:

### Option A: Run All in Parallel (Fast)
```bash
qsub run_parallel_array.sh
# Runtime: 2-4 hours (if GPUs available)
# Uses: 40 GPUs
```

### Option B: Run All Sequentially (Slow)
```bash
qsub run_all_experiments.sh
# Runtime: 40-80 hours
# Uses: 1 GPU
```

### Monitor Progress
```bash
# Check job status
./manage_jobs.sh status

# Check completed experiments
./manage_jobs.sh results

# View logs
./manage_jobs.sh logs
```

---

## Test Passed Checklist

Before running full experiments, confirm:

- [x] Local test passed (quick_test.sh on MacBook)
- [x] Code uploaded to cluster
- [x] Modules load without errors
- [x] PyTorch imports successfully
- [x] CUDA available = True
- [x] Test job submitted (qsub test_single.sh)
- [x] Test job completed (qstat shows nothing)
- [x] verify_test.sh shows 7/7 passed
- [x] Output files exist in cluster_test/

**If ALL checked → SAFE TO RUN FULL EXPERIMENTS! 🚀**

---

## Quick Reference

```bash
# Test the test (check syntax)
bash -n test_single.sh

# Submit test
qsub test_single.sh

# Check if running
qstat -u $USER

# Verify when done
./verify_test.sh

# If passed, run full experiment
qsub run_parallel_array.sh    # Fast
# OR
qsub run_all_experiments.sh   # Slow
```

---

## Need Help?

1. Check error file: `cat lazimpa_test.e*`
2. Check output file: `cat lazimpa_test.o*`
3. Run verification: `./verify_test.sh`
4. See QUICKSTART.md for setup
5. See README.md for troubleshooting
6. Contact: crcsupport@nd.edu (cluster issues)
