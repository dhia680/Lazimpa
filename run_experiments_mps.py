#!/usr/bin/env python
"""
MacBook GPU (MPS) experiment runner.
Runs experiments on Apple Silicon GPU for 3-5x speedup over CPU.
"""

import json
import os
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime
import sys
import torch

def check_mps():
    """Check if MPS is available."""
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("ERROR: MPS (Apple GPU) not available!")
        print()
        print("Your MacBook GPU cannot be used. Possible reasons:")
        print("  1. Not an Apple Silicon Mac (M1/M2/M3)")
        print("  2. PyTorch version too old (need 1.12+)")
        print("  3. macOS version too old (need 12.3+)")
        print()
        print("Current setup:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MPS backend built: {torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
        print()
        print("To use CPU instead, run:")
        print("  python run_experiments_local.py --config experiments_config_local.json")
        return False
    return True

def generate_command_args(experiment_name, config, common_params, arch_config, seed):
    """Generate training command arguments."""
    params = {**common_params, **arch_config, **config}
    args = []

    # Fields to exclude (metadata, not training params)
    exclude_fields = ['description', 'architecture']

    # Flags that use action='store_true' (no value needed, just include flag if True)
    store_true_flags = ['causal_sender', 'causal_receiver', 'no_cuda']

    for key, value in params.items():
        if key in exclude_fields:
            continue
        if isinstance(value, bool):
            if key in store_true_flags:
                # For action='store_true' flags, only include if True
                if value:
                    args.append(f"--{key}")
            else:
                # For type=bool flags (impatient, reg, no_cuda), always pass value
                args.append(f"--{key}={1 if value else 0}")
        else:
            args.append(f"--{key}={value}")

    args.append(f"--random_seed={seed}")
    args.append(f"--dir_save=results_mps/{experiment_name}/seed_{seed}")
    args.append(f"--name={experiment_name}_seed{seed}")

    return args

def run_experiment(exp_name, seed, args, total_runs, current_run):
    """Run a single experiment."""

    # Create all required directories
    base_dir = Path(f'results_mps/{exp_name}/seed_{seed}')
    (base_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (base_dir / 'sender').mkdir(parents=True, exist_ok=True)
    (base_dir / 'receiver').mkdir(parents=True, exist_ok=True)
    (base_dir / 'messages').mkdir(parents=True, exist_ok=True)
    (base_dir / 'accuracy').mkdir(parents=True, exist_ok=True)

    log_file = base_dir / 'logs' / 'training.log'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*70}")
    print(f"[{timestamp}] Run {current_run}/{total_runs}")
    print(f"Experiment: {exp_name} | Seed: {seed}")
    print(f"Device: MPS (Apple GPU)")
    print(f"Log: {log_file}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Set MPS environment variable to force it
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    with open(log_file, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Device: MPS (Apple GPU)\n")
        f.write(f"Started: {timestamp}\n")
        f.write(f"{'='*70}\n\n")
        f.flush()

        result = subprocess.run(
            ['python', '-m', 'egg.zoo.channel.train'] + args,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/60:.1f} min" if elapsed > 60 else f"{elapsed:.1f} sec"

    if result.returncode == 0:
        print(f"✓ Completed successfully in {elapsed_str}")
        return True, elapsed
    else:
        print(f"✗ Failed with return code {result.returncode} after {elapsed_str}")
        print(f"  Check log: {log_file}")
        return False, elapsed

def main():
    # Check MPS first
    if not check_mps():
        sys.exit(1)

    print("\n" + "="*70)
    print("MPS (Apple GPU) DETECTED - Ready to use!")
    print("Expected speedup: 3-5x faster than CPU")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments_config_mps.json')
    parser.add_argument('--experiments', nargs='+', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    common_params = config['common_params']
    arch_configs = config['architecture_configs']
    experiments = config['experiments']
    seeds = config['seeds']

    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}

    experiment_queue = []
    for exp_name, exp_config in experiments.items():
        exp_config_copy = exp_config.copy()
        arch_name = exp_config_copy.pop('architecture')
        arch_config = arch_configs[arch_name]

        for seed in seeds:
            cmd_args = generate_command_args(exp_name, exp_config_copy, common_params, arch_config, seed)
            experiment_queue.append({
                'name': exp_name,
                'seed': seed,
                'args': cmd_args
            })

    total_runs = len(experiment_queue)

    print(f"\n{'='*70}")
    print(f"MPS EXPERIMENT QUEUE")
    print(f"{'='*70}")
    print(f"Total runs: {total_runs} ({len(experiments)} experiments × {len(seeds)} seeds)")
    print(f"Device: MPS (Apple GPU)")
    print(f"{'='*70}\n")

    for exp in experiments.keys():
        print(f"  • {exp}: {len(seeds)} seeds")

    if args.dry_run:
        print(f"\n[DRY RUN]")
        for i, job in enumerate(experiment_queue, 1):
            print(f"{i:3d}. {job['name']:<30s} seed={job['seed']}")
        return

    # Estimate runtime (3-5x faster than CPU)
    cpu_time_per_run = 12  # minutes
    mps_time_per_run = cpu_time_per_run / 4  # ~4x speedup
    total_hours = (total_runs * mps_time_per_run) / 60

    print(f"\nEstimated runtime: ~{total_hours:.1f} hours (~{mps_time_per_run:.0f} min per run)")
    print(f"  (vs ~{total_runs * cpu_time_per_run / 60:.1f} hours on CPU)")

    response = input(f"\nProceed? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return

    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENTS ON APPLE GPU")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start_time = time.time()
    successful = 0
    failed = 0
    runtimes = []

    for i, job in enumerate(experiment_queue, 1):
        if args.resume:
            result_dir = Path(f"results_mps/{job['name']}/seed_{job['seed']}")
            if result_dir.exists() and (result_dir / 'messages').exists():
                print(f"[{i}/{total_runs}] Skipping {job['name']} seed {job['seed']} (completed)")
                successful += 1
                continue

        success, elapsed = run_experiment(
            job['name'],
            job['seed'],
            job['args'],
            total_runs,
            i
        )

        if success:
            successful += 1
            runtimes.append(elapsed)

            if len(runtimes) >= 3:
                avg_runtime = sum(runtimes) / len(runtimes)
                remaining = total_runs - i
                eta_seconds = avg_runtime * remaining
                eta_hours = eta_seconds / 3600
                print(f"  ETA for remaining {remaining} runs: ~{eta_hours:.1f} hours")
        else:
            failed += 1

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{total_runs}")
    print(f"Failed: {failed}/{total_runs}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg time per run: {total_time/total_runs/60:.1f} min")
    print(f"Speedup vs CPU: ~{(total_runs * cpu_time_per_run * 60) / total_time:.1f}x")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    summary_file = Path('results_mps/experiment_summary.json')
    summary_file.parent.mkdir(exist_ok=True)

    summary = {
        'total_runs': total_runs,
        'successful': successful,
        'failed': failed,
        'total_time_hours': total_time / 3600,
        'avg_time_per_run_minutes': total_time / total_runs / 60,
        'experiments': list(experiments.keys()),
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'device': 'MPS (Apple GPU)'
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    print(f"Results directory: results_mps/")

if __name__ == '__main__':
    main()
