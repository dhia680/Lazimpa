#!/usr/bin/env python
"""
Local GPU experiment runner with queue management.
Runs experiments sequentially on available GPU(s) with progress tracking.
"""

import json
import os
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime
import sys

def generate_command_args(experiment_name, config, common_params, arch_config, seed, gpu_id=0):
    """Generate training command arguments for a single experiment run."""

    # Merge all configs
    params = {**common_params, **arch_config, **config}

    # Build argument list
    args = []

    # Add all parameters
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}={value}")

    # Add experiment-specific params
    args.append(f"--random_seed={seed}")
    args.append(f"--dir_save=results/{experiment_name}/seed_{seed}")
    args.append(f"--name={experiment_name}_seed{seed}")

    return args

def run_experiment(exp_name, seed, args, gpu_id, total_runs, current_run):
    """Run a single experiment with GPU and logging."""

    # Set GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create log directory
    log_dir = Path(f'results/{exp_name}/seed_{seed}/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'

    # Print progress
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*70}")
    print(f"[{timestamp}] Run {current_run}/{total_runs}")
    print(f"Experiment: {exp_name} | Seed: {seed} | GPU: {gpu_id}")
    print(f"Log: {log_file}")
    print(f"{'='*70}\n")

    # Run experiment
    start_time = time.time()

    with open(log_file, 'w') as f:
        # Write header
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Started: {timestamp}\n")
        f.write(f"{'='*70}\n\n")
        f.flush()

        # Run training
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
        return True
    else:
        print(f"✗ Failed with return code {result.returncode} after {elapsed_str}")
        print(f"  Check log: {log_file}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments_config.json',
                       help='Path to experiments configuration JSON')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print experiment plan without running')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already completed experiments')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    common_params = config['common_params']
    arch_configs = config['architecture_configs']
    experiments = config['experiments']
    seeds = config['seeds']

    # Filter experiments if specified
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}

    # Build experiment queue
    experiment_queue = []
    for exp_name, exp_config in experiments.items():
        # Make a copy to avoid modifying the original
        exp_config_copy = exp_config.copy()
        arch_name = exp_config_copy.pop('architecture')
        arch_config = arch_configs[arch_name]

        for seed in seeds:
            cmd_args = generate_command_args(exp_name, exp_config_copy, common_params, arch_config, seed, args.gpu)
            experiment_queue.append({
                'name': exp_name,
                'seed': seed,
                'args': cmd_args,
                'description': exp_config.get('description', '')
            })

    total_runs = len(experiment_queue)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT QUEUE SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs: {total_runs} ({len(experiments)} experiments × {len(seeds)} seeds)")
    print(f"GPU: {args.gpu}")
    print(f"{'='*70}\n")

    for exp in experiments.keys():
        print(f"  • {exp}: {len(seeds)} seeds")

    if args.dry_run:
        print("\n[DRY RUN] - No experiments will be executed")
        for i, job in enumerate(experiment_queue, 1):
            print(f"{i:3d}. {job['name']:<25s} seed={job['seed']}")
        return

    # Confirm before starting
    print(f"\nEstimated runtime: ~{total_runs * 2} hours (assuming ~2h per run)")
    response = input("\nProceed with experiments? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return

    # Run experiments
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start_time = time.time()
    successful = 0
    failed = 0

    for i, job in enumerate(experiment_queue, 1):
        # Check if already completed (resume mode)
        if args.resume:
            result_dir = Path(f"results/{job['name']}/seed_{job['seed']}")
            if result_dir.exists() and (result_dir / 'messages').exists():
                print(f"[{i}/{total_runs}] Skipping {job['name']} seed {job['seed']} (already completed)")
                successful += 1
                continue

        # Run experiment
        success = run_experiment(
            job['name'],
            job['seed'],
            job['args'],
            args.gpu,
            total_runs,
            i
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{total_runs}")
    print(f"Failed: {failed}/{total_runs}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Save summary
    summary_file = Path('results/experiment_summary.json')
    summary_file.parent.mkdir(exist_ok=True)

    summary = {
        'total_runs': total_runs,
        'successful': successful,
        'failed': failed,
        'total_time_hours': total_time / 3600,
        'experiments': list(experiments.keys()),
        'seeds': seeds,
        'timestamp': datetime.now().isoformat()
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
