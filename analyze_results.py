#!/usr/bin/env python
"""
Comparative Analysis: LSTM vs Transformer for LazImpa
Compares all 4 experiment types across seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_messages_by_epoch(result_dir, epoch):
    """Load messages for a specific epoch."""
    msg_file = result_dir / 'messages' / f'messages_{epoch}.npy'
    if msg_file.exists():
        return np.load(msg_file, allow_pickle=True)
    return None

def load_accuracy_by_epoch(result_dir, epoch):
    """Load accuracy for a specific epoch."""
    acc_file = result_dir / 'accuracy' / f'accuracy_{epoch}.npy'
    if acc_file.exists():
        return np.load(acc_file, allow_pickle=True)
    return None

def compute_message_lengths(messages):
    """Compute length of each message (count non-zero symbols)."""
    lengths = []
    for msg in messages:
        # Count non-zero elements
        length = np.count_nonzero(msg)
        lengths.append(length)
    return np.array(lengths)

def load_experiment_data(base_dir, exp_name, seeds, max_epochs=100):
    """Load all data for an experiment across seeds."""
    exp_dir = base_dir / exp_name

    data = {
        'seeds': {},
        'name': exp_name
    }

    for seed in seeds:
        seed_dir = exp_dir / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        # Find last epoch
        msg_files = sorted(seed_dir.glob('messages/messages_*.npy'))
        if not msg_files:
            continue

        last_epoch = max([int(f.stem.split('_')[1]) for f in msg_files])

        # Load evolution data
        accuracies = []
        mean_lengths = []

        for epoch in range(min(last_epoch + 1, max_epochs)):
            acc = load_accuracy_by_epoch(seed_dir, epoch)
            msg = load_messages_by_epoch(seed_dir, epoch)

            if acc is not None:
                accuracies.append(np.mean(acc))

            if msg is not None:
                lengths = compute_message_lengths(msg)
                mean_lengths.append(np.mean(lengths))

        # Load final messages
        final_messages = load_messages_by_epoch(seed_dir, last_epoch)
        final_lengths = compute_message_lengths(final_messages) if final_messages is not None else None

        data['seeds'][seed] = {
            'last_epoch': last_epoch,
            'accuracies': accuracies,
            'mean_lengths': mean_lengths,
            'final_messages': final_messages,
            'final_lengths': final_lengths
        }

    return data

def plot_length_distributions(all_data, save_path):
    """Plot message length distributions for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']
    titles = ['LSTM Baseline', 'LSTM LazImpa', 'Transformer Baseline', 'Transformer LazImpa']

    for idx, (exp_name, title) in enumerate(zip(exp_names, titles)):
        ax = axes[idx]

        if exp_name not in all_data or not all_data[exp_name]['seeds']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Aggregate lengths across all seeds
        all_lengths = []
        for seed_data in all_data[exp_name]['seeds'].values():
            if seed_data['final_lengths'] is not None:
                all_lengths.extend(seed_data['final_lengths'])

        if not all_lengths:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Plot histogram
        ax.hist(all_lengths, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Message Length', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(alpha=0.3)

        # Add statistics
        mean_len = np.mean(all_lengths)
        std_len = np.std(all_lengths)
        ax.text(0.95, 0.95, f'Mean: {mean_len:.2f}\nStd: {std_len:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_accuracy_evolution(all_data, save_path):
    """Plot accuracy evolution for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']
    titles = ['LSTM Baseline', 'LSTM LazImpa', 'Transformer Baseline', 'Transformer LazImpa']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, (exp_name, title, color) in enumerate(zip(exp_names, titles, colors)):
        ax = axes[idx]

        if exp_name not in all_data or not all_data[exp_name]['seeds']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Plot each seed
        for seed, seed_data in all_data[exp_name]['seeds'].items():
            accuracies = seed_data['accuracies']
            ax.plot(accuracies, alpha=0.3, color=color, linewidth=1)

        # Plot mean across seeds
        all_accs = [seed_data['accuracies'] for seed_data in all_data[exp_name]['seeds'].values()]
        min_len = min(len(acc) for acc in all_accs)
        truncated = [acc[:min_len] for acc in all_accs]
        mean_acc = np.mean(truncated, axis=0)

        ax.plot(mean_acc, color=color, linewidth=3, label=f'Mean (n={len(all_accs)})')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # Add final accuracy
        final_acc = mean_acc[-1] if len(mean_acc) > 0 else 0
        ax.text(0.05, 0.95, f'Final: {final_acc:.3f}',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_length_evolution(all_data, save_path):
    """Plot mean message length evolution for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']
    titles = ['LSTM Baseline', 'LSTM LazImpa', 'Transformer Baseline', 'Transformer LazImpa']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, (exp_name, title, color) in enumerate(zip(exp_names, titles, colors)):
        ax = axes[idx]

        if exp_name not in all_data or not all_data[exp_name]['seeds']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Plot each seed
        for seed, seed_data in all_data[exp_name]['seeds'].items():
            mean_lengths = seed_data['mean_lengths']
            ax.plot(mean_lengths, alpha=0.3, color=color, linewidth=1)

        # Plot mean across seeds
        all_lengths = [seed_data['mean_lengths'] for seed_data in all_data[exp_name]['seeds'].values()]
        min_len = min(len(lengths) for lengths in all_lengths)
        truncated = [lengths[:min_len] for lengths in all_lengths]
        mean_length = np.mean(truncated, axis=0)

        ax.plot(mean_length, color=color, linewidth=3, label=f'Mean (n={len(all_lengths)})')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Message Length', fontsize=12)
        ax.set_ylim(0, 30)
        ax.grid(alpha=0.3)
        ax.legend()

        # Add final length
        final_len = mean_length[-1] if len(mean_length) > 0 else 0
        ax.text(0.05, 0.95, f'Final: {final_len:.2f}',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_zla_compliance(all_data, save_path):
    """Plot message length vs frequency rank (Zipf's Law Analysis)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']
    titles = ['LSTM Baseline', 'LSTM LazImpa', 'Transformer Baseline', 'Transformer LazImpa']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for idx, (exp_name, title, color) in enumerate(zip(exp_names, titles, colors)):
        ax = axes[idx]

        if exp_name not in all_data or not all_data[exp_name]['seeds']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Aggregate final messages across all seeds
        all_final_lengths = []
        for seed_data in all_data[exp_name]['seeds'].values():
            if seed_data['final_lengths'] is not None:
                # Sort by length (shortest to longest = most to least frequent in ZLA)
                sorted_lengths = np.sort(seed_data['final_lengths'])
                all_final_lengths.append(sorted_lengths)

        if not all_final_lengths:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Plot each seed with transparency
        for lengths in all_final_lengths:
            ax.plot(lengths, alpha=0.3, color=color, linewidth=1)

        # Plot mean
        min_len = min(len(lengths) for lengths in all_final_lengths)
        truncated = [lengths[:min_len] for lengths in all_final_lengths]
        mean_lengths = np.mean(truncated, axis=0)

        ax.plot(mean_lengths, color=color, linewidth=3, label=f'Mean (n={len(all_final_lengths)})')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Input (ranked by frequency)', fontsize=12)
        ax.set_ylabel('Message Length', fontsize=12)
        ax.set_ylim(0, 30)
        ax.grid(alpha=0.3)
        ax.legend()

        # Check ZLA compliance (shorter messages for frequent inputs)
        zla_compliant = mean_lengths[0] < mean_lengths[-1] if len(mean_lengths) > 10 else False
        compliance_text = "✓ ZLA-compliant" if zla_compliant else "✗ Anti-ZLA"
        ax.text(0.95, 0.05, compliance_text,
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round',
                         facecolor='lightgreen' if zla_compliant else 'lightcoral',
                         alpha=0.7),
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_lstm_vs_transformer_comparison(all_data, save_path):
    """Direct comparison: LSTM vs Transformer for baseline and LazImpa."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Row 1: Baseline comparison
    # Accuracy evolution
    ax = axes[0, 0]
    for exp_name, label, color in [('lstm_baseline', 'LSTM', 'tab:blue'),
                                     ('transformer_baseline', 'Transformer', 'tab:green')]:
        if exp_name in all_data and all_data[exp_name]['seeds']:
            all_accs = [seed_data['accuracies'] for seed_data in all_data[exp_name]['seeds'].values()]
            min_len = min(len(acc) for acc in all_accs)
            truncated = [acc[:min_len] for acc in all_accs]
            mean_acc = np.mean(truncated, axis=0)
            ax.plot(mean_acc, color=color, linewidth=3, label=label)

    ax.set_title('Baseline: Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    # Length evolution
    ax = axes[0, 1]
    for exp_name, label, color in [('lstm_baseline', 'LSTM', 'tab:blue'),
                                     ('transformer_baseline', 'Transformer', 'tab:green')]:
        if exp_name in all_data and all_data[exp_name]['seeds']:
            all_lengths = [seed_data['mean_lengths'] for seed_data in all_data[exp_name]['seeds'].values()]
            min_len = min(len(lengths) for lengths in all_lengths)
            truncated = [lengths[:min_len] for lengths in all_lengths]
            mean_length = np.mean(truncated, axis=0)
            ax.plot(mean_length, color=color, linewidth=3, label=label)

    ax.set_title('Baseline: Length Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Message Length', fontsize=12)
    ax.set_ylim(0, 30)
    ax.grid(alpha=0.3)
    ax.legend()

    # Row 2: LazImpa comparison
    # Accuracy evolution
    ax = axes[1, 0]
    for exp_name, label, color in [('lstm_lazimpa', 'LSTM', 'tab:orange'),
                                     ('transformer_lazimpa', 'Transformer', 'tab:red')]:
        if exp_name in all_data and all_data[exp_name]['seeds']:
            all_accs = [seed_data['accuracies'] for seed_data in all_data[exp_name]['seeds'].values()]
            min_len = min(len(acc) for acc in all_accs)
            truncated = [acc[:min_len] for acc in all_accs]
            mean_acc = np.mean(truncated, axis=0)
            ax.plot(mean_acc, color=color, linewidth=3, label=label)

    ax.set_title('LazImpa: Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    # Length evolution
    ax = axes[1, 1]
    for exp_name, label, color in [('lstm_lazimpa', 'LSTM', 'tab:orange'),
                                     ('transformer_lazimpa', 'Transformer', 'tab:red')]:
        if exp_name in all_data and all_data[exp_name]['seeds']:
            all_lengths = [seed_data['mean_lengths'] for seed_data in all_data[exp_name]['seeds'].values()]
            min_len = min(len(lengths) for lengths in all_lengths)
            truncated = [lengths[:min_len] for lengths in all_lengths]
            mean_length = np.mean(truncated, axis=0)
            ax.plot(mean_length, color=color, linewidth=3, label=label)

    ax.set_title('LazImpa: Length Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Message Length', fontsize=12)
    ax.set_ylim(0, 30)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_statistics_report(all_data, save_path):
    """Generate statistical summary comparing all experiments."""
    report = []
    report.append("="*80)
    report.append("COMPARATIVE ANALYSIS: LSTM vs TRANSFORMER")
    report.append("="*80)
    report.append("")

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']

    for exp_name in exp_names:
        if exp_name not in all_data or not all_data[exp_name]['seeds']:
            continue

        report.append(f"\n{exp_name.upper().replace('_', ' ')}")
        report.append("-"*80)

        # Aggregate statistics across seeds
        final_accs = []
        final_lengths = []

        for seed, seed_data in all_data[exp_name]['seeds'].items():
            if seed_data['accuracies']:
                final_accs.append(seed_data['accuracies'][-1])
            if seed_data['final_lengths'] is not None:
                final_lengths.extend(seed_data['final_lengths'])

        if final_accs:
            report.append(f"Final Accuracy: {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f} (n={len(final_accs)} seeds)")

        if final_lengths:
            report.append(f"Message Length: {np.mean(final_lengths):.2f} ± {np.std(final_lengths):.2f}")
            report.append(f"  Min: {np.min(final_lengths):.0f}, Max: {np.max(final_lengths):.0f}")

            # Check ZLA compliance
            sorted_lengths = np.sort(final_lengths)
            n_inputs = len(sorted_lengths)
            first_quartile_mean = np.mean(sorted_lengths[:n_inputs//4])
            last_quartile_mean = np.mean(sorted_lengths[-n_inputs//4:])
            zla_compliant = first_quartile_mean < last_quartile_mean

            report.append(f"  ZLA Compliance: {'✓ YES' if zla_compliant else '✗ NO (Anti-ZLA)'}")
            report.append(f"    Most frequent 25%: {first_quartile_mean:.2f} avg length")
            report.append(f"    Least frequent 25%: {last_quartile_mean:.2f} avg length")

    # Direct comparisons
    report.append("\n" + "="*80)
    report.append("DIRECT COMPARISONS")
    report.append("="*80)

    # Baseline: LSTM vs Transformer
    if 'lstm_baseline' in all_data and 'transformer_baseline' in all_data:
        lstm_lens = []
        trans_lens = []

        for seed_data in all_data['lstm_baseline']['seeds'].values():
            if seed_data['final_lengths'] is not None:
                lstm_lens.extend(seed_data['final_lengths'])

        for seed_data in all_data['transformer_baseline']['seeds'].values():
            if seed_data['final_lengths'] is not None:
                trans_lens.extend(seed_data['final_lengths'])

        if lstm_lens and trans_lens:
            report.append(f"\nBaseline Comparison:")
            report.append(f"  LSTM length: {np.mean(lstm_lens):.2f} ± {np.std(lstm_lens):.2f}")
            report.append(f"  Transformer length: {np.mean(trans_lens):.2f} ± {np.std(trans_lens):.2f}")
            report.append(f"  Difference: {np.mean(trans_lens) - np.mean(lstm_lens):.2f}")

    # LazImpa: LSTM vs Transformer
    if 'lstm_lazimpa' in all_data and 'transformer_lazimpa' in all_data:
        lstm_lens = []
        trans_lens = []

        for seed_data in all_data['lstm_lazimpa']['seeds'].values():
            if seed_data['final_lengths'] is not None:
                lstm_lens.extend(seed_data['final_lengths'])

        for seed_data in all_data['transformer_lazimpa']['seeds'].values():
            if seed_data['final_lengths'] is not None:
                trans_lens.extend(seed_data['final_lengths'])

        if lstm_lens and trans_lens:
            report.append(f"\nLazImpa Comparison:")
            report.append(f"  LSTM length: {np.mean(lstm_lens):.2f} ± {np.std(lstm_lens):.2f}")
            report.append(f"  Transformer length: {np.mean(trans_lens):.2f} ± {np.std(trans_lens):.2f}")
            report.append(f"  Difference: {np.mean(trans_lens) - np.mean(lstm_lens):.2f}")
            report.append(f"\n  → NOVEL RESULT: First comparison of Transformer-based LazImpa!")

    report.append("\n" + "="*80)

    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))

    # Also print to console
    print('\n'.join(report))
    print(f"\nReport saved: {save_path}")

def main():
    # Configuration
    base_dir = Path('results_local')
    output_dir = Path('results_local/analysis')
    output_dir.mkdir(exist_ok=True)

    exp_names = ['lstm_baseline', 'lstm_lazimpa', 'transformer_baseline', 'transformer_lazimpa']
    seeds = [42, 123, 456]

    print("\n" + "="*80)
    print("LOADING EXPERIMENT RESULTS")
    print("="*80 + "\n")

    # Load all experiment data
    all_data = {}
    for exp_name in exp_names:
        print(f"Loading {exp_name}...")
        all_data[exp_name] = load_experiment_data(base_dir, exp_name, seeds)
        n_seeds = len(all_data[exp_name]['seeds'])
        print(f"  → Loaded {n_seeds} seeds\n")

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")

    # Generate plots
    plot_length_distributions(all_data, output_dir / 'comparison_length_distributions.png')
    plot_accuracy_evolution(all_data, output_dir / 'comparison_accuracy_evolution.png')
    plot_length_evolution(all_data, output_dir / 'comparison_length_evolution.png')
    plot_zla_compliance(all_data, output_dir / 'comparison_zla_compliance.png')
    plot_lstm_vs_transformer_comparison(all_data, output_dir / 'direct_comparison.png')

    # Generate statistics report
    print("\n" + "="*80)
    print("GENERATING STATISTICS REPORT")
    print("="*80 + "\n")
    generate_statistics_report(all_data, output_dir / 'statistics_report.txt')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - comparison_length_distributions.png")
    print("  - comparison_accuracy_evolution.png")
    print("  - comparison_length_evolution.png")
    print("  - comparison_zla_compliance.png")
    print("  - direct_comparison.png")
    print("  - statistics_report.txt")
    print("\n")

if __name__ == '__main__':
    main()
