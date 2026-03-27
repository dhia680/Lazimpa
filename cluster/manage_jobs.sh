#!/bin/bash
# Helper script for managing LazImpa cluster jobs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

show_usage() {
    cat << EOF
LazImpa Cluster Job Manager

Usage: ./manage_jobs.sh [command]

Commands:
    test        Submit test job (single experiment, 10 epochs)
    all         Submit all experiments sequentially (1 GPU, 40-80 hours)
    parallel    Submit parallel array job (40 GPUs, 2-4 hours)
    single      Submit single experiment (interactive)
    status      Check job status
    results     Check completed experiments
    cancel      Cancel all running jobs
    logs        Show recent log output
    clean       Clean up old results/logs
    help        Show this help message

Examples:
    ./manage_jobs.sh test           # Quick test
    ./manage_jobs.sh parallel       # Run all experiments in parallel
    ./manage_jobs.sh status         # Check job status

EOF
}

check_status() {
    echo "========================================================================"
    echo "JOB STATUS"
    echo "========================================================================"
    qstat -u $USER
    echo ""

    # Count jobs by state
    RUNNING=$(qstat -u $USER | grep -c " r ")
    QUEUED=$(qstat -u $USER | grep -c " qw ")
    echo "Running: $RUNNING"
    echo "Queued: $QUEUED"
}

check_results() {
    echo "========================================================================"
    echo "COMPLETED EXPERIMENTS"
    echo "========================================================================"

    RESULTS_DIR="$HOME/Lazimpa/results"

    if [ ! -d "$RESULTS_DIR" ]; then
        echo "No results directory found at $RESULTS_DIR"
        return
    fi

    experiments=("lstm_baseline" "lstm_lazimpa" "transformer_baseline" "transformer_lazimpa")

    for exp in "${experiments[@]}"; do
        count=$(find "$RESULTS_DIR/$exp" -name "messages_*.npy" 2>/dev/null | wc -l)
        echo "$exp: $count/10 seeds completed"
    done

    echo ""
    total=$(find "$RESULTS_DIR" -name "messages_*.npy" 2>/dev/null | wc -l)
    echo "Total: $total/40 experiments completed"
}

show_logs() {
    echo "========================================================================"
    echo "RECENT LOG OUTPUT (last 50 lines)"
    echo "========================================================================"

    # Find most recent log file
    latest_log=$(ls -t logs/job_*.out 2>/dev/null | head -1)

    if [ -z "$latest_log" ]; then
        echo "No log files found in logs/"
    else
        echo "File: $latest_log"
        echo "---"
        tail -50 "$latest_log"
    fi
}

cancel_jobs() {
    echo "Canceling all running LazImpa jobs..."
    qstat -u $USER | grep lazimpa | awk '{print $1}' | xargs -r qdel
    echo "Done."
}

clean_results() {
    read -p "This will delete all results and logs. Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$HOME/Lazimpa/results/"
        rm -rf "$HOME/Lazimpa/cluster/logs/"
        echo "Cleaned up results and logs."
    else
        echo "Canceled."
    fi
}

submit_single() {
    echo "Available experiments:"
    echo "  1) lstm_baseline"
    echo "  2) lstm_lazimpa"
    echo "  3) transformer_baseline"
    echo "  4) transformer_lazimpa"
    echo ""
    read -p "Select experiment (1-4): " exp_num

    case $exp_num in
        1) EXPERIMENT="lstm_baseline" ;;
        2) EXPERIMENT="lstm_lazimpa" ;;
        3) EXPERIMENT="transformer_baseline" ;;
        4) EXPERIMENT="transformer_lazimpa" ;;
        *) echo "Invalid selection"; return ;;
    esac

    read -p "Enter seed (default: 42): " SEED
    SEED=${SEED:-42}

    echo "Submitting: $EXPERIMENT with seed $SEED"
    cd "$SCRIPT_DIR"
    qsub -v EXPERIMENT=$EXPERIMENT,SEED=$SEED run_single_experiment.sh
}

# Main command dispatcher
case "${1:-help}" in
    test)
        echo "Submitting test job..."
        cd "$SCRIPT_DIR"
        qsub test_single.sh
        ;;
    all)
        echo "Submitting all experiments (sequential)..."
        cd "$SCRIPT_DIR"
        qsub run_all_experiments.sh
        ;;
    parallel)
        echo "Submitting parallel array job (40 tasks)..."
        cd "$SCRIPT_DIR"
        mkdir -p logs
        qsub run_parallel_array.sh
        ;;
    single)
        submit_single
        ;;
    status)
        check_status
        ;;
    results)
        check_results
        ;;
    logs)
        show_logs
        ;;
    cancel)
        cancel_jobs
        ;;
    clean)
        clean_results
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
