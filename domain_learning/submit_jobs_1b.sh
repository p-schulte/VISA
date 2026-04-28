#!/bin/bash
#SBATCH --job-name=learning_from_states
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH --array=30-39
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gpus=0
#SBATCH --time=12:00:00

# Benchmarks
input_dir="./benchmark_learning_from_states"
input_files=("blocks.txt" "blocks4.txt" "delivery.txt" "driverlog.txt" "ferry.txt" "grid.txt" "gridlock.txt" "gripper.txt" "hanoi.txt" "logistics.txt" "miconic.txt" "npuzzle.txt" "cpuzzle.txt" "sokoban.txt" "sokoban_pull.txt")

file_index=$((SLURM_ARRAY_TASK_ID / 10))
run_index=$((SLURM_ARRAY_TASK_ID % 10))

input_file="${input_files[$file_index]}"
argument=$(head -n 1 "$input_dir/$input_file")

apptainer run --bind .:/graph-separator --bind /tmp:/tmp ../learning_from_states.sif /graph-separator/test.py $argument --run $run_index
exit_status=$?
exit $exit_status
