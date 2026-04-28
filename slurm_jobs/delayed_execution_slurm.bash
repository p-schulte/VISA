#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --gpus-per-node=0           # GPUs per node
#SBATCH --mem=1g                   # Memory allocation
#SBATCH --partition=rleap_gpu_48gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/paul.schulte/logs/misc/delayed_execution.txt  # Temporary placeholder




# Wait for 2 hours (2 * 60 * 60 = 7200 seconds)
echo "Waiting for 2 hours..."
sleep 7200

# Run your command here
./scripts/fcnn_train.sh
