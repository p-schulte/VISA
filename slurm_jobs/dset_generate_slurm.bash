#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=16           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_48gb  # Partition (queue) to use
#SBATCH --output=slurm-%x-%j.out  # Temporary placeholder


# activate conda environment
source miniconda3/bin/activate
conda activate ac_dataset

#  setup
cd ac_dsg/dataset

# start training process
python -u generate.py
# python -u visualize_distributions.py