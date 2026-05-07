#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=cluster_gpu_24gb  # Partition (queue) to use
#SBATCH --output=slurm-%x-%j.out  # Output log file

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

#  setup
cd ac_dsg/dsg_generator
export PYTHONPATH=$(pwd)


# Set correct CUDA architecture
export CUDA_HOME=/usr/local/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


nvcc --allow-unsupported-compiler --version



pip uninstall faster_rcnn -y
cd fasterRCNN/lib
rm -rf build/
rm model/_C.cpython-39-x86_64-linux-gnu.so
python setup.py clean
python setup.py build develop

