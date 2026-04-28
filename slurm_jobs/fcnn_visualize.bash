#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_48gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/paul.schulte/logs/fcnn_visualize/fcnn_visualize_log.txt  # Output log file

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

#  setup
cd ac_dsg/dsg_generator
./clean_cache.sh
export PYTHONPATH=$(pwd)

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"


# start training process
python -u fasterRCNN/test_net.py --dataset blocks_voc --net res101 --cuda --vis
