# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

DATASET_NAME=$(basename $(yq -r '.DATASET_NAME' ac_dsg/config/yaml_files/dsg_config.yaml))
sbatch --qos=cluster_deadline  --output="${LOG_ROOT}/fcnn_visualize/${DATASET_NAME}.txt" ac_dsg/slurm_jobs/fcnn_visualize.bash