# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

DATASET_NAME=$(basename $(yq -r '.DATASET_NAME' ac_dsg/config/yaml_files/dsg_config.yaml))
if [ ! -d "${LOG_ROOT}/ac_train/${DATASET_NAME}" ]; then
    echo "Directory ${LOG_ROOT}/ac_train/${DATASET_NAME} does not exist. Exiting."
    exit 1
fi

sbatch --output="${LOG_ROOT}/misc/${DATASET_NAME}_viz.txt" ac_dsg/slurm_jobs/training_viz_ac_train_slurm.bash
