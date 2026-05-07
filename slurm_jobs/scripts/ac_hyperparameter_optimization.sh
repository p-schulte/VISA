# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

RUN_NAME=$(basename $(yq -r '.run_name' ac_dsg/config/yaml_files/ac_config.yaml))
mkdir "${LOG_ROOT}/ac_hyperparameter_optimization/${RUN_NAME}"
cp ac_dsg/config/yaml_files/ac_config.yaml "${LOG_ROOT}/ac_hyperparameter_optimization/${RUN_NAME}/ac_config.yaml"
cp ac_dsg/config/yaml_files/dsg_config.yaml "${LOG_ROOT}/ac_hyperparameter_optimization/${RUN_NAME}/dsg_config.yaml"

sbatch --output="${LOG_ROOT}/ac_hyperparameter_optimization/${RUN_NAME}/run.txt" ac_dsg/slurm_jobs/ac_hyperparameter_optimization_slurm.bash
