#!/bin/bash

# activate conda environment for yq etc. (if needed here)
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Compute dataset name and run name based on *source* config
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")
RUN_NAME_RAW=$(yq -r '.run_name' "$AC_SRC")

BASE_DIR="${LOG_ROOT}/ac_train/${DATASET_NAME}"

# Fall back to something sane if run_name is empty/null
if [ -n "$RUN_NAME_RAW" ] && [ "$RUN_NAME_RAW" != "null" ]; then
    JOB_DIR="${BASE_DIR}/${RUN_NAME_RAW}"
else
    JOB_DIR="${BASE_DIR}/default"
fi

mkdir -p "$JOB_DIR"

# Snapshot configs for THIS job
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Submit job, telling it to use the copied configs
sbatch \
  --time=25:00:00 \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/ac_train_slurm.bash
# If you want the deadline queue, add this flag and move it up to be the first flag (does not work at the end):
#   --qos=cluster_deadline
