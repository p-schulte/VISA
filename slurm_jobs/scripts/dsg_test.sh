#!/bin/bash

# Activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Derive dataset name from DSG config
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")

# Job-specific directory for this DSG test
JOB_DIR="${LOG_ROOT}/dsg_test/${DATASET_NAME}"
mkdir -p "$JOB_DIR"

# Snapshot configs for THIS test job
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Submit DSG test job with frozen configs
sbatch \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/dsg_test_slurm.bash
# If you ever want a special queue:
#   --qos=cluster_deadline
