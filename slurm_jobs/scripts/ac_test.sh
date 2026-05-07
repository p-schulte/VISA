#!/bin/bash

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Derive dataset + run name from source config
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")
RUN_NAME_RAW=$(yq -r '.run_name' "$AC_SRC")

BASE_DIR="${LOG_ROOT}/ac_test/${DATASET_NAME}"

if [ -n "$RUN_NAME_RAW" ] && [ "$RUN_NAME_RAW" != "null" ]; then
    JOB_DIR="${BASE_DIR}/${RUN_NAME_RAW}"
else
    JOB_DIR="${BASE_DIR}/default"
fi

mkdir -p "$JOB_DIR"

# Snapshot configs for THIS test job
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Submit test job with frozen configs
sbatch \
  --time=5:00:00 \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/ac_test_slurm.bash
#   --qos=rleap_deadline
