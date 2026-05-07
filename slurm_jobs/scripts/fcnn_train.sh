#!/bin/bash

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Derive dataset name from DSG config
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")

# Job-specific directory for this FCNN training run
JOB_DIR="${LOG_ROOT}/fcnn_train/${DATASET_NAME}"
mkdir -p "$JOB_DIR"

# Snapshot configs for THIS job
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Submit FCNN training job with frozen configs
sbatch \
  --time=25:00:00 \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/fcnn_train_slurm.bash
# Optionally, if you want to use the deadline queue:
#  --qos=cluster_deadline
