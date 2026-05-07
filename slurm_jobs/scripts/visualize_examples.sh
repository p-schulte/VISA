#!/bin/bash

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

LOG_ROOT="${LOG_ROOT:-./logs}"

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Derive dataset + run name from config files
DATASET_DIR=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")
RUN_NAME_RAW=$(yq -r '.run_name' "$AC_SRC")

if [ -n "$RUN_NAME_RAW" ] && [ "$RUN_NAME_RAW" != "null" ]; then
    DATASET_DIR="${DATASET_DIR}/${RUN_NAME_RAW}"
else
    DATASET_DIR="${DATASET_DIR}/default"
fi

# Create a dedicated log directory
JOB_DIR="${LOG_ROOT}/ac_results/${DATASET_DIR}/example_visualization"
mkdir -p "$JOB_DIR"

# Snapshot current configs for this visualization job
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Submit SLURM job with frozen configs
sbatch \
  --qos=rleap_deadline \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/visualize_examples_slurm.bash
# Optionally enable deadline queue:
#  --qos=rleap_deadline
