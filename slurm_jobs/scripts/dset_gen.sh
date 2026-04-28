#!/bin/bash

# Activate Conda environment
source miniconda3/bin/activate
conda activate ac_dsg

DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Get relative domain_file path from dset_config
DOMAIN_REL_PATH=$(yq -r '.domain_file' "$DSET_SRC")
DOMAIN_SRC="ac_dsg/dataset/${DOMAIN_REL_PATH}"

# Extract identifiers from domain JSON
DOMAIN_NAME=$(jq -r '.domain_name' "$DOMAIN_SRC")
NAME_EXTENSION=$(jq -r '.name_extension' "$DOMAIN_SRC")

# Construct dataset and domain file names
DATASET_NAME="${DOMAIN_NAME}_${NAME_EXTENSION}"
DOMAIN_FILE_NAME="${DOMAIN_NAME}_${NAME_EXTENSION}.json"

# Create a job-specific directory
JOB_DIR="/work/rleap1/paul.schulte/logs/dataset_generation/${DATASET_NAME}"
mkdir -p "$JOB_DIR"

# Copy and rename the domain JSON (unique per dataset)
cp "$DOMAIN_SRC" "${JOB_DIR}/${DOMAIN_FILE_NAME}"

# Copy dset_config.yaml and patch it so domain_file points to the copied JSON
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"
# yq -y -i ".domain_file = \"${DOMAIN_FILE_NAME}\"" "${JOB_DIR}/dset_config.yaml"
yq -y -i ".domain_file = \"${DOMAIN_REL_PATH}\"" "${JOB_DIR}/dset_config.yaml"

# Submit SLURM job with frozen config
sbatch \
  --time=05:00:00 \
  --output="${JOB_DIR}/run.txt" \
  --export=ALL,DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
  ac_dsg/slurm_jobs/dset_generate_slurm.bash
