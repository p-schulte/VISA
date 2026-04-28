#!/bin/bash

# Activate Conda
source miniconda3/bin/activate
conda activate ac_dsg

# Base config paths (the "master" configs you edit by hand)
AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Dataset name from DSG config (same as before)
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")

# --- Define the sweep here ----------------------------------------------------

# Learning rates to test
LRS=("0.0001" "0.0003" "0.001")

# Batch sizes to test
BATCH_SIZES=(1 4 8)

# (You can add more things, e.g.
# OPTIMIZERS=("Adam" "AdamW")
# feature methods, etc.)

# ------------------------------------------------------------------------------

for LR in "${LRS[@]}"; do
  for BS in "${BATCH_SIZES[@]}"; do

    # Create a unique run name/tag for this combo
    RUN_TAG="lr${LR}_bs${BS}"

    # Directory for this particular experiment
    JOB_DIR="/work/rleap1/paul.schulte/logs/ac_train/${DATASET_NAME}/${RUN_TAG}"
    mkdir -p "$JOB_DIR"

    echo "Launching experiment: ${RUN_TAG}"

    # Snapshot configs
    cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
    cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
    cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

    # Modify the copied ac_config.yaml for this experiment
    # 1) Set run_name so it’s visible inside the code/logs
    yq -i ".run_name = \"${RUN_TAG}\"" "${JOB_DIR}/ac_config.yaml"

    # 2) Adjust learning rate
    yq -i ".optimizer_params.Adam.learning_rate = ${LR}" "${JOB_DIR}/ac_config.yaml"

    # 3) Adjust batch size
    yq -i ".batch_size = ${BS}" "${JOB_DIR}/ac_config.yaml"

    # (Example if you want to flip something else:)
    # yq -i ".use_early_stopping = false" "${JOB_DIR}/ac_config.yaml"

    # Submit job with *this* config snapshot
    sbatch \
      --output="${JOB_DIR}/run.txt" \
      --export=ALL,AC_CONFIG_FILE="${JOB_DIR}/ac_config.yaml",DSG_CONFIG_FILE="${JOB_DIR}/dsg_config.yaml",DSET_CONFIG_FILE="${JOB_DIR}/dset_config.yaml" \
      ac_dsg/slurm_jobs/ac_train_slurm.bash

  done
done
