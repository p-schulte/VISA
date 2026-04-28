#!/bin/bash

# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# Derive dataset name from DSG config
DATASET_NAME=$(basename "$(yq -r '.DATASET_NAME' "$DSG_SRC")")

# Base dir for this FCNN test run
JOB_DIR="/work/rleap1/paul.schulte/logs/fcnn_test/${DATASET_NAME}"
mkdir -p "$JOB_DIR"

# Snapshot configs for BOTH jobs (test set + train set)
cp "$AC_SRC"   "${JOB_DIR}/ac_config.yaml"
cp "$DSG_SRC"  "${JOB_DIR}/dsg_config.yaml"
cp "$DSET_SRC" "${JOB_DIR}/dset_config.yaml"

# Common export string so both sbatch calls get the same frozen configs
EXPORT_VARS="ALL,AC_CONFIG_FILE=${JOB_DIR}/ac_config.yaml,DSG_CONFIG_FILE=${JOB_DIR}/dsg_config.yaml,DSET_CONFIG_FILE=${JOB_DIR}/dset_config.yaml"

echo "Testing the FCNN on the test set"

# Submit the first job (test set)
JOB_ID=$(sbatch \
    --export="$EXPORT_VARS" \
    --output="${JOB_DIR}/${DATASET_NAME}_test_set.txt" \
    ac_dsg/slurm_jobs/fcnn_test_slurm.bash | awk '{print $4}')

echo "Submitted job with ID $JOB_ID"

# Wait for the first job to finish
while true; do
    STATE=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | awk 'NF {print $1}' | tr -d '\n')

    if [[ -z "$STATE" ]]; then
        echo "Job $JOB_ID not found in sacct yet. Waiting..."
    elif [[ "$STATE" == COMPLETED* || "$STATE" == FAILED* || "$STATE" == CANCELLED* ]]; then
        echo "Job $JOB_ID finished with state: $STATE"
        break
    else
        echo "Job $JOB_ID current state: $STATE"
    fi

    sleep 10
done

echo "Job $JOB_ID is finished!"
echo "Testing the FCNN on the train set"

# Submit the second job (train set) with the same frozen configs
sbatch \
  --export="$EXPORT_VARS" \
  --output="${JOB_DIR}/${DATASET_NAME}_train_set.txt" \
  ac_dsg/slurm_jobs/fcnn_test_training_set_slurm.bash
