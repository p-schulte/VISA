#!/bin/bash
set -euo pipefail

# Activate conda
source miniconda3/bin/activate
conda activate ac_dsg

# Template config files (your singletons)
AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# All datasets you want to sweep over
DATASETS=(
  # "blocks_1000"
  # "blocks_gen_test_1000"
  # "blocks_arity1_gen_test_1000"
  # "blocks_realistic_1000"
  # "blocks_realistic_gen_test_1000"
  # "blocks_interpolation_1000"
  # "blocks_interpolation_gen_test_1000"
  # "blocks_realistic_interpolation_1000"
  # "blocks_realistic_interpolation_gen_test_1000"
  # "hanoi_gripper_1000"
  # "hanoi_gripper_gen_test_2_1000"
  # "ball_hiding_1000"
  # "ball_hiding_gen_test_1000"
  # "ball_hiding_static_dsg_1000"
  # "dominoes_1000"
  # "grid_1000"
  # "gripper_1000"
  # "gripper_gen_test_1000"
  # "logistics_1000"      # PROBLEM
  # "logistics_gen_test_1000"
  # "slidetile_1000"
  # "slidetile_gen_test_1000"
  # "slidetile_with_positions_1000"
  # "sokoban_5x5_v2_1000"
  # "sokoban_5x5_v2_static_dsg_1000"
  # "sokoban_gen_test_1000"
  # "sokoban_gen_test_2_1000"
  # "grid_2_shapes_1000"
  # "grid_2_shapes_gen_test_1000"
  # "sokoban_new_format_1000"
  # "sokoban_new_format_gen_test_1000"
  # "sokoban_new_format_gen_test_2_1000"
  # "slidetile_with_positions_gen_test_1000"
  "sokoban_new_format_static_dsg_1000"
)


# Base log dirs
BASE_TRAIN_DIR="/work/rleap1/paul.schulte/logs/ac_train"
BASE_TEST_DIR="/work/rleap1/paul.schulte/logs/ac_test"

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=== DATASET: ${DATASET_NAME} ==="

    # Choose a run name (can be anything you like)
    RUN_NAME="final10" # "sweep_${DATASET_NAME}"

    # Per-dataset, per-run directories
    TRAIN_JOB_DIR="${BASE_TRAIN_DIR}/${DATASET_NAME}/${RUN_NAME}"
    TEST_JOB_DIR="${BASE_TEST_DIR}/${DATASET_NAME}/${RUN_NAME}"
    mkdir -p "${TRAIN_JOB_DIR}" "${TEST_JOB_DIR}"

    #### 1) CREATE TRAIN CONFIG SNAPSHOT ####

    AC_CFG_TRAIN="${TRAIN_JOB_DIR}/ac_config.yaml"
    DSG_CFG_TRAIN="${TRAIN_JOB_DIR}/dsg_config.yaml"
    DSET_CFG_TRAIN="${TRAIN_JOB_DIR}/dset_config.yaml"

    cp "${AC_SRC}"   "${AC_CFG_TRAIN}"
    cp "${DSG_SRC}"  "${DSG_CFG_TRAIN}"
    cp "${DSET_SRC}" "${DSET_CFG_TRAIN}"

    # Patch dataset + run name into the copied configs
    # (adjust keys if your YAML differs!)
    yq -y -i ".DATASET_NAME = \"${DATASET_NAME}\"" "${DSG_CFG_TRAIN}"
    yq -y -i ".RUN_NAME     = \"${RUN_NAME}\""     "${DSG_CFG_TRAIN}"
    # if your ac_config.yaml has a lowercase/other key for run name:
    yq -y -i ".run_name     = \"${RUN_NAME}\""     "${AC_CFG_TRAIN}"

    #### 2) SUBMIT TRAIN JOB ####

    TRAIN_JOB_ID=$(
        sbatch \
          --qos=rleap_deadline \
          --time=48:00:00 \
          --output="${TRAIN_JOB_DIR}/run.txt" \
          --export=ALL,AC_CONFIG_FILE="${AC_CFG_TRAIN}",DSG_CONFIG_FILE="${DSG_CFG_TRAIN}",DSET_CONFIG_FILE="${DSET_CFG_TRAIN}" \
          ac_dsg/slurm_jobs/ac_train_slurm.bash \
        | awk '{print $4}'
    )

    echo "Submitted TRAIN job ${TRAIN_JOB_ID} for ${DATASET_NAME}"

    #### 3) CREATE TEST CONFIG SNAPSHOT (OPTIONALLY IDENTICAL) ####

    AC_CFG_TEST="${TEST_JOB_DIR}/ac_config.yaml"
    DSG_CFG_TEST="${TEST_JOB_DIR}/dsg_config.yaml"
    DSET_CFG_TEST="${TEST_JOB_DIR}/dset_config.yaml"

    # You can reuse the exact same configs as train (most common):
    cp "${AC_CFG_TRAIN}"   "${AC_CFG_TEST}"
    cp "${DSG_CFG_TRAIN}"  "${DSG_CFG_TEST}"
    cp "${DSET_CFG_TRAIN}" "${DSET_CFG_TEST}"

    #### 4) SUBMIT TEST JOB THAT DEPENDS ON TRAIN JOB ####

    TEST_JOB_ID=$(
        sbatch \
          --qos=rleap_deadline \
          --time=05:00:00 \
          --dependency=afterany:${TRAIN_JOB_ID} \
          --output="${TEST_JOB_DIR}/run.txt" \
          --export=ALL,AC_CONFIG_FILE="${AC_CFG_TEST}",DSG_CONFIG_FILE="${DSG_CFG_TEST}",DSET_CONFIG_FILE="${DSET_CFG_TEST}" \
          ac_dsg/slurm_jobs/ac_test_slurm.bash \
        | awk '{print $4}'
    )

    echo "Submitted TEST  job ${TEST_JOB_ID} for ${DATASET_NAME} (depends on ${TRAIN_JOB_ID})"
    echo
done
echo "=== ALL SWEEP JOBS SUBMITTED ==="



# squeue -u $USER -o "%.12i %.30j %.10q %.10Q %.10Y %.10T"



# for id in $(squeue -u paul.schulte -n ac_train_slurm.bash -h -o %i); do
#     dataset=$(scontrol show job $id | awk -F'/' '/StdOut/ {print $(NF-2)}')
#     echo "$id → $dataset"
# done
