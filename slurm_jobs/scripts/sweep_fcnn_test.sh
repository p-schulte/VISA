#!/bin/bash
set -euo pipefail

# Activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# All datasets to sweep over
DATASETS=(
  # === BALL HIDING ===
  ball_hiding_1000
  ball_hiding_static_dsg_1000
  ball_hiding_gen_test_1000

  # === BLOCKSWORLD ===
  blocks_1000
  blocks_gen_test_1000
  blocks_arity1_gen_test_1000
  blocks_realistic_1000
  blocks_realistic_gen_test_1000
  blocks_interpolation_1000
  blocks_interpolation_gen_test_1000
  blocks_realistic_interpolation_1000
  blocks_realistic_interpolation_gen_test_1000

  # === DOMINOES ===
  dominoes_1000

  # === GRID ===
  grid_1000
  grid_gen_test_1000
  grid_2_shapes_1000
  grid_2_shapes_gen_test_1000

  # === GRIPPER ===
  gripper_1000
  gripper_gen_test_1000

  # === HANOI GRIPPER ===
  hanoi_gripper_1000
  hanoi_gripper_gen_test_2_1000

  # === LOGISTICS ===
  logistics_1000
  logistics_gen_test_1000

  # === SLIDETILE ===
  slidetile_1000
  slidetile_gen_test_1000
  slidetile_with_positions_1000
  slidetile_with_positions_gen_test_1000

  # === SOKOBAN ===
  sokoban_5x5_v2_1000
  sokoban_5x5_v2_static_dsg_1000
  sokoban_gen_test_1000
  sokoban_gen_test_2_1000
  sokoban_new_format_1000
  sokoban_new_format_gen_test_1000
  sokoban_new_format_gen_test_2_1000
)

BASE_FCNN_DIR="${LOG_ROOT:-./logs}/fcnn_test"

# To chain jobs sequentially
PREV_JOB_ID=""

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=== FCNN TEST DATASET: ${DATASET_NAME} ==="

    JOB_DIR="${BASE_FCNN_DIR}/${DATASET_NAME}"
    mkdir -p "${JOB_DIR}"

    # Per-dataset frozen config snapshots
    AC_CFG="${JOB_DIR}/ac_config.yaml"
    DSG_CFG="${JOB_DIR}/dsg_config.yaml"
    DSET_CFG="${JOB_DIR}/dset_config.yaml"

    cp "${AC_SRC}"   "${AC_CFG}"
    cp "${DSG_SRC}"  "${DSG_CFG}"
    cp "${DSET_SRC}" "${DSET_CFG}"

    # Patch dataset (and optionally a run name) into the copied configs
    yq -y -i ".DATASET_NAME = \"${DATASET_NAME}\"" "${DSG_CFG}" || true
    # Optional: set a run name so logs/metrics differentiate FCNN runs
    RUN_NAME="fcnn_test"
    yq -y -i ".RUN_NAME  = \"${RUN_NAME}\"" "${DSG_CFG}" || true
    yq -y -i ".run_name = \"${RUN_NAME}\"" "${AC_CFG}"   || true

    EXPORT_VARS="ALL,AC_CONFIG_FILE=${AC_CFG},DSG_CONFIG_FILE=${DSG_CFG},DSET_CONFIG_FILE=${DSET_CFG}"

    # Build sbatch command, adding dependency if there is a previous job
    SBATCH_CMD=(sbatch
        --qos=rleap_deadline
        --time=00:15:00
        --export="${EXPORT_VARS}"
        --output="${JOB_DIR}/${DATASET_NAME}_test_set.txt"
    )

    if [[ -n "${PREV_JOB_ID}" ]]; then
        SBATCH_CMD+=(--dependency=afterany:${PREV_JOB_ID})
    fi

    # Submit ONLY the FCNN test-on-test-set job
    JOB_ID=$("${SBATCH_CMD[@]}" ac_dsg/slurm_jobs/fcnn_test_slurm.bash | awk '{print $4}')
    echo "Submitted FCNN TEST job ${JOB_ID} for ${DATASET_NAME}"

    # Chain next dataset after this one finishes
    PREV_JOB_ID="${JOB_ID}"
    echo
done

echo "=== ALL FCNN TEST SWEEP JOBS SUBMITTED SEQUENTIALLY ==="
