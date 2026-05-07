#!/bin/bash
set -euo pipefail

# Activate conda
source miniconda3/bin/activate
conda activate ac_dsg

# Template config files (singletons)
AC_SRC="ac_dsg/config/yaml_files/ac_config.yaml"
DSG_SRC="ac_dsg/config/yaml_files/dsg_config.yaml"
DSET_SRC="ac_dsg/config/yaml_files/dset_config.yaml"

# All datasets you want to sweep over
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


# Base log dir for DSG tests
BASE_DSG_TEST_DIR="${LOG_ROOT:-./logs}/dsg_test"

for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=== DSG TEST DATASET: ${DATASET_NAME} ==="

    # Choose a run name
    RUN_NAME="final"  # or e.g. "dsg_${DATASET_NAME}"

    # Per-dataset, per-run directory
    JOB_DIR="${BASE_DSG_TEST_DIR}/${DATASET_NAME}/${RUN_NAME}"
    mkdir -p "${JOB_DIR}"

    # Snapshot configs for THIS DSG test job
    AC_CFG_TEST="${JOB_DIR}/ac_config.yaml"
    DSG_CFG_TEST="${JOB_DIR}/dsg_config.yaml"
    DSET_CFG_TEST="${JOB_DIR}/dset_config.yaml"

    cp "${AC_SRC}"   "${AC_CFG_TEST}"
    cp "${DSG_SRC}"  "${DSG_CFG_TEST}"
    cp "${DSET_SRC}" "${DSET_CFG_TEST}"

    # Patch dataset + run name into the copied configs
    yq -y -i ".DATASET_NAME = \"${DATASET_NAME}\"" "${DSG_CFG_TEST}"
    yq -y -i ".RUN_NAME     = \"${RUN_NAME}\""     "${DSG_CFG_TEST}"
    # If ac_config also tracks run name:
    yq -y -i ".run_name     = \"${RUN_NAME}\""     "${AC_CFG_TEST}"

    # Submit DSG test job with frozen configs
    JOB_ID=$(
      sbatch \
        --qos=rleap_deadline \
        --time=01:00:00 \
        --output="${JOB_DIR}/run.txt" \
        --export=ALL,AC_CONFIG_FILE="${AC_CFG_TEST}",DSG_CONFIG_FILE="${DSG_CFG_TEST}",DSET_CONFIG_FILE="${DSET_CFG_TEST}" \
        ac_dsg/slurm_jobs/dsg_test_slurm.bash \
      | awk '{print $4}'
    )

    echo "Submitted DSG TEST job ${JOB_ID} for ${DATASET_NAME}"
    echo
done

echo "=== ALL DSG TEST JOBS SUBMITTED ==="
