# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

DATASET_NAME=$(basename $(yq -r '.DATASET_NAME' ac_dsg/config/yaml_files/dsg_config.yaml))
if [ ! -d "/work/rleap1/paul.schulte/logs/ac_train/${DATASET_NAME}" ]; then
    echo "Directory /work/rleap1/paul.schulte/logs/ac_train/${DATASET_NAME} does not exist. Exiting."
    exit 1
fi

sbatch --output="/work/rleap1/paul.schulte/logs/misc/${DATASET_NAME}_viz.txt" ac_dsg/slurm_jobs/training_viz_ac_train_slurm.bash
