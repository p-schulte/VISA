# activate conda environment
source miniconda3/bin/activate
conda activate ac_dsg

DATASET_NAME=$(basename $(yq -r '.DATASET_NAME' ac_dsg/config/yaml_files/dsg_config.yaml))
sbatch --qos=rleap_deadline  --output="/work/rleap1/paul.schulte/logs/fcnn_visualize/${DATASET_NAME}.txt" ac_dsg/slurm_jobs/fcnn_visualize.bash