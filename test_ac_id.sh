cd action_classification
conda activate ac_dsg
export PYTHONPATH=$(cd ../dsg_generator && pwd)
python test.py