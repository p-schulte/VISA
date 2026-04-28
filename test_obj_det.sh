conda activate ac_dsg
cd dsg_generator
./clean_cache.sh
export PYTHONPATH=$(pwd)
python -u fasterRCNN/test_net.py