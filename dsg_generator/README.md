# Dynamic Scene Graph Generation

## Installation

For detailed installation instructions, refer to [Installation.md](Installation.md). And for training the faster-RCNN submodule, refer to [fasterRCNN](fasterRCNN).

## Running
In order to run the Dynamic Scene Graph Generator, the fasterRCNN object detector has to be trained first:

```bash
cd dsg_generator
```


### Training fasterRCNN
```bash
conda activate ac_dsg
./clean_cache.sh
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python fasterRCNN/trainval_net.py --dataset blocks_voc --net res101 --bs 2 --nw 0 --lr 0.0001 --lr_decay_step 5 --cuda
```

### Testing fasterRCNN
```bash
conda activate ac_dsg
./clean_cache.sh
export PYTHONPATH=$(pwd)
python fasterRCNN/test_net.py --dataset blocks_voc --net res101 --cuda
```
Adding the `--vis` enables visualizaiton (helpful for assessment later on). It renders the results of object detections to png files in the directory.

---
Now, if the fasterRCNN model works with a decent accuracy, the Dynamic Scene Graph Generation module can be trained as well:

### Training DSG-Generator
You can train the **DSG-DETR** with train.py:
```bash
conda activate ac_dsg
python train.py
```

### Testing DSG-Generator
You can evaluate the **DSG-DETR** with test.py.
```bash
conda activate ac_dsg
python test.py
```