# Installation Guide

## 0. Clone the repository
First of all, clone the repository and change the directory:
```bash
git clone https://github.com/VISA-sys/VISA
cd VISA
```


## 1. Create Environment for DSG-Generator
Depending on whether you want to run the code on you PC (with graphics card), on a cluster (with a newer graphics card), or on a CPU only, create a conda environment using one of the three given `environment.yaml` files. For example:
```bash
conda env create -f dsg_generator/environment_pc.yaml
conda activate ac_dsg
```


## 2. Verify Installation
Run the following line with the new conda environment activated to validate the installation and optionally whether a GPU is available:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```


## 3. Compile BBox Operations
Compile the necessary bounding box operations by running the following commands:
```bash
cd dsg_generator
cd lib/draw_rectangles
python setup.py build_ext --inplace
python setup.py build_ext --inplace # validate that it runs without errors
cd ..
```
AND:
```bash
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
python setup.py build_ext --inplace # validate that it runs without errors
cd ../../..
```

## 4. Compile Faster R-CNN libraries:
Now fasterRCNN must be compiled.
```bash
# create data folder
cd fasterRCNN && mkdir data

# compile libraries
cd lib
python setup.py build develop
```
To validate the installation, execute `pip list` and check whether fasterRCNN is listed (with a path to the directory, as it is a locally built python lib).

## 5. Downloading pretrained models:
In order to run fasterRCNN with our settings, the resnet101 model is needed. It can be downloaded from [this link](https://drive.google.com/file/d/0B7fNdx_jAqhtaXZ4aWppWV96czg/view?resourcekey=0-wQUfmnu6ntfoiyaR_i9tNw). It should be placed inside `dsg_generator/fasterRCNN/data/tmp/pretrained_model/`.