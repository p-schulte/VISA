# Detailed Overview and Instructions for Running VISA

---

## Overview

This repository contains two main components:

1. **Dataset Generator** – generates synthetic datasets with annotated object interactions  
2. **VISA Pipeline** – learns symbolic domains directly from image sequences  

The VISA pipeline consists of the following stages:

1. Object Detection  
2. Dynamic Scene Graph Generation  
3. Action Classification  
4. Domain Learning  

---

## Repository Structure

```
config/
dataset/
dsg_generator/
action_classification/
resouces/
```

### `config/`
Contains YAML configuration files and a loader script.  
This is the central place for adjusting parameters at runtime.

### `dataset/`
Contains dataset generation code and generated datasets.

### `dsg_generator/`
Contains object detection based on Faster R-CNN and dynamic scene graph generation based on DSG-DETR.  
[ADD CITATION / SOURCE CODE LINK]

### `action_classification/`
Contains training and evaluation code for action classification models based on dynamic scene graphs.

### `resources/`
This folder should contain our generated datasets and trained models. You can download them here: **TODO**.


---

## 📦 Installation

### 0. Clone the Repository
```bash
git clone https://github.com/VISA-sys/VISA
cd VISA
```

### 1. Setup Conda Environments
You will need two environments: one for the DSG generator and one for dataset generation.

#### Dataset Generator
```bash
conda env create -f dataset/environment.yaml
```

#### DSG Generator
Choose the appropriate environment file:
```bash
conda env create -f dsg_generator/environment_pc.yaml
```

##### 2. Verify Installation
```bash
conda activate ac_dsg
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

##### 3. Compile Required Libraries

###### Bounding Box Utilities
```bash
conda activate ac_dsg
cd dsg_generator
cd lib/draw_rectangles
python setup.py build_ext --inplace
python setup.py build_ext --inplace # validate that it runs without errors
cd ..
```
and
```bash
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
python setup.py build_ext --inplace # validate that it runs without errors
cd ../../..
```

###### Faster R-CNN C++ Extensions
```bash
conda activate ac_dsg
cd dsg_generator
# create data folder
cd fasterRCNN && mkdir data

# compile libraries
cd lib
python setup.py build develop
```
To validate the installation, execute `pip list` and check whether fasterRCNN is listed (with a path to the directory, as it is a locally built python lib).

##### 4. Download Pretrained Models
Download `resnet101` from [Google Drive](https://drive.google.com/file/d/0B7fNdx_jAqhtaXZ4aWppWV96czg/view?resourcekey=0-wQUfmnu6ntfoiyaR_i9tNw) and place it in:
```
dsg_generator/fasterRCNN/data/tmp/pretrained_model/
```

---

## Running the Project

The overall workflow consists of two main steps:

1. Dataset generation  
2. Running the VISA pipeline  

---

## 1. Dataset Generation

A new dataset can be generated using:

```bash
conda activate ac_dataset
cd dataset
python generate.py
```

This command loads the following configuration file:

```bash
config/yaml_files/dset_config.yaml
```

This configuration specifies which environment configuration to load, for example `"blocks"`, from:

```bash
dataset/configs/blocks/easy.json
```

The JSON file defines:
- number of sequences / images  
- object count distributions  
- domain-specific parameters  

The generated dataset is written to:

```
dataset/output/<domain_name>/
```

### Dataset Structure

```
dataset/output/<domain_name>/
├── annotations/
├── frames/
└── ImageSets/
```

### `annotations/`
Contains supervision data, including:
- XML annotations for Faster R-CNN  
- `object_bbox_and_relationships.json`  
- `action_descriptions.json`  

### `frames/`
Contains generated image frames.

### `ImageSets/`
Contains train/test splits in Faster R-CNN format.

---

## 2. Running the VISA Pipeline

Once the datasets are in place, the VISA pipeline can be executed.  
The pipeline proceeds in four stages: first the object detector is trained, then the DSG generator, then the action identification model, and finally the domain learning component.

---

### A: Object Detector

As with the DSG generator, the object detector is located in the `dsg_generator` folder. More specifically, the relevant implementation can be found in the `fasterRCNN` subfolder, which contains the files `trainval_net.py` and `test_net.py`.

Before training Faster R-CNN, please select the targeted dataset using the `dsg_config.yaml` file in the configuration directory mentioned above.

Training can then be started with:

```bash
conda activate ac_dsg
cd dsg_generator
./clean_cache.sh
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python -u fasterRCNN/trainval_net.py
```

Once training has finished, the final model `.pth` file can again be specified in `dsg_config.yaml`. Testing can then be run via:

```bash
conda activate ac_dsg
cd dsg_generator
./clean_cache.sh
export PYTHONPATH=$(pwd)
python -u fasterRCNN/test_net.py
```

The considered metric is average precision (AP), reported per object class as well as an overall score.

Once the object detector is in place, it can be used to build the DSG generator.

---

### B: DSG Generator

Once the object detector has been trained and is available, the DSG generator can easily be trained using:

```bash
conda activate ac_dsg
cd dsg_generator
python train.py
```

Testing works analogously:

```bash
conda activate ac_dsg
cd dsg_generator
python test.py
```

The reported metric is the F1 score, averaged over all relations and also reported per relation. This is important to avoid misleading conclusions caused by class imbalance.

---

### C: Action Identification

Once DSG generation is in place, the action identification module can be trained. Run:

```bash
cd action_classification
conda activate ac_dsg
export PYTHONPATH=$(cd ../dsg_generator && pwd)
python train.py
```

Testing can be run via:

```bash
cd action_classification
conda activate ac_dsg
export PYTHONPATH=$(cd ../dsg_generator && pwd)
python test.py
```

The reported metric is accuracy, both averaged and class-balanced.

---

### D: Domain Learning

**TODO**: add information here

---

## References

This repository is based on [pddlgym](https://github.com/tomsilver/pddlgym), [DSG-DETR](https://github.com/Shengyu-Feng/DSG-DETR), and [ST-GCN](https://github.com/yysijie/st-gcn).