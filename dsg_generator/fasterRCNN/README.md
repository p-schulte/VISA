# A *Faster* Pytorch Implementation of Faster R-CNN

## Running:
The following commands have to be executed from the base directory, i.e. `DSG-Generator`.

Training
```bash
./clean_cache.sh

export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python fasterRCNN/trainval_net.py --dataset blocks_voc --net res101 --bs 2 --nw 0 --lr 0.0001 --lr_decay_step 5 --cuda
```

Testing
```bash
./clean_cache.sh

export PYTHONPATH=$(pwd)

python fasterRCNN/test_net.py --dataset blocks_voc --net res101 --checksession 1 --checkepoch 3 --checkpoint 49 --cuda
```
`--vis` enables visualizaiton (helpful for assessment later on)

If something doesn't work, try
```bash
./clean_cache.sh
```

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
