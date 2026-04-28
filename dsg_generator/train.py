import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated:")
import traceback

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import CONFIG

from lib.object_detector import detector
from lib.config import Config as STTranConfig
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran
from lib.track import get_sequence
from lib.matcher import *
"""------------------------------------some settings----------------------------------------"""
sttran_config = STTranConfig()
print('Checkpoint save dir:', sttran_config.save_path)
if not os.path.exists(sttran_config.save_path):
    os.mkdir(sttran_config.save_path)
for i in sttran_config.args:
    print(i,':', sttran_config.args[i])
print()
"""-----------------------------------------------------------------------------------------"""


# load paths from config file
config = CONFIG.dsg
DATA_PATH = config['data_path']
MODEL_PATH = config['detector_model_path']

# Own dataset (BlocksWorld):
from dataloader.blocksworld import Blocksworld as BW, cuda_collate_fn as ccf
BW_dataset_train = BW(mode="train", datasize=sttran_config.datasize, data_path=DATA_PATH)
dataloader_train = torch.utils.data.DataLoader(BW_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=ccf, pin_memory=False)
BW_dataset_test = BW(mode="test", datasize=sttran_config.datasize, data_path=DATA_PATH)
dataloader_test = torch.utils.data.DataLoader(BW_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=ccf, pin_memory=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=BW_dataset_train.object_classes, use_SUPPLY=True, model_path=MODEL_PATH, mode=sttran_config.mode).to(device=device)
object_detector.eval()


model = STTran(mode=sttran_config.mode,
               spatial_class_num=len(BW_dataset_train.spatial_relationships),
               obj_classes=BW_dataset_train.object_classes,
               enc_layer_num=sttran_config.enc_layer,
               dec_layer_num=sttran_config.dec_layer,
               skip_temporal=CONFIG.dsg['skip_temporal']).to(device=device)
if sttran_config.ckpt:
    ckpt = torch.load(sttran_config.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)

evaluator = BasicSceneGraphEvaluator(mode=sttran_config.mode,
                                    BW_object_classes=BW_dataset_train.object_classes,
                                    BW_all_predicates=BW_dataset_train.relationship_classes,
                                    BW_spatial_predicates=BW_dataset_train.spatial_relationships,
                                    iou_threshold=0.5,
                                    save_file = os.path.join(sttran_config.save_path, "progress.txt"),
                                    constraint='semi', semithreshold=0.6)

# loss function, default Multi-label margin loss
if sttran_config.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if sttran_config.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=sttran_config.lr)
elif sttran_config.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=sttran_config.lr)
elif sttran_config.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=sttran_config.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
print()

tr = []
matcher= HungarianMatcher(0.5,1,1,0.5)
matcher.eval()
for epoch in range(sttran_config.nepoch):
    object_detector.is_train = True
    model.train()
    object_detector.train_x = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    for b in range(len(dataloader_train)):
        data = next(train_iter)

        im_data = copy.deepcopy(data[0]).to(device)
        im_info = copy.deepcopy(data[1]).to(device)
        gt_boxes = copy.deepcopy(data[2]).to(device)
        num_boxes = copy.deepcopy(data[3]).to(device)
        gt_annotation = BW_dataset_train.gt_annotations[data[4]]

        # prevent gradients to FasterRCNN
        with torch.no_grad():
            try:
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            except Exception as e:
                print("Error in object detector:", e)
                traceback.print_exc()
                continue


        # visualize prediction
        VIZ = False
        if VIZ:
            import lib.visualize_predictions as viz_pred
            viz_pred.visualize_object_detection(im_data, entry, BW_dataset_train, filename="predictions_visualization.png")

            
        # this only edits/adds the 'indices' key in the 'entry' dictionary which represents the *TRACKLETS* of the objects
        get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_config.mode) 


        # predict the scene graph
        pred = model(entry)


        VIZ_PREDICTIONS = False
        if VIZ_PREDICTIONS:
            import lib.visualize_predictions as viz_pred
            viz_pred.visualize_predictions_console(pred, BW_dataset_train)

        spatial_distribution = pred["spatial_distribution"]

        if not sttran_config.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), spatial_distribution.shape[1]], dtype=torch.long).to(device=spatial_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), spatial_distribution.shape[1]], dtype=torch.float32).to(device=spatial_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1

        losses = {}
        losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        criterion = torch.nn.BCEWithLogitsLoss()
        losses['spatial_relation_loss'] = criterion(spatial_distribution, spatial_label.float())


        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))


        print("loss:", loss.item())
        time_per_batch = (time.time() - start) / 1000
        print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                            time_per_batch, len(dataloader_train) * time_per_batch / 60))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start = time.time()

    if True:
        torch.save({"state_dict": model.state_dict()}, os.path.join(sttran_config.save_path, "model_{}.tar".format(epoch)))
        print("*" * 40)
        
        print("save the checkpoint after {} epochs".format(epoch))
        with open(evaluator.save_file, "a") as f:
            f.write("save the checkpoint after {} epochs\n".format(epoch))
        model.eval()
        object_detector.is_train = False
        with torch.no_grad():
            for b in range(len(dataloader_test)):
                data = next(test_iter)
                im_data = copy.deepcopy(data[0]).to(device)
                im_info = copy.deepcopy(data[1]).to(device)
                gt_boxes = copy.deepcopy(data[2]).to(device)
                num_boxes = copy.deepcopy(data[3]).to(device)
                gt_annotation = BW_dataset_test.gt_annotations[data[4]]
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, sttran_config.mode)
                pred = model(entry)
                

                # visualize prediction
                VIZ = False
                if VIZ:
                    import lib.visualize_predictions as viz_pred
                    viz_pred.visualize_object_detection(im_data, entry, BW_dataset_train, filename="predictions_visualization_test.png")

                evaluator.evaluate_scene_graph(gt_annotation, pred)
            print('-----------', flush=True)
        score = np.mean(evaluator.result_dict[sttran_config.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)
    
